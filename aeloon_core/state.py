"""Lightweight, explicit state shared by UASM agent nodes."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

from aeloon_core.loop_guard import LoopGuardDecision, LoopGuardState
from aeloon_core.transitions import TokenLedger, TransitionRecord

if TYPE_CHECKING:
    from aeloon_core.providers.base import LLMResponse, ToolCallRequest
    from aeloon_core.task_graph import TaskNode
    from aeloon_core.temporary_guard import GuardEvidence

Message = dict[str, Any]
LazyLoader = Callable[[], Any]
_UNRESOLVED = object()
_SAFE_REF_PREFIX = re.compile(r"[^a-zA-Z0-9._-]+")


class AgentNode(StrEnum):
    """Nodes in the MVP state-machine execution graph."""

    MASTER = "master"
    WORKER = "worker"
    CONTROL = "control"
    TOOL = "tool"
    TEMPORARY_GUARD = "temporary_guard"
    DONE = "done"


class RunStatus(StrEnum):
    """Lifecycle status, kept separate from the next node to execute."""

    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    TERMINATED_BY_RULE = "terminated_by_rule"
    TERMINATED_BY_GUARD = "terminated_by_guard"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.TERMINATED_BY_RULE,
            RunStatus.TERMINATED_BY_GUARD,
            RunStatus.FAILED,
        }


@dataclass(frozen=True)
class ProfileRef:
    """Immutable artifact provenance pinned for one runtime turn."""

    profile_id: str
    revision: int
    artifact_id: str
    generation: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "profile_id": self.profile_id,
            "revision": self.revision,
            "artifact_id": self.artifact_id,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class PendingHandoff:
    """Validated handoff intent waiting for profile-master routing."""

    from_agent_id: str
    summary: str
    recommended_agent_id: str | None = None


@dataclass
class StateMetadata:
    """Small, typed control plane for a ``LightweightState``."""

    phase: AgentNode = AgentNode.MASTER
    status: RunStatus = RunStatus.RUNNING
    iteration: int = 0
    finalization_iteration: int = 0
    finalization_prompt: Message | None = None
    final_content: str | None = None
    termination_reason: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase = AgentNode(self.phase)
        self.status = RunStatus(self.status)
        self.iteration = max(0, int(self.iteration))
        self.finalization_iteration = max(0, int(self.finalization_iteration))

    @property
    def is_terminal(self) -> bool:
        return self.status.terminal

    def finish(
        self,
        *,
        status: RunStatus,
        final_content: str,
        reason: str | None = None,
    ) -> None:
        """Move to the terminal node with an explicit visible final result."""

        resolved_status = RunStatus(status)
        if not resolved_status.terminal:
            raise ValueError(f"terminal status required, got {resolved_status.value}")
        if not final_content.strip():
            raise ValueError("terminal states require visible final_content")
        self.phase = AgentNode.DONE
        self.status = resolved_status
        self.final_content = final_content
        self.termination_reason = reason


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=_stable_json)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonicalize(model_dump())
    public_attributes = {
        key: item
        for key, item in vars(value).items()
        if not key.startswith("_") and not callable(item)
    }
    if public_attributes:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": _canonicalize(public_attributes),
        }
    raise TypeError(f"cannot create a stable digest for {type(value).__qualname__}")


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_hash(value: Any) -> str:
    payload = _stable_json(_canonicalize(value)).encode()
    return hashlib.sha256(payload).hexdigest()


def _value_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    canonical = _canonicalize(value)
    serialized = _stable_json(canonical)
    return {
        "type": type(value).__qualname__,
        "length": len(serialized),
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


@dataclass
class LazyValue:
    """A content-addressed value that may be resolved on first access."""

    ref: str
    value: Any = field(default=_UNRESOLVED, repr=False)
    loader: LazyLoader | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValueError("lazy ref must not be empty")

    @classmethod
    def from_value(cls, value: Any, *, prefix: str = "context") -> LazyValue:
        digest = _stable_hash(value)
        return cls(ref=_lazy_ref(prefix, digest), value=value)

    @classmethod
    def deferred(cls, ref: str, loader: LazyLoader) -> LazyValue:
        return cls(ref=ref, loader=loader)

    @property
    def is_loaded(self) -> bool:
        return self.value is not _UNRESOLVED

    def resolve(self) -> Any:
        if self.is_loaded:
            return self.value
        if self.loader is None:
            raise LookupError(f"no loader registered for {self.ref}")
        value = self.loader()
        if inspect.isawaitable(value):
            raise TypeError("LazyValue loaders must be synchronous")
        self.value = value
        return value

    def to_reference(self) -> dict[str, str]:
        return {"$ref": self.ref}

    def to_digest_value(self) -> dict[str, str]:
        """Keep state digests stable before and after deferred loading."""

        return {"ref": self.ref}


def _lazy_ref(prefix: str, digest: str) -> str:
    clean_prefix = _SAFE_REF_PREFIX.sub("-", prefix.strip()).strip("-") or "context"
    return f"lazy://{clean_prefix}/{digest}"


@dataclass
class LightweightState:
    """Canonical UASM data passed through every explicit agent-node transition."""

    messages: list[Message]
    minimal_context: list[Message] | None = None
    permissions: dict[str, Any] = field(default_factory=dict)
    active_tools: list[str] = field(default_factory=list)
    metadata: StateMetadata = field(default_factory=StateMetadata)
    guard_state: LoopGuardState = field(default_factory=LoopGuardState)
    pending_response: LLMResponse | None = None
    pending_tool_calls: list[ToolCallRequest] = field(default_factory=list)
    pending_tool_nodes: list[TaskNode] = field(default_factory=list)
    pending_guard_evidence: GuardEvidence | None = None
    pending_guard_fallback: LoopGuardDecision | None = None
    final_emitted: bool = False
    tools_used: list[str] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    lazy_values: dict[str, LazyValue] = field(default_factory=dict, repr=False)
    profile_ref: ProfileRef | None = None
    active_agent_id: str | None = None
    resume_agent_id: str | None = None
    handoff_count: int = 0
    pending_handoff: PendingHandoff | None = None
    pending_control_call: ToolCallRequest | None = None
    pending_profile_correction: str | None = None
    control_protocol_retries: int = 0

    def __post_init__(self) -> None:
        if self.minimal_context is None:
            self.minimal_context = list(self.messages)
        self.handoff_count = max(0, int(self.handoff_count))
        self.control_protocol_retries = max(0, int(self.control_protocol_retries))

    @classmethod
    def from_messages(
        cls,
        messages: list[Message],
        *,
        active_tools: list[str] | None = None,
        permissions: dict[str, Any] | None = None,
        max_iterations: int = 0,
        max_auto_continue_iterations: int = 0,
        max_finalization_iterations: int = 0,
        metadata: StateMetadata | None = None,
    ) -> LightweightState:
        """Build a state whose guard counters match the configured budgets."""

        return cls(
            messages=messages,
            minimal_context=list(messages),
            permissions=dict(permissions or {}),
            active_tools=list(active_tools or []),
            metadata=metadata or StateMetadata(),
            guard_state=LoopGuardState.from_limits(
                max_iterations=max_iterations,
                max_auto_continue_iterations=max_auto_continue_iterations,
                max_finalization_iterations=max_finalization_iterations,
            ),
        )

    @property
    def lazy_refs(self) -> dict[str, LazyValue]:
        """Read-only-by-convention alias exposing registered references."""

        return self.lazy_values

    def store_lazy(self, value: Any, *, prefix: str = "context") -> str:
        """Store a value by content digest and return its stable reference."""

        lazy = LazyValue.from_value(value, prefix=prefix)
        self.lazy_values.setdefault(lazy.ref, lazy)
        return lazy.ref

    def register_lazy(self, lazy: LazyValue) -> str:
        """Register an eager or deferred lazy value."""

        existing = self.lazy_values.get(lazy.ref)
        if existing is not None and existing is not lazy:
            raise ValueError(f"lazy ref already registered: {lazy.ref}")
        self.lazy_values[lazy.ref] = lazy
        return lazy.ref

    def resolve_lazy(self, ref: str) -> Any:
        try:
            lazy = self.lazy_values[ref]
        except KeyError as exc:
            raise KeyError(f"unknown lazy ref: {ref}") from exc
        return lazy.resolve()

    def stable_digest(self) -> str:
        """Hash a serializable state summary, excluding the trace itself."""

        minimal_context = self.minimal_context or []
        payload = {
            "messages": [_value_summary(message) for message in self.messages],
            "minimal_context": [_value_summary(message) for message in minimal_context],
            "permissions": _canonicalize(self.permissions),
            "active_tools": list(self.active_tools),
            "metadata": {
                "phase": self.metadata.phase.value,
                "status": self.metadata.status.value,
                "iteration": self.metadata.iteration,
                "finalization_iteration": self.metadata.finalization_iteration,
                "finalization_prompt": _value_summary(self.metadata.finalization_prompt),
                "final_content": _value_summary(self.metadata.final_content),
                "termination_reason": self.metadata.termination_reason,
                "session_id": self.metadata.session_id,
                "turn_id": self.metadata.turn_id,
                "extras": _canonicalize(self.metadata.extras),
            },
            "guard_state": self.guard_state.to_dict(),
            "pending_response": _value_summary(self.pending_response),
            "pending_tool_calls": [
                _value_summary(tool_call) for tool_call in self.pending_tool_calls
            ],
            "pending_tool_nodes": [
                _value_summary(tool_node) for tool_node in self.pending_tool_nodes
            ],
            "pending_guard_evidence": _value_summary(self.pending_guard_evidence),
            "pending_guard_fallback": _value_summary(self.pending_guard_fallback),
            "final_emitted": self.final_emitted,
            "tools_used": list(self.tools_used),
            "token_ledger": self.token_ledger.to_dict(),
            "lazy_values": {
                ref: lazy.to_digest_value()
                for ref, lazy in sorted(self.lazy_values.items())
            },
        }
        if self.profile_ref is not None:
            payload["profile"] = {
                "ref": self.profile_ref.to_dict(),
                "active_agent_id": self.active_agent_id,
                "resume_agent_id": self.resume_agent_id,
                "handoff_count": self.handoff_count,
                "pending_handoff": _value_summary(self.pending_handoff),
                "pending_control_call": _value_summary(self.pending_control_call),
                "pending_profile_correction": _value_summary(
                    self.pending_profile_correction
                ),
                "control_protocol_retries": self.control_protocol_retries,
            }
        return _stable_hash(payload)

    def digest(self) -> str:
        """Short alias used by transition recorders."""

        return self.stable_digest()

    def append_transition(self, transition: TransitionRecord) -> None:
        self.transitions.append(transition)
