"""Lightweight, explicit state shared by UASM agent nodes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

from aeloon_core.loop_guard import GuardEvidence, GuardRequest, GuardSource
from aeloon_core.transitions import TokenLedger, TransitionRecord

if TYPE_CHECKING:
    from aeloon_core.providers.base import LLMResponse, ToolCallRequest
    from aeloon_core.task_graph import TaskNode

Message = dict[str, Any]


class AgentNode(StrEnum):
    """Nodes in the MVP state-machine execution graph."""

    ROUTER = "router"
    MODEL = "model"
    TOOL = "tool"
    GUARD = "guard"
    DONE = "done"


class RunStatus(StrEnum):
    """Lifecycle status, kept separate from the next node to execute."""

    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    TERMINATED_BY_GUARD = "terminated_by_guard"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.TERMINATED_BY_GUARD,
            RunStatus.FAILED,
        }


@dataclass
class StateMetadata:
    """Small, typed control plane for a ``LightweightState``."""

    phase: AgentNode = AgentNode.ROUTER
    status: RunStatus = RunStatus.RUNNING
    iteration: int = 0
    iteration_limit: int = 0
    # Consecutive tool rounds that returned Error* results (reset on a clean round).
    consecutive_tool_failure_rounds: int = 0
    # How many times the host auto-extended the iteration budget without Guard.
    budget_auto_continues_used: int = 0
    # Host-owned recovery caps, keyed by a stable bounded Guard signature.
    guard_policy_retries: dict[str, int] = field(default_factory=dict)
    guard_recoveries: dict[str, int] = field(default_factory=dict)
    guard_total_recoveries: int = 0
    budget_guard_continues: int = 0
    finalization_prompt: Message | None = None
    finalization_source: GuardSource | None = None
    finalization_evidence: GuardEvidence | None = None
    final_content: str | None = None
    termination_reason: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase = AgentNode(self.phase)
        self.status = RunStatus(self.status)
        self.iteration = max(0, int(self.iteration))
        self.iteration_limit = max(0, int(self.iteration_limit))
        self.consecutive_tool_failure_rounds = max(0, int(self.consecutive_tool_failure_rounds))
        self.budget_auto_continues_used = max(0, int(self.budget_auto_continues_used))
        self.guard_policy_retries = {
            str(key): max(0, int(value))
            for key, value in self.guard_policy_retries.items()
        }
        self.guard_recoveries = {
            str(key): max(0, int(value)) for key, value in self.guard_recoveries.items()
        }
        self.guard_total_recoveries = max(0, int(self.guard_total_recoveries))
        self.budget_guard_continues = max(0, int(self.budget_guard_continues))

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
class LightweightState:
    """Canonical UASM data passed through every explicit agent-node transition."""

    messages: list[Message]
    permissions: dict[str, Any] = field(default_factory=dict)
    active_tools: list[str] = field(default_factory=list)
    metadata: StateMetadata = field(default_factory=StateMetadata)
    pending_response: LLMResponse | None = None
    pending_tool_calls: list[ToolCallRequest] = field(default_factory=list)
    pending_tool_nodes: list[TaskNode] = field(default_factory=list)
    pending_guard_request: GuardRequest | None = None
    final_emitted: bool = False
    tools_used: list[str] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    token_ledger: TokenLedger = field(default_factory=TokenLedger)

    @classmethod
    def from_messages(
        cls,
        messages: list[Message],
        *,
        active_tools: list[str] | None = None,
        permissions: dict[str, Any] | None = None,
        max_iterations: int = 0,
        metadata: StateMetadata | None = None,
    ) -> LightweightState:
        """Build a state whose iteration limit belongs to the agent loop."""

        resolved_metadata = metadata or StateMetadata()
        resolved_metadata.iteration_limit = max(1, int(max_iterations))

        return cls(
            messages=messages,
            permissions=dict(permissions or {}),
            active_tools=list(active_tools or []),
            metadata=resolved_metadata,
        )

    def stable_digest(self) -> str:
        """Hash a serializable state summary, excluding the trace itself."""

        payload = {
            "messages": [_value_summary(message) for message in self.messages],
            "permissions": _canonicalize(self.permissions),
            "active_tools": list(self.active_tools),
            "metadata": {
                "phase": self.metadata.phase.value,
                "status": self.metadata.status.value,
                "iteration": self.metadata.iteration,
                "iteration_limit": self.metadata.iteration_limit,
                "consecutive_tool_failure_rounds": self.metadata.consecutive_tool_failure_rounds,
                "budget_auto_continues_used": self.metadata.budget_auto_continues_used,
                "guard_policy_retries": dict(self.metadata.guard_policy_retries),
                "guard_recoveries": dict(self.metadata.guard_recoveries),
                "guard_total_recoveries": self.metadata.guard_total_recoveries,
                "budget_guard_continues": self.metadata.budget_guard_continues,
                "finalization_prompt": _value_summary(self.metadata.finalization_prompt),
                "finalization_source": self.metadata.finalization_source,
                "finalization_evidence": _value_summary(self.metadata.finalization_evidence),
                "final_content": _value_summary(self.metadata.final_content),
                "termination_reason": self.metadata.termination_reason,
                "session_id": self.metadata.session_id,
                "turn_id": self.metadata.turn_id,
                "extras": _canonicalize(self.metadata.extras),
            },
            "pending_response": _value_summary(self.pending_response),
            "pending_tool_calls": [
                _value_summary(tool_call) for tool_call in self.pending_tool_calls
            ],
            "pending_tool_nodes": [
                _value_summary(tool_node) for tool_node in self.pending_tool_nodes
            ],
            "pending_guard_request": _value_summary(self.pending_guard_request),
            "final_emitted": self.final_emitted,
            "tools_used": list(self.tools_used),
            "token_ledger": self.token_ledger.to_dict(),
        }
        return _stable_hash(payload)

    def digest(self) -> str:
        """Short alias used by transition recorders."""

        return self.stable_digest()

    def append_transition(self, transition: TransitionRecord) -> None:
        self.transitions.append(transition)
