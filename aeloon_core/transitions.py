"""Transition tracing and token attribution primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

from loguru import logger


class NodeKind(StrEnum):
    """Coarse node categories used for experiment-level token attribution."""

    DOMAIN = "domain"
    HARNESS = "harness"
    CONTEXT_PROCESSING = "context_processing"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_json_safe(item) for item in value), key=repr)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    return str(value)


def normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Normalize provider counters consistently across runtime surfaces."""

    normalized: dict[str, int] = {}
    for key, value in (usage or {}).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        normalized[str(key)] = max(0, int(value))

    if "total_tokens" not in normalized:
        for input_key, output_key in (
            ("prompt_tokens", "completion_tokens"),
            ("input_tokens", "output_tokens"),
        ):
            if input_key in normalized or output_key in normalized:
                normalized["total_tokens"] = normalized.get(input_key, 0) + normalized.get(
                    output_key, 0
                )
                break
    return normalized


def accumulate_usage(total: dict[str, int], usage: Mapping[str, Any] | None) -> None:
    """Add one normalized usage sample to an existing counter mapping."""

    for key, value in normalize_usage(usage).items():
        total[key] = total.get(key, 0) + value


@dataclass(frozen=True)
class TransitionRecord:
    """One explicit State -> Node -> State transition."""

    sequence: int
    iteration: int
    node: str
    node_kind: NodeKind
    before_digest: str
    after_digest: str
    session_id: str | None = None
    turn_id: str | None = None
    decision: Any = None
    token_usage: dict[str, int] = field(default_factory=dict)
    wall_time_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = 1
    component: str | None = None
    profile: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", max(0, int(self.sequence)))
        object.__setattr__(self, "iteration", max(0, int(self.iteration)))
        object.__setattr__(self, "node", str(self.node))
        object.__setattr__(self, "node_kind", NodeKind(self.node_kind))
        if self.component is not None:
            object.__setattr__(self, "component", str(self.component))
        if self.profile is not None:
            object.__setattr__(self, "profile", _json_safe(self.profile))
        object.__setattr__(self, "token_usage", normalize_usage(self.token_usage))
        object.__setattr__(self, "wall_time_ms", max(0.0, float(self.wall_time_ms)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize without leaking enum or dataclass implementation details."""

        payload = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "iteration": self.iteration,
            "node": self.node,
            "node_kind": self.node_kind.value,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "decision": _json_safe(self.decision),
            "token_usage": dict(self.token_usage),
            "wall_time_ms": self.wall_time_ms,
            "created_at": self.created_at,
        }
        if self.component is not None:
            payload["component"] = self.component
        if self.profile is not None:
            payload["profile"] = dict(self.profile)
        return payload


@dataclass
class TokenLedger:
    """Aggregate usage globally, by node category, and by exact component."""

    totals: dict[str, int] = field(default_factory=dict)
    by_node_kind: dict[NodeKind, dict[str, int]] = field(default_factory=dict)
    by_component: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(
        self,
        node_kind: NodeKind | str,
        usage: Mapping[str, Any] | None,
        *,
        component: str | None = None,
    ) -> dict[str, int]:
        """Add one usage sample and return its normalized counters."""

        kind = NodeKind(node_kind)
        normalized = normalize_usage(usage)
        bucket = self.by_node_kind.setdefault(kind, {})
        component_bucket = self.by_component.setdefault(component or kind.value, {})
        for key, value in normalized.items():
            self.totals[key] = self.totals.get(key, 0) + value
            bucket[key] = bucket.get(key, 0) + value
            component_bucket[key] = component_bucket.get(key, 0) + value
        return normalized

    add = record

    def merge(
        self,
        other: TokenLedger,
        *,
        component_prefix: str | None = None,
    ) -> dict[str, int]:
        """Merge one isolated ledger while preserving conservation and attribution."""

        normalized_totals = normalize_usage(other.totals)
        for key, value in normalized_totals.items():
            self.totals[key] = self.totals.get(key, 0) + value
        for kind, usage in other.by_node_kind.items():
            bucket = self.by_node_kind.setdefault(kind, {})
            for key, value in normalize_usage(usage).items():
                bucket[key] = bucket.get(key, 0) + value
        for component, usage in other.by_component.items():
            resolved = f"{component_prefix}:{component}" if component_prefix else component
            bucket = self.by_component.setdefault(resolved, {})
            for key, value in normalize_usage(usage).items():
                bucket[key] = bucket.get(key, 0) + value
        return normalized_totals

    def for_kind(self, node_kind: NodeKind | str) -> dict[str, int]:
        """Return a detached usage snapshot for one category."""

        return dict(self.by_node_kind.get(NodeKind(node_kind), {}))

    def for_component(self, component: str) -> dict[str, int]:
        """Return a detached usage snapshot for one runtime component."""

        return dict(self.by_component.get(component, {}))

    def is_conserved(self) -> bool:
        """Return whether every total is conserved in both attribution views."""

        keys = set(self.totals)
        return all(
            sum(bucket.get(key, 0) for bucket in self.by_node_kind.values())
            == self.totals[key]
            == sum(bucket.get(key, 0) for bucket in self.by_component.values())
            for key in keys
        )

    @property
    def total_tokens(self) -> int:
        return self.totals.get("total_tokens", 0)

    @property
    def total(self) -> dict[str, int]:
        """Compatibility alias for callers that prefer the singular name."""

        return self.totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": dict(self.totals),
            "by_node_kind": {
                kind.value: dict(usage)
                for kind, usage in sorted(self.by_node_kind.items(), key=lambda item: item[0].value)
            },
            "by_component": {
                component: dict(usage)
                for component, usage in sorted(self.by_component.items())
            },
        }


PersistTransition = Callable[[TransitionRecord], None]


class TransitionRecorder:
    """Collect transitions in memory and optionally persist each record."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        persist: PersistTransition | None = None,
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.persist = persist
        self.records: list[TransitionRecord] = []
        self.persistence_error: str | None = None

    def record(
        self,
        *,
        iteration: int,
        node: str | Enum,
        node_kind: NodeKind | str,
        component: str | None = None,
        profile: Mapping[str, Any] | None = None,
        before_digest: str,
        after_digest: str,
        decision: Any = None,
        token_usage: Mapping[str, Any] | None = None,
        wall_time_ms: float = 0.0,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> TransitionRecord:
        """Create, append, and optionally persist the next transition."""

        node_name = str(node.value) if isinstance(node, Enum) else str(node)
        transition = TransitionRecord(
            sequence=len(self.records) + 1,
            iteration=iteration,
            node=node_name,
            node_kind=NodeKind(node_kind),
            before_digest=before_digest,
            after_digest=after_digest,
            session_id=session_id if session_id is not None else self.session_id,
            turn_id=turn_id if turn_id is not None else self.turn_id,
            component=component,
            profile=dict(profile) if profile is not None else None,
            decision=decision,
            token_usage=normalize_usage(token_usage),
            wall_time_ms=wall_time_ms,
        )
        self.records.append(transition)
        self._persist(transition)
        return transition

    def append(self, transition: TransitionRecord) -> None:
        """Append an existing record, preserving its original sequence."""

        self.records.append(transition)
        self._persist(transition)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def _persist(self, transition: TransitionRecord) -> None:
        if self.persist is None:
            return
        try:
            self.persist(transition)
        except OSError as exc:
            self.persistence_error = str(exc)
            self.persist = None
            logger.warning(
                "Disabling transition persistence after trace write failed: {}",
                exc,
            )
