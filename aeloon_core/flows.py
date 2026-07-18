"""Durable first-class dynamic flows owned by a Master session."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

FLOW_SCHEMA_VERSION = 4
FLOW_STATE_VERSION = 2
DEFAULT_FLOW_TURN_LEASE_SECONDS = 60.0
MAX_FLOW_GOAL_CHARS = 16_000
MAX_FLOW_OBJECTIVE_CHARS = 32_000
MAX_FLOW_SUMMARY_CHARS = 16_000
MAX_FLOW_RUN_BINDINGS = 128

FlowId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
FlowNodeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]


class FlowStatus(StrEnum):
    """Master-owned lifecycle, separate from individual WorkerRun states."""

    OPEN = "open"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            FlowStatus.COMPLETED,
            FlowStatus.PARTIAL,
            FlowStatus.BLOCKED,
            FlowStatus.CANCELLED,
        }


class FlowNodeStatus(StrEnum):
    """Execution state of one semantic Flow node."""

    PENDING = "pending"
    STALE = "stale"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    WAITING_FOR_CONTEXT = "waiting_for_context"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def active(self) -> bool:
        return self in {FlowNodeStatus.STARTING, FlowNodeStatus.RUNNING}

    @property
    def terminal(self) -> bool:
        """Terminal here excludes the resumable waiting state."""

        return self in {
            FlowNodeStatus.COMPLETED,
            FlowNodeStatus.PARTIAL,
            FlowNodeStatus.FAILED,
            FlowNodeStatus.CANCELLED,
            FlowNodeStatus.SKIPPED,
        }


class DependencyPolicy(StrEnum):
    """How a node interprets dependency outcomes."""

    ALL_COMPLETED = "all_completed"
    ALL_TERMINAL = "all_terminal"


class WorkerSessionPolicy(StrEnum):
    """How a semantic node chooses a WorkerSession for a new execution epoch."""

    AUTO = "auto"
    FRESH = "fresh"


class WorkerSessionAction(StrEnum):
    """The resolved WorkerSession action recorded for one WorkerRun binding."""

    NEW = "new"
    REUSE = "reuse"
    RESUME = "resume"


class FlowCompletion(StrEnum):
    """Explicit terminal decisions authored by Master."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class FlowNodeSpec(_StrictModel):
    """Immutable semantic definition supplied by Master."""

    node_id: FlowNodeId
    worker_type_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=MAX_FLOW_OBJECTIVE_CHARS)
    depends_on: tuple[FlowNodeId, ...] = Field(default=(), max_length=64)
    dependency_policy: DependencyPolicy = DependencyPolicy.ALL_COMPLETED
    worker_session_policy: WorkerSessionPolicy = WorkerSessionPolicy.AUTO

    @field_validator("depends_on")
    @classmethod
    def _dependencies_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("depends_on must not contain duplicates")
        return value


class FlowRunBinding(_StrictModel):
    """Durable audit link from one node generation to one WorkerRun."""

    generation: int = Field(ge=1)
    attempt: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    source_run_id: str | None = Field(default=None, max_length=128)
    requested_session_policy: WorkerSessionPolicy = WorkerSessionPolicy.AUTO
    session_action: WorkerSessionAction = WorkerSessionAction.NEW
    session_reason: str = Field(default="legacy_binding", min_length=1, max_length=1_000)
    status: str = Field(min_length=1, max_length=64)
    created_at: str


class FlowNode(_StrictModel):
    """Mutable execution projection for a semantic node."""

    node_id: FlowNodeId
    worker_type_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=MAX_FLOW_OBJECTIVE_CHARS)
    depends_on: tuple[FlowNodeId, ...] = Field(default=(), max_length=64)
    dependency_policy: DependencyPolicy = DependencyPolicy.ALL_COMPLETED
    worker_session_policy: WorkerSessionPolicy = WorkerSessionPolicy.AUTO
    status: FlowNodeStatus = FlowNodeStatus.PENDING
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=0, ge=0)
    revision_feedback: str | None = Field(default=None, max_length=8_000)
    input_generations: dict[str, int] = Field(default_factory=dict)
    reuse_source_run_id: str | None = Field(default=None, max_length=128)
    pending_fresh_reason: str | None = Field(default=None, max_length=1_000)
    worker_id: str | None = Field(default=None, max_length=128)
    current_run_id: str | None = Field(default=None, max_length=128)
    result: dict[str, Any] | None = None
    runs: list[FlowRunBinding] = Field(
        default_factory=list,
        max_length=MAX_FLOW_RUN_BINDINGS,
    )

    @classmethod
    def from_spec(cls, spec: FlowNodeSpec) -> FlowNode:
        return cls(**spec.model_dump())

    def execution_objective(self) -> str:
        """Return the authoritative objective for the current generation."""

        if not self.revision_feedback:
            return self.objective
        return (
            self.objective
            + "\n\nMASTER REVISION FEEDBACK (authoritative):\n"
            + self.revision_feedback
        )


class MasterFlow(_StrictModel):
    """One appendable/revisable DAG persisted independently of Master history."""

    schema_version: int = FLOW_STATE_VERSION
    flow_id: FlowId
    base_session_id: str = Field(min_length=1, max_length=256)
    idempotency_key: str = Field(min_length=1, max_length=256)
    goal: str = Field(min_length=1, max_length=MAX_FLOW_GOAL_CHARS)
    status: FlowStatus = FlowStatus.OPEN
    revision: int = Field(default=1, ge=1)
    max_nodes: int = Field(default=64, ge=1, le=256)
    max_rounds: int = Field(default=12, ge=1, le=64)
    rounds_started: int = Field(default=0, ge=0)
    nodes: list[FlowNode] = Field(default_factory=list, max_length=256)
    cancellation_run_ids: list[str] = Field(default_factory=list, max_length=256)
    completion_summary: str | None = Field(default=None, max_length=MAX_FLOW_SUMMARY_CHARS)
    termination_reason: str | None = Field(default=None, max_length=2_000)
    created_at: str
    updated_at: str

    def node(self, node_id: str) -> FlowNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown Flow node: {node_id}")

    def ready_nodes(self) -> list[FlowNode]:
        by_id = {node.node_id: node for node in self.nodes}
        return [
            node
            for node in self.nodes
            if node.status in {FlowNodeStatus.PENDING, FlowNodeStatus.STALE}
            and _dependencies_satisfied(node, by_id)
        ]

    def starting_nodes(self) -> list[FlowNode]:
        return [node for node in self.nodes if node.status is FlowNodeStatus.STARTING]

    def active_nodes(self) -> list[FlowNode]:
        return [node for node in self.nodes if node.status.active]

    def to_view(self, *, include_results: bool = True) -> dict[str, Any]:
        """Return a bounded control-plane view safe for Master context."""

        by_id = {node.node_id: node for node in self.nodes}
        nodes: list[dict[str, Any]] = []
        for node in self.nodes:
            last_binding = node.runs[-1] if node.runs else None
            item: dict[str, Any] = {
                "node_id": node.node_id,
                "worker_type_id": node.worker_type_id,
                "objective": node.objective[:2_000],
                "depends_on": list(node.depends_on),
                "dependency_policy": node.dependency_policy.value,
                "status": node.status.value,
                "generation": node.generation,
                "attempt": node.attempt,
                "worker_id": node.worker_id,
                "run_id": node.current_run_id,
                "worker_session": {
                    "policy": node.worker_session_policy.value,
                    "pending_reuse_source_run_id": node.reuse_source_run_id,
                    "pending_fresh_reason": node.pending_fresh_reason,
                    "last_action": (
                        last_binding.session_action.value if last_binding is not None else None
                    ),
                    "last_reason": (
                        last_binding.session_reason if last_binding is not None else None
                    ),
                    "source_run_id": (
                        last_binding.source_run_id if last_binding is not None else None
                    ),
                },
            }
            if node.revision_feedback:
                item["revision_feedback"] = node.revision_feedback[:1_000]
            if include_results and node.result is not None:
                item["result"] = _bounded_result(node.result)
            nodes.append(item)
        return {
            "flow_id": self.flow_id,
            "goal": self.goal[:4_000],
            "status": self.status.value,
            "revision": self.revision,
            "rounds": {"started": self.rounds_started, "maximum": self.max_rounds},
            "nodes": nodes,
            "ready_node_ids": [node.node_id for node in self.ready_nodes()],
            "active_node_ids": [node.node_id for node in self.active_nodes()],
            "cancellation_run_count": len(self.cancellation_run_ids),
            "blocked_node_ids": [
                node.node_id
                for node in self.nodes
                if node.status in {FlowNodeStatus.PENDING, FlowNodeStatus.STALE}
                and not _dependencies_satisfied(node, by_id)
            ],
            "completion_summary": self.completion_summary,
            "termination_reason": self.termination_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FlowIdempotencyConflictError(ValueError):
    """One Flow operation key was reused for different arguments."""


class FlowSessionSealedError(ValueError):
    """A terminal Master decision already sealed this session turn."""


class FlowTurnConflictError(ValueError):
    """Another live Master turn owns this session's Flow mutation lease."""


class FlowTurnCommit(_StrictModel):
    """Durable terminal response used as the Master turn's linearization point."""

    commit_sequence: int = Field(ge=1)
    base_session_id: str = Field(min_length=1, max_length=256)
    turn_id: str = Field(min_length=1, max_length=256)
    payload: dict[str, Any]
    payload_digest: str = Field(min_length=64, max_length=64)
    created_at: str
    persisted_at: str | None = None


FlowMutation = Callable[[MasterFlow], None]


class FlowStore:
    """SQLite authority for dynamic Flow state and idempotent decisions."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "flow-control.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_flow(
        self,
        *,
        base_session_id: str,
        goal: str,
        nodes: Sequence[FlowNodeSpec],
        idempotency_key: str,
        max_nodes: int = 64,
        max_rounds: int = 12,
        turn_id: str | None = None,
    ) -> tuple[MasterFlow, bool]:
        base_session_id = _normalized_text(base_session_id, "base_session_id")
        idempotency_key = _normalized_text(idempotency_key, "idempotency_key")
        goal = _normalized_text(goal, "goal")
        if not nodes:
            raise ValueError("a Flow requires at least one initial node")
        if len(nodes) > max_nodes:
            raise ValueError("initial nodes exceed max_nodes")
        _validate_graph((), nodes)
        now = _now()
        request = {
            "goal": goal,
            "nodes": [flow_node_spec_payload(node) for node in nodes],
            "max_nodes": max_nodes,
            "max_rounds": max_rounds,
        }
        request_digest = _digest(request)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM flows WHERE base_session_id = ? AND idempotency_key = ?",
                (base_session_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if str(existing["request_digest"]) != request_digest:
                    raise FlowIdempotencyConflictError(
                        "Flow idempotency key was reused for a different create request"
                    )
                return _flow_from_row(existing), False
            self._require_unsealed_session(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            flow = MasterFlow(
                flow_id=uuid.uuid4().hex,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                goal=goal,
                max_nodes=max_nodes,
                max_rounds=max_rounds,
                nodes=[FlowNode.from_spec(spec) for spec in nodes],
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO flows(flow_id, base_session_id, idempotency_key, "
                "request_digest, status, state_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    flow.flow_id,
                    flow.base_session_id,
                    flow.idempotency_key,
                    request_digest,
                    flow.status.value,
                    _dump_flow(flow),
                    now,
                    now,
                ),
            )
            return flow, True

    def get_flow(
        self,
        flow_id: str,
        *,
        base_session_id: str | None = None,
    ) -> MasterFlow:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM flows WHERE flow_id = ?", (flow_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Flow: {flow_id}")
        flow = _flow_from_row(row)
        _validate_owner(flow, base_session_id)
        return flow

    def list_flows(
        self,
        base_session_id: str,
        *,
        include_terminal: bool = True,
    ) -> list[MasterFlow]:
        query = "SELECT * FROM flows WHERE base_session_id = ?"
        parameters: list[Any] = [base_session_id]
        if not include_terminal:
            query += " AND status IN (?, ?, ?)"
            parameters.extend(
                (
                    FlowStatus.OPEN.value,
                    FlowStatus.PAUSED.value,
                    FlowStatus.CANCELLING.value,
                )
            )
        query += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_flow_from_row(row) for row in rows]

    def mutate(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        operation: str,
        idempotency_key: str,
        payload: Any,
        mutation: FlowMutation,
        turn_id: str | None = None,
    ) -> tuple[MasterFlow, bool]:
        """Apply one model-authored state decision exactly once."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        operation = _normalized_text(operation, "operation")
        idempotency_key = _normalized_text(idempotency_key, "idempotency_key")
        payload_digest = _digest(payload)
        with self._transaction() as connection:
            row = self._required_flow_row(connection, flow_id)
            flow = _flow_from_row(row)
            _validate_owner(flow, base_session_id)
            existing = connection.execute(
                "SELECT * FROM flow_operations WHERE flow_id = ? AND idempotency_key = ?",
                (flow_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["operation"]) != operation
                    or str(existing["payload_digest"]) != payload_digest
                ):
                    raise FlowIdempotencyConflictError(
                        "Flow idempotency key was reused for a different operation"
                    )
                return flow, False
            self._require_unsealed_session(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            mutation(flow)
            flow.updated_at = _now()
            connection.execute(
                "INSERT INTO flow_operations(flow_id, idempotency_key, operation, "
                "payload_digest, created_at) VALUES (?, ?, ?, ?, ?)",
                (flow_id, idempotency_key, operation, payload_digest, flow.updated_at),
            )
            self._save(connection, flow)
            return flow, True

    def replay_operation(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        operation: str,
        idempotency_key: str,
        payload: Any,
        turn_id: str | None = None,
    ) -> MasterFlow | None:
        """Return current state for an exact completed operation replay."""

        flow, replayed = self.operation_state(
            flow_id,
            base_session_id=base_session_id,
            operation=operation,
            idempotency_key=idempotency_key,
            payload=payload,
            turn_id=turn_id,
        )
        return flow if replayed else None

    def operation_state(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        operation: str,
        idempotency_key: str,
        payload: Any,
        turn_id: str | None = None,
    ) -> tuple[MasterFlow, bool]:
        """Atomically read current Flow state and one operation replay decision."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        operation = _normalized_text(operation, "operation")
        idempotency_key = _normalized_text(idempotency_key, "idempotency_key")
        payload_digest = _digest(payload)
        # BEGIN IMMEDIATE prevents a writer from committing the operation
        # between reading Flow state and checking its idempotency record.
        with self._transaction() as connection:
            row = self._required_flow_row(connection, flow_id)
            flow = _flow_from_row(row)
            _validate_owner(flow, base_session_id)
            existing = connection.execute(
                "SELECT * FROM flow_operations WHERE flow_id = ? AND idempotency_key = ?",
                (flow_id, idempotency_key),
            ).fetchone()
            if existing is None:
                self._require_unsealed_session(
                    connection,
                    base_session_id,
                    turn_id=turn_id,
                )
                return flow, False
            if (
                str(existing["operation"]) != operation
                or str(existing["payload_digest"]) != payload_digest
            ):
                raise FlowIdempotencyConflictError(
                    "Flow idempotency key was reused for a different operation"
                )
            return flow, True

    def update_runtime(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        mutation: FlowMutation,
        turn_id: str | None = None,
    ) -> MasterFlow:
        """Persist a deterministic runtime projection without a model decision key."""

        with self._transaction() as connection:
            row = self._required_flow_row(connection, flow_id)
            flow = _flow_from_row(row)
            _validate_owner(flow, base_session_id)
            self._require_unsealed_session(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            mutation(flow)
            flow.updated_at = _now()
            self._save(connection, flow)
            return flow

    def begin_turn(
        self,
        base_session_id: str,
        turn_id: str,
        *,
        lease_seconds: float = DEFAULT_FLOW_TURN_LEASE_SECONDS,
    ) -> None:
        """Acquire one durable Master-turn lease without admitting duplicates."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM flow_session_state WHERE base_session_id = ?",
                (base_session_id,),
            ).fetchone()
            if row is not None and row["active_turn_id"] is not None:
                active_turn_id = str(row["active_turn_id"])
                expired = _lease_expired(row["lease_expires_at"], now_text)
                if not expired:
                    raise FlowTurnConflictError(
                        "another Master turn is already active for this session"
                    )
                if active_turn_id == turn_id:
                    raise FlowTurnConflictError(
                        "the Master turn lease expired and cannot be revived"
                    )
            connection.execute(
                "INSERT INTO flow_session_state(base_session_id, sealed, "
                "active_turn_id, lease_expires_at, updated_at) VALUES (?, 0, ?, ?, ?) "
                "ON CONFLICT(base_session_id) DO UPDATE SET sealed = 0, "
                "active_turn_id = excluded.active_turn_id, "
                "lease_expires_at = excluded.lease_expires_at, "
                "updated_at = excluded.updated_at",
                (base_session_id, turn_id, lease_expires_at, now_text),
            )

    def refresh_turn_lease(
        self,
        base_session_id: str,
        turn_id: str,
        *,
        lease_seconds: float = DEFAULT_FLOW_TURN_LEASE_SECONDS,
    ) -> None:
        """Renew the current turn without allowing an expired owner to revive."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE flow_session_state SET lease_expires_at = ?, updated_at = ? "
                "WHERE base_session_id = ? AND active_turn_id = ? "
                "AND lease_expires_at > ?",
                (
                    lease_expires_at,
                    now_text,
                    base_session_id,
                    turn_id,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                raise FlowTurnConflictError("the Master turn lease was lost or expired")

    def end_turn(self, base_session_id: str, turn_id: str) -> None:
        """Release only the caller's turn ownership after response persistence."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        with self._transaction() as connection:
            connection.execute(
                "UPDATE flow_session_state SET active_turn_id = NULL, "
                "lease_expires_at = NULL, updated_at = ? "
                "WHERE base_session_id = ? AND active_turn_id = ?",
                (_now(), base_session_id, turn_id),
            )

    def commit_turn_response(
        self,
        base_session_id: str,
        turn_id: str,
        payload: dict[str, Any],
    ) -> tuple[FlowTurnCommit, bool]:
        """Linearize one terminal response while the caller still owns the lease.

        The response and the final quiescence check live in the same transaction.
        A turn that loses its lease before this transaction commits cannot persist
        or return a terminal response. Exact replays return the durable commit so a
        process crash after the commit does not require rerunning the model.
        """

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        if payload.get("session_id") != base_session_id:
            raise ValueError("turn response session_id does not match its owner")
        if payload.get("turn_id") != turn_id:
            raise ValueError("turn response turn_id does not match its owner")
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_digest = hashlib.sha256(payload_json.encode()).hexdigest()

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
                (base_session_id, turn_id),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_digest"]) != payload_digest:
                    raise FlowIdempotencyConflictError(
                        "Master turn commit was replayed with a different response"
                    )
                return self._turn_commit_from_row(existing), False

            self._require_active_turn_owner(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            rows = connection.execute(
                "SELECT flow_id FROM flows WHERE base_session_id = ? "
                "AND status IN (?, ?) ORDER BY created_at",
                (
                    base_session_id,
                    FlowStatus.OPEN.value,
                    FlowStatus.CANCELLING.value,
                ),
            ).fetchall()
            open_flow_ids = [str(row["flow_id"]) for row in rows]
            if open_flow_ids:
                raise ValueError(
                    f"cannot commit the Master response while Flows remain open: {open_flow_ids}"
                )

            now = _now()
            connection.execute(
                "UPDATE flow_session_state SET sealed = 1, updated_at = ? "
                "WHERE base_session_id = ?",
                (now, base_session_id),
            )
            connection.execute(
                "INSERT INTO flow_turn_commits(base_session_id, turn_id, "
                "payload_json, payload_digest, created_at) VALUES (?, ?, ?, ?, ?)",
                (base_session_id, turn_id, payload_json, payload_digest, now),
            )
            row = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
                (base_session_id, turn_id),
            ).fetchone()
            assert row is not None
            return self._turn_commit_from_row(row), True

    def get_turn_commit(
        self,
        base_session_id: str,
        turn_id: str,
    ) -> FlowTurnCommit | None:
        """Return one already-linearized response for exact retry recovery."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
                (base_session_id, turn_id),
            ).fetchone()
        return self._turn_commit_from_row(row) if row is not None else None

    def list_unpersisted_turn_commits(
        self,
        base_session_id: str,
    ) -> list[FlowTurnCommit]:
        """Return durable responses still awaiting the JSONL history projection."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? "
                "AND persisted_at IS NULL ORDER BY commit_sequence",
                (base_session_id,),
            ).fetchall()
        return [self._turn_commit_from_row(row) for row in rows]

    def persist_turn_commit(
        self,
        base_session_id: str,
        turn_id: str,
        projector: Callable[[FlowTurnCommit], None],
    ) -> FlowTurnCommit:
        """Project one commit idempotently, then record a short durable marker.

        The projector must itself be idempotent by ``turn_id``. If the process
        stops after appending the JSONL record but before this transaction commits,
        recovery invokes it again and the existing record is verified instead of
        duplicated.
        """

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        commit = self.get_turn_commit(base_session_id, turn_id)
        if commit is None:
            raise KeyError(f"unknown Master turn commit: {turn_id}")
        if commit.persisted_at is not None:
            return commit

        # File projection is deliberately outside the SQLite writer transaction.
        # The projector serializes per session and is idempotent by turn_id, so a
        # crash or concurrent recovery can safely retry before this short marker CAS.
        projector(commit)
        with self._transaction() as connection:
            connection.execute(
                "UPDATE flow_turn_commits SET persisted_at = ? "
                "WHERE base_session_id = ? AND turn_id = ? "
                "AND persisted_at IS NULL",
                (_now(), base_session_id, turn_id),
            )
            row = self._required_turn_commit_row(
                connection,
                base_session_id,
                turn_id,
            )
            return self._turn_commit_from_row(row)

    def seal_session_if_quiescent(
        self,
        base_session_id: str,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Atomically prevent new Flow work once no live Flow remains."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._transaction() as connection:
            self._require_turn_owner(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            rows = connection.execute(
                "SELECT flow_id FROM flows WHERE base_session_id = ? "
                "AND status IN (?, ?) ORDER BY created_at",
                (
                    base_session_id,
                    FlowStatus.OPEN.value,
                    FlowStatus.CANCELLING.value,
                ),
            ).fetchall()
            open_flow_ids = [str(row["flow_id"]) for row in rows]
            if open_flow_ids:
                raise ValueError(
                    f"cannot finish the Master turn while Flows remain open: {open_flow_ids}"
                )
            now = _now()
            connection.execute(
                "INSERT INTO flow_session_state(base_session_id, sealed, "
                "active_turn_id, lease_expires_at, updated_at) "
                "VALUES (?, 1, NULL, NULL, ?) ON CONFLICT(base_session_id) "
                "DO UPDATE SET sealed = 1, updated_at = excluded.updated_at",
                (base_session_id, now),
            )

    def _initialize(self) -> None:
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > FLOW_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Flow store schema v{version} is newer than supported v{FLOW_SCHEMA_VERSION}"
                )
            if version == 0:
                connection.execute(
                    """CREATE TABLE flows (
                      flow_id TEXT PRIMARY KEY,
                      base_session_id TEXT NOT NULL,
                      idempotency_key TEXT NOT NULL,
                      request_digest TEXT NOT NULL,
                      status TEXT NOT NULL,
                      state_json TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL,
                      UNIQUE(base_session_id, idempotency_key)
                    )"""
                )
                connection.execute(
                    "CREATE INDEX flows_session_idx ON flows(base_session_id, created_at)"
                )
                connection.execute(
                    """CREATE TABLE flow_operations (
                      flow_id TEXT NOT NULL REFERENCES flows(flow_id) ON DELETE CASCADE,
                      idempotency_key TEXT NOT NULL,
                      operation TEXT NOT NULL,
                      payload_digest TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      PRIMARY KEY(flow_id, idempotency_key)
                    )"""
                )
                self._create_session_state_table(connection)
                self._create_turn_commits_table(connection)
                connection.execute(f"PRAGMA user_version={FLOW_SCHEMA_VERSION}")
            elif version == 1:
                self._create_session_state_table(connection)
                self._create_turn_commits_table(connection)
                connection.execute(f"PRAGMA user_version={FLOW_SCHEMA_VERSION}")
            elif version == 2:
                connection.execute("ALTER TABLE flow_session_state ADD COLUMN active_turn_id TEXT")
                connection.execute(
                    "ALTER TABLE flow_session_state ADD COLUMN lease_expires_at TEXT"
                )
                self._create_turn_commits_table(connection)
                connection.execute(f"PRAGMA user_version={FLOW_SCHEMA_VERSION}")
            elif version == 3:
                self._create_turn_commits_table(connection)
                connection.execute(f"PRAGMA user_version={FLOW_SCHEMA_VERSION}")
            self._validate_schema(connection)

    @staticmethod
    def _create_session_state_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS flow_session_state (
              base_session_id TEXT PRIMARY KEY,
              sealed INTEGER NOT NULL CHECK(sealed IN (0, 1)),
              active_turn_id TEXT,
              lease_expires_at TEXT,
              updated_at TEXT NOT NULL
            )"""
        )

    @staticmethod
    def _create_turn_commits_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS flow_turn_commits (
              commit_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              base_session_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              payload_digest TEXT NOT NULL,
              created_at TEXT NOT NULL,
              persisted_at TEXT,
              UNIQUE(base_session_id, turn_id)
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS flow_turn_commits_pending_idx "
            "ON flow_turn_commits(base_session_id, persisted_at, commit_sequence)"
        )

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        required = {
            "flows": {
                "flow_id",
                "base_session_id",
                "idempotency_key",
                "request_digest",
                "status",
                "state_json",
                "created_at",
                "updated_at",
            },
            "flow_operations": {
                "flow_id",
                "idempotency_key",
                "operation",
                "payload_digest",
                "created_at",
            },
            "flow_session_state": {
                "base_session_id",
                "sealed",
                "active_turn_id",
                "lease_expires_at",
                "updated_at",
            },
            "flow_turn_commits": {
                "commit_sequence",
                "base_session_id",
                "turn_id",
                "payload_json",
                "payload_digest",
                "created_at",
                "persisted_at",
            },
        }
        for table, columns in required.items():
            actual = {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            missing = columns - actual
            if missing:
                raise RuntimeError(
                    f"Flow store v{FLOW_SCHEMA_VERSION} is invalid: "
                    f"{table} missing {sorted(missing)}"
                )

    @staticmethod
    def _required_flow_row(
        connection: sqlite3.Connection,
        flow_id: str,
    ) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM flows WHERE flow_id = ?", (flow_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Flow: {flow_id}")
        return row

    @staticmethod
    def _save(connection: sqlite3.Connection, flow: MasterFlow) -> None:
        connection.execute(
            "UPDATE flows SET status = ?, state_json = ?, updated_at = ? WHERE flow_id = ?",
            (flow.status.value, _dump_flow(flow), flow.updated_at, flow.flow_id),
        )

    @staticmethod
    def _required_turn_commit_row(
        connection: sqlite3.Connection,
        base_session_id: str,
        turn_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
            (base_session_id, turn_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Master turn commit: {turn_id}")
        return row

    @staticmethod
    def _turn_commit_from_row(row: sqlite3.Row) -> FlowTurnCommit:
        return FlowTurnCommit(
            commit_sequence=int(row["commit_sequence"]),
            base_session_id=str(row["base_session_id"]),
            turn_id=str(row["turn_id"]),
            payload=json.loads(str(row["payload_json"])),
            payload_digest=str(row["payload_digest"]),
            created_at=str(row["created_at"]),
            persisted_at=(str(row["persisted_at"]) if row["persisted_at"] is not None else None),
        )

    @staticmethod
    def _require_unsealed_session(
        connection: sqlite3.Connection,
        base_session_id: str,
        *,
        turn_id: str | None,
    ) -> None:
        FlowStore._require_turn_owner(
            connection,
            base_session_id,
            turn_id=turn_id,
        )
        row = connection.execute(
            "SELECT sealed FROM flow_session_state WHERE base_session_id = ?",
            (base_session_id,),
        ).fetchone()
        if row is not None and bool(row["sealed"]):
            raise FlowSessionSealedError(
                "this Master turn is sealed; start a new turn before mutating Flows"
            )

    @staticmethod
    def _require_turn_owner(
        connection: sqlite3.Connection,
        base_session_id: str,
        *,
        turn_id: str | None,
    ) -> None:
        row = connection.execute(
            "SELECT active_turn_id, lease_expires_at FROM flow_session_state "
            "WHERE base_session_id = ?",
            (base_session_id,),
        ).fetchone()
        active_turn_id = (
            str(row["active_turn_id"])
            if row is not None and row["active_turn_id"] is not None
            else None
        )
        if active_turn_id is None:
            return
        if turn_id != active_turn_id:
            raise FlowTurnConflictError("this Flow mutation does not own the active Master turn")
        if _lease_expired(row["lease_expires_at"], _now()):
            raise FlowTurnConflictError("the active Master turn lease expired")

    @staticmethod
    def _require_active_turn_owner(
        connection: sqlite3.Connection,
        base_session_id: str,
        *,
        turn_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT active_turn_id, lease_expires_at FROM flow_session_state "
            "WHERE base_session_id = ?",
            (base_session_id,),
        ).fetchone()
        if row is None or row["active_turn_id"] is None:
            raise FlowTurnConflictError("the Master turn no longer owns this session")
        if str(row["active_turn_id"]) != turn_id:
            raise FlowTurnConflictError(
                "this terminal response does not own the active Master turn"
            )
        if _lease_expired(row["lease_expires_at"], _now()):
            raise FlowTurnConflictError("the active Master turn lease expired")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


def add_flow_nodes(flow: MasterFlow, specs: Sequence[FlowNodeSpec]) -> None:
    """Append one validated dynamic graph revision."""

    _require_open(flow)
    if not specs:
        raise ValueError("add_flow_nodes requires at least one node")
    if len(flow.nodes) + len(specs) > flow.max_nodes:
        raise ValueError(f"Flow node limit reached ({flow.max_nodes})")
    existing = [
        FlowNodeSpec(
            node_id=node.node_id,
            worker_type_id=node.worker_type_id,
            objective=node.objective,
            depends_on=node.depends_on,
            dependency_policy=node.dependency_policy,
            worker_session_policy=node.worker_session_policy,
        )
        for node in flow.nodes
    ]
    _validate_graph(existing, specs)
    flow.nodes.extend(FlowNode.from_spec(spec) for spec in specs)
    flow.revision += 1


def revise_flow_node(
    flow: MasterFlow,
    node_id: str,
    feedback: str,
    *,
    fresh_worker: bool = False,
    fresh_reason: str | None = None,
) -> None:
    """Create a new node generation and invalidate only affected descendants."""

    _require_open(flow)
    target = flow.node(node_id)
    if target.status.active or target.status is FlowNodeStatus.WAITING_FOR_CONTEXT:
        raise ValueError("an active or waiting node cannot be revised")
    _prepare_node_session_rerun(
        target,
        default_reason="same_node_revision",
        fresh_worker=fresh_worker,
        fresh_reason=fresh_reason,
    )
    target.generation += 1
    target.attempt = 0
    target.status = FlowNodeStatus.PENDING
    target.revision_feedback = feedback
    target.input_generations = {}
    target.worker_id = None
    target.current_run_id = None
    target.result = None
    _invalidate_descendants(
        flow,
        node_id,
        feedback=f"Upstream node {node_id!r} was revised: {feedback}",
    )
    flow.revision += 1


def retry_flow_node(
    flow: MasterFlow,
    node_id: str,
    *,
    fresh_worker: bool = False,
    fresh_reason: str | None = None,
) -> None:
    """Retry a technical/non-success outcome without changing its generation."""

    _require_open(flow)
    node = flow.node(node_id)
    if node.status not in {
        FlowNodeStatus.PARTIAL,
        FlowNodeStatus.FAILED,
        FlowNodeStatus.CANCELLED,
    }:
        raise ValueError("only a partial, failed, or cancelled node can be retried")
    _prepare_node_session_rerun(
        node,
        default_reason="same_node_retry",
        fresh_worker=fresh_worker,
        fresh_reason=fresh_reason,
    )
    _invalidate_descendants(
        flow,
        node_id,
        feedback=f"Upstream node {node_id!r} is being retried",
    )
    node.status = FlowNodeStatus.PENDING
    node.worker_id = None
    node.current_run_id = None
    node.result = None
    flow.revision += 1


def skip_flow_node(flow: MasterFlow, node_id: str, reason: str) -> None:
    """Explicitly waive a non-active node so successful dependencies can proceed."""

    _require_open(flow)
    node = flow.node(node_id)
    if node.status.active or node.status is FlowNodeStatus.COMPLETED:
        raise ValueError("an active or completed node cannot be skipped")
    _invalidate_descendants(
        flow,
        node_id,
        feedback=f"Upstream node {node_id!r} was explicitly skipped: {reason}",
    )
    node.status = FlowNodeStatus.SKIPPED
    node.reuse_source_run_id = None
    node.worker_id = None
    node.current_run_id = None
    node.result = {"summary": reason[:4_000], "status": FlowNodeStatus.SKIPPED.value}
    flow.revision += 1


def pause_flow(flow: MasterFlow, reason: str) -> None:
    _require_open(flow)
    if flow.active_nodes():
        raise ValueError("cannot pause while Flow nodes are active")
    flow.status = FlowStatus.PAUSED
    flow.termination_reason = reason[:2_000]


def reopen_flow(flow: MasterFlow) -> None:
    if flow.status is not FlowStatus.PAUSED:
        raise ValueError("only a paused Flow can be resumed")
    flow.status = FlowStatus.OPEN
    flow.termination_reason = None


def finish_flow(flow: MasterFlow, outcome: FlowCompletion, summary: str) -> None:
    """Apply Master's explicit terminal decision after validating live work."""

    _require_open(flow)
    if flow.active_nodes():
        raise ValueError("cannot finish while Flow nodes are active")
    if outcome is FlowCompletion.COMPLETED:
        incomplete = [
            node.node_id
            for node in flow.nodes
            if node.status not in {FlowNodeStatus.COMPLETED, FlowNodeStatus.SKIPPED}
        ]
        if incomplete:
            raise ValueError(
                "completed Flow requires every node to complete or be skipped; "
                f"incomplete={incomplete}"
            )
    else:
        for node in flow.nodes:
            if node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT:
                continue
            node.status = FlowNodeStatus.CANCELLED
            if node.result is not None:
                node.result["status"] = FlowNodeStatus.CANCELLED.value
            for binding in reversed(node.runs):
                if binding.run_id == node.current_run_id:
                    binding.status = FlowNodeStatus.CANCELLED.value
                    break
    flow.status = FlowStatus(outcome.value)
    flow.completion_summary = summary
    flow.termination_reason = None if outcome is FlowCompletion.COMPLETED else summary[:2_000]


def cancel_flow_state(
    flow: MasterFlow,
    reason: str,
    *,
    run_ids: Sequence[str] = (),
) -> None:
    if flow.status.terminal:
        return
    flow.cancellation_run_ids = list(
        dict.fromkeys(
            [
                *flow.cancellation_run_ids,
                *(node.current_run_id for node in flow.nodes if node.current_run_id),
                *run_ids,
            ]
        )
    )
    if flow.status is FlowStatus.CANCELLING:
        return
    flow.status = FlowStatus.CANCELLING
    flow.completion_summary = reason
    flow.termination_reason = reason[:2_000]
    for node in flow.nodes:
        if node.status in {
            FlowNodeStatus.PENDING,
            FlowNodeStatus.STALE,
            FlowNodeStatus.STARTING,
            FlowNodeStatus.WAITING_FOR_CONTEXT,
        }:
            node.status = FlowNodeStatus.CANCELLED
            node.reuse_source_run_id = None
            node.pending_fresh_reason = None


def _invalidate_descendants(
    flow: MasterFlow,
    node_id: str,
    *,
    feedback: str,
) -> None:
    """Invalidate transitive consumers of a changed dependency outcome."""

    affected = {node_id}
    changed = True
    while changed:
        changed = False
        for node in flow.nodes:
            if node.node_id in affected or not affected.intersection(node.depends_on):
                continue
            affected.add(node.node_id)
            changed = True
            if node.status.active or node.status is FlowNodeStatus.WAITING_FOR_CONTEXT:
                raise ValueError(
                    f"cannot change {node_id!r} while descendant "
                    f"{node.node_id!r} is active or waiting"
                )
            if node.attempt > 0 or node.status not in {
                FlowNodeStatus.PENDING,
                FlowNodeStatus.STALE,
            }:
                node.generation += 1
                node.attempt = 0
            node.status = FlowNodeStatus.STALE
            node.revision_feedback = f"Re-evaluate because {feedback}"[:8_000]
            node.input_generations = {}
            node.reuse_source_run_id = None
            node.pending_fresh_reason = "upstream_context_changed"
            node.worker_id = None
            node.current_run_id = None
            node.result = None


def _prepare_node_session_rerun(
    node: FlowNode,
    *,
    default_reason: str,
    fresh_worker: bool,
    fresh_reason: str | None,
) -> None:
    """Persist the next epoch's session intent before clearing its live binding."""

    if fresh_reason is not None and not fresh_worker:
        raise ValueError("fresh_reason requires fresh_worker=true")
    if fresh_worker:
        normalized_reason = fresh_reason.strip() if fresh_reason is not None else ""
        if not normalized_reason:
            raise ValueError("fresh_worker=true requires a concrete fresh_reason")
        node.reuse_source_run_id = None
        node.pending_fresh_reason = normalized_reason[:1_000]
        return
    if node.worker_session_policy is WorkerSessionPolicy.FRESH:
        node.reuse_source_run_id = None
        node.pending_fresh_reason = "policy_fresh"
        return
    if node.status is FlowNodeStatus.SKIPPED and node.pending_fresh_reason is not None:
        node.reuse_source_run_id = None
        return
    candidate_run_id = node.current_run_id
    if candidate_run_id is None and node.status is FlowNodeStatus.SKIPPED and node.runs:
        candidate_run_id = node.runs[-1].run_id
    node.reuse_source_run_id = candidate_run_id
    node.pending_fresh_reason = None if candidate_run_id is not None else default_reason


def _dependencies_satisfied(
    node: FlowNode,
    by_id: dict[str, FlowNode],
) -> bool:
    dependencies = [by_id[dependency] for dependency in node.depends_on]
    if node.dependency_policy is DependencyPolicy.ALL_COMPLETED:
        allowed = {FlowNodeStatus.COMPLETED, FlowNodeStatus.SKIPPED}
    else:
        allowed = {
            FlowNodeStatus.COMPLETED,
            FlowNodeStatus.PARTIAL,
            FlowNodeStatus.FAILED,
            FlowNodeStatus.CANCELLED,
            FlowNodeStatus.SKIPPED,
        }
    return all(dependency.status in allowed for dependency in dependencies)


def _validate_graph(
    existing: Sequence[FlowNodeSpec],
    additions: Sequence[FlowNodeSpec],
) -> None:
    specs = [*existing, *additions]
    ids = [spec.node_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Flow node ids must be unique")
    known = set(ids)
    for spec in specs:
        if spec.node_id in spec.depends_on:
            raise ValueError(f"Flow node {spec.node_id!r} cannot depend on itself")
        unknown = set(spec.depends_on) - known
        if unknown:
            raise ValueError(
                f"Flow node {spec.node_id!r} has unknown dependencies: {sorted(unknown)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()
    dependencies = {spec.node_id: spec.depends_on for spec in specs}

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("Flow graph must be acyclic")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in dependencies[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in ids:
        visit(node_id)


def _require_open(flow: MasterFlow) -> None:
    if flow.status is not FlowStatus.OPEN:
        raise ValueError(f"Flow is {flow.status.value}, not open")


def _validate_owner(flow: MasterFlow, base_session_id: str | None) -> None:
    if base_session_id is not None and flow.base_session_id != base_session_id:
        raise ValueError("cannot access a Flow owned by another Master session")


def _bounded_result(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "status",
        "summary",
        "artifacts",
        "evidence",
        "unresolved",
        "waiting_request",
        "tool_outcome",
        "usage",
    ):
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, str):
            result[key] = item[:4_000]
        elif isinstance(item, list):
            result[key] = [str(entry)[:500] for entry in item[:20]]
        else:
            result[key] = item
    return result


def _dump_flow(flow: MasterFlow) -> str:
    return flow.model_dump_json()


def _flow_from_row(row: sqlite3.Row) -> MasterFlow:
    payload = json.loads(str(row["state_json"]))
    version = int(payload.get("schema_version", 1))
    if version not in {1, FLOW_STATE_VERSION}:
        raise RuntimeError(f"unsupported Flow state version: {version}")
    if version == 1:
        for node in payload.get("nodes", []):
            node.setdefault("worker_session_policy", WorkerSessionPolicy.AUTO.value)
            node.setdefault("reuse_source_run_id", None)
            node.setdefault("pending_fresh_reason", None)
            for binding in node.get("runs", []):
                binding.setdefault(
                    "session_action",
                    (
                        WorkerSessionAction.RESUME.value
                        if binding.get("source_run_id") is not None
                        else WorkerSessionAction.NEW.value
                    ),
                )
                binding.setdefault("session_reason", "legacy_binding")
                binding.setdefault(
                    "requested_session_policy",
                    WorkerSessionPolicy.AUTO.value,
                )
        payload["schema_version"] = FLOW_STATE_VERSION
    return MasterFlow.model_validate(payload)


def flow_node_spec_payload(node: FlowNodeSpec) -> dict[str, Any]:
    """Serialize a node request without changing pre-v2 default idempotency digests."""

    payload = node.model_dump(mode="json")
    if node.worker_session_policy is WorkerSessionPolicy.AUTO:
        payload.pop("worker_session_policy", None)
    return payload


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be nonempty")
    return normalized


def _lease_expired(value: Any, now: str) -> bool:
    return value is None or str(value) <= now


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DependencyPolicy",
    "DEFAULT_FLOW_TURN_LEASE_SECONDS",
    "FlowCompletion",
    "FlowIdempotencyConflictError",
    "FlowSessionSealedError",
    "FlowTurnCommit",
    "FlowTurnConflictError",
    "MAX_FLOW_RUN_BINDINGS",
    "FlowNode",
    "FlowNodeSpec",
    "FlowNodeStatus",
    "FlowRunBinding",
    "FlowStatus",
    "FlowStore",
    "MasterFlow",
    "WorkerSessionAction",
    "WorkerSessionPolicy",
    "add_flow_nodes",
    "cancel_flow_state",
    "finish_flow",
    "flow_node_spec_payload",
    "pause_flow",
    "reopen_flow",
    "retry_flow_node",
    "revise_flow_node",
    "skip_flow_node",
]
