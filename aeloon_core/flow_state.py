"""Pure domain contracts and state transitions for dynamic Master Flows."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from aeloon_core.worker_state import BudgetGrant, RelatedContextSection

FLOW_SCHEMA_VERSION = 5
FLOW_STATE_VERSION = 5
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


class FlowAdvanceMode(StrEnum):
    """Whether one advance call stops after one frontier or a predictable chain."""

    CHECKPOINTED = "checkpointed"
    AUTO = "auto"


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


class FlowContextRef(_StrictModel):
    """Master-authored reference to bounded context from related work."""

    kind: Literal["worker_run", "flow_node"]
    id: str = Field(min_length=1, max_length=128)
    relation: str = Field(min_length=1, max_length=128)
    include: tuple[RelatedContextSection, ...] = (
        "objective",
        "summary",
        "artifacts",
        "evidence",
        "unresolved",
    )

    @field_validator("include")
    @classmethod
    def _sections_are_unique(
        cls,
        value: tuple[RelatedContextSection, ...],
    ) -> tuple[RelatedContextSection, ...]:
        if len(set(value)) != len(value):
            raise ValueError("context reference include sections must be unique")
        if not value:
            raise ValueError("context reference include requires at least one section")
        return value


class FlowNodeSpec(_StrictModel):
    """Immutable semantic definition supplied by Master."""

    node_id: FlowNodeId
    worker_type_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=MAX_FLOW_OBJECTIVE_CHARS)
    depends_on: tuple[FlowNodeId, ...] = Field(default=(), max_length=64)
    dependency_policy: DependencyPolicy = DependencyPolicy.ALL_COMPLETED
    worker_session_policy: WorkerSessionPolicy = WorkerSessionPolicy.AUTO
    context_refs: tuple[FlowContextRef, ...] = Field(default=(), max_length=4)

    @field_validator("depends_on")
    @classmethod
    def _dependencies_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("depends_on must not contain duplicates")
        return value

    @field_validator("context_refs")
    @classmethod
    def _context_refs_are_unique(
        cls,
        value: tuple[FlowContextRef, ...],
    ) -> tuple[FlowContextRef, ...]:
        identities = [(item.kind, item.id, item.relation) for item in value]
        if len(set(identities)) != len(identities):
            raise ValueError("context_refs must not contain duplicates")
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
    budget: BudgetGrant | None = None
    status: str = Field(min_length=1, max_length=64)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class FlowNode(_StrictModel):
    """Mutable execution projection for a semantic node."""

    node_id: FlowNodeId
    worker_type_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=MAX_FLOW_OBJECTIVE_CHARS)
    depends_on: tuple[FlowNodeId, ...] = Field(default=(), max_length=64)
    dependency_policy: DependencyPolicy = DependencyPolicy.ALL_COMPLETED
    worker_session_policy: WorkerSessionPolicy = WorkerSessionPolicy.AUTO
    context_refs: tuple[FlowContextRef, ...] = Field(default=(), max_length=4)
    status: FlowNodeStatus = FlowNodeStatus.PENDING
    generation: int = Field(default=1, ge=1)
    attempt: int = Field(default=0, ge=0)
    revision_feedback: str | None = Field(default=None, max_length=8_000)
    input_generations: dict[str, int] = Field(default_factory=dict)
    reuse_source_run_id: str | None = Field(default=None, max_length=128)
    pending_fresh_reason: str | None = Field(default=None, max_length=1_000)
    pending_budget: BudgetGrant | None = None
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
    advance_mode: FlowAdvanceMode = FlowAdvanceMode.CHECKPOINTED
    auto_advance_max_frontiers: int = Field(default=4, ge=1, le=4)
    frontier_widths: list[int] = Field(default_factory=list, max_length=64)
    auto_advanced_frontiers: int = Field(default=0, ge=0)
    last_stop_reason: str | None = Field(default=None, max_length=128)
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

    def telemetry(self) -> dict[str, Any]:
        """Derive bounded Flow telemetry from durable node results."""

        roles: dict[str, int] = {}
        models: dict[str, int] = {}
        tokens: dict[str, int] = {}
        request_count = 0
        tool_call_count = 0
        duration_ms = 0
        budget_increase_count = 0
        utilizations: list[float] = []
        partial_reasons: list[str] = []
        for node in self.nodes:
            for binding in node.runs:
                roles[node.worker_type_id] = roles.get(node.worker_type_id, 0) + 1
                telemetry = binding.telemetry
                model_name = telemetry.get("resolved_model")
                if isinstance(model_name, str) and model_name:
                    models[model_name] = models.get(model_name, 0) + 1
                request_count += _nonnegative_int(telemetry.get("request_count"))
                tool_call_count += _nonnegative_int(telemetry.get("tool_call_count"))
                duration_ms += _nonnegative_int(telemetry.get("duration_ms"))
                utilization = telemetry.get("budget_request_utilization")
                if isinstance(utilization, int | float) and utilization >= 0:
                    utilizations.append(float(utilization))
                partial_reason = telemetry.get("partial_reason")
                if isinstance(partial_reason, str) and partial_reason:
                    partial_reasons.append(partial_reason[:500])
                binding_tokens = telemetry.get("tokens")
                if isinstance(binding_tokens, dict):
                    for key, value in binding_tokens.items():
                        if isinstance(key, str):
                            tokens[key] = tokens.get(key, 0) + _nonnegative_int(value)
                budget_increase_count += _nonnegative_int(
                    telemetry.get("budget_increase_count")
                )
        return {
            "roles": roles,
            "resolved_models": models,
            "request_count": request_count,
            "tool_call_count": tool_call_count,
            "duration_ms": duration_ms,
            "tokens": tokens,
            "budget_request_utilization_mean": (
                sum(utilizations) / len(utilizations) if utilizations else None
            ),
            "partial_reasons": partial_reasons[:20],
            "budget_increase_count": budget_increase_count,
            "frontier_widths": list(self.frontier_widths),
            "auto_advanced_frontiers": self.auto_advanced_frontiers,
            "revision_count": sum(max(0, node.generation - 1) for node in self.nodes),
            "stop_reason": self.last_stop_reason,
        }

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
                "context_refs": [
                    reference.model_dump(mode="json") for reference in node.context_refs
                ],
                "budget": {
                    "pending": (
                        node.pending_budget.model_dump(mode="json")
                        if node.pending_budget is not None
                        else None
                    ),
                    "last": (
                        last_binding.budget.model_dump(mode="json")
                        if last_binding is not None and last_binding.budget is not None
                        else None
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
            "execution": {
                "advance_mode": self.advance_mode.value,
                "auto_advance_max_frontiers": self.auto_advance_max_frontiers,
                "frontier_widths": list(self.frontier_widths),
                "auto_advanced_frontiers": self.auto_advanced_frontiers,
                "last_stop_reason": self.last_stop_reason,
            },
            "telemetry": self.telemetry(),
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
            context_refs=node.context_refs,
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
    budget: BudgetGrant | None = None,
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
    target.pending_budget = budget
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
    budget: BudgetGrant | None = None,
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
    node.pending_budget = budget
    flow.revision += 1


def skip_flow_node(flow: MasterFlow, node_id: str, reason: str) -> None:
    """Explicitly waive a non-active node so successful dependencies can proceed."""

    _require_open(flow)
    node = flow.node(node_id)
    if node.status.active or node.status is FlowNodeStatus.COMPLETED:
        raise ValueError("an active or completed node cannot be skipped")
    if node.status is FlowNodeStatus.PARTIAL:
        raise ValueError(
            "a partial node cannot be skipped; retry it with a Master-authored "
            "budget increase or finish the Flow as partial"
        )
    _invalidate_descendants(
        flow,
        node_id,
        feedback=f"Upstream node {node_id!r} was explicitly skipped: {reason}",
    )
    node.status = FlowNodeStatus.SKIPPED
    node.reuse_source_run_id = None
    node.pending_budget = None
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
            node.pending_budget = None


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
            node.pending_budget = None
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

    def ancestors(node_id: str) -> set[str]:
        result: set[str] = set()
        pending = list(dependencies[node_id])
        while pending:
            dependency = pending.pop()
            if dependency in result:
                continue
            result.add(dependency)
            pending.extend(dependencies[dependency])
        return result

    for spec in specs:
        allowed_flow_nodes = ancestors(spec.node_id)
        for reference in spec.context_refs:
            if reference.kind != "flow_node":
                continue
            if reference.id not in known:
                raise ValueError(
                    f"Flow node {spec.node_id!r} references unknown context node "
                    f"{reference.id!r}"
                )
            if reference.id not in allowed_flow_nodes:
                raise ValueError(
                    f"Flow node {spec.node_id!r} may only reference an ancestor for context; "
                    f"got {reference.id!r}"
                )


def _require_open(flow: MasterFlow) -> None:
    if flow.status is not FlowStatus.OPEN:
        raise ValueError(f"Flow is {flow.status.value}, not open")


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
        "telemetry",
        "budget_increase_count",
    ):
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, str):
            result[key] = item[:4_000]
        elif isinstance(item, list):
            if key == "evidence":
                result[key] = [
                    _bounded_evidence_payload(entry)
                    for entry in item[:20]
                ]
            else:
                result[key] = [str(entry)[:500] for entry in item[:20]]
        elif key == "telemetry" and isinstance(item, dict):
            result[key] = flow_run_telemetry_payload(value)
        elif key == "usage" and isinstance(item, dict):
            result[key] = {
                str(usage_key)[:128]: _nonnegative_int(usage_value)
                for usage_key, usage_value in list(item.items())[:32]
                if isinstance(usage_value, int | float)
                and not isinstance(usage_value, bool)
            }
        else:
            result[key] = item
    return result


def _bounded_evidence_payload(value: Any) -> Any:
    if isinstance(value, str):
        text = value[:500]
        return {
            "kind": "legacy",
            "locator": text,
            "claim": text,
            "status": "observed",
            "method": None,
            "finding_id": None,
        }
    if not isinstance(value, dict):
        return str(value)[:500]
    result = dict(value)
    for key, limit in (("locator", 500), ("claim", 500), ("method", 1_000)):
        if result.get(key) is not None:
            result[key] = str(result[key])[:limit]
    return result


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0, int(value))
    return 0


def flow_run_telemetry_payload(run_view: dict[str, Any]) -> dict[str, Any]:
    """Extract the bounded telemetry persisted with one Flow Run binding."""

    raw = run_view.get("telemetry")
    telemetry = raw if isinstance(raw, dict) else {}
    usage = run_view.get("usage")
    tokens = (
        {
            str(key)[:128]: _nonnegative_int(value)
            for key, value in list(usage.items())[:32]
            if isinstance(key, str)
            and "token" in key
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        }
        if isinstance(usage, dict)
        else {}
    )
    return {
        "role": _bounded_optional_text(telemetry.get("role"), 128),
        "resolved_model": _bounded_optional_text(
            telemetry.get("resolved_model"), 256
        ),
        "request_count": _nonnegative_int(telemetry.get("request_count")),
        "tool_call_count": _nonnegative_int(telemetry.get("tool_call_count")),
        "duration_ms": _nonnegative_int(telemetry.get("duration_ms")),
        "budget_request_limit": (
            _nonnegative_int(telemetry.get("budget_request_limit")) or None
        ),
        "budget_request_utilization": _nonnegative_float(
            telemetry.get("budget_request_utilization")
        ),
        "partial_reason": _bounded_optional_text(
            telemetry.get("partial_reason"), 500
        ),
        "budget_increase_count": _nonnegative_int(
            run_view.get("budget_increase_count")
        ),
        "tokens": tokens,
    }


def _bounded_optional_text(value: Any, limit: int) -> str | None:
    return str(value)[:limit] if isinstance(value, str) and value else None


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def flow_node_spec_payload(node: FlowNodeSpec) -> dict[str, Any]:
    """Serialize a node request without changing pre-v2 default idempotency digests."""

    payload = node.model_dump(mode="json")
    if node.worker_session_policy is WorkerSessionPolicy.AUTO:
        payload.pop("worker_session_policy", None)
    if not node.context_refs:
        payload.pop("context_refs", None)
    return payload


__all__ = [
    "DependencyPolicy",
    "DEFAULT_FLOW_TURN_LEASE_SECONDS",
    "FLOW_SCHEMA_VERSION",
    "FLOW_STATE_VERSION",
    "FlowCompletion",
    "FlowAdvanceMode",
    "FlowContextRef",
    "FlowId",
    "FlowIdempotencyConflictError",
    "FlowSessionSealedError",
    "FlowTurnCommit",
    "FlowTurnConflictError",
    "MAX_FLOW_GOAL_CHARS",
    "MAX_FLOW_OBJECTIVE_CHARS",
    "MAX_FLOW_RUN_BINDINGS",
    "MAX_FLOW_SUMMARY_CHARS",
    "FlowNode",
    "FlowNodeId",
    "FlowNodeSpec",
    "FlowNodeStatus",
    "FlowRunBinding",
    "FlowStatus",
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
