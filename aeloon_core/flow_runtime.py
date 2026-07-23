"""Pure runtime projections used by the Flow control service."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from aeloon_core.flow_state import (
    MAX_FLOW_RUN_BINDINGS,
    FlowNode,
    FlowNodeStatus,
    FlowRunBinding,
    FlowStatus,
    MasterFlow,
    WorkerSessionAction,
    WorkerSessionPolicy,
    flow_run_telemetry_payload,
)
from aeloon_core.worker_control import WorkerControlService
from aeloon_core.worker_state import BudgetGrant, WorkerRunStatus


def _claim_ready_frontier(flow: MasterFlow) -> None:
    if flow.status is not FlowStatus.OPEN:
        return
    if flow.starting_nodes():
        return
    ready = flow.ready_nodes()
    if not ready:
        return
    if flow.rounds_started >= flow.max_rounds:
        # Do not declare a terminal Flow while an earlier timeout=0 frontier is
        # still mutating the workspace. The current advance will join it first.
        if flow.active_nodes():
            return
        flow.status = FlowStatus.BLOCKED
        flow.completion_summary = (
            f"Flow stopped before starting another frontier because max_rounds="
            f"{flow.max_rounds} was reached."
        )
        flow.termination_reason = "Flow execution round limit reached"
        return
    generations = {node.node_id: node.generation for node in flow.nodes}
    claimed = 0
    for node in ready:
        if len(node.runs) >= MAX_FLOW_RUN_BINDINGS:
            node.status = FlowNodeStatus.FAILED
            node.reuse_source_run_id = None
            node.pending_fresh_reason = None
            node.pending_budget = None
            node.worker_id = None
            node.current_run_id = None
            node.result = {
                "status": FlowNodeStatus.FAILED.value,
                "summary": (
                    "Flow node Run history limit reached; no additional WorkerRun "
                    "was created. Finish this Flow or author a replacement node."
                ),
                "tool_outcome": "none",
            }
            continue
        node.status = FlowNodeStatus.STARTING
        node.attempt += 1
        node.input_generations = {
            dependency: generations[dependency] for dependency in node.depends_on
        }
        node.worker_id = None
        node.current_run_id = None
        node.result = None
        claimed += 1
    if claimed:
        flow.rounds_started += 1
        flow.frontier_widths.append(claimed)


def _fail_nodes_with_missing_runs(
    flow: MasterFlow,
    *,
    missing_run_ids: frozenset[str],
) -> None:
    """Project lost active/waiting Worker records into retryable unknown failures."""

    if flow.status is not FlowStatus.OPEN:
        return
    for node in flow.nodes:
        if node.current_run_id not in missing_run_ids or node.status not in {
            FlowNodeStatus.STARTING,
            FlowNodeStatus.RUNNING,
            FlowNodeStatus.WAITING_FOR_CONTEXT,
        }:
            continue
        node.status = FlowNodeStatus.FAILED
        node.result = _missing_run_view(node.current_run_id)
        for binding in reversed(node.runs):
            if binding.run_id == node.current_run_id:
                binding.status = WorkerRunStatus.FAILED.value
                break


def _missing_run_view(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": WorkerRunStatus.FAILED.value,
        "summary": (
            "The bound WorkerRun record is missing, so its execution and private "
            "context outcome are unknown. Retry with a fresh WorkerSession only after "
            "checking possible side effects."
        ),
        "artifacts": [],
        "evidence": [],
        "unresolved": ["The exact WorkerSession context can no longer be resumed."],
        "waiting_request": None,
        "tool_outcome": "unknown",
        "usage": {},
    }


def _worker_run_exists(workers: WorkerControlService, run_id: str) -> bool:
    try:
        workers.manager.store.get_run(run_id)
    except (KeyError, ValueError):
        return False
    return True


def _fail_node_for_binding_limit(
    flow: MasterFlow,
    *,
    node_id: str,
    source_run_id: str,
) -> None:
    if flow.status is not FlowStatus.OPEN:
        return
    node = flow.node(node_id)
    if (
        node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT
        or node.current_run_id != source_run_id
        or len(node.runs) < MAX_FLOW_RUN_BINDINGS
    ):
        return
    node.status = FlowNodeStatus.FAILED
    node.reuse_source_run_id = None
    node.pending_fresh_reason = None
    node.pending_budget = None
    node.worker_id = None
    node.current_run_id = None
    node.result = {
        "status": FlowNodeStatus.FAILED.value,
        "summary": (
            "Flow node Run history limit reached while adopting an external "
            "continuation. The continuation was cancelled; finish this Flow or "
            "author a replacement node."
        ),
        "tool_outcome": "unknown",
    }


def _node_status(worker_status: str) -> FlowNodeStatus:
    if worker_status in {"queued", "running"}:
        return FlowNodeStatus.RUNNING
    mapping = {
        "completed": FlowNodeStatus.COMPLETED,
        "partial": FlowNodeStatus.PARTIAL,
        "waiting_for_context": FlowNodeStatus.WAITING_FOR_CONTEXT,
        "failed": FlowNodeStatus.FAILED,
        "cancelled": FlowNodeStatus.CANCELLED,
    }
    try:
        return mapping[worker_status]
    except KeyError as exc:
        raise ValueError(f"unknown WorkerRun status: {worker_status}") from exc


def _node_run_key(flow_id: str, node: FlowNode) -> str:
    return (
        f"flow:{flow_id}:node:{node.node_id}:generation:{node.generation}:"
        f"attempt:{node.attempt}"
    )


def _requested_fresh_reason(node: FlowNode) -> str | None:
    if node.pending_fresh_reason is not None:
        return node.pending_fresh_reason
    if node.worker_session_policy is WorkerSessionPolicy.FRESH:
        return "policy_fresh"
    return None


def _reuse_session_reason(node: FlowNode) -> str:
    source_run_id = node.reuse_source_run_id
    source = next(
        (
            binding
            for binding in reversed(node.runs)
            if binding.run_id == source_run_id
        ),
        None,
    )
    if source is not None and node.generation > source.generation:
        return "same_node_revision"
    return "same_node_retry"


def _annotate_recovered_session_decision(
    node: FlowNode,
    run_view: dict[str, Any],
    *,
    fallback_reason: str | None = None,
) -> None:
    source_run_id = run_view.get("source_run_id")
    if (
        node.reuse_source_run_id is not None
        and source_run_id == node.reuse_source_run_id
    ):
        run_view["worker_session_action"] = WorkerSessionAction.REUSE.value
        run_view["worker_session_reason"] = _reuse_session_reason(node)
        return
    run_view["worker_session_action"] = WorkerSessionAction.NEW.value
    run_view["worker_session_reason"] = (
        _requested_fresh_reason(node)
        or fallback_reason
        or ("first_run" if not node.runs else "reuse_unavailable")
    )


def _resume_run_key(flow_id: str, node_id: str, source_run_id: str) -> str:
    return f"flow:{flow_id}:node:{node_id}:resume:{source_run_id}"


def _flow_turn_id(flow_id: str) -> str:
    """Stable Worker ownership key for a Flow that may outlive one Master turn."""

    return f"flow:{flow_id}"


def _flow_summary(flow: MasterFlow) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for node in flow.nodes:
        counts[node.status.value] = counts.get(node.status.value, 0) + 1
    return {
        "flow_id": flow.flow_id,
        "goal": flow.goal[:1_000],
        "status": flow.status.value,
        "revision": flow.revision,
        "rounds": {"started": flow.rounds_started, "maximum": flow.max_rounds},
        "execution": {
            "advance_mode": flow.advance_mode.value,
            "auto_advance_max_frontiers": flow.auto_advance_max_frontiers,
            "auto_advanced_frontiers": flow.auto_advanced_frontiers,
            "last_stop_reason": flow.last_stop_reason,
        },
        "telemetry": flow.telemetry(),
        "node_counts": counts,
        "ready_node_ids": [node.node_id for node in flow.ready_nodes()],
        "active_node_ids": [node.node_id for node in flow.active_nodes()],
        "fresh_worker_node_ids": [
            node.node_id
            for node in flow.nodes
            if node.worker_session_policy is WorkerSessionPolicy.FRESH
        ],
        "completion_summary": flow.completion_summary,
    }


def _attach_run_to_flow(
    flow: MasterFlow,
    *,
    node_id: str,
    generation: int,
    attempt: int,
    run_view: dict[str, Any],
) -> None:
    """Compare-and-set one durable Worker binding into an open Flow."""

    node = flow.node(node_id)
    run_id = str(run_view["run_id"])
    source_run_id = run_view.get("source_run_id")
    if flow.status is not FlowStatus.OPEN:
        raise ValueError("cannot attach a WorkerRun to a non-open Flow")
    existing = next(
        (binding for binding in node.runs if binding.run_id == run_id),
        None,
    )
    if existing is not None:
        if (
            existing.generation != generation
            or existing.attempt != attempt
            or existing.worker_id != str(run_view["worker_id"])
            or node.generation != generation
            or node.attempt != attempt
        ):
            raise ValueError("stale WorkerRun binding cannot replace the current node epoch")
        if node.current_run_id == run_id:
            return
        if node.current_run_id is not None or node.status not in {
            FlowNodeStatus.STARTING,
            FlowNodeStatus.WAITING_FOR_CONTEXT,
        }:
            raise ValueError("existing WorkerRun binding is not attachable in this state")
        node.current_run_id = run_id
        node.worker_id = str(run_view["worker_id"])
        node.status = _node_status(str(run_view.get("status") or "queued"))
        node.reuse_source_run_id = None
        node.pending_fresh_reason = None
        node.pending_budget = None
        return
    raw_action = run_view.get("worker_session_action")
    if raw_action is None:
        if node.status is FlowNodeStatus.WAITING_FOR_CONTEXT and source_run_id is not None:
            action = WorkerSessionAction.RESUME
        elif (
            node.reuse_source_run_id is not None
            and source_run_id == node.reuse_source_run_id
        ):
            action = WorkerSessionAction.REUSE
        else:
            action = WorkerSessionAction.NEW
    else:
        action = WorkerSessionAction(str(raw_action))
    if action is WorkerSessionAction.REUSE and source_run_id != node.reuse_source_run_id:
        raise ValueError("reused WorkerRun does not match the node's exact source Run")
    if (
        action is WorkerSessionAction.RESUME
        and node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT
    ):
        raise ValueError("a resumed WorkerRun can only attach to its waiting Flow node")
    session_reason = str(
        run_view.get("worker_session_reason")
        or (
            "waiting_exact_resume"
            if action is WorkerSessionAction.RESUME
            else _reuse_session_reason(node)
            if action is WorkerSessionAction.REUSE
            else _requested_fresh_reason(node)
            or ("first_run" if not node.runs else "reuse_unavailable")
        )
    )[:1_000]
    if node.generation != generation or node.attempt != attempt:
        raise ValueError("Flow node changed while its WorkerRun was being created")
    if node.status not in {
        FlowNodeStatus.STARTING,
        FlowNodeStatus.WAITING_FOR_CONTEXT,
    }:
        raise ValueError(f"cannot attach a WorkerRun while node is {node.status.value}")
    if len(node.runs) >= MAX_FLOW_RUN_BINDINGS:
        raise ValueError(f"Flow node Run history limit reached ({MAX_FLOW_RUN_BINDINGS})")
    node.worker_id = str(run_view["worker_id"])
    node.current_run_id = run_id
    node.status = _node_status(str(run_view.get("status") or "queued"))
    node.runs = [
        *node.runs,
        FlowRunBinding(
            generation=generation,
            attempt=attempt,
            worker_id=node.worker_id,
            run_id=run_id,
            source_run_id=source_run_id,
            requested_session_policy=node.worker_session_policy,
            session_action=action,
            session_reason=session_reason,
            budget=BudgetGrant.model_validate(run_view["budget"]),
            status=str(run_view.get("status") or "queued"),
            telemetry=flow_run_telemetry_payload(run_view),
            created_at=str(run_view.get("created_at") or _now()),
        ),
    ]
    node.reuse_source_run_id = None
    node.pending_fresh_reason = None
    node.pending_budget = None


def _unique(values: Sequence[str | None] | Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _now() -> str:
    return datetime.now(UTC).isoformat()
