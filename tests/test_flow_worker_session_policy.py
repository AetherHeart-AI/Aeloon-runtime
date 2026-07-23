"""Outcome tests for Flow-owned WorkerSession allocation policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.flow_control import FlowControlService
from aeloon_core.flows import (
    FlowCompletion,
    FlowIdempotencyConflictError,
    FlowNodeSpec,
    FlowNodeStatus,
    FlowStore,
)
from aeloon_core.master_flow_tools import build_master_flow_tools
from aeloon_core.worker_control import WorkerControlService
from aeloon_core.worker_manager import WorkerExecutionOutcome, WorkerSessionManager
from aeloon_core.worker_sessions import (
    BudgetIncrease,
    WaitingRequest,
    WorkerReport,
    WorkerRunStatus,
    WorkerStore,
)
from aeloon_core.workers import WorkerRegistry
from tests.message_helpers import checkpoint as message_checkpoint


class PolicyExecutor:
    """Return deterministic checkpoints for each policy scenario."""

    def __init__(self) -> None:
        self.unknown_attempts = 0
        self.known_failure_attempts = 0
        self.cancel_attempts = 0

    async def __call__(self, run: Any, worker: Any) -> WorkerExecutionOutcome:
        del worker
        await asyncio.sleep(0)
        objective = run.context.objective
        checkpoint = message_checkpoint(f"checkpoint for {objective}")
        if objective == "wait for context":
            request = WaitingRequest(
                summary="a decision is required",
                question="Which option should I use?",
            )
            return WorkerExecutionOutcome(
                status=WorkerRunStatus.WAITING_FOR_CONTEXT,
                report=WorkerReport(
                    summary=request.summary,
                    unresolved=(request.question,),
                ),
                tool_outcome="known",
                checkpoint=checkpoint,
                waiting_request=request,
            )
        if objective == "partial work":
            return WorkerExecutionOutcome(
                status=WorkerRunStatus.PARTIAL,
                report=WorkerReport(summary="partial result"),
                tool_outcome="known",
                checkpoint=checkpoint,
            )
        if objective == "unknown failure":
            self.unknown_attempts += 1
            if self.unknown_attempts == 1:
                return WorkerExecutionOutcome(
                    status=WorkerRunStatus.FAILED,
                    report=WorkerReport(summary="execution outcome is unknown"),
                    tool_outcome="unknown",
                )
        if objective == "known failure":
            self.known_failure_attempts += 1
            if self.known_failure_attempts == 1:
                return WorkerExecutionOutcome(
                    status=WorkerRunStatus.FAILED,
                    report=WorkerReport(summary="known failure without a checkpoint"),
                    tool_outcome="known",
                )
        if objective == "cancel once":
            self.cancel_attempts += 1
            if self.cancel_attempts == 1:
                await asyncio.Event().wait()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary=f"completed: {objective}"),
            tool_outcome="known",
            checkpoint=checkpoint,
        )


def _control(tmp_path: Path) -> FlowControlService:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = WorkerSessionManager(
        store=WorkerStore(data_dir),
        executor=PolicyExecutor(),
        max_concurrency=4,
    )
    workers = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(workspace),
    )
    return FlowControlService(store=FlowStore(data_dir), workers=workers)


def _node(
    node_id: str,
    objective: str,
    *,
    worker_type_id: str = "builder",
    depends_on: tuple[str, ...] = (),
    worker_session_policy: str = "auto",
) -> FlowNodeSpec:
    return FlowNodeSpec(
        node_id=node_id,
        worker_type_id=worker_type_id,
        objective=objective,
        depends_on=depends_on,
        worker_session_policy=worker_session_policy,
    )


async def _advance(
    control: FlowControlService,
    flow_id: str,
    *,
    turn_id: str,
) -> dict[str, Any]:
    return await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id=turn_id,
        timeout_seconds=2,
    )


def _view_node(view: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(node for node in view["nodes"] if node["node_id"] == node_id)


def _stored_node(control: FlowControlService, flow_id: str, node_id: str) -> Any:
    return control.store.get_flow(flow_id, base_session_id="master").node(node_id)


def _historical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.asyncio
async def test_independent_siblings_get_distinct_worker_sessions(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="build independent branches",
        idempotency_key="create",
        nodes=[
            _node("left", "build left"),
            _node("right", "build right"),
        ],
    )

    completed = await _advance(control, created["flow_id"], turn_id="turn-1")
    left = _view_node(completed, "left")
    right = _view_node(completed, "right")

    assert left["status"] == right["status"] == "completed"
    assert left["worker_id"] != right["worker_id"]
    assert len(control.workers.manager.store.list_workers("master")) == 2


@pytest.mark.asyncio
async def test_completed_revision_reuses_worker_with_new_sourced_run(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="revise one result",
        idempotency_key="create",
        nodes=[_node("build", "build result")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "build")

    await control.revise_node(
        flow_id,
        "build",
        feedback="apply review feedback",
        base_session_id="master",
        idempotency_key="revise",
    )
    second = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "build")
    binding = _stored_node(control, flow_id, "build").runs[-1]

    assert second["worker_id"] == first["worker_id"]
    assert second["run_id"] != first["run_id"]
    assert binding.source_run_id == first["run_id"]
    assert binding.session_action.value == "reuse"
    assert binding.session_reason == "same_node_revision"
    assert second["worker_session"]["last_action"] == "reuse"


@pytest.mark.asyncio
async def test_partial_retry_reuses_worker_session(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="retry partial work",
        idempotency_key="create",
        nodes=[_node("build", "partial work")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "build")
    assert first["status"] == "partial"

    with pytest.raises(ValueError, match="requires a Master-authored budget_increase"):
        await control.retry_node(
            flow_id,
            "build",
            base_session_id="master",
            idempotency_key="retry-without-budget",
        )
    with pytest.raises(ValueError, match="must increase from 25"):
        await control.retry_node(
            flow_id,
            "build",
            base_session_id="master",
            idempotency_key="retry-with-same-budget",
            budget_increase=BudgetIncrease(max_requests=25),
        )

    increased = await control.retry_node(
        flow_id,
        "build",
        base_session_id="master",
        idempotency_key="retry",
        budget_increase=BudgetIncrease(max_requests=50),
    )
    assert _view_node(increased, "build")["budget"]["pending"]["max_requests"] == 50
    second = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "build")
    binding = _stored_node(control, flow_id, "build").runs[-1]
    second_run = control.workers.manager.store.get_run(second["run_id"])

    assert second["worker_id"] == first["worker_id"]
    assert second["run_id"] != first["run_id"]
    assert binding.source_run_id == first["run_id"]
    assert binding.session_action.value == "reuse"
    assert binding.session_reason == "same_node_retry"
    assert binding.budget is not None
    assert binding.budget.max_requests == 50
    assert second_run.context.budget.max_requests == 50


@pytest.mark.asyncio
async def test_fresh_policy_reviewer_revision_gets_new_worker(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="perform independent reviews",
        idempotency_key="create",
        nodes=[
            _node(
                "review",
                "review result",
                worker_type_id="reviewer",
                worker_session_policy="fresh",
            )
        ],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "review")

    await control.revise_node(
        flow_id,
        "review",
        feedback="audit independently",
        base_session_id="master",
        idempotency_key="revise",
    )
    second = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "review")
    binding = _stored_node(control, flow_id, "review").runs[-1]

    assert second["worker_id"] != first["worker_id"]
    assert binding.source_run_id is None
    assert binding.session_action.value == "new"
    assert binding.session_reason == "policy_fresh"


@pytest.mark.asyncio
async def test_fresh_worker_override_applies_to_only_one_revision(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="replace polluted context once",
        idempotency_key="create",
        nodes=[_node("build", "build result")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "build")

    await control.revise_node(
        flow_id,
        "build",
        feedback="discard polluted context",
        base_session_id="master",
        idempotency_key="fresh-revision",
        fresh_worker=True,
        fresh_reason="the prior Worker context is polluted",
    )
    fresh = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "build")
    assert fresh["worker_id"] != first["worker_id"]
    assert _stored_node(control, flow_id, "build").runs[-1].source_run_id is None

    await control.revise_node(
        flow_id,
        "build",
        feedback="ordinary follow-up revision",
        base_session_id="master",
        idempotency_key="ordinary-revision",
    )
    reused = _view_node(await _advance(control, flow_id, turn_id="turn-3"), "build")
    binding = _stored_node(control, flow_id, "build").runs[-1]

    assert reused["worker_id"] == fresh["worker_id"]
    assert reused["run_id"] != fresh["run_id"]
    assert binding.source_run_id == fresh["run_id"]


@pytest.mark.asyncio
async def test_waiting_fresh_policy_still_resumes_exact_worker(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="continue an exact waiting context",
        idempotency_key="create",
        nodes=[
            _node(
                "work",
                "wait for context",
                worker_session_policy="fresh",
            )
        ],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    assert waiting["status"] == "waiting_for_context"

    resumed_view = await control.resume_node(
        flow_id,
        "work",
        response="Use option A",
        base_session_id="master",
        base_turn_id="turn-2",
        idempotency_key="resume",
        budget_increase=BudgetIncrease(max_requests=50),
    )
    resumed_run_id = _view_node(resumed_view, "work")["run_id"]
    await control.workers.await_workers(
        [resumed_run_id],
        timeout=2,
        base_session_id="master",
    )
    settled = _view_node(
        await control.inspect_flow(flow_id, base_session_id="master"),
        "work",
    )
    binding = _stored_node(control, flow_id, "work").runs[-1]

    assert settled["worker_id"] == waiting["worker_id"]
    assert settled["run_id"] != waiting["run_id"]
    assert binding.source_run_id == waiting["run_id"]
    assert binding.session_action.value == "resume"
    assert binding.session_reason == "waiting_exact_resume"
    assert binding.budget is not None
    assert binding.budget.max_requests == 50


@pytest.mark.asyncio
async def test_unknown_failed_retry_uses_fresh_worker(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover from an unknown Worker outcome",
        idempotency_key="create",
        nodes=[_node("work", "unknown failure")],
    )
    flow_id = created["flow_id"]
    failed = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    assert failed["status"] == "failed"
    assert failed["result"]["tool_outcome"] == "unknown"

    await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="retry",
    )
    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]

    assert retried["status"] == "completed"
    assert retried["worker_id"] != failed["worker_id"]
    assert binding.source_run_id is None
    assert binding.session_action.value == "new"
    assert binding.session_reason == "worker_state_unknown"


@pytest.mark.asyncio
async def test_known_failed_retry_keeps_session_with_clean_context(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="retry a known failure",
        idempotency_key="create",
        nodes=[_node("work", "known failure")],
    )
    flow_id = created["flow_id"]
    failed = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    assert failed["status"] == "failed"

    await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="retry",
    )
    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]

    assert retried["status"] == "completed"
    assert retried["worker_id"] == failed["worker_id"]
    assert binding.source_run_id == failed["run_id"]
    assert binding.session_action.value == "reuse"


@pytest.mark.asyncio
async def test_clean_cancelled_retry_keeps_worker_session(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="retry clean cancellation",
        idempotency_key="create",
        nodes=[_node("work", "cancel once")],
    )
    flow_id = created["flow_id"]
    running = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-1",
        timeout_seconds=0,
    )
    first = _view_node(running, "work")
    await asyncio.sleep(0.05)
    await control.workers.cancel_worker(
        first["run_id"],
        base_session_id="master",
    )
    cancelled = _view_node(
        await control.inspect_flow(flow_id, base_session_id="master"),
        "work",
    )
    assert cancelled["status"] == "cancelled"

    await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="retry",
    )
    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]

    assert retried["status"] == "completed"
    assert retried["worker_id"] == first["worker_id"]
    assert binding.source_run_id == first["run_id"]
    assert binding.session_action.value == "reuse"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loss_mode", "expected_reason"),
    [
        ("archived", "worker_archived"),
        ("missing", "source_run_missing"),
    ],
)
async def test_lost_worker_retry_falls_back_to_new_session(
    tmp_path: Path,
    loss_mode: str,
    expected_reason: str,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="replace an unavailable Worker",
        idempotency_key="create",
        nodes=[_node("work", "partial work")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    live_tasks = tuple(control.workers.manager._tasks.values())
    if live_tasks:
        await asyncio.gather(*live_tasks, return_exceptions=True)

    if loss_mode == "archived":
        control.workers.manager.archive_worker(first["worker_id"])
    else:
        with sqlite3.connect(control.workers.manager.store.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                "DELETE FROM worker_sessions WHERE worker_id = ?",
                (first["worker_id"],),
            )

    await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="retry",
    )
    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]

    assert retried["worker_id"] != first["worker_id"]
    assert binding.session_action.value == "new"
    assert binding.session_reason == expected_reason


@pytest.mark.asyncio
async def test_reuse_reservation_recovers_after_attach_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover an exact reuse reservation",
        idempotency_key="create",
        nodes=[_node("work", "build result")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    await control.revise_node(
        flow_id,
        "work",
        feedback="revise after a controller restart",
        base_session_id="master",
        idempotency_key="revise",
    )

    original_attach = control._attach_run

    def crash_attach(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("crash after exact reuse reservation")

    monkeypatch.setattr(control, "_attach_run", crash_attach)
    with pytest.raises(RuntimeError, match="exact reuse reservation"):
        await _advance(control, flow_id, turn_id="turn-2")
    worker_runs = control.workers.manager.store.list_runs(first["worker_id"])
    reserved_run_id = worker_runs[-1].run_id
    assert len(worker_runs) == 2

    monkeypatch.setattr(control, "_attach_run", original_attach)
    recovered = _view_node(await _advance(control, flow_id, turn_id="turn-3"), "work")

    assert recovered["worker_id"] == first["worker_id"]
    assert recovered["run_id"] == reserved_run_id
    assert len(control.workers.manager.store.list_workers("master")) == 1
    assert len(control.workers.manager.store.list_runs(first["worker_id"])) == 2


@pytest.mark.asyncio
async def test_concurrent_controllers_share_one_reuse_run(tmp_path: Path) -> None:
    control = _control(tmp_path)
    second = FlowControlService(
        store=FlowStore(control.store.path.parent),
        workers=control.workers,
    )
    created = control.create_flow(
        base_session_id="master",
        goal="reuse once across controllers",
        idempotency_key="create",
        nodes=[_node("work", "build result")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    await control.revise_node(
        flow_id,
        "work",
        feedback="one concurrent revision",
        base_session_id="master",
        idempotency_key="revise",
    )

    left, right = await asyncio.gather(
        _advance(control, flow_id, turn_id="turn-2"),
        _advance(second, flow_id, turn_id="turn-3"),
    )
    left_node = _view_node(left, "work")
    right_node = _view_node(right, "work")

    assert left_node["run_id"] == right_node["run_id"]
    assert left_node["worker_id"] == right_node["worker_id"] == first["worker_id"]
    assert len(control.workers.manager.store.list_runs(first["worker_id"])) == 2


@pytest.mark.asyncio
async def test_v1_flow_state_migrates_without_breaking_create_replay(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    spec = _node("work", "build result")
    created = control.create_flow(
        base_session_id="master",
        goal="migrate an old Flow",
        idempotency_key="create",
        nodes=[spec],
    )
    flow_id = created["flow_id"]
    await _advance(control, flow_id, turn_id="turn-1")

    with sqlite3.connect(control.store.path) as connection:
        row = connection.execute(
            "SELECT state_json, request_digest FROM flows WHERE flow_id = ?",
            (flow_id,),
        ).fetchone()
        assert row is not None
        state = json.loads(str(row[0]))
        assert str(row[1]) == _historical_digest(
            {
                "goal": "migrate an old Flow",
                "nodes": [
                    {
                        "node_id": "work",
                        "worker_type_id": "builder",
                        "objective": "build result",
                        "depends_on": [],
                        "dependency_policy": "all_completed",
                    }
                ],
                "max_nodes": 64,
                "max_rounds": 12,
            }
        )
        state["schema_version"] = 1
        for node in state["nodes"]:
            node.pop("worker_session_policy", None)
            node.pop("reuse_source_run_id", None)
            node.pop("pending_fresh_reason", None)
            for binding in node["runs"]:
                binding.pop("requested_session_policy", None)
                binding.pop("session_action", None)
                binding.pop("session_reason", None)
        connection.execute(
            "UPDATE flows SET state_json = ? WHERE flow_id = ?",
            (json.dumps(state), flow_id),
        )

    migrated = control.store.get_flow(flow_id, base_session_id="master")
    binding = migrated.node("work").runs[-1]
    assert migrated.schema_version == 3
    assert migrated.node("work").worker_session_policy.value == "auto"
    assert binding.session_action.value == "new"
    assert binding.session_reason == "legacy_binding"

    replayed = control.create_flow(
        base_session_id="master",
        goal="migrate an old Flow",
        idempotency_key="create",
        nodes=[spec],
    )
    assert replayed["flow_id"] == flow_id
    assert replayed["created"] is False


@pytest.mark.asyncio
async def test_default_flow_mutations_keep_v1_operation_digests(tmp_path: Path) -> None:
    control = _control(tmp_path)
    retry_flow = control.create_flow(
        base_session_id="master",
        goal="old add and retry payloads",
        idempotency_key="create-retry",
        nodes=[_node("work", "known failure")],
    )
    await _advance(control, retry_flow["flow_id"], turn_id="turn-1")
    await control.add_nodes(
        retry_flow["flow_id"],
        base_session_id="master",
        nodes=[_node("after", "follow-up", depends_on=("work",))],
        idempotency_key="old-add",
    )
    await control.retry_node(
        retry_flow["flow_id"],
        "work",
        base_session_id="master",
        idempotency_key="old-retry",
    )

    revise_flow = control.create_flow(
        base_session_id="master",
        goal="old revise payload",
        idempotency_key="create-revise",
        nodes=[_node("build", "build result")],
    )
    await _advance(control, revise_flow["flow_id"], turn_id="turn-2")
    await control.revise_node(
        revise_flow["flow_id"],
        "build",
        feedback="historical feedback",
        base_session_id="master",
        idempotency_key="old-revise",
    )

    with sqlite3.connect(control.store.path) as connection:
        rows = connection.execute(
            "SELECT idempotency_key, payload_digest FROM flow_operations "
            "WHERE idempotency_key IN ('old-add', 'old-retry', 'old-revise')"
        ).fetchall()
    digests = {str(key): str(digest) for key, digest in rows}
    assert digests == {
        "old-add": _historical_digest(
            [
                {
                    "node_id": "after",
                    "worker_type_id": "builder",
                    "objective": "follow-up",
                    "depends_on": ["work"],
                    "dependency_policy": "all_completed",
                }
            ]
        ),
        "old-retry": _historical_digest({"node_id": "work"}),
        "old-revise": _historical_digest(
            {"node_id": "build", "feedback": "historical feedback"}
        ),
    }


def test_master_tool_schema_exposes_session_policy_fresh_override_and_budget_increase(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    registry = build_master_flow_tools(
        control=control,
        base_session_id="master",
        base_turn_id="turn",
    )
    definitions = {
        definition["name"]: definition["input_schema"]
        for definition in registry.get_definitions()
    }

    for tool_name in ("create_flow", "add_flow_nodes"):
        schema = definitions[tool_name]
        policy = schema["$defs"]["WorkerSessionPolicy"]
        node = schema["$defs"]["FlowNodeSpec"]
        assert policy["enum"] == ["auto", "fresh"]
        assert node["properties"]["worker_session_policy"]["default"] == "auto"
    for tool_name in ("revise_flow_node", "retry_flow_node", "resume_flow_node"):
        schema = definitions[tool_name]
        properties = schema["properties"]
        if tool_name != "resume_flow_node":
            assert properties["fresh_worker"]["default"] is False
            assert properties["fresh_reason"]["default"] is None
            assert properties["fresh_reason"]["anyOf"][0]["maxLength"] == 1_000
        assert properties["budget_increase"]["default"] is None
        increase = schema["$defs"]["BudgetIncrease"]["properties"]
        assert set(increase) == {
            "max_requests",
            "max_output_tokens",
            "max_tokens",
            "max_seconds",
            "max_tool_calls",
        }


@pytest.mark.asyncio
async def test_retry_fresh_override_is_observable_and_idempotent(tmp_path: Path) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="fresh retry once",
        idempotency_key="create",
        nodes=[_node("work", "partial work")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")

    changed = await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="fresh-retry",
        fresh_worker=True,
        fresh_reason="discard a polluted retry context",
    )
    replayed = await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="fresh-retry",
        fresh_worker=True,
        fresh_reason="discard a polluted retry context",
    )
    assert changed["changed"] is True
    assert replayed["changed"] is False
    with pytest.raises(FlowIdempotencyConflictError):
        await control.retry_node(
            flow_id,
            "work",
            base_session_id="master",
            idempotency_key="fresh-retry",
            fresh_worker=True,
            fresh_reason="a different claimed context problem",
        )

    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]
    assert retried["worker_id"] != first["worker_id"]
    assert binding.session_action.value == "new"
    assert binding.session_reason == "discard a polluted retry context"
    assert retried["worker_session"]["last_reason"] == binding.session_reason


@pytest.mark.asyncio
async def test_cancel_fences_an_inert_reservation_missed_by_its_initial_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="cancel a raced reservation",
        idempotency_key="create",
        nodes=[_node("work", "build result")],
    )
    flow_id = created["flow_id"]

    def claim(flow: Any) -> None:
        node = flow.node("work")
        node.status = FlowNodeStatus.STARTING
        node.attempt = 1

    control.store.update_runtime(
        flow_id,
        base_session_id="master",
        mutation=claim,
    )
    reserved = await control.workers.spawn_worker(
        base_session_id="master",
        base_turn_id=f"flow:{flow_id}",
        worker_type_id="builder",
        objective="build result",
        idempotency_key=f"flow:{flow_id}:node:work:generation:1:attempt:1",
        start=False,
    )
    monkeypatch.setattr(control, "_recover_cancellable_run_ids", lambda *args, **kwargs: [])

    cancelled = await control.cancel(
        flow_id,
        reason="cancel wins the reserve race",
        base_session_id="master",
        idempotency_key="cancel",
    )
    run = control.workers.manager.store.get_run(reserved["run_id"])

    assert cancelled["status"] == "cancelled"
    assert run.status is WorkerRunStatus.CANCELLED
    archived = control.workers.manager.archive_worker(reserved["worker_id"])
    assert archived.status.value == "archived"


@pytest.mark.asyncio
async def test_reconcile_preserves_an_inert_exact_waiting_continuation(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover a waiting continuation",
        idempotency_key="create",
        nodes=[_node("work", "wait for context")],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    reserved = await control.workers._reserve_flow_resume(
        waiting["run_id"],
        flow_id=flow_id,
        response="Use option A",
        idempotency_key=f"flow:{flow_id}:node:work:resume:{waiting['run_id']}",
        base_session_id="master",
    )

    swept = await control.reconcile_legacy_runs("master", wait=True)
    assert swept == []
    assert (
        control.workers.manager.store.get_run(reserved["run_id"]).status
        is WorkerRunStatus.QUEUED
    )

    await control.inspect_flow(flow_id, base_session_id="master")
    await control.workers.await_workers(
        [reserved["run_id"]],
        timeout=2,
        base_session_id="master",
    )
    settled = _view_node(
        await control.inspect_flow(flow_id, base_session_id="master"),
        "work",
    )
    assert settled["status"] == "completed"
    assert settled["worker_id"] == waiting["worker_id"]
    assert settled["run_id"] == reserved["run_id"]
    assert await control.reconcile_legacy_runs("master", wait=True) == []
    assert (
        control.workers.manager.store.get_run(waiting["run_id"]).status
        is WorkerRunStatus.WAITING_FOR_CONTEXT
    )
    assert _stored_node(control, flow_id, "work").runs[0].status == "waiting_for_context"


@pytest.mark.asyncio
async def test_lost_waiting_binding_becomes_unknown_then_retries_fresh(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover a lost waiting Worker",
        idempotency_key="create",
        nodes=[_node("work", "wait for context")],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    with sqlite3.connect(control.workers.manager.store.path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "DELETE FROM worker_sessions WHERE worker_id = ?",
            (waiting["worker_id"],),
        )

    failed = _view_node(
        await control.inspect_flow(flow_id, base_session_id="master"),
        "work",
    )
    assert failed["status"] == "failed"
    assert failed["result"]["tool_outcome"] == "unknown"

    await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="retry",
    )
    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]
    assert retried["worker_id"] != waiting["worker_id"]
    assert binding.session_action.value == "new"
    assert binding.session_reason == "source_run_missing"


@pytest.mark.asyncio
async def test_cancel_lost_waiting_binding_reaches_blocked_unknown_state(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="cancel a lost waiting Worker",
        idempotency_key="create",
        nodes=[_node("work", "wait for context")],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    with sqlite3.connect(control.workers.manager.store.path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "DELETE FROM worker_sessions WHERE worker_id = ?",
            (waiting["worker_id"],),
        )

    cancelled = await control.cancel(
        flow_id,
        reason="stop despite the lost Worker",
        base_session_id="master",
        idempotency_key="cancel",
    )
    node = _view_node(cancelled, "work")
    assert cancelled["status"] == "blocked"
    assert node["status"] == "failed"
    assert node["result"]["tool_outcome"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action", ["cancel", "complete"])
async def test_terminal_flow_settles_waiting_worker_for_archive(
    tmp_path: Path,
    terminal_action: str,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="stop waiting permanently",
        idempotency_key="create",
        nodes=[_node("work", "wait for context")],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")

    if terminal_action == "cancel":
        terminal = await control.cancel(
            flow_id,
            reason="stop waiting",
            base_session_id="master",
            idempotency_key="cancel",
        )
        assert terminal["status"] == "cancelled"
    else:
        terminal = await control.complete(
            flow_id,
            outcome=FlowCompletion.PARTIAL,
            summary="finish without the requested context",
            base_session_id="master",
            idempotency_key="complete",
        )
        assert terminal["status"] == "partial"
    node = _view_node(terminal, "work")
    assert node["status"] == "cancelled"
    assert node["result"]["status"] == "cancelled"
    assert _stored_node(control, flow_id, "work").runs[-1].status == "cancelled"
    run = control.workers.manager.store.get_run(waiting["run_id"])
    assert run.status is WorkerRunStatus.CANCELLED
    listed = next(
        worker
        for worker in control.workers.list_workers("master")
        if worker["worker_id"] == waiting["worker_id"]
    )
    assert listed["recommended_action"] is None
    archived = control.workers.archive_worker(
        waiting["worker_id"],
        base_session_id="master",
    )
    assert archived["status"] == "archived"


@pytest.mark.asyncio
async def test_reconcile_cancels_waiting_run_abandoned_by_skip(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="skip a waiting node",
        idempotency_key="create",
        nodes=[_node("work", "wait for context")],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")

    skipped = await control.skip_node(
        flow_id,
        "work",
        reason="the answer is no longer needed",
        base_session_id="master",
        idempotency_key="skip",
    )
    assert _view_node(skipped, "work")["status"] == "skipped"
    completed = await control.complete(
        flow_id,
        outcome=FlowCompletion.COMPLETED,
        summary="the waiting work was explicitly waived",
        base_session_id="master",
        idempotency_key="complete",
    )
    assert completed["status"] == "completed"

    assert await control.reconcile_legacy_runs("master", wait=True) == [
        waiting["run_id"]
    ]
    assert (
        control.workers.manager.store.get_run(waiting["run_id"]).status
        is WorkerRunStatus.CANCELLED
    )
    stored_node = _stored_node(control, flow_id, "work")
    assert stored_node.status is FlowNodeStatus.SKIPPED
    assert stored_node.runs[-1].status == WorkerRunStatus.CANCELLED.value
    listed = next(
        worker
        for worker in control.workers.list_workers("master")
        if worker["worker_id"] == waiting["worker_id"]
    )
    assert listed["recommended_action"] is None
    assert control.workers.archive_worker(
        waiting["worker_id"],
        base_session_id="master",
    )["status"] == "archived"


@pytest.mark.asyncio
async def test_partial_skip_then_revision_reuses_last_healthy_session(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="revise skipped partial work",
        idempotency_key="create",
        nodes=[_node("work", "partial work")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    with pytest.raises(ValueError, match="partial node cannot be skipped"):
        await control.skip_node(
            flow_id,
            "work",
            reason="temporarily waive it",
            base_session_id="master",
            idempotency_key="skip",
        )
    await control.revise_node(
        flow_id,
        "work",
        feedback="restore and revise it",
        base_session_id="master",
        idempotency_key="revise",
    )
    revised = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    assert revised["worker_id"] == first["worker_id"]
    assert _stored_node(control, flow_id, "work").runs[-1].session_action.value == "reuse"


@pytest.mark.asyncio
async def test_stale_skip_then_revision_preserves_fresh_context_requirement(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="never reuse stale downstream context",
        idempotency_key="create",
        nodes=[
            _node("build", "build result"),
            _node("review", "review result", worker_type_id="reviewer", depends_on=("build",)),
        ],
    )
    flow_id = created["flow_id"]
    await _advance(control, flow_id, turn_id="turn-1")
    original_review = _view_node(
        await _advance(control, flow_id, turn_id="turn-2"),
        "review",
    )
    await control.revise_node(
        flow_id,
        "build",
        feedback="change the upstream artifact",
        base_session_id="master",
        idempotency_key="revise-build",
    )
    await control.skip_node(
        flow_id,
        "review",
        reason="defer the stale review",
        base_session_id="master",
        idempotency_key="skip-review",
    )
    await control.revise_node(
        flow_id,
        "review",
        feedback="run the deferred review now",
        base_session_id="master",
        idempotency_key="revise-review",
    )
    await _advance(control, flow_id, turn_id="turn-3")
    new_review = _view_node(
        await _advance(control, flow_id, turn_id="turn-4"),
        "review",
    )
    binding = _stored_node(control, flow_id, "review").runs[-1]
    assert new_review["worker_id"] != original_review["worker_id"]
    assert binding.session_action.value == "new"
    assert binding.session_reason == "upstream_context_changed"


@pytest.mark.asyncio
async def test_source_loss_between_eligibility_and_reserve_falls_back_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="survive exact-source deletion",
        idempotency_key="create",
        nodes=[_node("work", "partial work")],
    )
    flow_id = created["flow_id"]
    first = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    await control.retry_node(
        flow_id,
        "work",
        base_session_id="master",
        idempotency_key="retry",
        budget_increase=BudgetIncrease(max_requests=50),
    )
    original_reserve = control.workers._reserve_flow_reuse

    async def delete_source_then_reserve(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with sqlite3.connect(control.workers.manager.store.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                "DELETE FROM worker_sessions WHERE worker_id = ?",
                (first["worker_id"],),
            )
        return await original_reserve(*args, **kwargs)

    monkeypatch.setattr(control.workers, "_reserve_flow_reuse", delete_source_then_reserve)
    retried = _view_node(await _advance(control, flow_id, turn_id="turn-2"), "work")
    binding = _stored_node(control, flow_id, "work").runs[-1]

    assert retried["worker_id"] != first["worker_id"]
    assert binding.session_action.value == "new"
    assert binding.session_reason == "source_run_missing"


@pytest.mark.asyncio
async def test_finish_turn_recovers_waiting_cleanup_after_complete_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover terminal waiting cleanup",
        idempotency_key="create",
        nodes=[_node("work", "wait for context")],
    )
    flow_id = created["flow_id"]
    waiting = _view_node(await _advance(control, flow_id, turn_id="turn-1"), "work")
    original_cancel = control.workers.cancel_worker

    async def crash_cancel(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise RuntimeError("crash after terminal Flow commit")

    monkeypatch.setattr(control.workers, "cancel_worker", crash_cancel)
    with pytest.raises(RuntimeError, match="terminal Flow commit"):
        await control.complete(
            flow_id,
            outcome=FlowCompletion.BLOCKED,
            summary="stop without the missing context",
            base_session_id="master",
            idempotency_key="complete",
        )
    assert control.store.get_flow(flow_id).status.value == "blocked"
    assert (
        control.workers.manager.store.get_run(waiting["run_id"]).status
        is WorkerRunStatus.WAITING_FOR_CONTEXT
    )

    monkeypatch.setattr(control.workers, "cancel_worker", original_cancel)
    assert (
        await control.finish_turn("terminal cleanup recovered", base_session_id="master")
        == "terminal cleanup recovered"
    )
    assert (
        control.workers.manager.store.get_run(waiting["run_id"]).status
        is WorkerRunStatus.CANCELLED
    )
    archived = control.workers.archive_worker(
        waiting["worker_id"],
        base_session_id="master",
    )
    assert archived["status"] == "archived"


@pytest.mark.asyncio
async def test_upstream_revision_reruns_old_descendant_with_fresh_worker(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="rebuild and review changed upstream work",
        idempotency_key="create",
        nodes=[
            _node("build", "build result"),
            _node(
                "review",
                "review result",
                worker_type_id="reviewer",
                depends_on=("build",),
            ),
        ],
    )
    flow_id = created["flow_id"]
    await _advance(control, flow_id, turn_id="turn-1")
    original_review = _view_node(
        await _advance(control, flow_id, turn_id="turn-2"),
        "review",
    )

    await control.revise_node(
        flow_id,
        "build",
        feedback="change the upstream result",
        base_session_id="master",
        idempotency_key="revise-build",
    )
    await _advance(control, flow_id, turn_id="turn-3")
    new_review = _view_node(
        await _advance(control, flow_id, turn_id="turn-4"),
        "review",
    )
    binding = _stored_node(control, flow_id, "review").runs[-1]

    assert new_review["worker_id"] != original_review["worker_id"]
    assert binding.source_run_id is None
    assert binding.session_action.value == "new"
    assert binding.session_reason == "upstream_context_changed"
