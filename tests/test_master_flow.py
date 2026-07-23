"""Outcome tests for the durable dynamic Master Flow runtime."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from aeloon_core.config import (
    AgentDefaultsConfig,
    AgentsConfig,
    Config,
    ContextCompactionConfig,
    SkillsConfig,
)
from aeloon_core.flow_control import FlowControlService
from aeloon_core.flows import (
    MAX_FLOW_RUN_BINDINGS,
    DependencyPolicy,
    FlowAdvanceMode,
    FlowCompletion,
    FlowContextRef,
    FlowNodeSpec,
    FlowNodeStatus,
    FlowRunBinding,
    FlowStatus,
    FlowStore,
    FlowTurnConflictError,
    cancel_flow_state,
    finish_flow,
)
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.session import SessionStore
from aeloon_core.worker_control import WorkerControlService
from aeloon_core.worker_manager import WorkerExecutionOutcome, WorkerSessionManager
from aeloon_core.worker_sessions import (
    BudgetGrant,
    WaitingRequest,
    WorkerReport,
    WorkerRunStatus,
    WorkerStore,
)
from aeloon_core.workers import WorkerRegistry
from tests.message_helpers import checkpoint as message_checkpoint


class RecordingExecutor:
    def __init__(self) -> None:
        self.objectives: list[str] = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, run: Any, worker: Any) -> WorkerExecutionOutcome:
        del worker
        self.objectives.append(run.context.objective)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        if run.context.objective == "wait for Master":
            request = WaitingRequest(summary="need a decision", question="Which option?")
            return WorkerExecutionOutcome(
                status=WorkerRunStatus.WAITING_FOR_CONTEXT,
                report=WorkerReport(summary=request.summary, unresolved=(request.question,)),
                checkpoint=message_checkpoint("waiting"),
                waiting_request=request,
            )
        if run.context.objective == "return partial":
            return WorkerExecutionOutcome(
                status=WorkerRunStatus.PARTIAL,
                report=WorkerReport(summary="partial result"),
                checkpoint=message_checkpoint("partial"),
            )
        if run.context.objective == "return failed":
            return WorkerExecutionOutcome(
                status=WorkerRunStatus.FAILED,
                report=WorkerReport(summary="failed result"),
                tool_outcome="known",
            )
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(
                summary=f"completed: {run.context.objective}",
                evidence=("verified",),
            ),
            checkpoint=message_checkpoint("done"),
        )


def _flow_control(tmp_path: Path) -> tuple[FlowControlService, RecordingExecutor]:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = RecordingExecutor()
    manager = WorkerSessionManager(
        store=WorkerStore(data_dir),
        executor=executor,
        max_concurrency=4,
    )
    workers = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(workspace),
    )
    return FlowControlService(store=FlowStore(data_dir), workers=workers), executor


def _node(
    node_id: str,
    objective: str,
    *,
    worker_type_id: str = "builder",
    depends_on: tuple[str, ...] = (),
    dependency_policy: DependencyPolicy = DependencyPolicy.ALL_COMPLETED,
    context_refs: tuple[FlowContextRef, ...] = (),
) -> FlowNodeSpec:
    return FlowNodeSpec(
        node_id=node_id,
        worker_type_id=worker_type_id,
        objective=objective,
        depends_on=depends_on,
        dependency_policy=dependency_policy,
        context_refs=context_refs,
    )


def _start_flow_tool_owner_process(
    data_dir: Path,
    run_id: str,
    ready_path: Path,
) -> subprocess.Popen[str]:
    code = """
import sys
import time
from pathlib import Path
from aeloon_core.worker_sessions import WorkerStore

store = WorkerStore(Path(sys.argv[1]))
run, claimed = store.try_start_run(sys.argv[2])
if not claimed:
    raise RuntimeError(f"unable to claim {run.run_id}: {run.status.value}")
store.begin_tool_execution(run.run_id)
Path(sys.argv[3]).write_text(store.execution_owner_token, encoding="utf-8")
while True:
    time.sleep(60)
"""
    return subprocess.Popen(
        [sys.executable, "-c", code, str(data_dir), run_id, str(ready_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_flow_tool_owner(
    process: subprocess.Popen[str],
    ready_path: Path,
) -> None:
    deadline = time.monotonic() + 5
    while not ready_path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"tool owner exited before ready: stdout={stdout!r}, stderr={stderr!r}"
            )
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for tool owner process")
        time.sleep(0.01)


def test_flow_store_migrates_v1_session_state(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "flow-control.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE flows (
              flow_id TEXT PRIMARY KEY,
              base_session_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              request_digest TEXT NOT NULL,
              status TEXT NOT NULL,
              state_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(base_session_id, idempotency_key)
            );
            CREATE INDEX flows_session_idx ON flows(base_session_id, created_at);
            CREATE TABLE flow_operations (
              flow_id TEXT NOT NULL REFERENCES flows(flow_id) ON DELETE CASCADE,
              idempotency_key TEXT NOT NULL,
              operation TEXT NOT NULL,
              payload_digest TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY(flow_id, idempotency_key)
            );
            PRAGMA user_version=1;
            """
        )

    FlowStore(data_dir)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "flow_session_state" in tables


def test_flow_store_migrates_v2_turn_lease_without_losing_seal(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FlowStore(data_dir)
    store.seal_session_if_quiescent("master")
    with sqlite3.connect(store.path) as connection:
        connection.execute("ALTER TABLE flow_session_state DROP COLUMN active_turn_id")
        connection.execute("ALTER TABLE flow_session_state DROP COLUMN lease_expires_at")
        connection.execute("PRAGMA user_version=2")

    reopened = FlowStore(data_dir)

    with sqlite3.connect(reopened.path) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        row = connection.execute(
            "SELECT * FROM flow_session_state WHERE base_session_id = 'master'"
        ).fetchone()
    assert row is not None
    assert row["sealed"] == 1
    assert row["active_turn_id"] is None
    assert row["lease_expires_at"] is None


def test_flow_store_migrates_v3_terminal_commit_table(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = FlowStore(data_dir)
    store.seal_session_if_quiescent("master")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE flow_turn_commits")
        connection.execute("PRAGMA user_version=3")

    reopened = FlowStore(data_dir)

    with sqlite3.connect(reopened.path) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        state = connection.execute(
            "SELECT * FROM flow_session_state WHERE base_session_id = 'master'"
        ).fetchone()
    assert "flow_turn_commits" in tables
    assert state is not None
    assert state["sealed"] == 1


async def _create_legacy_flow_continuation(
    control: FlowControlService,
    *,
    flow_id: str,
    source_run_id: str,
    response: str,
    idempotency_key: str,
    start: bool = True,
) -> dict[str, str]:
    """Simulate a continuation persisted before the Flow-only guard existed."""

    source = control.workers.manager.store.get_run(source_run_id)
    context = source.context.model_copy(
        update={
            "objective": response,
            "budget": control.workers.default_budget,
        }
    )
    run, _ = await control.workers.manager.resume_worker(
        source_run_id=source_run_id,
        context=context,
        idempotency_key=idempotency_key,
        base_turn_id=f"flow:{flow_id}",
        start=start,
    )
    return {"run_id": run.run_id}


@pytest.mark.asyncio
async def test_plan_parallel_builds_and_review_execute_one_frontier_at_a_time(
    tmp_path: Path,
) -> None:
    control, executor = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="plan, build independent parts, then review",
        idempotency_key="create",
        nodes=[
            _node("plan", "make plan", worker_type_id="explorer"),
            _node("build_1", "build one", depends_on=("plan",)),
            _node("build_2", "build two", depends_on=("plan",)),
            _node(
                "review",
                "review both",
                worker_type_id="reviewer",
                depends_on=("build_1", "build_2"),
            ),
        ],
    )
    flow_id = created["flow_id"]

    first = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    assert first["frontier_node_ids"] == ["plan"]
    assert first["ready_node_ids"] == ["build_1", "build_2"]
    assert "build one" not in executor.objectives

    executor.max_active = 0
    second = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    assert second["frontier_node_ids"] == ["build_1", "build_2"]
    assert second["ready_node_ids"] == ["review"]
    assert executor.max_active >= 2
    assert "review both" not in executor.objectives

    third = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    assert third["frontier_node_ids"] == ["review"]
    assert {node["status"] for node in third["nodes"]} == {"completed"}


@pytest.mark.asyncio
async def test_auto_advance_runs_predictable_chain_and_stops_after_review(
    tmp_path: Path,
) -> None:
    control, executor = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="build and review a predictable graph",
        idempotency_key="create-auto",
        advance_mode=FlowAdvanceMode.AUTO,
        nodes=[
            _node("inspect", "inspect", worker_type_id="explorer"),
            _node("build_1", "build one", depends_on=("inspect",)),
            _node("build_2", "build two", depends_on=("inspect",)),
            _node(
                "review",
                "review both",
                worker_type_id="reviewer",
                depends_on=("build_1", "build_2"),
            ),
        ],
    )

    result = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert result["frontiers_executed"] == [
        ["inspect"],
        ["build_1", "build_2"],
        ["review"],
    ]
    assert result["auto_advanced_count"] == 2
    assert result["stop_reason"] == "reviewer_frontier"
    assert result["execution"]["auto_advanced_frontiers"] == 2
    assert executor.max_active >= 2
    assert {node["status"] for node in result["nodes"]} == {"completed"}


@pytest.mark.asyncio
async def test_auto_advance_honors_frontier_limit(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="run a bounded chain",
        idempotency_key="create-auto-limit",
        advance_mode=FlowAdvanceMode.AUTO,
        auto_advance_max_frontiers=2,
        nodes=[
            _node("one", "one"),
            _node("two", "two", depends_on=("one",)),
            _node("three", "three", depends_on=("two",)),
        ],
    )

    result = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert result["frontiers_executed"] == [["one"], ["two"]]
    assert result["ready_node_ids"] == ["three"]
    assert result["auto_advanced_count"] == 1
    assert result["stop_reason"] == "auto_advance_limit"


@pytest.mark.asyncio
async def test_auto_advance_stops_on_partial_frontier(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="stop when a node is partial",
        idempotency_key="create-auto-partial",
        advance_mode=FlowAdvanceMode.AUTO,
        nodes=[
            _node("partial", "return partial"),
            _node("downstream", "must not run", depends_on=("partial",)),
        ],
    )

    result = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert result["frontiers_executed"] == [["partial"]]
    assert result["stop_reason"] == "frontier_non_success"
    assert result["nodes"][1]["status"] == "pending"


@pytest.mark.parametrize(
    ("objective", "expected_status", "expected_reason"),
    [
        ("wait for Master", "waiting_for_context", "waiting_for_context"),
        ("return failed", "failed", "frontier_non_success"),
    ],
)
@pytest.mark.asyncio
async def test_auto_advance_stops_on_waiting_or_failed_frontier(
    tmp_path: Path,
    objective: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    control, executor = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="stop a predictable chain safely",
        idempotency_key=f"create-auto-{expected_status}",
        advance_mode=FlowAdvanceMode.AUTO,
        nodes=[
            _node("first", objective),
            _node("downstream", "must not run", depends_on=("first",)),
        ],
    )

    result = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert result["nodes"][0]["status"] == expected_status
    assert result["nodes"][1]["status"] == "pending"
    assert result["stop_reason"] == expected_reason
    assert "must not run" not in executor.objectives


@pytest.mark.asyncio
async def test_auto_advance_stops_at_default_four_frontiers_and_max_rounds(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    chain = [
        _node(
            f"step_{index}",
            f"step {index}",
            depends_on=((f"step_{index - 1}",) if index else ()),
        )
        for index in range(5)
    ]
    created = control.create_flow(
        base_session_id="master",
        goal="enforce the structural auto limit",
        idempotency_key="create-auto-four",
        advance_mode=FlowAdvanceMode.AUTO,
        nodes=chain,
    )

    limited = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert len(limited["frontiers_executed"]) == 4
    assert limited["ready_node_ids"] == ["step_4"]
    assert limited["stop_reason"] == "auto_advance_limit"

    round_limited = control.create_flow(
        base_session_id="master",
        goal="enforce max rounds during auto advance",
        idempotency_key="create-auto-round-limit",
        advance_mode=FlowAdvanceMode.AUTO,
        max_rounds=2,
        nodes=chain[:3],
    )
    blocked = await control.advance_flow(
        round_limited["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    assert blocked["status"] == "blocked"
    assert blocked["stop_reason"] == "max_rounds"
    assert len(blocked["frontiers_executed"]) == 2


@pytest.mark.asyncio
async def test_auto_advance_timeout_returns_without_starting_downstream(
    tmp_path: Path,
) -> None:
    control, executor = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="honor the caller timeout",
        idempotency_key="create-auto-timeout",
        advance_mode=FlowAdvanceMode.AUTO,
        nodes=[
            _node("first", "slow first"),
            _node("downstream", "must not run", depends_on=("first",)),
        ],
    )

    timed_out = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )

    assert timed_out["stop_reason"] == "active_or_timeout"
    assert timed_out["nodes"][1]["status"] == "pending"
    assert "must not run" not in executor.objectives
    run_id = timed_out["nodes"][0]["run_id"]
    await control.workers.await_workers([run_id], timeout=1, base_session_id="master")


@pytest.mark.asyncio
async def test_auto_advance_replays_a_crash_after_worker_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover and continue a predictable chain",
        idempotency_key="create-auto-replay",
        advance_mode=FlowAdvanceMode.AUTO,
        nodes=[
            _node("first", "first"),
            _node("second", "second", depends_on=("first",)),
        ],
    )
    original_attach = control._attach_run

    def crash_attach(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("crash after durable Worker reservation")

    monkeypatch.setattr(control, "_attach_run", crash_attach)
    with pytest.raises(RuntimeError, match="durable Worker reservation"):
        await control.advance_flow(
            created["flow_id"],
            base_session_id="master",
            base_turn_id="turn-one",
            timeout_seconds=2,
        )
    monkeypatch.setattr(control, "_attach_run", original_attach)

    recovered = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn-two",
        timeout_seconds=2,
    )

    assert recovered["frontiers_executed"] == [["first"], ["second"]]
    assert {node["status"] for node in recovered["nodes"]} == {"completed"}
    assert len(control.workers.manager.store.list_workers("master")) == 2


@pytest.mark.asyncio
async def test_flow_uses_role_budget_overrides_and_exposes_telemetry(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    control.workers.default_budgets["explorer"] = BudgetGrant(max_requests=7)
    created = control.create_flow(
        base_session_id="master",
        goal="compare role budgets",
        idempotency_key="create-role-budgets",
        nodes=[
            _node("inspect", "inspect budget", worker_type_id="explorer"),
            _node("build", "build budget", worker_type_id="builder"),
        ],
    )

    result = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    by_id = {node["node_id"]: node for node in result["nodes"]}
    assert by_id["inspect"]["budget"]["last"]["max_requests"] == 7
    assert (
        by_id["build"]["budget"]["last"]["max_requests"]
        == control.workers.default_budget.max_requests
    )
    assert by_id["inspect"]["result"]["telemetry"]["role"] == "explorer"
    assert result["telemetry"]["roles"] == {"explorer": 1, "builder": 1}
    assert result["telemetry"]["frontier_widths"] == [2]


@pytest.mark.asyncio
async def test_fresh_follow_up_receives_explicit_prior_run_context(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    original = control.create_flow(
        base_session_id="master",
        goal="build the original implementation",
        idempotency_key="create-original",
        nodes=[_node("original_build", "build original")],
    )
    completed = await control.advance_flow(
        original["flow_id"],
        base_session_id="master",
        base_turn_id="turn-original",
        timeout_seconds=2,
    )
    source_node = completed["nodes"][0]

    follow_up = control.create_flow(
        base_session_id="master",
        goal="fix a newly reported problem",
        idempotency_key="create-follow-up",
        nodes=[
            _node(
                "fix",
                "fix the follow-up defect",
                context_refs=(
                    FlowContextRef(
                        kind="worker_run",
                        id=source_node["run_id"],
                        relation="original implementation",
                        include=("summary", "evidence"),
                    ),
                ),
            )
        ],
    )
    fixed = await control.advance_flow(
        follow_up["flow_id"],
        base_session_id="master",
        base_turn_id="turn-follow-up",
        timeout_seconds=2,
    )
    fixed_node = fixed["nodes"][0]
    run = control.workers.manager.store.get_run(fixed_node["run_id"])

    assert fixed_node["worker_id"] != source_node["worker_id"]
    assert run.source_run_id is None
    assert len(run.context.related_contexts) == 1
    related = run.context.related_contexts[0]
    assert related.source_kind == "worker_run"
    assert related.source_id == source_node["run_id"]
    assert related.run_id == source_node["run_id"]
    assert related.relation == "original implementation"
    assert related.included_sections == ("summary", "evidence")
    assert related.objective is None
    assert related.summary == "completed: build original"
    assert len(related.evidence) == 1
    assert related.evidence[0].kind == "legacy"
    assert related.evidence[0].locator == "verified"

    with pytest.raises(ValueError, match="another Master session"):
        control.create_flow(
            base_session_id="different-master",
            goal="must not import another session's Worker context",
            idempotency_key="cross-session-association",
            nodes=[
                _node(
                    "invalid",
                    "consume inaccessible context",
                    context_refs=(
                        FlowContextRef(
                            kind="worker_run",
                            id=source_node["run_id"],
                            relation="cross-session attempt",
                        ),
                    ),
                )
            ],
        )


@pytest.mark.asyncio
async def test_flow_node_context_ref_resolves_only_an_ancestor_run(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="build then independently review with bounded context",
        idempotency_key="create-associated-flow",
        nodes=[
            _node("build", "build feature"),
            _node(
                "review",
                "review feature",
                worker_type_id="reviewer",
                depends_on=("build",),
                context_refs=(
                    FlowContextRef(
                        kind="flow_node",
                        id="build",
                        relation="implementation under review",
                        include=("summary", "artifacts", "evidence"),
                    ),
                ),
            ),
        ],
    )
    built = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn-build",
        timeout_seconds=2,
    )
    reviewed = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn-review",
        timeout_seconds=2,
    )
    by_id = {node["node_id"]: node for node in reviewed["nodes"]}
    review_run = control.workers.manager.store.get_run(by_id["review"]["run_id"])
    related = review_run.context.related_contexts[0]

    assert by_id["review"]["worker_id"] != by_id["build"]["worker_id"]
    assert related.source_kind == "flow_node"
    assert related.source_id == "build"
    assert related.run_id == built["nodes"][0]["run_id"]
    assert related.summary == "completed: build feature"

    with pytest.raises(ValueError, match="only reference an ancestor"):
        control.create_flow(
            base_session_id="master",
            goal="reject an implicit peer context pipe",
            idempotency_key="create-invalid-association",
            nodes=[
                _node("peer", "peer work"),
                _node(
                    "consumer",
                    "consume peer",
                    context_refs=(
                        FlowContextRef(
                            kind="flow_node",
                            id="peer",
                            relation="not a dependency",
                        ),
                    ),
                ),
            ],
        )


@pytest.mark.asyncio
async def test_settled_partial_does_not_unlock_default_join(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="do work then review",
        idempotency_key="create",
        nodes=[
            _node("work", "return partial"),
            _node("strict_review", "strict review", depends_on=("work",)),
            _node(
                "diagnose",
                "diagnose terminal result",
                depends_on=("work",),
                dependency_policy=DependencyPolicy.ALL_TERMINAL,
            ),
        ],
    )
    flow_id = created["flow_id"]

    result = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert result["ready_node_ids"] == ["diagnose"]
    assert result["blocked_node_ids"] == ["strict_review"]
    assert result["nodes"][0]["status"] == "partial"


@pytest.mark.asyncio
async def test_revision_reruns_only_changed_branch_and_stale_descendants(
    tmp_path: Path,
) -> None:
    control, executor = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="build two branches and review",
        idempotency_key="create",
        nodes=[
            _node("build_1", "build one"),
            _node("build_2", "build two"),
            _node(
                "review",
                "review both",
                worker_type_id="reviewer",
                depends_on=("build_1", "build_2"),
            ),
        ],
    )
    flow_id = created["flow_id"]
    await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    revised = await control.revise_node(
        flow_id,
        "build_1",
        feedback="fix the reviewed defect",
        base_session_id="master",
        idempotency_key="revise-build-1",
    )
    by_id = {node["node_id"]: node for node in revised["nodes"]}
    assert by_id["build_1"]["generation"] == 2
    assert by_id["build_1"]["status"] == "pending"
    assert by_id["build_2"]["generation"] == 1
    assert by_id["build_2"]["status"] == "completed"
    assert by_id["review"]["generation"] == 2
    assert by_id["review"]["status"] == "stale"

    await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    counts = Counter(objective.split("\n", 1)[0] for objective in executor.objectives)
    assert counts == Counter({"build one": 2, "review both": 2, "build two": 1})


@pytest.mark.asyncio
async def test_late_old_binding_cannot_resurrect_a_revised_generation(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="revise after a completed run",
        idempotency_key="create",
        nodes=[_node("build", "build once")],
    )
    flow_id = created["flow_id"]
    completed = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    old_run_id = completed["nodes"][0]["run_id"]
    old_view = (
        await control.workers.await_workers([old_run_id], timeout=0, base_session_id="master")
    )[0]

    await control.revise_node(
        flow_id,
        "build",
        feedback="produce a different result",
        base_session_id="master",
        idempotency_key="revise",
    )
    with pytest.raises(ValueError, match="stale WorkerRun binding"):
        control._attach_run(
            flow_id,
            base_session_id="master",
            node_id="build",
            generation=1,
            attempt=1,
            run_view=old_view,
        )

    node = control.store.get_flow(flow_id, base_session_id="master").node("build")
    assert node.generation == 2
    assert node.status is FlowNodeStatus.PENDING
    assert node.current_run_id is None


@pytest.mark.asyncio
async def test_waiting_node_resumes_exact_run_binding(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="wait then continue",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    waiting = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    first_run_id = waiting["nodes"][0]["run_id"]
    assert waiting["nodes"][0]["status"] == "waiting_for_context"

    with pytest.raises(ValueError, match="belongs to a Flow"):
        await control.workers.resume_worker(
            first_run_id,
            response="bypass the Flow",
            idempotency_key="generic-resume",
            base_session_id="master",
            base_turn_id="ordinary-master-turn",
        )
    with pytest.raises(ValueError, match="belongs to a Flow"):
        await control.workers.resume_worker(
            first_run_id,
            response="matching ownership must not bypass the Flow",
            idempotency_key="generic-resume-matching-flow",
            base_session_id="master",
            base_turn_id=f"flow:{flow_id}",
        )

    resumed = await control.resume_node(
        flow_id,
        "work",
        response="Use option A",
        base_session_id="master",
        base_turn_id="turn-2",
        idempotency_key="resume-work",
    )
    second_run_id = resumed["nodes"][0]["run_id"]

    assert second_run_id != first_run_id
    assert resumed["nodes"][0]["status"] in {"running", "completed"}
    stored = control.store.get_flow(flow_id, base_session_id="master").node("work")
    assert [binding.run_id for binding in stored.runs] == [first_run_id, second_run_id]
    assert stored.runs[-1].source_run_id == first_run_id


@pytest.mark.asyncio
async def test_flow_worker_cannot_be_reused_outside_its_flow(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="keep Flow ownership",
        idempotency_key="create",
        nodes=[_node("work", "complete work")],
    )
    completed = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    worker_id = completed["nodes"][0]["worker_id"]

    listed = next(
        worker
        for worker in control.workers.list_workers("master")
        if worker["worker_id"] == worker_id
    )
    assert listed["flow_owned"] is True
    assert listed["reusable"] is False
    with pytest.raises(ValueError, match="belongs to a Flow"):
        await control.workers.reuse_worker(
            base_session_id="master",
            worker_id=worker_id,
            objective="bypass the Flow",
            idempotency_key="reuse-bypass",
        )


@pytest.mark.asyncio
async def test_flow_adopts_a_matching_low_level_continuation_idempotently(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="adopt a continuation",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    waiting = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-one",
        timeout_seconds=2,
    )
    source_run_id = waiting["nodes"][0]["run_id"]
    external = await _create_legacy_flow_continuation(
        control,
        flow_id=flow_id,
        source_run_id=source_run_id,
        response="Use option A",
        idempotency_key="low-level-resume",
    )

    adopted = await control.resume_node(
        flow_id,
        "work",
        response="Use option A",
        base_session_id="master",
        base_turn_id="turn-three",
        idempotency_key="flow-resume",
    )
    replayed = await control.resume_node(
        flow_id,
        "work",
        response="Use option A",
        base_session_id="master",
        base_turn_id="turn-four",
        idempotency_key="flow-resume",
    )

    assert adopted["adopted_external_continuation"] is True
    assert adopted["nodes"][0]["run_id"] == external["run_id"]
    assert replayed["nodes"][0]["run_id"] == external["run_id"]


@pytest.mark.asyncio
async def test_flow_surfaces_a_conflicting_low_level_continuation(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="detect a conflicting continuation",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    waiting = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-one",
        timeout_seconds=2,
    )
    source_run_id = waiting["nodes"][0]["run_id"]
    external = await _create_legacy_flow_continuation(
        control,
        flow_id=flow_id,
        source_run_id=source_run_id,
        response="Use option B",
        idempotency_key="low-level-resume",
    )

    with pytest.raises(ValueError, match="different response"):
        await control.resume_node(
            flow_id,
            "work",
            response="Use option A",
            base_session_id="master",
            base_turn_id="turn-three",
            idempotency_key="flow-resume",
        )

    node = control.store.get_flow(flow_id, base_session_id="master").node("work")
    assert node.current_run_id == external["run_id"]
    assert node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT


@pytest.mark.asyncio
async def test_dispatch_and_resume_recover_after_attach_crash_across_master_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="recover dispatch",
        idempotency_key="create-dispatch",
        nodes=[_node("work", "ordinary work")],
    )
    original_attach = control._attach_run

    def crash_attach(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("crash after durable Worker creation")

    monkeypatch.setattr(control, "_attach_run", crash_attach)
    with pytest.raises(RuntimeError, match="crash after durable"):
        await control.advance_flow(
            created["flow_id"],
            base_session_id="master",
            base_turn_id="turn-one",
            timeout_seconds=2,
        )
    monkeypatch.setattr(control, "_attach_run", original_attach)
    recovered = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn-two",
        timeout_seconds=2,
    )
    assert recovered["nodes"][0]["status"] == "completed"
    assert len(control.workers.manager.store.list_workers("master")) == 1

    waiting_flow = control.create_flow(
        base_session_id="master",
        goal="recover resume",
        idempotency_key="create-resume",
        nodes=[_node("wait", "wait for Master")],
    )
    waiting = await control.advance_flow(
        waiting_flow["flow_id"],
        base_session_id="master",
        base_turn_id="turn-three",
        timeout_seconds=2,
    )
    waiting_run_id = waiting["nodes"][0]["run_id"]

    original_mutate = control.store.mutate

    def crash_resume_mutation(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("operation") == "resume_node":
            raise RuntimeError("crash after durable Worker creation")
        return original_mutate(*args, **kwargs)

    monkeypatch.setattr(control.store, "mutate", crash_resume_mutation)
    with pytest.raises(RuntimeError, match="crash after durable"):
        await control.resume_node(
            waiting_flow["flow_id"],
            "wait",
            response="Use option A",
            base_session_id="master",
            base_turn_id="turn-four",
            idempotency_key="resume-after-crash",
        )
    monkeypatch.setattr(control.store, "mutate", original_mutate)
    resumed = await control.resume_node(
        waiting_flow["flow_id"],
        "wait",
        response="Use option A",
        base_session_id="master",
        base_turn_id="turn-five",
        idempotency_key="resume-after-crash",
    )
    assert resumed["nodes"][0]["run_id"] != waiting_run_id


@pytest.mark.asyncio
async def test_cancel_tracks_a_run_created_before_flow_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="cancel an interrupted launch",
        idempotency_key="create",
        nodes=[_node("work", "work")],
    )

    def crash_attach(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("crash after Worker reservation")

    monkeypatch.setattr(control, "_attach_run", crash_attach)
    with pytest.raises(RuntimeError, match="Worker reservation"):
        await control.advance_flow(
            created["flow_id"],
            base_session_id="master",
            base_turn_id="turn",
            timeout_seconds=0,
        )
    monkeypatch.undo()

    cancelled = await control.cancel(
        created["flow_id"],
        reason="stop after interrupted launch",
        base_session_id="master",
        idempotency_key="cancel",
    )

    assert cancelled["status"] == FlowStatus.CANCELLED.value
    assert cancelled["cancellation_run_count"] == 1
    worker = control.workers.manager.store.list_workers("master")[0]
    run = control.workers.manager.store.list_runs(worker.worker_id)[0]
    assert run.status is WorkerRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_does_not_activate_an_attached_flow_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, executor = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="cancel an attached reservation",
        idempotency_key="create",
        nodes=[_node("work", "must remain inert")],
    )
    original_start = control.workers.start_worker_run

    def crash_before_activation(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("crash before Worker activation")

    monkeypatch.setattr(control.workers, "start_worker_run", crash_before_activation)
    with pytest.raises(RuntimeError, match="before Worker activation"):
        await control.advance_flow(
            created["flow_id"],
            base_session_id="master",
            base_turn_id="turn",
            timeout_seconds=0,
        )
    monkeypatch.setattr(control.workers, "start_worker_run", original_start)

    worker = control.workers.manager.store.list_workers("master")[0]
    reserved = control.workers.manager.store.list_runs(worker.worker_id)[0]
    assert reserved.status is WorkerRunStatus.QUEUED
    assert reserved.activated_at is None

    cancelled = await control.cancel(
        created["flow_id"],
        reason="stop before activation",
        base_session_id="master",
        idempotency_key="cancel",
    )

    settled = control.workers.manager.store.get_run(reserved.run_id)
    assert cancelled["status"] == FlowStatus.CANCELLED.value
    node = cancelled["nodes"][0]
    assert node["status"] == FlowNodeStatus.CANCELLED.value
    assert node["result"]["status"] == WorkerRunStatus.CANCELLED.value
    assert (
        control.store.get_flow(created["flow_id"]).node("work").runs[-1].status
        == WorkerRunStatus.CANCELLED.value
    )
    assert settled.status is WorkerRunStatus.CANCELLED
    assert settled.activated_at is None
    assert executor.objectives == []


@pytest.mark.asyncio
async def test_cancel_intent_fences_stale_activation_across_controllers(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reserving_manager = WorkerSessionManager(store=WorkerStore(data_dir))
    reserving_workers = WorkerControlService(
        manager=reserving_manager,
        worker_types=WorkerRegistry.discover(workspace),
    )
    first = FlowControlService(store=FlowStore(data_dir), workers=reserving_workers)
    created = first.create_flow(
        base_session_id="master",
        goal="fence a stale controller",
        idempotency_key="create",
        nodes=[_node("work", "must never activate after cancellation")],
    )
    launched = await first.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )
    run_id = launched["nodes"][0]["run_id"]
    reservation = reserving_manager.store.get_run(run_id)
    assert reservation.status is WorkerRunStatus.QUEUED
    assert reservation.activated_at is not None

    second = FlowControlService(
        store=FlowStore(data_dir),
        workers=WorkerControlService(
            manager=WorkerSessionManager(store=WorkerStore(data_dir)),
            worker_types=reserving_workers.worker_types,
        ),
    )
    # SIGKILL equivalent: cancellation intent commits under the shared fence,
    # then the controller dies before it can fence the WorkerStore reservation.
    with second._activation_fence(created["flow_id"]):
        second.store.mutate(
            created["flow_id"],
            base_session_id="master",
            operation="cancel",
            idempotency_key="cancel",
            payload={"reason": "cancel wins"},
            mutation=lambda flow: cancel_flow_state(flow, "cancel wins"),
        )

    executor = RecordingExecutor()
    detached_runner = WorkerSessionManager(
        store=WorkerStore(data_dir),
        executor=executor,
    )
    assert [run.run_id for run in detached_runner.start_queued()] == [run_id]
    settled = (await detached_runner.await_workers([run_id], timeout=1))[0]

    assert settled.status is WorkerRunStatus.CANCELLED
    assert settled.activated_at is not None
    assert executor.objectives == []
    cancelled = await first.inspect_flow(
        created["flow_id"],
        base_session_id="master",
    )
    assert cancelled["status"] == FlowStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_concurrent_exact_resume_calls_share_one_flow_operation(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    second = FlowControlService(
        store=FlowStore(tmp_path / "data"),
        workers=control.workers,
    )
    created = control.create_flow(
        base_session_id="master",
        goal="resume once across controllers",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-one",
        timeout_seconds=2,
    )

    first_result, second_result = await asyncio.gather(
        control.resume_node(
            flow_id,
            "work",
            response="Use option A",
            base_session_id="master",
            base_turn_id="turn-two",
            idempotency_key="resume",
        ),
        second.resume_node(
            flow_id,
            "work",
            response="Use option A",
            base_session_id="master",
            base_turn_id="turn-three",
            idempotency_key="resume",
        ),
    )

    assert first_result["nodes"][0]["run_id"] == second_result["nodes"][0]["run_id"]
    node = control.store.get_flow(flow_id, base_session_id="master").node("work")
    assert len(node.runs) == 2


@pytest.mark.asyncio
async def test_resume_attach_race_cancels_the_unbound_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="terminal race during resume",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-one",
        timeout_seconds=2,
    )

    async def blocked_executor(run: Any, worker: Any) -> WorkerExecutionOutcome:
        del run, worker
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    control.workers.manager.executor = blocked_executor
    original_mutate = control.store.mutate

    def terminal_before_attach(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("operation") == "resume_node":
            control.store.update_runtime(
                flow_id,
                base_session_id="master",
                mutation=lambda flow: finish_flow(
                    flow,
                    FlowCompletion.PARTIAL,
                    "terminal race",
                ),
            )
        return original_mutate(*args, **kwargs)

    monkeypatch.setattr(control.store, "mutate", terminal_before_attach)
    with pytest.raises(ValueError, match="non-open Flow"):
        await control.resume_node(
            flow_id,
            "work",
            response="Use option A",
            base_session_id="master",
            base_turn_id="turn-two",
            idempotency_key="resume",
        )

    latest_worker = control.workers.manager.store.list_workers("master")[-1]
    latest_run = control.workers.manager.store.list_runs(latest_worker.worker_id)[-1]
    assert latest_run.status is WorkerRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_sync_does_not_resurrect_a_concurrently_cancelled_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = _flow_control(tmp_path)
    release = asyncio.Event()

    async def blocked_executor(run: Any, worker: Any) -> WorkerExecutionOutcome:
        del run, worker
        await release.wait()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="done"),
            checkpoint=message_checkpoint("done"),
        )

    control.workers.manager.executor = blocked_executor
    created = control.create_flow(
        base_session_id="master",
        goal="cancel during synchronization",
        idempotency_key="create",
        nodes=[_node("work", "long work")],
    )
    flow_id = created["flow_id"]
    running = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )
    run_id = running["nodes"][0]["run_id"]
    original_await = control.workers.await_workers
    raced = False

    async def cancel_after_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal raced
        views = await original_await(*args, **kwargs)
        if not raced:
            raced = True
            control.store.update_runtime(
                flow_id,
                base_session_id="master",
                mutation=lambda flow: cancel_flow_state(flow, "concurrent cancel"),
            )
        return views

    monkeypatch.setattr(control.workers, "await_workers", cancel_after_read)
    inspected = await control.inspect_flow(flow_id, base_session_id="master")

    assert inspected["status"] == FlowStatus.CANCELLING.value
    assert inspected["nodes"][0]["status"] == FlowNodeStatus.RUNNING.value
    with pytest.raises(ValueError, match="remain open"):
        await control.finish_turn("too early", base_session_id="master")
    monkeypatch.setattr(control.workers, "await_workers", original_await)
    cancelled = await control.cancel(
        flow_id,
        reason="finish cancellation",
        base_session_id="master",
        idempotency_key="cancel",
    )
    assert cancelled["status"] == FlowStatus.CANCELLED.value
    assert await control.finish_turn("safe now", base_session_id="master") == "safe now"
    release.set()
    assert control.workers.manager.store.get_run(run_id).status is WorkerRunStatus.CANCELLED


def test_graph_validation_and_idempotent_dynamic_expansion(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    with pytest.raises(ValueError, match="unknown dependencies"):
        control.create_flow(
            base_session_id="master",
            goal="invalid",
            idempotency_key="unknown",
            nodes=[_node("build", "build", depends_on=("missing",))],
        )
    with pytest.raises(ValueError, match="acyclic"):
        control.create_flow(
            base_session_id="master",
            goal="cycle",
            idempotency_key="cycle",
            nodes=[
                _node("one", "one", depends_on=("two",)),
                _node("two", "two", depends_on=("one",)),
            ],
        )

    created = control.create_flow(
        base_session_id="master",
        goal="expand",
        idempotency_key="create",
        nodes=[_node("plan", "plan")],
    )

    async def expand_twice() -> tuple[dict[str, Any], dict[str, Any]]:
        first = await control.add_nodes(
            created["flow_id"],
            base_session_id="master",
            nodes=[_node("build", "build", depends_on=("plan",))],
            idempotency_key="add-build",
        )
        second = await control.add_nodes(
            created["flow_id"],
            base_session_id="master",
            nodes=[_node("build", "build", depends_on=("plan",))],
            idempotency_key="add-build",
        )
        return first, second

    first, second = asyncio.run(expand_twice())
    assert first["changed"] is True
    assert second["changed"] is False
    assert len(second["nodes"]) == 2


@pytest.mark.asyncio
async def test_frontier_history_limit_fails_before_creating_an_orphan_run(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="bounded run history",
        idempotency_key="create",
        nodes=[_node("work", "work")],
    )

    def saturate(flow: Any) -> None:
        flow.node("work").runs = [
            FlowRunBinding(
                generation=1,
                attempt=index + 1,
                worker_id=f"worker-{index}",
                run_id=f"run-{index}",
                status=WorkerRunStatus.COMPLETED.value,
                created_at="2026-01-01T00:00:00+00:00",
            )
            for index in range(MAX_FLOW_RUN_BINDINGS)
        ]

    control.store.update_runtime(
        created["flow_id"],
        base_session_id="master",
        mutation=saturate,
    )
    result = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )

    assert result["nodes"][0]["status"] == FlowNodeStatus.FAILED.value
    assert control.workers.manager.store.list_workers("master") == []


@pytest.mark.asyncio
async def test_external_continuation_over_history_limit_is_cancelled_and_bounded(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="bound external continuation history",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    waiting = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-one",
        timeout_seconds=2,
    )
    source_run_id = waiting["nodes"][0]["run_id"]

    def saturate(flow: Any) -> None:
        node = flow.node("work")
        node.runs = [
            *node.runs,
            *(
                FlowRunBinding(
                    generation=1,
                    attempt=index + 2,
                    worker_id=f"audit-worker-{index}",
                    run_id=f"audit-run-{index}",
                    status=WorkerRunStatus.COMPLETED.value,
                    created_at="2026-01-01T00:00:00+00:00",
                )
                for index in range(MAX_FLOW_RUN_BINDINGS - 1)
            ),
        ]

    control.store.update_runtime(
        flow_id,
        base_session_id="master",
        mutation=saturate,
    )
    external = await _create_legacy_flow_continuation(
        control,
        flow_id=flow_id,
        source_run_id=source_run_id,
        response="Use option A",
        idempotency_key="external-resume",
        start=False,
    )
    inspected = await control.inspect_flow(flow_id, base_session_id="master")

    assert inspected["nodes"][0]["status"] == FlowNodeStatus.FAILED.value
    stored = control.store.get_flow(flow_id, base_session_id="master").node("work")
    assert len(stored.runs) == MAX_FLOW_RUN_BINDINGS
    assert (
        control.workers.manager.store.get_run(external["run_id"]).status
        is WorkerRunStatus.CANCELLED
    )


@pytest.mark.asyncio
async def test_paused_flow_fences_external_resume_until_it_is_adopted(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="pause around external continuation",
        idempotency_key="create",
        nodes=[_node("work", "wait for Master")],
    )
    flow_id = created["flow_id"]
    waiting = await control.advance_flow(
        flow_id,
        base_session_id="master",
        base_turn_id="turn-one",
        timeout_seconds=2,
    )
    await control.pause(
        flow_id,
        reason="ask the user",
        base_session_id="master",
        idempotency_key="pause",
    )
    external = await _create_legacy_flow_continuation(
        control,
        flow_id=flow_id,
        source_run_id=waiting["nodes"][0]["run_id"],
        response="Use option A",
        idempotency_key="external-resume",
        start=False,
    )
    control.workers.manager.store.activate_run(external["run_id"])
    fenced, claimed = control.workers.manager.store.try_start_run(external["run_id"])
    assert claimed is False
    assert fenced.status is WorkerRunStatus.CANCELLED

    paused = await control.inspect_flow(flow_id, base_session_id="master")
    assert paused["status"] == FlowStatus.PAUSED.value
    assert paused["nodes"][0]["run_id"] != external["run_id"]

    assert await control.finish_turn("paused safely", base_session_id="master") == ("paused safely")
    assert (
        control.workers.manager.store.get_run(external["run_id"]).status
        is WorkerRunStatus.CANCELLED
    )

    control.store.begin_turn("master", "next-turn")
    await control.resume(
        flow_id,
        base_session_id="master",
        idempotency_key="resume-flow",
        turn_id="next-turn",
    )
    adopted = await control.inspect_flow(
        flow_id,
        base_session_id="master",
        turn_id="next-turn",
    )
    assert adopted["nodes"][0]["run_id"] == external["run_id"]


@pytest.mark.asyncio
async def test_round_limit_blocks_before_unbounded_dynamic_loop(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="bounded",
        idempotency_key="create",
        max_rounds=1,
        nodes=[
            _node("plan", "plan"),
            _node("build", "build", depends_on=("plan",)),
        ],
    )
    await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )
    blocked = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=2,
    )

    assert blocked["status"] == FlowStatus.BLOCKED.value
    assert blocked["nodes"][1]["status"] == FlowNodeStatus.PENDING.value
    assert "max_rounds=1" in blocked["completion_summary"]


class MasterScriptModel(FunctionModel):
    def __init__(self) -> None:
        self.responses = deque(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            "create_flow",
                            {
                                "goal": "prove completion is gated",
                                "nodes": [
                                    {
                                        "node_id": "work",
                                        "worker_type_id": "builder",
                                        "objective": "do the work",
                                    }
                                ],
                                "idempotency_key": "create-flow",
                            },
                            "create-flow",
                        )
                    ],
                ),
                ModelResponse(parts=[TextPart("I am done too early.")]),
                "complete_flow",
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            "finish_turn",
                            {"final_content": "Honest partial result."},
                            "finish-turn",
                        )
                    ],
                ),
            ]
        )
        self.call_count = 0

        async def function(messages: list[ModelMessage], _info: Any) -> ModelResponse:
            self.call_count += 1
            response = self.responses.popleft()
            if response == "complete_flow":
                flow_result = next(
                    part
                    for message in reversed(messages)
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, ToolReturnPart)
                    and part.tool_call_id == "create-flow"
                )
                import json

                flow_id = json.loads(str(flow_result.content))["flow_id"]
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "complete_flow",
                            {
                                "flow_id": flow_id,
                                "outcome": FlowCompletion.PARTIAL.value,
                                "summary": "stopped intentionally for the test",
                                "idempotency_key": "complete-flow",
                            },
                            "complete-flow",
                        )
                    ]
                )
            assert isinstance(response, ModelResponse)
            return response

        super().__init__(function=function)


class FinishTurnModel(FunctionModel):
    def __init__(self, final_content: str) -> None:
        self.final_content = final_content
        self.call_count = 0
        self.seen_messages: list[list[ModelMessage]] = []

        async def function(messages: list[ModelMessage], _info: Any) -> ModelResponse:
            self.seen_messages.append(messages)
            self.call_count += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "finish_turn",
                        {"final_content": self.final_content},
                        f"finish-{self.call_count}",
                    )
                ]
            )

        super().__init__(function=function)


class FailIfCalledModel(FunctionModel):
    def __init__(self) -> None:
        self.call_count = 0

        async def function(_messages: list[ModelMessage], _info: Any) -> ModelResponse:
            self.call_count += 1
            raise AssertionError("a durable committed turn must not rerun the model")

        super().__init__(function=function)


class TurnProgress:
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.blocks: list[dict[str, Any]] = []
        self.finals: list[str] = []

    async def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def on_final(
        self,
        content: str,
        **_kwargs: Any,
    ) -> None:
        self.finals.append(content)


class EventTurnProgress(TurnProgress):
    def __init__(self, turn_id: str, events: list[str]) -> None:
        super().__init__(turn_id)
        self._events = events
        self.emit = self._emit

    async def _emit(self, event: str, _payload: dict[str, Any]) -> None:
        self._events.append(event)

    async def on_final(self, content: str, **_kwargs: Any) -> None:
        self.blocks.append({"type": "text", "content": content})
        await self.emit("chat.turn.end", {"final": content, "blocks": self.blocks})


@pytest.mark.asyncio
async def test_open_flow_rejects_premature_master_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = AeloonCoreOrchestrator(
        Config(
            workspace=workspace,
            data_dir=tmp_path / "data",
            skills=SkillsConfig(enabled=False, external=False, claude_code=False),
        ).normalized()
    )
    model = MasterScriptModel()
    app.model = model

    result = await app.run_turn("Use a Flow for this work", session_id="master")

    assert result.final_content == "Honest partial result."
    assert model.call_count == 4
    assert "finish_turn" in str(result.messages)
    flows = app.flow_control.list_flows("master", include_terminal=True)
    assert flows[0]["status"] == FlowStatus.PARTIAL.value


@pytest.mark.asyncio
async def test_lease_takeover_fences_stalled_terminal_response_and_final_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config(
        workspace=workspace,
        data_dir=tmp_path / "data",
        agents=AgentsConfig(
            defaults=AgentDefaultsConfig(context_compaction=ContextCompactionConfig(enabled=False))
        ),
        skills=SkillsConfig(enabled=False, external=False, claude_code=False),
    ).normalized()
    old = AeloonCoreOrchestrator(config)
    successor = AeloonCoreOrchestrator(config)
    old.model = FinishTurnModel("stale terminal")
    successor.model = FinishTurnModel("successor terminal")

    old_events: list[str] = []
    successor_events: list[str] = []

    old_progress = EventTurnProgress("turn-one", old_events)
    successor_progress = EventTurnProgress("turn-two", successor_events)

    reached_commit = asyncio.Event()
    release_commit = asyncio.Event()
    commit = old._commit_turn_result

    async def stall_before_commit(prompt: str, result: Any) -> Any:
        reached_commit.set()
        await release_commit.wait()
        return await commit(prompt, result)

    monkeypatch.setattr(old, "_commit_turn_result", stall_before_commit)
    old_task = asyncio.create_task(
        old.run_turn("old prompt", session_id="master", on_progress=old_progress)
    )
    commit_wait = asyncio.create_task(reached_commit.wait())
    done, _ = await asyncio.wait(
        {old_task, commit_wait},
        timeout=2,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if old_task in done:
        await old_task
    assert commit_wait in done

    assert "chat.turn.end" not in old_events
    assert any(block.get("content") == "stale terminal" for block in old_progress.blocks)
    with sqlite3.connect(old.flow_store.path) as connection:
        connection.execute(
            "UPDATE flow_session_state SET lease_expires_at = ? "
            "WHERE base_session_id = ? AND active_turn_id = ?",
            ("2000-01-01T00:00:00+00:00", "master", "turn-one"),
        )

    successor_result = await successor.run_turn(
        "new prompt",
        session_id="master",
        on_progress=successor_progress,
    )
    release_commit.set()
    with pytest.raises(FlowTurnConflictError):
        await old_task

    history = successor.sessions.history("master")
    assert successor_result.final_content == "successor terminal"
    assert [record["final_content"] for record in history] == ["successor terminal"]
    assert "chat.turn.end" not in old_events
    assert successor_events.count("chat.turn.end") == 1
    assert any(block.get("content") == "successor terminal" for block in successor_result.blocks)


@pytest.mark.asyncio
async def test_committed_turn_recovers_after_crash_without_rerunning_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config(
        workspace=workspace,
        data_dir=tmp_path / "data",
        agents=AgentsConfig(
            defaults=AgentDefaultsConfig(context_compaction=ContextCompactionConfig(enabled=False))
        ),
        skills=SkillsConfig(enabled=False, external=False, claude_code=False),
    ).normalized()
    crashed = AeloonCoreOrchestrator(config)
    crashed.model = FinishTurnModel("durable terminal")
    first_progress = TurnProgress("stable-turn")

    async def crash_after_commit(
        _committed: Any,
        *,
        expected_prompt: str | None,
    ) -> Any:
        del expected_prompt
        raise RuntimeError("process stopped after durable commit")

    monkeypatch.setattr(crashed, "_recover_turn_commit", crash_after_commit)
    with pytest.raises(RuntimeError, match="after durable commit"):
        await crashed.run_turn(
            "same prompt",
            session_id="master",
            on_progress=first_progress,
        )

    durable = crashed.flow_store.get_turn_commit("master", "stable-turn")
    assert durable is not None
    assert durable.persisted_at is None
    assert crashed.sessions.history("master") == []
    assert first_progress.finals == []

    recovered = AeloonCoreOrchestrator(config)
    recovered.model = FailIfCalledModel()
    second_progress = TurnProgress("stable-turn")
    result = await recovered.run_turn(
        "same prompt",
        session_id="master",
        on_progress=second_progress,
    )

    assert result.final_content == "durable terminal"
    assert recovered.model.call_count == 0
    assert second_progress.finals == ["durable terminal"]
    assert len(recovered.sessions.history("master")) == 1
    assert recovered.flow_store.get_turn_commit("master", "stable-turn").persisted_at is not None

    conflicting = TurnProgress("stable-turn")
    with pytest.raises(ValueError, match="different user prompt"):
        await recovered.run_turn(
            "different prompt",
            session_id="master",
            on_progress=conflicting,
        )


@pytest.mark.asyncio
async def test_conversation_fork_resumes_from_event_head_without_copying_jsonl(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config(
        workspace=workspace,
        data_dir=tmp_path / "data",
        agents=AgentsConfig(
            defaults=AgentDefaultsConfig(
                context_compaction=ContextCompactionConfig(enabled=False)
            )
        ),
        skills=SkillsConfig(enabled=False, external=False, claude_code=False),
    ).normalized()
    app = AeloonCoreOrchestrator(config)
    app.model = FinishTurnModel("source answer")

    await app.run_turn(
        "source request",
        session_id="source",
        on_progress=TurnProgress("source-turn"),
    )
    source_head = app.flow_store.get_session_head("source")
    assert source_head is not None

    legacy_target = app.sessions.session_path("legacy-target")
    legacy_target.write_text(
        '{"type":"turn","session_id":"legacy-target","user_prompt":"legacy request"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a pristine Master session"):
        app._fork_conversation_only_session("source", "legacy-target")
    assert app.flow_store.get_session_head("legacy-target") is None

    fork_head = app._fork_conversation_only_session("source", "fork")

    assert fork_head.head_event_id == source_head.head_event_id
    assert app.sessions.history("fork") == []
    fork_model = FinishTurnModel("fork answer")
    app.model = fork_model
    result = await app.run_turn(
        "fork request",
        session_id="fork",
        on_progress=TurnProgress("fork-turn"),
    )

    assert result.final_content == "fork answer"
    assert len(fork_model.seen_messages) == 1
    assert "source request" in str(fork_model.seen_messages[0])
    assert "source answer" in str(fork_model.seen_messages[0])
    assert [event.turn_id for event in app.flow_store.session_event_ancestry("fork")] == [
        "source-turn",
        "fork-turn",
    ]
    assert [record["turn_id"] for record in app.sessions.history("fork")] == ["fork-turn"]


@pytest.mark.asyncio
async def test_concurrent_duplicate_turn_id_cannot_join_live_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config(
        workspace=workspace,
        data_dir=tmp_path / "data",
        agents=AgentsConfig(
            defaults=AgentDefaultsConfig(context_compaction=ContextCompactionConfig(enabled=False))
        ),
        skills=SkillsConfig(enabled=False, external=False, claude_code=False),
    ).normalized()
    first = AeloonCoreOrchestrator(config)
    duplicate = AeloonCoreOrchestrator(config)
    first.model = FinishTurnModel("first terminal")
    duplicate.model = FinishTurnModel("duplicate terminal")
    entered = asyncio.Event()
    release = asyncio.Event()
    commit = first._commit_turn_result

    async def block_commit(prompt: str, result: Any) -> Any:
        entered.set()
        await release.wait()
        return await commit(prompt, result)

    monkeypatch.setattr(first, "_commit_turn_result", block_commit)
    first_task = asyncio.create_task(
        first.run_turn(
            "same prompt",
            session_id="master",
            on_progress=TurnProgress("shared-turn"),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)

    with pytest.raises(FlowTurnConflictError, match="already active"):
        await duplicate.run_turn(
            "same prompt",
            session_id="master",
            on_progress=TurnProgress("shared-turn"),
        )

    release.set()
    result = await first_task
    assert result.final_content == "first terminal"
    assert duplicate.model.call_count == 0


def test_session_projection_repairs_partial_tail_before_marking_turn_durable(
    tmp_path: Path,
) -> None:
    store = SessionStore(data_dir=tmp_path / "data", workspace=tmp_path)
    path = store.session_path("master")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"type":"turn","turn_id":"crash')
    messages = message_checkpoint("answer")["messages"]

    created = store.append_turn_once(
        session_id="master",
        user_prompt="prompt",
        final_content="answer",
        tools_used=["finish_turn"],
        messages=messages,
        blocks=[{"type": "text", "content": "answer"}],
        usage={"total_tokens": 1},
        turn_id="turn-one",
    )
    replayed = store.append_turn_once(
        session_id="master",
        user_prompt="prompt",
        final_content="answer",
        tools_used=["finish_turn"],
        messages=messages,
        blocks=[{"type": "text", "content": "answer"}],
        usage={"total_tokens": 1},
        turn_id="turn-one",
    )

    assert created is True
    assert replayed is False
    assert path.read_bytes().endswith(b"\n")
    assert [record["turn_id"] for record in store.history("master")] == ["turn-one"]


@pytest.mark.asyncio
async def test_invalid_session_path_is_rejected_before_flow_turn_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = AeloonCoreOrchestrator(
        Config(
            workspace=workspace,
            data_dir=tmp_path / "data",
            skills=SkillsConfig(enabled=False, external=False, claude_code=False),
        ).normalized()
    )
    oversized_session_id = "/" * 200

    with pytest.raises(ValueError, match="too long"):
        await app.run_turn("must not start", session_id=oversized_session_id)

    assert app.flow_store.get_turn_commit(oversized_session_id, "unused") is None
    with sqlite3.connect(app.flow_store.path) as connection:
        state_rows = connection.execute(
            "SELECT COUNT(*) FROM flow_session_state WHERE base_session_id = ?",
            (oversized_session_id,),
        ).fetchone()[0]
        commit_rows = connection.execute(
            "SELECT COUNT(*) FROM flow_turn_commits WHERE base_session_id = ?",
            (oversized_session_id,),
        ).fetchone()[0]
        flow_rows = connection.execute(
            "SELECT COUNT(*) FROM flows WHERE base_session_id = ?",
            (oversized_session_id,),
        ).fetchone()[0]
    assert (state_rows, commit_rows, flow_rows) == (0, 0, 0)


@pytest.mark.asyncio
async def test_completed_flow_requires_successful_or_skipped_nodes(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="must really complete",
        idempotency_key="create",
        nodes=[_node("work", "work")],
    )
    with pytest.raises(ValueError, match="incomplete"):
        await control.complete(
            created["flow_id"],
            outcome=FlowCompletion.COMPLETED,
            summary="not actually complete",
            base_session_id="master",
            idempotency_key="complete",
        )


@pytest.mark.asyncio
async def test_complete_flow_decision_is_idempotent(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="finish consistently",
        idempotency_key="create",
        nodes=[_node("work", "work")],
    )
    first = await control.complete(
        created["flow_id"],
        outcome=FlowCompletion.PARTIAL,
        summary="partial",
        base_session_id="master",
        idempotency_key="complete",
    )
    assert first["changed"] is True
    with pytest.raises(ValueError, match="idempotency key"):
        await control.complete(
            created["flow_id"],
            outcome=FlowCompletion.PARTIAL,
            summary="different partial decision",
            base_session_id="master",
            idempotency_key="complete",
        )


@pytest.mark.asyncio
async def test_finish_turn_cannot_abandon_another_open_flow(tmp_path: Path) -> None:
    control, _ = _flow_control(tmp_path)
    first = control.create_flow(
        base_session_id="master",
        goal="first",
        idempotency_key="create-first",
        nodes=[_node("first", "first")],
    )
    second = control.create_flow(
        base_session_id="master",
        goal="second",
        idempotency_key="create-second",
        nodes=[_node("second", "second")],
    )
    await control.complete(
        first["flow_id"],
        outcome=FlowCompletion.PARTIAL,
        summary="first stopped",
        base_session_id="master",
        idempotency_key="complete-first",
    )

    with pytest.raises(ValueError, match="remain open"):
        await control.finish_turn("too early", base_session_id="master")

    await control.complete(
        second["flow_id"],
        outcome=FlowCompletion.PARTIAL,
        summary="second stopped",
        base_session_id="master",
        idempotency_key="complete-second",
    )
    assert await control.finish_turn("all settled", base_session_id="master") == ("all settled")


@pytest.mark.asyncio
async def test_finish_turn_atomically_seals_paused_flows_until_next_turn(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    created = control.create_flow(
        base_session_id="master",
        goal="pause across turns",
        idempotency_key="create",
        nodes=[_node("work", "work")],
    )
    await control.pause(
        created["flow_id"],
        reason="wait for user input",
        base_session_id="master",
        idempotency_key="pause",
    )

    assert await control.finish_turn("What should I do?", base_session_id="master") == (
        "What should I do?"
    )
    second_controller = FlowControlService(
        store=FlowStore(tmp_path / "data"),
        workers=control.workers,
    )
    with pytest.raises(ValueError, match="turn is sealed"):
        await second_controller.resume(
            created["flow_id"],
            base_session_id="master",
            idempotency_key="resume-in-sealed-turn",
        )

    second_controller.store.begin_turn("master", "turn-two")
    resumed = await second_controller.resume(
        created["flow_id"],
        base_session_id="master",
        idempotency_key="resume-next-turn",
        turn_id="turn-two",
    )
    assert resumed["status"] == FlowStatus.OPEN.value


@pytest.mark.asyncio
async def test_overlapping_master_turn_cannot_reopen_after_terminal_seal(
    tmp_path: Path,
) -> None:
    control, _ = _flow_control(tmp_path)
    control.store.begin_turn("master", "turn-one")
    created = control.create_flow(
        base_session_id="master",
        goal="pause without overlapping turns",
        idempotency_key="create",
        nodes=[_node("work", "work")],
        turn_id="turn-one",
    )
    await control.pause(
        created["flow_id"],
        reason="wait",
        base_session_id="master",
        idempotency_key="pause",
        turn_id="turn-one",
    )
    assert await control.finish_turn(
        "first response",
        base_session_id="master",
        turn_id="turn-one",
    ) == ("first response")

    with pytest.raises(ValueError, match="already active"):
        control.store.begin_turn("master", "turn-one")
    with pytest.raises(ValueError, match="turn is sealed"):
        await control.resume(
            created["flow_id"],
            base_session_id="master",
            idempotency_key="same-turn-resume",
            turn_id="turn-one",
        )
    with pytest.raises(ValueError, match="already active"):
        control.store.begin_turn("master", "turn-two")
    with pytest.raises(ValueError, match="active Master turn"):
        await control.resume(
            created["flow_id"],
            base_session_id="master",
            idempotency_key="overlapping-resume",
            turn_id="turn-two",
        )

    control.store.end_turn("master", "turn-one")
    control.store.begin_turn("master", "turn-two")
    resumed = await control.resume(
        created["flow_id"],
        base_session_id="master",
        idempotency_key="next-turn-resume",
        turn_id="turn-two",
    )
    assert resumed["status"] == FlowStatus.OPEN.value


@pytest.mark.asyncio
async def test_detached_flow_cancellation_blocks_finish_until_teardown(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    master_store = WorkerStore(data_dir)
    runner_store = WorkerStore(data_dir)
    master_manager = WorkerSessionManager(
        store=master_store,
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    workers = WorkerControlService(
        manager=master_manager,
        worker_types=WorkerRegistry.discover(workspace),
    )
    control = FlowControlService(store=FlowStore(data_dir), workers=workers)
    started = asyncio.Event()
    teardown_started = asyncio.Event()
    allow_teardown = asyncio.Event()
    stopped = asyncio.Event()

    async def executor(run: Any, worker: Any) -> WorkerExecutionOutcome:
        del run, worker
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            teardown_started.set()
            await allow_teardown.wait()
            stopped.set()

    runner_manager = WorkerSessionManager(
        store=runner_store,
        executor=executor,
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    created = control.create_flow(
        base_session_id="master",
        goal="cancel detached work safely",
        idempotency_key="create",
        nodes=[_node("work", "long detached work")],
    )
    running = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )
    run_id = running["nodes"][0]["run_id"]
    runner_manager.start(run_id)
    await asyncio.wait_for(started.wait(), timeout=1)

    # Simulate a process death after the Flow cancellation intent commits but
    # before the Worker cancellation request is sent.
    control.store.mutate(
        created["flow_id"],
        base_session_id="master",
        operation="cancel",
        idempotency_key="cancel",
        payload={"reason": "user cancelled"},
        mutation=lambda flow: cancel_flow_state(flow, "user cancelled"),
    )
    cancelling = await control.inspect_flow(
        created["flow_id"],
        base_session_id="master",
    )
    assert cancelling["status"] == FlowStatus.CANCELLING.value
    with pytest.raises(ValueError, match="remain open"):
        await control.finish_turn("too early", base_session_id="master")

    await asyncio.wait_for(teardown_started.wait(), timeout=1)
    assert runner_store.get_run(run_id).cancel_requested_at is not None
    await asyncio.sleep(0.12)
    still_cancelling = await control.inspect_flow(
        created["flow_id"],
        base_session_id="master",
    )
    assert still_cancelling["status"] == FlowStatus.CANCELLING.value

    allow_teardown.set()
    await asyncio.wait_for(stopped.wait(), timeout=1)
    await runner_manager.await_workers([run_id], timeout=1)
    cancelled = await control.inspect_flow(
        created["flow_id"],
        base_session_id="master",
    )
    assert cancelled["status"] == FlowStatus.CANCELLED.value
    assert await control.finish_turn("safe", base_session_id="master") == "safe"


@pytest.mark.asyncio
async def test_crashed_flow_owner_converges_after_its_lease_expires(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = WorkerSessionManager(
        store=WorkerStore(data_dir),
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.03,
    )
    workers = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(workspace),
    )
    control = FlowControlService(store=FlowStore(data_dir), workers=workers)
    created = control.create_flow(
        base_session_id="master",
        goal="recover cancellation after owner crash",
        idempotency_key="create",
        nodes=[_node("work", "detached work")],
    )
    launched = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )
    run_id = launched["nodes"][0]["run_id"]
    running, claimed = manager.store.try_start_run(run_id)
    assert claimed is True
    assert running.status is WorkerRunStatus.RUNNING

    cancelling = await control.cancel(
        created["flow_id"],
        reason="stop after detached owner disappeared",
        base_session_id="master",
        idempotency_key="cancel",
    )
    assert cancelling["status"] == FlowStatus.CANCELLING.value

    await asyncio.sleep(0.05)
    cancelled = await control.inspect_flow(
        created["flow_id"],
        base_session_id="master",
    )

    assert manager.store.get_run(run_id).status is WorkerRunStatus.CANCELLED
    assert cancelled["nodes"][0]["status"] == FlowNodeStatus.CANCELLED.value
    assert cancelled["status"] == FlowStatus.CANCELLED.value
    assert await control.finish_turn("cancelled safely", base_session_id="master") == (
        "cancelled safely"
    )


@pytest.mark.asyncio
async def test_cancelled_flow_blocks_when_tool_owner_dies_with_unknown_side_effects(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = WorkerSessionManager(
        store=WorkerStore(data_dir),
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.03,
    )
    workers = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(workspace),
    )
    control = FlowControlService(store=FlowStore(data_dir), workers=workers)
    created = control.create_flow(
        base_session_id="master",
        goal="report uncertain cancellation honestly",
        idempotency_key="create-unknown-cancellation",
        nodes=[_node("work", "launch a side-effecting tool")],
    )
    launched = await control.advance_flow(
        created["flow_id"],
        base_session_id="master",
        base_turn_id="turn",
        timeout_seconds=0,
    )
    run_id = launched["nodes"][0]["run_id"]
    ready_path = tmp_path / "flow-owner-ready"
    process = _start_flow_tool_owner_process(data_dir, run_id, ready_path)

    try:
        _wait_for_flow_tool_owner(process, ready_path)
        running = manager.store.get_run(run_id)
        assert running.status is WorkerRunStatus.RUNNING
        assert running.active_tool_count == 1

        cancelling = await control.cancel(
            created["flow_id"],
            reason="stop the side-effecting work",
            base_session_id="master",
            idempotency_key="cancel-unknown-cancellation",
        )
        assert cancelling["status"] == FlowStatus.CANCELLING.value

        process.kill()
        process.wait(timeout=5)
        await asyncio.sleep(0.05)

        blocked = await control.inspect_flow(
            created["flow_id"],
            base_session_id="master",
        )
        recovered = manager.store.get_run(run_id)

        assert recovered.status is WorkerRunStatus.FAILED
        assert recovered.result is not None
        assert recovered.result.tool_outcome == "unknown"
        assert blocked["status"] == FlowStatus.BLOCKED.value
        assert blocked["nodes"][0]["status"] == FlowNodeStatus.FAILED.value
        assert blocked["nodes"][0]["result"]["tool_outcome"] == "unknown"
        assert "Cancellation outcome is unknown" in blocked["completion_summary"]
        assert "inspect side effects" in blocked["completion_summary"]
        assert "Cancellation outcome is unknown" in blocked["termination_reason"]
        assert await control.finish_turn(
            "Cancellation is blocked pending side-effect inspection.",
            base_session_id="master",
        ) == "Cancellation is blocked pending side-effect inspection."
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
