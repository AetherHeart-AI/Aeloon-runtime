"""Durable v2 Worker lifecycle, migration, and concurrency tests."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from aeloon_core.flows import (
    FlowNodeSpec,
    FlowNodeStatus,
    FlowRunBinding,
    FlowStore,
    cancel_flow_state,
)
from aeloon_core.runner import run_worker_runner
from aeloon_core.tools.base import FunctionTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.worker_control import WorkerControlService
from aeloon_core.worker_manager import WorkerExecutionOutcome, WorkerSessionManager
from aeloon_core.worker_sessions import (
    BudgetGrant,
    ContextEnvelope,
    IdempotencyConflictError,
    PermissionSnapshot,
    ResultEnvelope,
    WaitingRequest,
    WorkerReport,
    WorkerRunFencedError,
    WorkerRunStatus,
    WorkerStore,
)
from aeloon_core.workers import WorkerRegistry, parse_worker


def _snapshot():
    return parse_worker(
        "---\nid: test-worker\ndescription: Test work\n---\nDeliver a verified result.\n"
    )


def _context(
    objective: str,
    *,
    tools: tuple[str, ...] = ("read", "complete_work", "request_master"),
) -> ContextEnvelope:
    return ContextEnvelope(
        objective=objective,
        permissions=PermissionSnapshot(
            tool_names=tools,
            skills_enabled=False,
        ),
        budget=BudgetGrant(max_tokens=10_000, max_seconds=60, max_tool_calls=10),
    )


def _control(
    store: WorkerStore,
    tmp_path: Path,
    *,
    default_budget: BudgetGrant,
) -> WorkerControlService:
    return WorkerControlService(
        manager=WorkerSessionManager(store=store),
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
        default_budget=default_budget,
    )


def _create_running(store: WorkerStore, *, objective: str = "do work"):
    worker, run, created = store.create_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context(objective),
        idempotency_key=f"spawn:{objective}",
    )
    assert created
    running, claimed = store.try_start_run(run.run_id)
    assert claimed
    return worker, running


def _complete(store: WorkerStore, run_id: str) -> None:
    run = store.get_run(run_id)
    store.complete_run(
        run_id,
        ResultEnvelope(
            worker_id=run.worker_id,
            run_id=run.run_id,
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="done"),
            tool_outcome="known",
        ),
        checkpoint={"messages": [{"role": "assistant", "content": "checkpoint"}]},
    )


def _create_bound_flow_running(
    data_dir: Path,
) -> tuple[FlowStore, WorkerStore, str, object]:
    flow_store = FlowStore(data_dir)
    flow, _ = flow_store.create_flow(
        base_session_id="master",
        goal="runtime execution fence",
        nodes=[
            FlowNodeSpec(
                node_id="work",
                worker_type_id=_snapshot().id,
                objective="runtime execution fence",
            )
        ],
        idempotency_key="create-runtime-fence-flow",
    )
    store = WorkerStore(data_dir)
    worker, run, _ = store.create_worker(
        base_session_id="master",
        base_turn_id=f"flow:{flow.flow_id}",
        snapshot=_snapshot(),
        context=_context("runtime execution fence"),
        idempotency_key="runtime-fence-run",
    )

    def bind(current):
        node = current.node("work")
        node.status = FlowNodeStatus.RUNNING
        node.attempt = 1
        node.worker_id = worker.worker_id
        node.current_run_id = run.run_id
        node.runs = [
            FlowRunBinding(
                generation=node.generation,
                attempt=node.attempt,
                worker_id=worker.worker_id,
                run_id=run.run_id,
                status=WorkerRunStatus.QUEUED.value,
                created_at=run.created_at,
            )
        ]

    flow_store.update_runtime(
        flow.flow_id,
        base_session_id="master",
        mutation=bind,
    )
    store.activate_run(run.run_id)
    running, claimed = store.try_start_run(run.run_id)
    assert claimed is True
    return flow_store, store, flow.flow_id, running


def _commit_flow_cancellation_only(flow_store: FlowStore, flow_id: str) -> None:
    flow_store.mutate(
        flow_id,
        base_session_id="master",
        operation="cancel",
        idempotency_key="runtime-fence-cancel",
        payload={"reason": "cancel before Worker fence"},
        mutation=lambda flow: cancel_flow_state(flow, "cancel before Worker fence"),
    )


def _start_tool_owner_process(
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


def _wait_for_process_ready(process: subprocess.Popen[str], ready_path: Path) -> None:
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


def test_cancellation_revokes_tool_execution_authority(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path / "data")
    _, running = _create_running(store)

    store.require_run_execution_authority(running.run_id)
    requested, changed = store.try_cancel_run(running.run_id)

    assert changed is True
    assert requested.status is WorkerRunStatus.RUNNING
    with pytest.raises(WorkerRunFencedError, match="authority was revoked"):
        store.require_run_execution_authority(running.run_id)


def test_flow_cancel_between_guard_and_begin_fences_the_tool_boundary(
    tmp_path: Path,
) -> None:
    flow_store, store, flow_id, running = _create_bound_flow_running(tmp_path / "data")
    store.require_run_execution_authority(running.run_id)

    _commit_flow_cancellation_only(flow_store, flow_id)

    with pytest.raises(WorkerRunFencedError, match="Flow execution authority was revoked"):
        store.begin_tool_execution(running.run_id)
    fenced = store.get_run(running.run_id)
    assert fenced.cancel_requested_at is not None
    assert fenced.active_tool_count == 0


def test_flow_cancel_does_not_block_inflight_tool_teardown(tmp_path: Path) -> None:
    flow_store, store, flow_id, running = _create_bound_flow_running(tmp_path / "data")
    store.begin_tool_execution(running.run_id)

    _commit_flow_cancellation_only(flow_store, flow_id)

    store.end_tool_execution(running.run_id)
    assert store.get_run(running.run_id).active_tool_count == 0
    with pytest.raises(WorkerRunFencedError, match="Flow execution authority was revoked"):
        store.require_run_execution_authority(running.run_id)
    cancelled, settled = store.acknowledge_cancel_run(running.run_id)
    assert settled is True
    assert cancelled.status is WorkerRunStatus.CANCELLED


def test_flow_cancel_wins_race_with_worker_finalization(tmp_path: Path) -> None:
    flow_store, store, flow_id, running = _create_bound_flow_running(tmp_path / "data")
    _commit_flow_cancellation_only(flow_store, flow_id)
    result = ResultEnvelope(
        worker_id=running.worker_id,
        run_id=running.run_id,
        status=WorkerRunStatus.COMPLETED,
        report=WorkerReport(summary="must not commit"),
        tool_outcome="known",
    )

    cancelled, changed = store.try_finalize_run(
        running.run_id,
        result,
        checkpoint={"messages": [{"role": "assistant", "content": "must not persist"}]},
    )

    assert changed is True
    assert cancelled.status is WorkerRunStatus.CANCELLED
    assert cancelled.cancel_requested_at is not None
    assert cancelled.result is None
    assert store.load_checkpoint(running.run_id) is None


@pytest.mark.asyncio
async def test_lease_expiry_waits_for_inflight_tool_teardown(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path / "data")
    _, running = _create_running(store)
    started = asyncio.Event()
    release = asyncio.Event()
    late_file = tmp_path / "late.txt"

    class NoArgs(BaseModel):
        pass

    async def mutating_tool() -> str:
        started.set()
        try:
            await release.wait()
        finally:
            late_file.write_text("cleanup finished", encoding="utf-8")
        return "done"

    tools = ToolRegistry(
        execution_guard=lambda _tool: store.require_run_execution_authority(
            running.run_id
        ),
        execution_started=lambda _tool: store.begin_tool_execution(running.run_id),
        execution_finished=lambda _tool: store.end_tool_execution(running.run_id),
    )
    tools.register(
        FunctionTool(
            name="mutate",
            description="Run a blocking mutation.",
            args_model=NoArgs,
            handler=mutating_tool,
            concurrency_mode="mutating",
        )
    )

    execution = asyncio.create_task(tools.execute("mutate", {}))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert store.get_run(running.run_id).active_tool_count == 1

    requested, changed = store.try_cancel_run(running.run_id)
    assert changed is True
    assert requested.status is WorkerRunStatus.RUNNING
    assert store.expire_stale_running_runs(stale_before="9999-12-31T23:59:59+00:00") == []
    assert store.get_run(running.run_id).status is WorkerRunStatus.RUNNING
    assert late_file.exists() is False

    release.set()
    await execution
    assert late_file.read_text(encoding="utf-8") == "cleanup finished"
    assert store.get_run(running.run_id).active_tool_count == 0
    cancelled, settled = store.acknowledge_cancel_run(running.run_id)
    assert settled is True
    assert cancelled.status is WorkerRunStatus.CANCELLED


@pytest.mark.parametrize(
    "request_cancel",
    (False, True),
)
def test_dead_tool_owner_is_recovered_only_after_kernel_releases_its_lease(
    tmp_path: Path,
    request_cancel: bool,
) -> None:
    data_dir = tmp_path / "data"
    controller = WorkerStore(data_dir)
    _, queued, created = controller.create_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("hard-crash tool"),
        idempotency_key=f"hard-crash:{request_cancel}",
    )
    assert created
    ready_path = tmp_path / "owner-ready"
    process = _start_tool_owner_process(data_dir, queued.run_id, ready_path)
    try:
        _wait_for_process_ready(process, ready_path)
        observer = WorkerStore(data_dir)
        live = observer.get_run(queued.run_id)
        assert live.status is WorkerRunStatus.RUNNING
        assert live.active_tool_count == 1
        assert live.execution_owner_token == ready_path.read_text(encoding="utf-8")

        if request_cancel:
            requested, changed = controller.try_cancel_run(queued.run_id)
            assert changed is True
            assert requested.status is WorkerRunStatus.RUNNING

        # Even an arbitrarily old timestamp is insufficient while the exact
        # owner process still holds its kernel lease.
        assert (
            observer.expire_stale_running_runs(
                stale_before="9999-12-31T23:59:59+00:00"
            )
            == []
        )
        assert observer.get_run(queued.run_id).active_tool_count == 1

        process.kill()
        process.wait(timeout=5)
        expired = observer.expire_stale_running_runs(
            stale_before="9999-12-31T23:59:59+00:00"
        )

        assert [run.run_id for run in expired] == [queued.run_id]
        recovered = observer.get_run(queued.run_id)
        assert recovered.status is WorkerRunStatus.FAILED
        assert recovered.active_tool_count == 0
        assert recovered.result is not None
        assert recovered.result.tool_outcome == "unknown"
        assert recovered.result.report is not None
        assert "owner process exited while a tool was in flight" in (
            recovered.result.report.summary
        )
        assert "External or descendant side effects" in recovered.result.report.summary
        assert "does not roll them back" in recovered.result.report.summary
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _make_v1_database(data_dir: Path) -> Path:
    path = data_dir / "worker-control.sqlite3"
    data_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE worker_sessions (
              worker_id TEXT PRIMARY KEY,
              base_session_id TEXT NOT NULL,
              profile_json TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE worker_runs (
              run_id TEXT PRIMARY KEY,
              worker_id TEXT NOT NULL,
              status TEXT NOT NULL,
              context_json TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE worker_checkpoints (
              run_id TEXT PRIMARY KEY,
              checkpoint_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE worker_ui_events (
              sequence INTEGER PRIMARY KEY,
              run_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE worker_ui_state (
              singleton INTEGER PRIMARY KEY,
              event_count INTEGER NOT NULL
            );
            INSERT INTO worker_sessions VALUES ('old-worker', 'master', '{}', 'idle', 'now');
            INSERT INTO worker_runs
              VALUES ('old-run', 'old-worker', 'completed', '{}', 'key', 'now');
            INSERT INTO worker_checkpoints VALUES ('old-run', '{}', 'now');
            INSERT INTO worker_ui_events VALUES (1, 'old-run', 'tool', '{}', 'now');
            INSERT INTO worker_ui_state VALUES (1, 1);
            PRAGMA user_version=1;
            """
        )
    transcript = data_dir / "worker-sessions" / "old-worker" / "transcript.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("legacy private data\n", encoding="utf-8")
    return path


def test_v1_migration_is_destructive_atomic_and_idempotent(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = _make_v1_database(data_dir)

    store = WorkerStore(data_dir)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {row[1] for row in connection.execute("PRAGMA table_info(worker_sessions)")}
        assert "snapshot_json" in columns
        assert "profile_json" not in columns
        for table in (
            "worker_sessions",
            "worker_runs",
            "worker_checkpoints",
            "worker_ui_events",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert connection.execute("SELECT event_count FROM worker_ui_state").fetchone()[0] == 0
    assert not (data_dir / "worker-sessions").exists()

    worker, _, _ = store.create_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("new work"),
        idempotency_key="new",
    )
    current_transcript = data_dir / "worker-sessions" / "current.txt"
    current_transcript.parent.mkdir(parents=True)
    current_transcript.write_text("v2", encoding="utf-8")

    reopened = WorkerStore(data_dir)
    assert reopened.get_worker(worker.worker_id).snapshot.digest == worker.snapshot.digest
    assert current_transcript.read_text(encoding="utf-8") == "v2"


def test_failed_transcript_cleanup_rolls_back_schema_migration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = _make_v1_database(data_dir)

    with patch(
        "aeloon_core.worker_sessions.shutil.rmtree",
        side_effect=OSError("cannot remove transcript"),
    ):
        with pytest.raises(OSError, match="cannot remove transcript"):
            WorkerStore(data_dir)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM worker_sessions").fetchone()[0] == 1


def test_v5_schema_missing_a_runtime_column_blocks_startup(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute("ALTER TABLE worker_runs DROP COLUMN cancel_requested_at")

    with pytest.raises(RuntimeError, match="cancel_requested_at"):
        WorkerStore(tmp_path)


def test_v2_activation_migration_preserves_worker_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = WorkerStore(data_dir)
    worker, run, _ = store.create_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("preserve this run"),
        idempotency_key="preserved",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("ALTER TABLE worker_runs DROP COLUMN activated_at")
        connection.execute("PRAGMA user_version=2")

    migrated = WorkerStore(data_dir)

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {row[1] for row in connection.execute("PRAGMA table_info(worker_runs)")}
        assert "activated_at" in columns
        assert "active_tool_count" in columns
        assert "execution_owner_token" in columns
    assert migrated.get_worker(worker.worker_id) == worker
    assert migrated.get_run(run.run_id).context.objective == "preserve this run"
    assert migrated.get_run(run.run_id).activated_at is None


def test_v3_execution_fence_migration_preserves_worker_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = WorkerStore(data_dir)
    worker, run, _ = store.create_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("preserve v3 data"),
        idempotency_key="preserved-v3",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("ALTER TABLE worker_runs DROP COLUMN active_tool_count")
        connection.execute("PRAGMA user_version=3")

    migrated = WorkerStore(data_dir)

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {row[1] for row in connection.execute("PRAGMA table_info(worker_runs)")}
        assert "active_tool_count" in columns
        assert "execution_owner_token" in columns
    assert migrated.get_worker(worker.worker_id) == worker
    assert migrated.get_run(run.run_id).context.objective == "preserve v3 data"
    assert migrated.get_run(run.run_id).active_tool_count == 0


def test_v4_unknown_active_tool_owner_migrates_fail_closed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    legacy = WorkerStore(data_dir)
    _, run = _create_running(legacy, objective="legacy in-flight tool")
    legacy.begin_tool_execution(run.run_id)
    with sqlite3.connect(legacy.path) as connection:
        connection.execute("ALTER TABLE worker_runs DROP COLUMN execution_owner_token")
        connection.execute("PRAGMA user_version=4")

    migrated = WorkerStore(data_dir)

    with sqlite3.connect(migrated.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {row[1] for row in connection.execute("PRAGMA table_info(worker_runs)")}
        assert "execution_owner_token" in columns
    current = migrated.get_run(run.run_id)
    assert current.execution_owner_token is None
    assert current.active_tool_count == 1
    assert (
        migrated.expire_stale_running_runs(
            stale_before="9999-12-31T23:59:59+00:00"
        )
        == []
    )
    assert migrated.get_run(run.run_id).status is WorkerRunStatus.RUNNING


def test_run_execution_mutations_are_fenced_to_the_claiming_owner(
    tmp_path: Path,
) -> None:
    owner = WorkerStore(tmp_path / "data")
    _, run = _create_running(owner, objective="owner-fenced work")
    observer = WorkerStore(tmp_path / "data")

    assert run.execution_owner_token == owner.execution_owner_token
    assert observer.execution_owner_token != owner.execution_owner_token
    assert observer.refresh_run_lease(run.run_id) is False
    with pytest.raises(WorkerRunFencedError, match="authority was revoked"):
        observer.require_run_execution_authority(run.run_id)
    with pytest.raises(WorkerRunFencedError, match="authority was revoked"):
        observer.begin_tool_execution(run.run_id)

    owner.begin_tool_execution(run.run_id)
    with pytest.raises(RuntimeError, match="marker is unbalanced"):
        observer.end_tool_execution(run.run_id)
    assert owner.get_run(run.run_id).active_tool_count == 1
    owner.end_tool_execution(run.run_id)

    foreign_result = ResultEnvelope(
        worker_id=run.worker_id,
        run_id=run.run_id,
        status=WorkerRunStatus.FAILED,
        report=WorkerReport(summary="foreign result"),
        tool_outcome="unknown",
    )
    with pytest.raises(WorkerRunFencedError, match="another owner"):
        observer.try_finalize_run(run.run_id, foreign_result)

    owner.try_cancel_run(run.run_id)
    unchanged, settled = observer.acknowledge_cancel_run(run.run_id)
    assert settled is False
    assert unchanged.status is WorkerRunStatus.RUNNING
    cancelled, settled = owner.acknowledge_cancel_run(run.run_id)
    assert settled is True
    assert cancelled.status is WorkerRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_followup_runs_use_current_budget_instead_of_legacy_cap(
    tmp_path: Path,
) -> None:
    store = WorkerStore(tmp_path / "data")
    legacy_budget = BudgetGrant(
        max_tokens=128_000,
        max_seconds=60,
        max_tool_calls=25,
    )
    current_budget = BudgetGrant(max_seconds=120)
    legacy = _control(store, tmp_path, default_budget=legacy_budget)

    completed_spawn = await legacy.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="legacy completed run",
        idempotency_key="completed",
        detached=True,
    )
    completed_run, claimed = store.try_start_run(completed_spawn["run_id"])
    assert claimed
    _complete(store, completed_run.run_id)

    waiting_spawn = await legacy.spawn_worker(
        base_session_id="master",
        worker_type_id="explorer",
        objective="legacy waiting run",
        idempotency_key="waiting",
        detached=True,
    )
    waiting_run, claimed = store.try_start_run(waiting_spawn["run_id"])
    assert claimed
    waiting_request = WaitingRequest(summary="need context", question="Which target?")
    store.complete_run(
        waiting_run.run_id,
        ResultEnvelope(
            worker_id=waiting_run.worker_id,
            run_id=waiting_run.run_id,
            status=WorkerRunStatus.WAITING_FOR_CONTEXT,
            report=WorkerReport(
                summary=waiting_request.summary,
                unresolved=(waiting_request.question,),
            ),
            tool_outcome="known",
        ),
        checkpoint={"messages": [{"role": "assistant", "content": "checkpoint"}]},
        waiting_request=waiting_request,
    )

    current = _control(store, tmp_path, default_budget=current_budget)
    reused = await current.reuse_worker(
        base_session_id="master",
        worker_id=completed_spawn["worker_id"],
        objective="continue without the legacy cap",
        idempotency_key="reuse",
    )
    resumed = await current.resume_worker(
        waiting_run.run_id,
        response="Use target A",
        idempotency_key="resume",
        base_session_id="master",
    )

    assert store.get_run(reused["run_id"]).context.budget == current_budget
    assert store.get_run(resumed["run_id"]).context.budget == current_budget
    assert current_budget.max_tokens is None
    assert current_budget.max_tool_calls is None


@pytest.mark.asyncio
async def test_waiting_is_settled_and_checkpoint_request_result_are_atomic(
    tmp_path: Path,
) -> None:
    store = WorkerStore(tmp_path)
    _, run = _create_running(store)
    request = WaitingRequest(summary="inspected the inputs", question="Which target?")
    result = ResultEnvelope(
        worker_id=run.worker_id,
        run_id=run.run_id,
        status=WorkerRunStatus.WAITING_FOR_CONTEXT,
        report=WorkerReport(
            summary=request.summary,
            unresolved=(request.question,),
        ),
        tool_outcome="known",
    )

    waiting, changed = store.try_finalize_run(
        run.run_id,
        result,
        checkpoint={"messages": [{"role": "assistant", "content": "exact state"}]},
        waiting_request=request,
    )

    assert changed
    assert waiting.status is WorkerRunStatus.WAITING_FOR_CONTEXT
    assert waiting.waiting_request == request
    assert store.load_checkpoint(run.run_id) == {
        "messages": [{"role": "assistant", "content": "exact state"}]
    }
    manager = WorkerSessionManager(store=store)
    returned = await manager.await_workers([run.run_id], timeout=0.01)
    assert returned[0].status is WorkerRunStatus.WAITING_FOR_CONTEXT

    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT status, result_json, waiting_request_json FROM worker_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()
        assert row == (
            "waiting_for_context",
            result.model_dump_json(),
            request.model_dump_json(),
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM worker_checkpoints WHERE run_id = ?", (run.run_id,)
            ).fetchone()[0]
            == 1
        )


@pytest.mark.asyncio
async def test_running_lease_heartbeats_without_a_progress_observer(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    runner_store = WorkerStore(data_dir)
    started = asyncio.Event()
    release = asyncio.Event()

    async def executor(run, worker):
        del run, worker
        started.set()
        await release.wait()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="done"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    runner = WorkerSessionManager(
        store=runner_store,
        executor=executor,
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    _, run, _ = await runner.spawn_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("long work without a UI observer"),
        idempotency_key="spawn",
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    observer = WorkerSessionManager(
        store=WorkerStore(data_dir),
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    await asyncio.sleep(0.12)
    observed = await observer.await_workers([run.run_id], timeout=0)
    assert observed[0].status is WorkerRunStatus.RUNNING

    release.set()
    settled = await runner.await_workers([run.run_id], timeout=1)
    assert settled[0].status is WorkerRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_expired_running_lease_fails_without_replaying_the_run(
    tmp_path: Path,
) -> None:
    store = WorkerStore(tmp_path / "data")
    worker, run = _create_running(store, objective="orphaned work")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE worker_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run.run_id),
        )

    restarted = WorkerSessionManager(
        store=WorkerStore(tmp_path / "data"),
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    settled = await restarted.await_workers([run.run_id], timeout=0)

    assert settled[0].status is WorkerRunStatus.FAILED
    assert settled[0].result is not None
    assert settled[0].result.tool_outcome == "unknown"
    assert "lease expired" in settled[0].result.report.summary
    assert store.get_worker(worker.worker_id).status.value == "idle"
    assert store.refresh_run_lease(run.run_id) is False

    late_result = ResultEnvelope(
        worker_id=worker.worker_id,
        run_id=run.run_id,
        status=WorkerRunStatus.COMPLETED,
        report=WorkerReport(summary="late result"),
        tool_outcome="known",
    )
    unchanged, changed = store.try_finalize_run(
        run.run_id,
        late_result,
        checkpoint={"messages": [{"role": "assistant", "content": "late"}]},
    )
    assert changed is False
    assert unchanged.status is WorkerRunStatus.FAILED


@pytest.mark.asyncio
async def test_expired_cancel_request_settles_as_cancelled(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path / "data")
    _, run = _create_running(store, objective="orphaned cancellation")
    requested, changed = store.try_cancel_run(run.run_id)
    assert changed
    assert requested.status is WorkerRunStatus.RUNNING
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE worker_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run.run_id),
        )

    restarted = WorkerSessionManager(
        store=WorkerStore(tmp_path / "data"),
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    settled = await restarted.await_workers([run.run_id], timeout=0)

    assert settled[0].status is WorkerRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_runner_startup_sweeps_orphaned_running_leases(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path / "data")
    _, run = _create_running(store, objective="orphaned runner work")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE worker_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run.run_id),
        )

    events: list[str] = []

    async def lifecycle(event, current):
        del current
        events.append(event)

    restarted = WorkerSessionManager(
        store=WorkerStore(tmp_path / "data"),
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
        on_lifecycle=lifecycle,
    )
    assert await restarted.run_queued() == []
    assert store.get_run(run.run_id).status is WorkerRunStatus.FAILED
    assert events == ["failed"]


def test_reusable_outcomes_require_an_atomic_checkpoint(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path)
    _, run = _create_running(store)

    with pytest.raises(ValueError, match="completed requires an atomic checkpoint"):
        store.try_finalize_run(
            run.run_id,
            ResultEnvelope(
                worker_id=run.worker_id,
                run_id=run.run_id,
                status=WorkerRunStatus.COMPLETED,
                report=WorkerReport(summary="done"),
            ),
        )


def test_concurrent_resume_creates_one_exact_continuation(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path)
    _, source = _create_running(store)
    request = WaitingRequest(summary="blocked", question="Choose A or B")
    store.try_finalize_run(
        source.run_id,
        ResultEnvelope(
            worker_id=source.worker_id,
            run_id=source.run_id,
            status=WorkerRunStatus.WAITING_FOR_CONTEXT,
            report=WorkerReport(summary="blocked", unresolved=(request.question,)),
        ),
        checkpoint={"messages": [{"role": "tool", "content": "waiting"}]},
        waiting_request=request,
    )
    response = _context("Choose A")

    def resume():
        return store.create_resume_run(
            source_run_id=source.run_id,
            context=response,
            idempotency_key="same-resume",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: resume(), range(2)))

    assert len({run.run_id for run, _ in outcomes}) == 1
    assert sorted(created for _, created in outcomes) == [False, True]
    continuation = outcomes[0][0]
    assert continuation.source_run_id == source.run_id
    assert continuation.run_sequence == source.run_sequence + 1
    assert store.get_run(source.run_id).status is WorkerRunStatus.WAITING_FOR_CONTEXT
    assert store.load_checkpoint(source.run_id) == {
        "messages": [{"role": "tool", "content": "waiting"}]
    }
    with pytest.raises(ValueError, match="newer run"):
        store.create_resume_run(
            source_run_id=source.run_id,
            context=_context("Choose B"),
            idempotency_key="different-resume",
        )


def test_reuse_requires_checkpoint_and_cannot_expand_permissions(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path)
    _, running = _create_running(store)
    _complete(store, running.run_id)

    expanded = _context(
        "follow up",
        tools=("read", "exec", "complete_work", "request_master"),
    )
    with pytest.raises(ValueError, match="expand Worker permissions"):
        store.create_run(
            worker_id=running.worker_id,
            context=expanded,
            idempotency_key="expanded",
        )

    followup = _context("follow up")
    first, created = store.create_run(
        worker_id=running.worker_id,
        context=followup,
        idempotency_key="reuse",
    )
    replay, replay_created = store.create_run(
        worker_id=running.worker_id,
        context=followup,
        idempotency_key="reuse",
    )
    assert created is True
    assert replay_created is False
    assert replay.run_id == first.run_id


def test_spawn_idempotency_conflict_checks_snapshot_and_objective(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path)
    store.create_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("first"),
        idempotency_key="same-key",
    )
    with pytest.raises(IdempotencyConflictError, match="context envelope"):
        store.create_worker(
            base_session_id="master",
            snapshot=_snapshot(),
            context=_context("different"),
            idempotency_key="same-key",
        )


def test_reuse_keeps_persisted_worker_snapshot_after_catalog_change(
    tmp_path: Path,
) -> None:
    definition = tmp_path / ".aeloon-core" / "workers" / "builder.md"
    definition.parent.mkdir(parents=True)
    definition.write_text(
        "---\nid: builder\ndescription: Project builder v1\n---\nUse workflow v1.\n",
        encoding="utf-8",
    )
    first_catalog = WorkerRegistry.discover(tmp_path)
    store = WorkerStore(tmp_path / "data")
    worker, run, _ = store.create_worker(
        base_session_id="master",
        snapshot=first_catalog.get("builder"),
        context=_context("first"),
        idempotency_key="spawn",
    )
    store.try_start_run(run.run_id)
    _complete(store, run.run_id)

    definition.write_text(
        "---\nid: builder\ndescription: Project builder v2\n---\nUse workflow v2.\n",
        encoding="utf-8",
    )
    current_catalog = WorkerRegistry.discover(tmp_path)
    followup, _ = store.create_run(
        worker_id=worker.worker_id,
        context=_context("follow up"),
        idempotency_key="reuse",
    )

    persisted = store.get_worker(worker.worker_id).snapshot
    assert persisted.digest == first_catalog.get("builder").digest
    assert persisted.prompt == "Use workflow v1."
    assert persisted.digest != current_catalog.get("builder").digest
    assert followup.source_run_id == run.run_id


@pytest.mark.asyncio
async def test_cross_worker_runs_execute_concurrently(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path / "data")
    active = 0
    maximum = 0
    both_started = asyncio.Event()
    lock = asyncio.Lock()

    async def executor(run, worker):
        nonlocal active, maximum
        del worker
        async with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        async with lock:
            active -= 1
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary=f"done {run.run_id}"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    manager = WorkerSessionManager(store=store, executor=executor, max_concurrency=2)
    control = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
    )
    first, second = await asyncio.gather(
        control.spawn_worker(
            base_session_id="master",
            worker_type_id="builder",
            objective="first",
            idempotency_key="first",
        ),
        control.spawn_worker(
            base_session_id="master",
            worker_type_id="explorer",
            objective="second",
            idempotency_key="second",
        ),
    )
    settled = await control.await_workers([first["run_id"], second["run_id"]], timeout=2)

    assert maximum == 2
    assert {item["status"] for item in settled} == {"completed"}


@pytest.mark.asyncio
async def test_runner_cannot_start_a_flow_reservation_without_current_binding(
    tmp_path: Path,
) -> None:
    store = WorkerStore(tmp_path / "data")
    executed = asyncio.Event()

    async def executor(run, worker):
        del run, worker
        executed.set()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="done"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    manager = WorkerSessionManager(store=store, executor=executor)
    _, run, _ = await manager.spawn_worker(
        base_session_id="master",
        base_turn_id="flow:test-flow",
        snapshot=_snapshot(),
        context=_context("reserved Flow work"),
        idempotency_key="flow-reservation",
        start=False,
    )

    reserved, claimed = store.try_start_run(run.run_id)
    assert claimed is False
    assert reserved.status is WorkerRunStatus.QUEUED
    assert manager.start_queued() == []
    await asyncio.sleep(0)
    assert executed.is_set() is False
    assert store.get_run(run.run_id).status is WorkerRunStatus.QUEUED

    manager.start_existing(run.run_id)
    assert store.get_run(run.run_id).activated_at is not None
    settled = await manager.await_workers([run.run_id], timeout=1)
    assert settled[0].status is WorkerRunStatus.CANCELLED
    assert executed.is_set() is False


@pytest.mark.asyncio
async def test_detached_runner_recovers_an_activated_flow_run(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    flow_store = FlowStore(data_dir)
    flow, _ = flow_store.create_flow(
        base_session_id="master",
        goal="recover activated Flow work",
        nodes=[
            FlowNodeSpec(
                node_id="work",
                worker_type_id=_snapshot().id,
                objective="recover activated Flow work",
            )
        ],
        idempotency_key="create-flow",
    )
    master = WorkerSessionManager(store=WorkerStore(data_dir))
    _, run, _ = await master.spawn_worker(
        base_session_id="master",
        base_turn_id=f"flow:{flow.flow_id}",
        snapshot=_snapshot(),
        context=_context("recover activated Flow work"),
        idempotency_key="flow-recovery",
        start=False,
    )

    def bind(current):
        node = current.node("work")
        node.status = FlowNodeStatus.RUNNING
        node.attempt = 1
        node.worker_id = run.worker_id
        node.current_run_id = run.run_id
        node.runs = [
            FlowRunBinding(
                generation=node.generation,
                attempt=node.attempt,
                worker_id=run.worker_id,
                run_id=run.run_id,
                status=WorkerRunStatus.QUEUED.value,
                created_at=run.created_at,
            )
        ]

    flow_store.update_runtime(
        flow.flow_id,
        base_session_id="master",
        mutation=bind,
    )

    # Flow binding commits before this activation. Simulate the Master process
    # disappearing immediately after the durable activation but before claim.
    master.start_existing(run.run_id)
    assert master.store.get_run(run.run_id).status is WorkerRunStatus.QUEUED

    executed = asyncio.Event()

    async def executor(current, worker):
        del current, worker
        executed.set()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="recovered"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    runner = WorkerSessionManager(
        store=WorkerStore(data_dir),
        executor=executor,
    )
    queued = runner.start_queued()
    settled = await runner.await_workers([run.run_id], timeout=1)

    assert [item.run_id for item in queued] == [run.run_id]
    assert executed.is_set()
    assert settled[0].status is WorkerRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_busy_worker_cannot_be_reused_and_cancel_then_archive_is_normal(
    tmp_path: Path,
) -> None:
    store = WorkerStore(tmp_path / "data")
    started = asyncio.Event()
    release = asyncio.Event()

    async def executor(run, worker):
        del run, worker
        started.set()
        await release.wait()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="done"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    manager = WorkerSessionManager(store=store, executor=executor)
    control = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
    )
    spawned = await control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="first",
        idempotency_key="spawn",
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    with pytest.raises(ValueError, match="completed or partial"):
        await control.reuse_worker(
            base_session_id="master",
            worker_id=spawned["worker_id"],
            objective="too soon",
            idempotency_key="reuse",
        )
    release.set()
    await control.await_workers([spawned["run_id"]], timeout=1)
    reused = await control.reuse_worker(
        base_session_id="master",
        worker_id=spawned["worker_id"],
        objective="follow up",
        idempotency_key="reuse",
    )
    await control.cancel_worker(reused["run_id"], base_session_id="master")
    archived = control.archive_worker(spawned["worker_id"], base_session_id="master")
    assert archived["status"] == "archived"


@pytest.mark.asyncio
async def test_detached_cancellation_waits_for_runner_teardown(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    master_store = WorkerStore(data_dir)
    runner_store = WorkerStore(data_dir)
    started = asyncio.Event()
    teardown_started = asyncio.Event()
    allow_teardown = asyncio.Event()
    stopped = asyncio.Event()

    async def executor(run, worker):
        del run, worker
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            teardown_started.set()
            await allow_teardown.wait()
            stopped.set()

    master_manager = WorkerSessionManager(
        store=master_store,
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    runner_manager = WorkerSessionManager(
        store=runner_store,
        executor=executor,
        heartbeat_interval_seconds=0.01,
        lease_timeout_seconds=0.05,
    )
    control = WorkerControlService(
        manager=master_manager,
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
    )
    spawned = await control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="long detached work",
        idempotency_key="spawn",
        detached=True,
    )
    runner_manager.start(spawned["run_id"])
    await asyncio.wait_for(started.wait(), timeout=1)

    requested = await control.cancel_worker(spawned["run_id"], base_session_id="master")

    assert requested["status"] == "running"
    assert requested["action"] == "cancellation_requested"
    assert requested["cancel_requested"] is True
    with pytest.raises(ValueError, match="active worker runs"):
        control.archive_worker(spawned["worker_id"], base_session_id="master")
    await asyncio.wait_for(teardown_started.wait(), timeout=1)
    await asyncio.sleep(0.12)
    still_tearing_down = (
        await control.await_workers([spawned["run_id"]], timeout=0, base_session_id="master")
    )[0]
    assert still_tearing_down["status"] == "running"
    assert master_store.get_run(spawned["run_id"]).status is WorkerRunStatus.RUNNING
    allow_teardown.set()
    await asyncio.wait_for(stopped.wait(), timeout=1)
    settled = (
        await control.await_workers([spawned["run_id"]], timeout=1, base_session_id="master")
    )[0]
    assert settled["status"] == "cancelled"
    assert (
        control.archive_worker(spawned["worker_id"], base_session_id="master")["status"]
        == "archived"
    )


@pytest.mark.asyncio
async def test_detached_cancellation_is_observed_while_waiting_for_slot(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    runner_store = WorkerStore(data_dir)
    master_store = WorkerStore(data_dir)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_executed = False

    async def executor(run, worker):
        nonlocal second_executed
        del worker
        if run.context.objective == "first":
            first_started.set()
            await release_first.wait()
        else:
            second_executed = True
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary="done"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    runner_manager = WorkerSessionManager(
        store=runner_store,
        executor=executor,
        max_concurrency=1,
    )
    master_manager = WorkerSessionManager(store=master_store)
    control = WorkerControlService(
        manager=master_manager,
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
    )
    first = await control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="first",
        idempotency_key="first",
        detached=True,
    )
    second = await control.spawn_worker(
        base_session_id="master",
        worker_type_id="explorer",
        objective="second",
        idempotency_key="second",
        detached=True,
    )
    runner_manager.start(first["run_id"])
    await asyncio.wait_for(first_started.wait(), timeout=1)
    runner_manager.start(second["run_id"])
    while runner_store.get_run(second["run_id"]).status is WorkerRunStatus.QUEUED:
        await asyncio.sleep(0.01)

    requested = await control.cancel_worker(second["run_id"], base_session_id="master")
    settled = (
        await control.await_workers([second["run_id"]], timeout=1, base_session_id="master")
    )[0]

    assert requested["action"] == "cancellation_requested"
    assert settled["status"] == "cancelled"
    assert second_executed is False
    release_first.set()
    await control.await_workers([first["run_id"]], timeout=1, base_session_id="master")


@pytest.mark.asyncio
async def test_immediate_local_queued_cancel_emits_once_and_drops_task(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    async def lifecycle(event, run):
        del run
        events.append(event)

    async def executor(run, worker):
        raise AssertionError(f"queued cancellation executed {run.run_id} for {worker.worker_id}")

    store = WorkerStore(tmp_path / "data")
    manager = WorkerSessionManager(
        store=store,
        executor=executor,
        on_lifecycle=lifecycle,
    )
    _, run, _ = await manager.spawn_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("cancel immediately"),
        idempotency_key="spawn",
    )

    cancelled = await manager.cancel_worker(run.run_id)

    assert cancelled.status is WorkerRunStatus.CANCELLED
    assert events.count("cancelled") == 1
    assert run.run_id not in manager._tasks


@pytest.mark.asyncio
async def test_cancel_on_settled_run_reports_no_change(tmp_path: Path) -> None:
    store = WorkerStore(tmp_path / "data")
    worker, running = _create_running(store)
    _complete(store, running.run_id)
    control = WorkerControlService(
        manager=WorkerSessionManager(store=store),
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
    )

    result = await control.cancel_worker(
        running.run_id,
        base_session_id=worker.base_session_id,
    )

    assert result["status"] == "completed"
    assert result["action"] == "already_settled"
    assert result["cancel_requested"] is False


@pytest.mark.asyncio
async def test_cancel_does_not_interrupt_terminal_lifecycle_projection(
    tmp_path: Path,
) -> None:
    terminal_started = asyncio.Event()
    release_terminal = asyncio.Event()
    terminal_finished = asyncio.Event()

    async def lifecycle(event, run):
        del run
        if event == "completed":
            terminal_started.set()
            await release_terminal.wait()
            terminal_finished.set()

    async def executor(run, worker):
        del worker
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary=f"done {run.context.objective}"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    store = WorkerStore(tmp_path / "data")
    manager = WorkerSessionManager(
        store=store,
        executor=executor,
        on_lifecycle=lifecycle,
    )
    _, run, _ = await manager.spawn_worker(
        base_session_id="master",
        snapshot=_snapshot(),
        context=_context("finish while cancel is requested"),
        idempotency_key="spawn",
    )
    await asyncio.wait_for(terminal_started.wait(), timeout=1)
    task = manager._tasks[run.run_id]

    unchanged = await manager.cancel_worker(run.run_id)

    assert unchanged.status is WorkerRunStatus.COMPLETED
    assert manager._tasks.get(run.run_id) is task
    assert not task.done()
    release_terminal.set()
    await asyncio.wait_for(task, timeout=1)
    assert terminal_finished.is_set()
    assert run.run_id not in manager._tasks


@pytest.mark.asyncio
async def test_continuous_runner_polls_while_other_workers_are_active(
    tmp_path: Path,
) -> None:
    store = WorkerStore(tmp_path / "data")
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()

    async def executor(run, worker):
        del worker
        if run.context.objective == "first":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(summary=f"done {run.context.objective}"),
            checkpoint={"messages": [{"role": "assistant", "content": "done"}]},
        )

    manager = WorkerSessionManager(store=store, executor=executor, max_concurrency=2)
    control = WorkerControlService(
        manager=manager,
        worker_types=WorkerRegistry.discover(tmp_path),
        worker_tool_names=("complete_work", "request_master"),
        skills_enabled=False,
    )
    first = await control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="first",
        idempotency_key="first",
        detached=True,
    )
    runner = asyncio.create_task(
        run_worker_runner(
            SimpleNamespace(worker_manager=manager),  # type: ignore[arg-type]
            poll_seconds=0.02,
        )
    )
    try:
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = await control.spawn_worker(
            base_session_id="master",
            worker_type_id="explorer",
            objective="second",
            idempotency_key="second",
            detached=True,
        )

        await asyncio.wait_for(second_started.wait(), timeout=0.5)
        assert store.get_run(first["run_id"]).status is WorkerRunStatus.RUNNING
        release_first.set()
        settled = await control.await_workers(
            [first["run_id"], second["run_id"]],
            timeout=1,
            base_session_id="master",
        )
        assert {run["status"] for run in settled} == {"completed"}
    finally:
        release_first.set()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
