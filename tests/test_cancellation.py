from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from aeloon_runtime.blocking import run_blocking
from aeloon_runtime.runtime.coordinator import Operation
from aeloon_runtime.runtime.service import RuntimeService
from aeloon_runtime.tool import BashTool, ToolContext


async def _assert_reaped(pid: int) -> None:
    """Wait for the process group to actually go away.

    Reaping is asynchronous, so a single fixed sleep only passes while the
    machine is idle: under load these assertions fail for a reason that has
    nothing to do with cancellation. Poll to a deadline instead.
    """
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"process {pid} was still alive after 2s")


@pytest.mark.asyncio
async def test_run_blocking_waits_for_worker_before_propagating_cancel() -> None:
    finished = threading.Event()

    def work() -> str:
        import time

        time.sleep(0.05)
        finished.set()
        return "done"

    task = asyncio.create_task(run_blocking(work))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_run_blocking_preserves_cancel_when_worker_fails() -> None:
    started = threading.Event()

    def work() -> str:
        import time

        started.set()
        time.sleep(0.05)
        raise ValueError("worker failed")

    task = asyncio.create_task(run_blocking(work))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_bash_cancel_stops_the_entire_process_group(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"
    tool = BashTool(ToolContext.create(tmp_path), shell_path="/bin/bash")
    task = asyncio.create_task(
        tool.execute(
            "bash",
            {"command": f"sleep 30 & echo $! > {child_pid_file}; wait"},
            None,
        )
    )
    for _ in range(100):
        if child_pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_reaped(child_pid)


@pytest.mark.asyncio
async def test_bash_cancel_cleans_background_process_after_shell_exits(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "orphan.pid"
    tool = BashTool(ToolContext.create(tmp_path), shell_path="/bin/bash")
    task = asyncio.create_task(
        tool.execute(
            "bash",
            {"command": f"sleep 30 & echo $! > {child_pid_file}"},
            None,
        )
    )
    for _ in range(100):
        if child_pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    await _assert_reaped(child_pid)


@pytest.mark.asyncio
async def test_bash_timeout_stops_the_entire_process_group(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "timeout-child.pid"
    tool = BashTool(ToolContext.create(tmp_path), shell_path="/bin/bash")
    with pytest.raises(TimeoutError):
        await tool.execute(
            "bash",
            {
                "command": f"sleep 30 & echo $! > {child_pid_file}; wait",
                "timeout": 0.02,
            },
            None,
        )
    assert child_pid_file.exists()
    await _assert_reaped(int(child_pid_file.read_text(encoding="utf-8")))


@pytest.mark.asyncio
async def test_turn_cancel_is_idempotent_and_emits_cancelling_once(tmp_path: Path) -> None:
    runtime = RuntimeService(config_path=tmp_path / "config.json", data_dir=tmp_path / "data")
    session = await runtime.repository.create(cwd=tmp_path, session_id="cancel-session")
    operation = Operation(
        id="cancel-operation",
        session_id=session.id,
        workspace=str(tmp_path),
        kind="turn",
        input={"kind": "prompt", "text": "cancel"},
    )
    runtime._runtime(session.id).operations[operation.id] = operation
    events: list[str] = []
    runtime.add_event_listener(lambda event: events.append(event.name))

    first = await runtime.turn_cancel({"operation_id": operation.id})
    second = await runtime.turn_cancel({"operation_id": operation.id})

    assert first == {"operation_id": operation.id, "cancelled": True, "status": "cancelling"}
    assert second == first
    assert events == ["operation.cancelling"]

    operation.status = "completed"
    settled = await runtime.turn_cancel({"operation_id": operation.id})
    assert settled == {"operation_id": operation.id, "cancelled": False, "status": "completed"}
    await runtime.close()
