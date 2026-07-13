"""Worker scheduling, execution, and lifecycle projection.

The manager is deliberately usable both in-process and from a detached Runner:
the SQLite WorkerStore is the authority, while this class owns only live tasks.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from loguru import logger

from aeloon_core.worker_sessions import (
    ContextEnvelope,
    ProfileHandle,
    ResultEnvelope,
    WorkerReport,
    WorkerRunRecord,
    WorkerRunStatus,
    WorkerSessionRecord,
    WorkerStore,
)

ToolOutcome = Literal["known", "unknown", "none"]


@dataclass(frozen=True)
class WorkerExecutionOutcome:
    """Host-owned outcome returned by a Worker execution engine."""

    status: WorkerRunStatus
    report: WorkerReport
    tool_outcome: ToolOutcome = "none"
    usage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {
            WorkerRunStatus.COMPLETED,
            WorkerRunStatus.PARTIAL,
            WorkerRunStatus.FAILED,
        }:
            raise ValueError("Worker execution outcome requires a finalizable status")
        object.__setattr__(self, "usage", copy.deepcopy(self.usage))


WorkerExecutor = Callable[
    [WorkerRunRecord, WorkerSessionRecord],
    Awaitable[WorkerExecutionOutcome | WorkerReport],
]
LifecycleHook = Callable[[str, WorkerRunRecord], Any]


class WorkerSessionManager:
    """Atomic Worker primitives shared by Base tools, TUI, and Runner."""

    def __init__(
        self,
        *,
        store: WorkerStore,
        executor: WorkerExecutor | None = None,
        on_lifecycle: LifecycleHook | None = None,
        max_concurrency: int = 4,
        heartbeat_interval_seconds: float = 10.0,
    ) -> None:
        self.store = store
        self.executor = executor
        self.on_lifecycle = on_lifecycle
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._progress: dict[str, Any] = {}
        # Hot Worker context belongs to the WorkerSession, never to Base history.
        # Checkpoints remain the durable fallback after a process restart.
        self._contexts: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        self._worker_locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.heartbeat_interval_seconds = max(0.01, heartbeat_interval_seconds)

    def progress_for(self, run_id: str) -> Any | None:
        """Return the exact Base-turn observer attached to one WorkerRun."""

        return self._progress.get(run_id)

    def load_live_context(
        self,
        worker_id: str,
        *,
        source_run_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return an isolated copy of one Worker's in-process message history."""

        cached = self._contexts.get(worker_id)
        if cached is None:
            return None
        cached_run_id, messages = cached
        if source_run_id is not None and cached_run_id != source_run_id:
            return None
        return copy.deepcopy(messages)

    def save_live_context(
        self,
        worker_id: str,
        source_run_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Remember private Worker messages for a later related WorkerRun."""

        self._contexts[worker_id] = (source_run_id, copy.deepcopy(messages))

    async def spawn_worker(
        self,
        *,
        base_session_id: str,
        profile: ProfileHandle,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None = None,
        start: bool = True,
        progress: Any | None = None,
    ) -> tuple[WorkerSessionRecord, WorkerRunRecord, bool]:
        session, run, created = self.store.create_worker(
            base_session_id=base_session_id,
            profile=profile,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
        )
        self._bind_progress(run, progress, created=created)
        if created:
            await self._emit("created", run)
        if start:
            self.start(run.run_id)
        return session, run, created

    async def send_worker(
        self,
        *,
        worker_id: str,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None = None,
        start: bool = True,
        progress: Any | None = None,
    ) -> tuple[WorkerRunRecord, bool]:
        run, created = self.store.create_run(
            worker_id=worker_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
        )
        self._bind_progress(run, progress, created=created)
        if created:
            await self._emit("created", run)
        if start:
            self.start(run.run_id)
        return run, created

    def start(self, run_id: str) -> None:
        """Start a queued Run once; duplicate starts share the same live task."""

        if self.executor is None or run_id in self._tasks:
            return
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id))

    def _bind_progress(self, run: WorkerRunRecord, progress: Any, *, created: bool) -> None:
        """Attach one owning Base turn without rerouting idempotent replays."""

        if progress is None or run.status.terminal:
            return
        if created:
            self._progress[run.run_id] = progress
        else:
            self._progress.setdefault(run.run_id, progress)

    async def await_workers(
        self,
        run_ids: list[str],
        *,
        timeout: float | None = None,
    ) -> list[WorkerRunRecord]:
        deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
        while True:
            runs = [self.store.get_run(run_id) for run_id in run_ids]
            if all(run.status.terminal for run in runs):
                return runs

            remaining = None if deadline is None else deadline - asyncio.get_running_loop().time()
            if remaining is not None and remaining <= 0:
                return runs

            # A detached Runner has no Task in this manager, so SQLite remains
            # the only shared completion signal. Poll it without ever wrapping
            # or cancelling the Worker task when this wait times out.
            poll_interval = 0.05 if remaining is None else min(0.05, remaining)
            local_tasks = [
                self._tasks[run.run_id]
                for run in runs
                if not run.status.terminal and run.run_id in self._tasks
            ]
            if local_tasks:
                await asyncio.wait(
                    local_tasks,
                    timeout=poll_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                await asyncio.sleep(poll_interval)

    def inspect_worker(self, worker_id: str) -> tuple[WorkerSessionRecord, list[WorkerRunRecord]]:
        session = self.store.get_worker(worker_id)
        return session, self.store.list_runs(worker_id)

    async def cancel_worker(self, run_id: str) -> WorkerRunRecord:
        cancelled, changed = self.store.try_cancel_run(run_id)
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            if task is not asyncio.current_task():
                await asyncio.gather(task, return_exceptions=True)
        # A local cancellation is only shown after its executor and owned tools
        # have torn down, so no tool result can appear after the terminal line.
        if changed:
            await self._emit("cancelled", cancelled)
        return cancelled

    async def resume_worker(self, run_id: str) -> WorkerRunRecord:
        run = self.store.get_run(run_id)
        if run.status is WorkerRunStatus.WAITING_FOR_CONTEXT:
            raise ValueError("send_worker must provide the requested context before resuming")
        if not run.status.terminal:
            self.start(run_id)
            return self.store.get_run(run_id)
        raise ValueError("terminal WorkerRuns resume through a new continuation run")

    def archive_worker(self, worker_id: str) -> WorkerSessionRecord:
        worker = self.store.archive_worker(worker_id)
        self._contexts.pop(worker_id, None)
        for run in self.store.list_runs(worker_id):
            self._progress.pop(run.run_id, None)
        return worker

    async def run_queued(self) -> list[WorkerRunRecord]:
        """Runner entry point: claim every currently queued Run in this process."""

        runs = self.store.list_queued_runs()
        for run in runs:
            self.start(run.run_id)
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
        return [self.store.get_run(run.run_id) for run in runs]

    async def _execute(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        lock = self._worker_locks.setdefault(run.worker_id, asyncio.Lock())
        try:
            # WorkerRuns share one WorkerSession context and therefore execute in
            # order. Different Workers still use the global concurrency budget.
            async with lock:
                await self._execute_serial(run_id)
        except asyncio.CancelledError:
            cancelled, changed = self.store.try_cancel_run(run_id)
            if changed:
                await self._emit("cancelled", cancelled)
            raise
        except Exception as exc:
            logger.exception("Unhandled WorkerRun failure for {}: {}", run_id, exc)
            current = self.store.get_run(run_id)
            if not current.status.terminal:
                session = self.inspect_worker(current.worker_id)[0]
                result = ResultEnvelope(
                    worker_id=current.worker_id,
                    run_id=current.run_id,
                    status=WorkerRunStatus.FAILED,
                    profile=session.profile,
                    report=WorkerReport(summary=_safe_failure_summary(exc)),
                    tool_outcome="unknown",
                )
                failed, changed = self.store.try_finalize_run(run_id, result)
                if changed:
                    await self._emit("failed", failed)
        finally:
            self._tasks.pop(run_id, None)

    async def _execute_serial(self, run_id: str) -> None:
        while True:
            running, claimed = self.store.try_start_run(run_id)
            if claimed:
                break
            if running.status is not WorkerRunStatus.QUEUED:
                return
            # Another process owns this WorkerSession. Keep this queued Run
            # alive until the durable per-Worker execution slot is available.
            await asyncio.sleep(0.05)
        await self._emit("running", running)
        started = perf_counter()
        session = self.inspect_worker(running.worker_id)[0]
        heartbeat = self._start_heartbeat(running, started)
        try:
            async with asyncio.timeout(running.context.budget.max_seconds):
                async with self._semaphore:
                    assert self.executor is not None
                    execution = _normalize_execution_outcome(
                        await self.executor(running, session)
                    )
            result = ResultEnvelope(
                worker_id=running.worker_id,
                run_id=running.run_id,
                status=execution.status,
                profile=session.profile,
                report=execution.report,
                tool_outcome=execution.tool_outcome,
                usage=execution.usage,
                duration_ms=max(0, int((perf_counter() - started) * 1_000)),
            )
            finalized, changed = self.store.try_finalize_run(run_id, result)
            if changed:
                await self._emit(_lifecycle_event(execution.status), finalized)
        except asyncio.CancelledError:
            cancelled, changed = self.store.try_cancel_run(run_id)
            if changed:
                await self._emit("cancelled", cancelled)
            raise
        except TimeoutError:
            duration_ms = max(0, int((perf_counter() - started) * 1_000))
            result = ResultEnvelope(
                worker_id=running.worker_id,
                run_id=running.run_id,
                status=WorkerRunStatus.FAILED,
                profile=session.profile,
                report=WorkerReport(
                    summary=(
                        "Worker timed out after "
                        f"{running.context.budget.max_seconds} seconds."
                    )
                ),
                tool_outcome="unknown",
                duration_ms=duration_ms,
            )
            failed, changed = self.store.try_finalize_run(run_id, result)
            if changed:
                await self._emit("timed_out", failed)
        except Exception as exc:
            result = ResultEnvelope(
                worker_id=running.worker_id,
                run_id=running.run_id,
                status=WorkerRunStatus.FAILED,
                profile=session.profile,
                report=WorkerReport(summary=_safe_failure_summary(exc)),
                tool_outcome="unknown",
                duration_ms=max(0, int((perf_counter() - started) * 1_000)),
            )
            failed, changed = self.store.try_finalize_run(run_id, result)
            if changed:
                await self._emit("failed", failed)
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    def _start_heartbeat(
        self,
        run: WorkerRunRecord,
        started: float,
    ) -> asyncio.Task[None] | None:
        if run.run_id not in self._progress:
            return None
        return asyncio.create_task(self._heartbeat(run.run_id, started))

    async def _heartbeat(self, run_id: str, started: float) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_seconds)
            run = self.store.get_run(run_id)
            if run.status is not WorkerRunStatus.RUNNING:
                return
            await self._emit_heartbeat(
                run,
                elapsed_ms=max(0, int((perf_counter() - started) * 1_000)),
            )

    async def _emit(self, event: str, run: WorkerRunRecord) -> None:
        if self.on_lifecycle is not None:
            await self._invoke_observer(self.on_lifecycle, event, run)
        progress = self._progress.get(run.run_id)
        if progress is not None:
            hook = getattr(progress, "on_worker_lifecycle", None)
            if hook is not None:
                worker = self.store.get_worker(run.worker_id)
                await self._invoke_observer(
                    hook,
                    event=event,
                    worker_id=run.worker_id,
                    run_id=run.run_id,
                    profile_id=worker.profile.profile_id,
                    status=run.status.value,
                    duration_ms=run.result.duration_ms if run.result is not None else None,
                )
        if run.status.terminal:
            self._progress.pop(run.run_id, None)

    async def _emit_heartbeat(self, run: WorkerRunRecord, *, elapsed_ms: int) -> None:
        progress = self._progress.get(run.run_id)
        hook = getattr(progress, "on_worker_heartbeat", None) if progress is not None else None
        if hook is None:
            return
        worker = self.store.get_worker(run.worker_id)
        await self._invoke_observer(
            hook,
            worker_id=run.worker_id,
            run_id=run.run_id,
            profile_id=worker.profile.profile_id,
            status=run.status.value,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    async def _invoke_observer(observer: Any, *args: Any, **kwargs: Any) -> None:
        try:
            result = observer(*args, **kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Ignoring Worker observability failure: {}", exc)


def _safe_failure_summary(exc: Exception) -> str:
    """Describe host failure without copying potentially sensitive exception text."""

    return f"Worker execution failed ({type(exc).__name__})."


def _normalize_execution_outcome(
    execution: WorkerExecutionOutcome | WorkerReport,
) -> WorkerExecutionOutcome:
    """Keep legacy executors compatible while preserving typed terminal outcomes."""

    if isinstance(execution, WorkerExecutionOutcome):
        return execution
    if isinstance(execution, WorkerReport):
        return WorkerExecutionOutcome(
            status=WorkerRunStatus.COMPLETED,
            report=execution,
            tool_outcome="known",
        )
    raise TypeError("Worker executor must return WorkerExecutionOutcome or WorkerReport")


def _lifecycle_event(status: WorkerRunStatus) -> str:
    if status is WorkerRunStatus.COMPLETED:
        return "completed"
    if status is WorkerRunStatus.PARTIAL:
        return "partial"
    if status is WorkerRunStatus.FAILED:
        return "failed"
    raise ValueError(f"unsupported Worker execution status: {status.value}")
