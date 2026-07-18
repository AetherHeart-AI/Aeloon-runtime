"""Detached-capable local Worker runner."""

from __future__ import annotations

import asyncio

from aeloon_core.orchestrator import AeloonCoreOrchestrator


async def run_worker_runner(
    orchestrator: AeloonCoreOrchestrator,
    *,
    once: bool = False,
    poll_seconds: float = 0.5,
) -> None:
    """Run queued WorkerRuns until interrupted, or drain once for automation/tests."""

    if once:
        await orchestrator.worker_manager.run_queued()
        return

    while True:
        await orchestrator.worker_manager.reconcile_stale_runs()
        orchestrator.worker_manager.start_queued()
        await asyncio.sleep(max(0.05, poll_seconds))
