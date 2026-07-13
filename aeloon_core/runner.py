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

    while True:
        await orchestrator.worker_manager.run_queued()
        if once:
            return
        await asyncio.sleep(max(0.05, poll_seconds))
