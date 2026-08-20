"""Cancellation-aware wrappers for blocking work dispatched to a thread."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def run_blocking(function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run blocking work without abandoning the worker when cancelled."""

    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(worker)
        except BaseException:
            # Cancellation is the caller's control signal. The worker's
            # exception is intentionally not allowed to replace it while we
            # finish joining the thread.
            pass
        raise


__all__ = ["run_blocking"]
