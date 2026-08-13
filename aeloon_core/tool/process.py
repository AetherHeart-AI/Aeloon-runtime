"""Process-group lifecycle helpers shared by subprocess-backed tools."""

from __future__ import annotations

import asyncio
import os
import signal
from asyncio.subprocess import Process


async def terminate_process_group(process: Process, *, grace_seconds: float = 1.0) -> None:
    """Stop a process and its ordinary descendants, then force-kill if needed."""

    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), grace_seconds)
        return
    except TimeoutError:
        pass
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    await process.wait()


__all__ = ["terminate_process_group"]
