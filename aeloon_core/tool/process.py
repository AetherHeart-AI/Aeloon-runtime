"""Process-group lifecycle helpers shared by subprocess-backed tools."""

from __future__ import annotations

import asyncio
import os
import signal
from asyncio.subprocess import Process


async def terminate_process_group(process: Process, *, grace_seconds: float = 1.0) -> None:
    """Stop a process and its ordinary descendants, then force-kill if needed."""

    if os.name == "posix":
        pgid = process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            await process.wait()
            return
        deadline = asyncio.get_running_loop().time() + max(0.0, grace_seconds)
        while True:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                await process.wait()
                return
            except PermissionError:
                # A process in our own group should be reachable, but retain the
                # conservative cleanup path if the platform reports otherwise.
                pass
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        return
    else:
        if process.returncode is not None:
            return
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), grace_seconds)
        return
    except TimeoutError:
        pass
    if process.returncode is not None:
        return
    process.kill()
    await process.wait()


__all__ = ["terminate_process_group"]
