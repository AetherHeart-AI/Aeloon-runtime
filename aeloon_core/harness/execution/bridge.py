"""Async JSONL bridge from Python to the Bun-hosted Pi runtime."""

from __future__ import annotations

import asyncio
import inspect
import json
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

BridgeHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]
EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]

_STREAM_LIMIT = 32 * 1024 * 1024
_STDERR_LIMIT = 128_000


class PiBridgeError(RuntimeError):
    """Raised when the Bun runtime cannot start or violates the bridge protocol."""


class PiRuntimeBridge:
    """Run one isolated pi-core loop and service its Python tool callbacks."""

    def __init__(
        self,
        *,
        bun_executable: str | None = None,
        runtime_path: Path | None = None,
    ) -> None:
        self.bun_executable = bun_executable or shutil.which("bun") or "bun"
        self.runtime_path = runtime_path or (
            Path(__file__).resolve().parents[2] / "pi_runtime" / "runtime.ts"
        )

    async def run(
        self,
        payload: dict[str, Any],
        *,
        on_rpc: BridgeHandler,
        on_event: EventHandler,
    ) -> dict[str, Any]:
        """Execute one request, preserving concurrent Pi tool callbacks."""

        self._validate_installation()
        process = await asyncio.create_subprocess_exec(
            self.bun_executable,
            "run",
            str(self.runtime_path),
            cwd=str(self.runtime_path.parent),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        write_lock = asyncio.Lock()
        rpc_tasks: set[asyncio.Task[None]] = set()
        stderr_task = asyncio.create_task(_bounded_stderr(process.stderr))

        async def write(message: dict[str, Any]) -> None:
            encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            async with write_lock:
                process.stdin.write(encoded.encode("utf-8") + b"\n")
                await process.stdin.drain()

        async def answer_rpc(message: dict[str, Any]) -> None:
            rpc_id = message.get("id")
            try:
                response = on_rpc(message)
                if inspect.isawaitable(response):
                    response = await response
                await write({"type": "rpc_result", "id": rpc_id, "result": response})
            except Exception as exc:
                await write(
                    {
                        "type": "rpc_result",
                        "id": rpc_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        result: dict[str, Any] | None = None
        try:
            await write({"type": "start", "request": payload})
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    break
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PiBridgeError(
                        f"Pi runtime emitted invalid JSONL: {raw[:500]!r}"
                    ) from exc
                if not isinstance(message, dict):
                    raise PiBridgeError("Pi runtime emitted a non-object protocol message")
                message_type = message.get("type")
                if message_type == "rpc":
                    task = asyncio.create_task(answer_rpc(message))
                    rpc_tasks.add(task)
                    task.add_done_callback(rpc_tasks.discard)
                    continue
                if message_type == "event":
                    emitted = on_event(message)
                    if inspect.isawaitable(emitted):
                        await emitted
                    continue
                if message_type == "result":
                    value = message.get("result")
                    if not isinstance(value, dict):
                        raise PiBridgeError("Pi runtime result must be an object")
                    result = value
                    break
                if message_type == "fatal":
                    raise PiBridgeError(str(message.get("error") or "Pi runtime failed"))
                raise PiBridgeError(f"unknown Pi runtime protocol message: {message_type!r}")

            if rpc_tasks:
                await asyncio.gather(*rpc_tasks)
            process.stdin.close()
            await process.stdin.wait_closed()
            return_code = await process.wait()
            stderr = await stderr_task
            if result is None:
                detail = stderr or f"exit code {return_code}"
                raise PiBridgeError(f"Pi runtime exited without a result: {detail}")
            if return_code != 0:
                raise PiBridgeError(
                    f"Pi runtime exited with code {return_code}: {stderr or '(no stderr)'}"
                )
            return result
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        finally:
            for task in rpc_tasks:
                if not task.done():
                    task.cancel()
            if not stderr_task.done():
                stderr_task.cancel()

    def _validate_installation(self) -> None:
        if not self.runtime_path.is_file():
            raise PiBridgeError(f"Pi runtime entrypoint is missing: {self.runtime_path}")
        dependency = self.runtime_path.parent / "node_modules" / "@earendil-works"
        if not dependency.is_dir():
            raise PiBridgeError(
                "Pi runtime dependencies are not installed; run "
                f"`bun install --cwd {self.runtime_path.parent}`"
            )


async def _bounded_stderr(reader: asyncio.StreamReader) -> str:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await reader.read(8_192)
        if not chunk:
            break
        if size < _STDERR_LIMIT:
            remaining = _STDERR_LIMIT - size
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


__all__ = ["PiBridgeError", "PiRuntimeBridge"]
