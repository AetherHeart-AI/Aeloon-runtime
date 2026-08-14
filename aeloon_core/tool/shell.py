"""Built-in shell execution tool."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from aeloon_core.blocking import run_blocking
from aeloon_core.core.types import ToolResult, ToolUpdateCallback
from aeloon_core.tool.base import (
    DEFAULT_MAX_BYTES,
    ToolContext,
    WorkspaceTool,
    object_schema,
    truncate_tail,
)
from aeloon_core.tool.process import terminate_process_group

_LOG_RETENTION_SECONDS = 7 * 24 * 60 * 60
_LOG_CAP_BYTES = 512 * 1024 * 1024


def bash_log_directory() -> Path:
    return Path(tempfile.gettempdir()) / "aeloon-core"


def _write_bytes(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view) :]


def prune_bash_logs(
    directory: Path | None = None,
    *,
    now: float | None = None,
    retention_seconds: int = _LOG_RETENTION_SECONDS,
    cap_bytes: int = _LOG_CAP_BYTES,
) -> None:
    """Remove only completed bash logs, oldest first; active .tmp files are untouched."""

    root = directory or bash_log_directory()
    if not root.is_dir():
        return
    cutoff = (time.time() if now is None else now) - retention_seconds
    logs: list[tuple[float, int, Path]] = []
    for path in root.glob("bash-*.log"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime < cutoff:
            path.unlink(missing_ok=True)
        else:
            logs.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _, size, _ in logs)
    for _, size, path in sorted(logs):
        if total <= cap_bytes:
            break
        path.unlink(missing_ok=True)
        total -= size


class BashTool(WorkspaceTool):
    name = "bash"
    label = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns combined output, "
        "truncated to the last 2000 lines or 50KB."
    )
    prompt_snippet = "Execute shell commands"
    parameters = object_schema(
        {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)",
            },
        },
        ("command",),
    )

    def __init__(self, context: ToolContext, *, shell_path: str | None = None) -> None:
        super().__init__(context)
        self.executable = shell_path or os.environ.get("SHELL") or "/bin/bash"

    async def execute(
        self,
        _call_id: str,
        arguments: dict[str, Any],
        on_update: ToolUpdateCallback | None,
    ) -> ToolResult:
        command = str(arguments["command"])
        timeout = arguments.get("timeout")
        if timeout is not None and (
            not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
            or timeout > 2_147_483.647
        ):
            raise ValueError("Invalid timeout: must be a positive finite number of seconds")
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.context.cwd),
            executable=self.executable,
            env=dict(os.environ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        directory = bash_log_directory()
        await run_blocking(directory.mkdir, parents=True, exist_ok=True)
        descriptor, temporary_name = await run_blocking(
            tempfile.mkstemp, prefix="bash-", suffix=".tmp", dir=directory
        )
        temporary_path = Path(temporary_name)
        tail = bytearray()
        output_bytes = 0
        newline_count = 0
        ended_with_newline = False

        async def pump(stream: asyncio.StreamReader | None) -> None:
            nonlocal output_bytes, newline_count, ended_with_newline
            if stream is None:
                return
            while True:
                data = await stream.read(8_192)
                if not data:
                    return
                await run_blocking(_write_bytes, descriptor, data)
                output_bytes += len(data)
                newline_count += data.count(b"\n")
                ended_with_newline = data.endswith(b"\n")
                tail.extend(data)
                if len(tail) > DEFAULT_MAX_BYTES + 4:
                    del tail[: len(tail) - DEFAULT_MAX_BYTES - 4]
                text = data.decode("utf-8", errors="replace")
                if on_update is not None:
                    updated = on_update(ToolResult.text(text))
                    if inspect.isawaitable(updated):
                        await updated

        pumps = [
            asyncio.create_task(pump(process.stdout)),
            asyncio.create_task(pump(process.stderr)),
        ]
        completed = False
        try:
            if timeout is None:
                await process.wait()
            else:
                await asyncio.wait_for(process.wait(), float(timeout))
            completed = True
        except TimeoutError:
            await terminate_process_group(process)
            await asyncio.gather(*pumps)
            raise TimeoutError(f"Command timed out after {timeout} seconds") from None
        except asyncio.CancelledError:
            await terminate_process_group(process)
            raise
        finally:
            if not all(task.done() for task in pumps):
                await asyncio.gather(*pumps, return_exceptions=True)
            await run_blocking(os.close, descriptor)
            if not completed:
                await run_blocking(temporary_path.unlink, missing_ok=True)
        output_lines = newline_count + (1 if output_bytes and not ended_with_newline else 0)
        output = bytes(tail).decode("utf-8", errors="replace").rstrip("\n")
        visible, tail_truncated = truncate_tail(output)
        truncated = output_bytes > DEFAULT_MAX_BYTES or output_lines > 2_000 or tail_truncated
        full_output_path: str | None = None
        if truncated:
            full_path = directory / f"bash-{uuid.uuid4().hex}.log"
            await run_blocking(temporary_path.replace, full_path)
            full_output_path = str(full_path)
        else:
            await run_blocking(temporary_path.unlink, missing_ok=True)
        rendered = visible or "(no output)"
        if process.returncode:
            rendered += f"\n\nCommand exited with code {process.returncode}"
        if full_output_path:
            rendered += f"\n\n[Output truncated. Full output: {full_output_path}]"
        return ToolResult.text(
            rendered,
            details={
                "command": command,
                "output": visible,
                "outputBytes": output_bytes,
                "outputLines": output_lines,
                "exitCode": process.returncode,
                "cancelled": False,
                "truncated": truncated,
                "fullOutputPath": full_output_path,
            },
        )


__all__ = ["BashTool", "bash_log_directory", "prune_bash_logs"]
