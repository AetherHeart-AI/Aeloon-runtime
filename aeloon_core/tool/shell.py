"""Built-in shell execution tool."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from aeloon_core.core.types import ToolResult, ToolUpdateCallback
from aeloon_core.tool.base import ToolContext, WorkspaceTool, object_schema, truncate_tail
from aeloon_core.tool.process import terminate_process_group


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
        chunks: list[str] = []

        async def pump(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                data = await stream.read(8_192)
                if not data:
                    return
                text = data.decode("utf-8", errors="replace")
                chunks.append(text)
                if on_update is not None:
                    updated = on_update(ToolResult.text(text))
                    if inspect.isawaitable(updated):
                        await updated

        pumps = [
            asyncio.create_task(pump(process.stdout)),
            asyncio.create_task(pump(process.stderr)),
        ]
        try:
            if timeout is None:
                await process.wait()
            else:
                await asyncio.wait_for(process.wait(), float(timeout))
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
        output = "".join(chunks).rstrip("\n")
        visible, truncated = truncate_tail(output)
        full_output_path: str | None = None
        if truncated:
            directory = Path(tempfile.gettempdir()) / "aeloon-core"
            directory.mkdir(parents=True, exist_ok=True)
            full_path = directory / f"bash-{uuid.uuid4().hex}.log"
            full_path.write_text(output, encoding="utf-8")
            full_output_path = str(full_path)
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
                "outputBytes": len(output.encode("utf-8")),
                "outputLines": len(output.splitlines()),
                "exitCode": process.returncode,
                "cancelled": False,
                "truncated": truncated,
                "fullOutputPath": full_output_path,
            },
        )


__all__ = ["BashTool"]
