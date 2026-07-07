"""Shell execution tool."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

from aeloon_core.tools.base import Tool


class ExecTool(Tool):
    """Execute shell commands in the configured workspace."""

    name = "exec"
    description = (
        "Execute a shell command in the workspace and return stdout, stderr, "
        "and exit code. Quote paths because they may contain spaces."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "working_dir": {
                "type": "string",
                "description": "Optional working directory. Relative paths resolve from workspace.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds.",
                "minimum": 1,
                "maximum": 600,
            },
        },
        "required": ["command"],
    }

    _MAX_OUTPUT = 12_000

    def __init__(self, *, workspace: Path, timeout: int = 60) -> None:
        self.workspace = workspace
        self.timeout = timeout

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        cwd = self._resolve_dir(working_dir)
        effective_timeout = min(timeout or self.timeout, 600)
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=os.environ.copy(),
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except TimeoutError:
                self._kill_process_group(process)
                return f"Error: Command timed out after {effective_timeout} seconds"

            parts: list[str] = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    parts.append(f"STDERR:\n{stderr_text}")
            parts.append(f"\nExit code: {process.returncode}")
            result = "\n".join(parts) if parts else "(no output)"
            return self._truncate(result)
        except Exception as exc:
            return f"Error executing command: {exc}"

    def _resolve_dir(self, working_dir: str | None) -> Path:
        if not working_dir:
            return self.workspace
        path = Path(working_dir).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve(strict=False)

    @staticmethod
    def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        pid = process.pid
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()

    def _truncate(self, result: str) -> str:
        if len(result) <= self._MAX_OUTPUT:
            return result
        half = self._MAX_OUTPUT // 2
        omitted = len(result) - self._MAX_OUTPUT
        return result[:half] + f"\n\n... ({omitted:,} chars truncated) ...\n\n" + result[-half:]
