"""Shell execution tool."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from pydantic import BaseModel, Field

from aeloon_core.tools.base import WorkspaceTool


class ExecArgs(BaseModel):
    command: str = Field(description="Shell command to execute.")
    working_dir: str | None = Field(
        default=None,
        description="Optional working directory. Relative paths resolve from workspace.",
    )
    timeout: int | None = Field(default=None, ge=1, le=600, description="Timeout in seconds.")


class ExecTool(WorkspaceTool):
    """Execute shell commands in the configured workspace."""

    name = "exec"
    description = (
        "Execute a shell command in the workspace and return stdout, stderr, "
        "and exit code. Quote paths because they may contain spaces."
    )
    args_model = ExecArgs

    _MAX_OUTPUT = 12_000

    def __init__(self, *, workspace: Path, timeout: int = 60) -> None:
        super().__init__(workspace=workspace)
        self.timeout = timeout

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> str:
        cwd = self._resolve(working_dir)
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
