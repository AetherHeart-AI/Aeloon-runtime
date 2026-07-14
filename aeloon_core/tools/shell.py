"""Shell execution tool."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from aeloon_core.tools.base import WorkspaceTool

MAX_EXEC_COMMAND_CHARS = 8_192


class ExecArgs(BaseModel):
    command: str = Field(
        description=(
            "Shell command to execute, at most 8192 characters. Do not put generated file or "
            "long script bodies here; use write for new files and str_replace for edits."
        ),
        json_schema_extra={"maxLength": MAX_EXEC_COMMAND_CHARS},
    )
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
        "and exit code. Quote paths because they may contain spaces. Use write or str_replace "
        "instead of inline Python, heredocs, or redirection for generated file contents."
    )
    args_model = ExecArgs

    _MAX_OUTPUT = 12_000

    def __init__(
        self,
        *,
        workspace: Path,
        timeout: int = 60,
        denied_paths: Iterable[Path] = (),
    ) -> None:
        super().__init__(workspace=workspace, denied_paths=denied_paths)
        self.timeout = timeout

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if len(command) > MAX_EXEC_COMMAND_CHARS:
            return _payload_violation_error(len(command))
        cwd = self._resolve(working_dir)
        effective_timeout = min(timeout or self.timeout, 600)
        try:
            sandbox_argv = self._sandbox_argv(command)
            if sandbox_argv is None and self.denied_paths:
                return (
                    "Error: exec is disabled because this host cannot isolate protected "
                    "runtime data from shell commands"
                )
            if sandbox_argv is None:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd),
                    env=os.environ.copy(),
                    start_new_session=True,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *sandbox_argv,
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
                await self._terminate_process_group(process)
                return f"Error: Command timed out after {effective_timeout} seconds"
            except asyncio.CancelledError:
                await self._terminate_process_group(process)
                raise

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

    def _sandbox_argv(self, command: str) -> list[str] | None:
        if not self.denied_paths:
            return None
        sandbox_exec = Path("/usr/bin/sandbox-exec")
        if sys.platform == "darwin" and sandbox_exec.exists():
            denied_rules = "\n".join(
                "(deny file-read* file-write* (subpath "
                f'"{_sandbox_literal(path)}"))'
                for path in self.denied_paths
            )
            profile = f"(version 1)\n(allow default)\n{denied_rules}\n"
            return [str(sandbox_exec), "-p", profile, "/bin/sh", "-c", command]

        bubblewrap = shutil.which("bwrap")
        if bubblewrap is None:
            return None
        argv = [
            bubblewrap,
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(self.workspace),
            str(self.workspace),
        ]
        for path in self.denied_paths:
            if path.exists() and path.is_dir():
                argv.extend(("--tmpfs", str(path)))
        return [*argv, "/bin/sh", "-c", command]

    @staticmethod
    async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        ExecTool._signal_process_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            ExecTool._signal_process_group(process, signal.SIGKILL)
            await process.wait()

    @staticmethod
    def _signal_process_group(
        process: asyncio.subprocess.Process,
        requested_signal: signal.Signals,
    ) -> None:
        pid = process.pid
        if pid is None:
            return
        try:
            os.killpg(pid, requested_signal)
        except ProcessLookupError:
            return
        except Exception:
            if requested_signal is signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    def _truncate(self, result: str) -> str:
        if len(result) <= self._MAX_OUTPUT:
            return result
        half = self._MAX_OUTPUT // 2
        omitted = len(result) - self._MAX_OUTPUT
        return result[:half] + f"\n\n... ({omitted:,} chars truncated) ...\n\n" + result[-half:]


def _sandbox_literal(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _payload_violation_error(actual_chars: int) -> str:
    return (
        f"Error [EXEC_COMMAND_TOO_LARGE]: field=command; actual={actual_chars}; "
        f"limit={MAX_EXEC_COMMAND_CHARS}; next_action=run a short command by path, or use write "
        "for a new file and str_replace for an existing file."
    )
