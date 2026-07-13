"""Launch the OpenTUI frontend with the active Python runtime configuration."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from aeloon_core.config import Config

TUI_CONFIG_ENV = "AELOON_CORE_TUI_CONFIG_JSON"
TUI_BRIDGE_FD_ENV = "AELOON_CORE_TUI_BRIDGE_FD"
TUI_SESSION_ENV = "AELOON_CORE_TUI_SESSION_ID"
TUI_WORKSPACE_ENV = "AELOON_CORE_WORKSPACE"
TUI_PYTHON_ENV = "AELOON_CORE_PYTHON"
TUI_INITIAL_VIEW_ENV = "AELOON_CORE_TUI_INITIAL_VIEW"
TUI_LOG_DETAIL_ENV = "AELOON_CORE_TUI_LOG_DETAIL"
TUI_LOG_LEVEL_ENV = "AELOON_CORE_TUI_LOG_LEVEL"

_MINIMUM_BUN_VERSION = (1, 3, 0)
_REQUIRED_PACKAGES = ("@opentui/core", "@opentui/solid", "solid-js")
_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
_FRONTEND_ENV_ALLOWLIST = {
    "BUN_INSTALL",
    "BUN_RUNTIME_TRANSPILER_CACHE_PATH",
    "COLORTERM",
    "DYLD_LIBRARY_PATH",
    "FORCE_COLOR",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "NO_COLOR",
    "PATH",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TERM",
    "TERMINFO",
    "TERMINFO_DIRS",
    "TMP",
    "TMPDIR",
    "TZ",
    "USER",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}


class OpenTuiLaunchError(RuntimeError):
    """A user-actionable failure that prevented the TUI from starting."""


async def run_opentui(
    config: Config,
    *,
    session_id: str | None = None,
    show_gateway_logs: bool = False,
    gateway_log_level: str = "INFO",
    gateway_log_detail: bool = False,
    tui_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Run the bundled OpenTUI application attached to the current terminal."""

    root = (tui_dir or Path(__file__).resolve().parent / "tui").resolve()
    entry = root / "src" / "index.tsx"
    _validate_tui_entry(entry)
    bun = _resolve_bun()
    _validate_tui_dependencies(root)
    parent_socket, frontend_socket = socket.socketpair()
    reader, writer = await asyncio.open_connection(sock=parent_socket)
    child_env = build_opentui_environment(
        config,
        session_id=session_id,
        show_gateway_logs=show_gateway_logs,
        gateway_log_level=gateway_log_level,
        gateway_log_detail=gateway_log_detail,
        environ=environ,
    )
    bridge_fd = frontend_socket.fileno()
    child_env[TUI_BRIDGE_FD_ENV] = str(bridge_fd)

    try:
        process = await asyncio.create_subprocess_exec(
            bun,
            "run",
            "--conditions=browser",
            "--preload",
            "@opentui/solid/runtime-plugin-support",
            str(entry),
            cwd=str(root),
            env=child_env,
            pass_fds=(bridge_fd,),
        )
    except OSError as exc:
        frontend_socket.close()
        await _close_stream(writer)
        raise OpenTuiLaunchError(
            f"Could not start OpenTUI with Bun ({exc}). "
            "Verify that `bun --version` works in this terminal."
        ) from None
    frontend_socket.close()

    bridge_task = asyncio.create_task(
        _serve_bridge(
            config,
            reader,
            writer,
            session_id=session_id,
            gateway_log_level=gateway_log_level,
        )
    )
    process_wait = asyncio.create_task(process.wait())
    try:
        done, _pending = await asyncio.wait(
            {bridge_task, process_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if bridge_task in done:
            bridge_error = bridge_task.exception()
            await _close_stream(writer)
            if bridge_error is not None:
                await _terminate(process)
                await process_wait
                raise OpenTuiLaunchError(
                    "The runtime bridge stopped unexpectedly."
                ) from None
            returncode = await process_wait
        else:
            returncode = process_wait.result()
            await _close_stream(writer)
            await _settle_bridge(bridge_task)
    except asyncio.CancelledError:
        await _terminate(process)
        await _close_stream(writer)
        await _cancel_task(bridge_task)
        raise

    if returncode not in {0, 130, -signal.SIGINT}:
        raise OpenTuiLaunchError(
            f"OpenTUI exited with status {returncode}. "
            f"Run `bun install --cwd {json.dumps(str(root))}` and try again."
        )


def build_opentui_environment(
    config: Config,
    *,
    session_id: str | None = None,
    show_gateway_logs: bool = False,
    gateway_log_level: str = "INFO",
    gateway_log_detail: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the subprocess environment without putting config secrets in argv."""

    source = os.environ if environ is None else environ
    values = {
        name: source[name]
        for name in _FRONTEND_ENV_ALLOWLIST
        if name in source
    }
    normalized = config.normalized()
    values.pop(TUI_CONFIG_ENV, None)
    values.pop(TUI_BRIDGE_FD_ENV, None)
    values[TUI_PYTHON_ENV] = sys.executable
    values[TUI_WORKSPACE_ENV] = str(normalized.workspace)
    values[TUI_LOG_LEVEL_ENV] = gateway_log_level.upper()
    if session_id:
        values[TUI_SESSION_ENV] = session_id
    else:
        values.pop(TUI_SESSION_ENV, None)
    if show_gateway_logs:
        values[TUI_INITIAL_VIEW_ENV] = "logs"
    else:
        values.pop(TUI_INITIAL_VIEW_ENV, None)
    if gateway_log_detail:
        values[TUI_LOG_DETAIL_ENV] = "1"
    else:
        values.pop(TUI_LOG_DETAIL_ENV, None)

    package_root = str(Path(__file__).resolve().parent.parent)
    existing = source.get("PYTHONPATH", "")
    paths = [package_root, *(item for item in existing.split(os.pathsep) if item)]
    values["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return values


def _validate_tui_entry(entry: Path) -> None:
    if not entry.is_file():
        raise OpenTuiLaunchError(
            "The bundled OpenTUI entrypoint is missing. Reinstall `aeloon-core` and try again."
        )


def _validate_tui_dependencies(root: Path) -> None:
    missing = [
        package
        for package in _REQUIRED_PACKAGES
        if not (root / "node_modules" / package / "package.json").is_file()
    ]
    if missing:
        packages = ", ".join(missing)
        raise OpenTuiLaunchError(
            f"OpenTUI dependencies are not installed ({packages}). "
            f"Run `bun install --cwd {json.dumps(str(root))}` first."
        )


def _resolve_bun() -> str:
    bun = shutil.which("bun")
    if bun is None:
        raise OpenTuiLaunchError(
            "OpenTUI requires Bun >= 1.3, but `bun` was not found. "
            "Install Bun from https://bun.sh/docs/installation, then run "
            "`bun install --cwd aeloon_core/tui`."
        )
    try:
        result = subprocess.run(
            [bun, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        raise OpenTuiLaunchError(
            "Bun is installed but its version could not be read. "
            "Verify that `bun --version` works in this terminal."
        ) from None

    raw_version = result.stdout.strip()
    match = _VERSION_PATTERN.match(raw_version)
    if match is None:
        raise OpenTuiLaunchError(
            f"Could not parse Bun version {raw_version!r}; OpenTUI requires Bun >= 1.3."
        )
    version = tuple(int(part or 0) for part in match.groups())
    if version < _MINIMUM_BUN_VERSION:
        raise OpenTuiLaunchError(
            f"Bun {raw_version} is too old; OpenTUI requires Bun >= 1.3. "
            "Upgrade Bun and try again."
        )
    return bun


async def _serve_bridge(
    config: Config,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    session_id: str | None,
    gateway_log_level: str,
) -> None:
    from aeloon_core.tui_bridge import NDJSONStreamWriter, run_bridge

    await run_bridge(
        config,
        input_stream=reader,
        session_id=session_id,
        sink=NDJSONStreamWriter(writer),
        gateway_log_level=gateway_log_level,
    )


async def _close_stream(writer: asyncio.StreamWriter) -> None:
    if not writer.is_closing():
        writer.close()
    try:
        await writer.wait_closed()
    except (BrokenPipeError, ConnectionError):
        return


async def _settle_bridge(task: asyncio.Task[None]) -> None:
    done, _pending = await asyncio.wait({task}, timeout=2)
    if not done:
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _cancel_task(task: asyncio.Task[None]) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except TimeoutError:
        process.kill()
        await process.wait()
