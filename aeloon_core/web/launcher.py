"""Launch Aeloon Core's local browser interface."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from secrets import token_urlsafe
from typing import Any
from urllib.parse import urlsplit

from aeloon_core.config import Config
from aeloon_core.web.bridge import WEB_CONFIG_ENV, WEB_LOG_LEVEL_ENV, WEB_SESSION_ENV

WEB_HOST_ENV = "AELOON_CORE_WEB_HOST"
WEB_PORT_ENV = "AELOON_CORE_WEB_PORT"
WEB_TOKEN_ENV = "AELOON_CORE_WEB_TOKEN"
WEB_PYTHON_ENV = "AELOON_CORE_WEB_PYTHON"
WEB_WORKSPACE_ENV = "AELOON_CORE_WEB_WORKSPACE"

_MINIMUM_BUN_VERSION = (1, 3, 0)
_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
_ENV_ALLOWLIST = {
    "BUN_INSTALL",
    "BUN_RUNTIME_TRANSPILER_CACHE_PATH",
    "DYLD_LIBRARY_PATH",
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
    "TMP",
    "TMPDIR",
    "TZ",
    "USER",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}


class WebLaunchError(RuntimeError):
    """A user-actionable failure that prevented the Web UI from starting."""


async def run_web_ui(
    config: Config,
    *,
    session_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7331,
    open_browser: bool = True,
    gateway_log_level: str = "INFO",
    web_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Run the bundled local server until it exits or the caller cancels."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise WebLaunchError("The Web UI only binds to the local loopback interface.")
    if not 0 <= port <= 65_535:
        raise WebLaunchError("Web UI port must be between 0 and 65535.")

    root = (web_dir or Path(__file__).resolve().parent).resolve()
    entry = root / "server.js"
    if not entry.is_file():
        raise WebLaunchError(
            "The bundled Web UI server is missing. Reinstall aeloon-core and try again."
        )

    bun = _resolve_bun()
    token = token_urlsafe(24)
    child_env = build_web_environment(
        config,
        host=host,
        port=port,
        token=token,
        session_id=session_id,
        gateway_log_level=gateway_log_level,
        environ=environ,
    )
    try:
        process = await asyncio.create_subprocess_exec(
            bun,
            "run",
            str(entry),
            cwd=str(root),
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise WebLaunchError(
            f"Could not start the Web UI with Bun ({exc}). "
            "Verify that `bun --version` works."
        ) from None

    assert process.stdout is not None
    assert process.stderr is not None
    try:
        ready = await _read_server_ready(process)
        url = str(ready["url"])
        if open_browser:
            parsed = urlsplit(url)
            origin = f"{parsed.scheme}://{parsed.netloc}/"
            print(f"Aeloon Web UI: {origin} (opening secure one-time link)", flush=True)
            opened = await asyncio.to_thread(webbrowser.open, url)
            if not opened:
                print(f"Browser could not be opened automatically; use {url}", flush=True)
        else:
            print(f"Aeloon Web UI: {url}", flush=True)

        stdout_task = asyncio.create_task(_copy_stream(process.stdout, sys.stdout))
        stderr_task = asyncio.create_task(_copy_stream(process.stderr, sys.stderr))
        returncode = await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
    except asyncio.CancelledError:
        await _terminate(process)
        raise
    except BaseException:
        await _terminate(process)
        raise

    if returncode not in {0, 130, -signal.SIGINT}:
        raise WebLaunchError(f"Web UI server exited with status {returncode}.")


def build_web_environment(
    config: Config,
    *,
    host: str,
    port: int,
    token: str,
    session_id: str | None = None,
    gateway_log_level: str = "INFO",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal child environment without putting secrets in argv."""

    source = os.environ if environ is None else environ
    values = {name: source[name] for name in _ENV_ALLOWLIST if name in source}
    normalized = config.normalized()
    values.update(
        {
            WEB_CONFIG_ENV: normalized.model_dump_json(),
            WEB_HOST_ENV: host,
            WEB_PORT_ENV: str(port),
            WEB_TOKEN_ENV: token,
            WEB_PYTHON_ENV: sys.executable,
            WEB_WORKSPACE_ENV: str(normalized.workspace),
            WEB_LOG_LEVEL_ENV: gateway_log_level.upper(),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if session_id:
        values[WEB_SESSION_ENV] = session_id

    package_root = str(Path(__file__).resolve().parents[2])
    existing = source.get("PYTHONPATH", "")
    paths = [package_root, *(item for item in existing.split(os.pathsep) if item)]
    values["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return values


def _resolve_bun() -> str:
    bun = shutil.which("bun")
    if bun is None:
        raise WebLaunchError(
            "The Web UI requires Bun >= 1.3. Install Bun and try again."
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
        raise WebLaunchError(
            "Bun is installed but its version could not be read."
        ) from None
    raw_version = result.stdout.strip()
    match = _VERSION_PATTERN.match(raw_version)
    if match is None:
        raise WebLaunchError(f"Could not parse Bun version {raw_version!r}.")
    version = tuple(int(part or 0) for part in match.groups())
    if version < _MINIMUM_BUN_VERSION:
        raise WebLaunchError(
            f"Bun {raw_version} is too old; the Web UI requires Bun >= 1.3."
        )
    return bun


async def _read_server_ready(process: asyncio.subprocess.Process) -> dict[str, Any]:
    assert process.stdout is not None
    for _ in range(20):
        try:
            raw = await asyncio.wait_for(process.stdout.readline(), timeout=10)
        except TimeoutError:
            raise WebLaunchError("Timed out while starting the Web UI server.") from None
        if not raw:
            returncode = await process.wait()
            raise WebLaunchError(
                f"Web UI server stopped during startup (status {returncode})."
            )
        text = raw.decode("utf-8", errors="replace").strip()
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            print(text, file=sys.stderr)
            continue
        if record.get("type") == "server.ready" and record.get("url"):
            return record
        print(text, file=sys.stderr)
    raise WebLaunchError("Web UI server did not publish a startup URL.")


async def _copy_stream(
    stream: asyncio.StreamReader,
    target: Any,
) -> None:
    while chunk := await stream.readline():
        target.write(chunk.decode("utf-8", errors="replace"))
        target.flush()


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except TimeoutError:
        process.kill()
        await process.wait()
