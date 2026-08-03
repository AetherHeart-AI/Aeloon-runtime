# ruff: noqa: E501
"""Unix-domain-socket JSON-RPC 2.0 daemon for Bridge v2."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from aeloon_core.bridge.protocol import PROTOCOL_VERSION, BridgeError
from aeloon_core.config import load_config, resolve_config_path
from aeloon_core.service import CoreService

MAX_REQUEST_BYTES = 1024 * 1024


def runtime_directory(data_dir: Path | str) -> Path:
    directory = Path(data_dir).expanduser().resolve(strict=False) / "runtime"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    return directory


def default_socket_path(data_dir: Path | str) -> Path:
    return runtime_directory(data_dir) / "bridge-v2.sock"


class BridgeConnection:
    def __init__(
        self,
        daemon: BridgeDaemon,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.daemon = daemon
        self.reader = reader
        self.writer = writer
        self.attachment_roots: tuple[Path, ...] = ()
        self.session_ids: set[str] = set()
        self.subscribed = False
        self._write_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        try:
            while not self.reader.at_eof():
                try:
                    line = await self.reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        await self._error(None, BridgeError("invalid_argument", "Incomplete NDJSON request"))
                    break
                except asyncio.LimitOverrunError:
                    await self._error(None, BridgeError("invalid_argument", "Bridge request exceeds 1 MiB"))
                    break
                if len(line) > MAX_REQUEST_BYTES:
                    await self._error(None, BridgeError("invalid_argument", "Bridge request exceeds 1 MiB"))
                    break
                task = asyncio.create_task(self._handle(line))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self.daemon.connections.discard(self)
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self.subscribed:
            return
        if event.get("session_id") is not None and event.get("session_id") not in self.session_ids:
            return
        await self._send({"jsonrpc": "2.0", "method": "event", "params": event})

    async def _handle(self, raw: bytes) -> None:
        request_id: Any = None
        try:
            try:
                request = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BridgeError("invalid_argument", "Invalid UTF-8 JSON request") from None
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise BridgeError("invalid_argument", "Request must use JSON-RPC 2.0")
            request_id = request.get("id")
            if not isinstance(request_id, str | int):
                raise BridgeError("invalid_argument", "Request id must be a string or integer")
            method = request.get("method")
            params = request.get("params") or {}
            if not isinstance(method, str) or not isinstance(params, dict):
                raise BridgeError("invalid_argument", "Invalid method or params")
            result = await self.daemon.service.dispatch(
                method,
                params,
                attachment_roots=self.attachment_roots,
            )
            if method == "system.handshake":
                roots = params.get("attachment_roots") or []
                self.attachment_roots = tuple(
                    Path(item).expanduser().resolve(strict=False) for item in roots
                )
            if method == "events.subscribe":
                self.session_ids = set(params.get("session_ids") or [])
                self.subscribed = True
                replay = result.pop("events", [])
                for event in replay:
                    await self._send({"jsonrpc": "2.0", "method": "event", "params": event})
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except BridgeError as exc:
            await self._error(request_id, exc)

    async def _error(self, request_id: Any, error: BridgeError) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "error": error.to_rpc()})

    async def _send(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        async with self._write_lock:
            self.writer.write(encoded)
            await self.writer.drain()


class BridgeDaemon:
    def __init__(self, service: CoreService, socket_path: Path) -> None:
        self.service = service
        self.socket_path = socket_path.expanduser().resolve(strict=False)
        self.connections: set[BridgeConnection] = set()
        self.server: asyncio.AbstractServer | None = None
        self._remove_listener = service.add_event_listener(self._broadcast)

    async def run(self) -> None:
        if len(os.fsencode(self.socket_path)) >= 104:
            raise BridgeError(
                "invalid_argument",
                "Unix socket path is too long; choose a shorter --socket path",
            )
        parent_existed = self.socket_path.parent.exists()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.socket_path.parent.chmod(0o700)
        if self.socket_path.exists():
            if await _socket_alive(self.socket_path):
                raise BridgeError("daemon_config_conflict", f"Bridge daemon is already running at {self.socket_path}")
            self.socket_path.unlink()
        self.server = await asyncio.start_unix_server(
            self._accept,
            path=str(self.socket_path),
            limit=MAX_REQUEST_BYTES + 1,
        )
        self.socket_path.chmod(0o600)
        self._write_metadata()
        loop = asyncio.get_running_loop()
        for name in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(name, self._signal_shutdown)
        try:
            await self.service.shutdown_signal.wait()
        finally:
            self.server.close()
            for connection in tuple(self.connections):
                connection.writer.close()
            await asyncio.gather(
                *(connection.writer.wait_closed() for connection in tuple(self.connections)),
                return_exceptions=True,
            )
            await self.server.wait_closed()
            await self.service.close()
            self._remove_listener()
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            with contextlib.suppress(FileNotFoundError):
                self._metadata_path().unlink()

    def _signal_shutdown(self) -> None:
        self.service.shutdown_requested.set()
        self.service.shutdown_signal.set()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = BridgeConnection(self, reader, writer)
        self.connections.add(connection)
        await connection.run()

    async def _broadcast(self, event: dict[str, Any]) -> None:
        await asyncio.gather(
            *(connection.send_event(event) for connection in tuple(self.connections)),
            return_exceptions=True,
        )

    def _metadata_path(self) -> Path:
        return self.socket_path.parent / "bridge-v2.json"

    def _write_metadata(self) -> None:
        path = self._metadata_path()
        path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "socket_path": str(self.socket_path),
                    "config_path": str(self.service.config_path),
                    "data_dir": str(self.service.data_dir),
                    "server_instance_id": self.service.server_instance_id,
                    "protocol_version": PROTOCOL_VERSION,
                },
                separators=(",", ":"),
            ) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)


async def run_daemon(
    *,
    config_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    socket_path: Path | str | None = None,
    max_concurrent_operations: int = 4,
) -> None:
    config = load_config(config_path)
    if data_dir is not None:
        config = config.model_copy(update={"data_dir": Path(data_dir)}).normalized()
    resolved_socket = Path(socket_path) if socket_path is not None else default_socket_path(config.data_dir)
    service = CoreService(
        config_path=config_path,
        data_dir=data_dir,
        max_concurrent_operations=max_concurrent_operations,
    )
    await BridgeDaemon(service, resolved_socket).run()


async def bridge_request(
    socket_path: Path | str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 3.0,
) -> dict[str, Any]:
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(Path(socket_path))),
        timeout,
    )
    request = {"jsonrpc": "2.0", "id": "cli", "method": method, "params": params or {}}
    writer.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    await writer.drain()
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout)
        response = json.loads(raw)
        if response.get("error"):
            error = response["error"]
            raise BridgeError(str((error.get("data") or {}).get("code") or "internal_error"), str(error.get("message") or "Bridge request failed"))
        return dict(response.get("result") or {})
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def ensure_daemon(
    *,
    config_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    socket_path: Path | str | None = None,
    max_concurrent_operations: int = 4,
    required_methods: tuple[str, ...] = (),
) -> dict[str, Any]:
    config = load_config(config_path)
    if data_dir is not None:
        config = config.model_copy(update={"data_dir": Path(data_dir)}).normalized()
    resolved_config = resolve_config_path(config_path).resolve(strict=False)
    resolved_data = config.data_dir.resolve(strict=False)
    resolved_socket = (Path(socket_path) if socket_path is not None else default_socket_path(resolved_data)).expanduser().resolve(strict=False)
    runtime = runtime_directory(resolved_data)
    lock_path = runtime / "bridge-v2.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        metadata_path = runtime / "bridge-v2.json"
        if metadata_path.is_file():
            try:
                recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
                recorded_socket = Path(str(recorded.get("socket_path"))).resolve(strict=False)
            except (OSError, ValueError, json.JSONDecodeError):
                recorded_socket = resolved_socket
            if recorded_socket != resolved_socket and await _existing(recorded_socket) is not None:
                raise BridgeError(
                    "daemon_config_conflict",
                    f"Daemon for {resolved_data} is already running at a different socket: {recorded_socket}",
                )
        existing = await _existing(resolved_socket)
        if existing is not None and existing.get("status") == "stopping":
            for _ in range(100):
                await asyncio.sleep(0.05)
                existing = await _existing(resolved_socket)
                if existing is None:
                    break
        if existing is not None:
            _verify_identity(existing, resolved_config, resolved_data, resolved_socket)
            missing_methods = sorted(set(required_methods) - set(existing.get("methods") or ()))
            if missing_methods:
                if int(existing.get("active_operations") or 0) > 0:
                    raise BridgeError(
                        "invalid_state",
                        "The running Bridge daemon must be upgraded before using "
                        f"{', '.join(missing_methods)}; wait for active operations to finish",
                    )
                await bridge_request(resolved_socket, "system.shutdown")
                for _ in range(100):
                    await asyncio.sleep(0.05)
                    existing = await _existing(resolved_socket)
                    if existing is None:
                        break
                if existing is not None:
                    raise BridgeError(
                        "internal_error", "Timed out upgrading the Aeloon Core daemon"
                    )
            else:
                return {"socket_path": str(resolved_socket), **existing, "status": "running"}
        if resolved_socket.exists():
            resolved_socket.unlink()
        command = [
            sys.executable,
            "-m",
            "aeloon_core",
            "bridge",
            "serve",
            "--config",
            str(resolved_config),
            "--data-dir",
            str(resolved_data),
            "--socket",
            str(resolved_socket),
            "--max-concurrent-operations",
            str(max_concurrent_operations),
        ]
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        for _ in range(100):
            await asyncio.sleep(0.05)
            existing = await _existing(resolved_socket)
            if existing is not None:
                _verify_identity(existing, resolved_config, resolved_data, resolved_socket)
                return {"socket_path": str(resolved_socket), **existing, "status": "started"}
        raise BridgeError("internal_error", "Timed out starting the Aeloon Core daemon")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


async def daemon_status(socket_path: Path | str) -> dict[str, Any]:
    resolved = Path(socket_path).expanduser().resolve(strict=False)
    existing = await _existing(resolved)
    if existing is None:
        return {"status": "stopped", "socket_path": str(resolved)}
    return {"socket_path": str(resolved), **existing}


async def stop_daemon(socket_path: Path | str) -> dict[str, Any]:
    resolved = Path(socket_path).expanduser().resolve(strict=False)
    existing = await _existing(resolved)
    if existing is None:
        _remove_stale_socket(resolved)
        return {"status": "stopped", "socket_path": str(resolved)}
    try:
        await bridge_request(resolved, "system.shutdown")
    except (OSError, TimeoutError):
        pass
    for _ in range(100):
        if await _existing(resolved) is None:
            _remove_stale_socket(resolved)
            return {"status": "stopped", "socket_path": str(resolved)}
        await asyncio.sleep(0.05)
    raise BridgeError("internal_error", "Timed out stopping the Aeloon Core daemon")


async def _existing(socket_path: Path) -> dict[str, Any] | None:
    try:
        handshake = await bridge_request(
            socket_path,
            "system.handshake",
            {"protocol_versions": [PROTOCOL_VERSION], "client": {"name": "aeloon-core-cli", "version": "0.3.0"}, "attachment_roots": []},
            timeout=0.5,
        )
        health = await bridge_request(socket_path, "system.health", timeout=0.5)
        return {**handshake, **health}
    except (OSError, TimeoutError, json.JSONDecodeError):
        return None


async def _socket_alive(socket_path: Path) -> bool:
    return await _existing(socket_path) is not None


def _verify_identity(
    value: dict[str, Any],
    config_path: Path,
    data_dir: Path,
    socket_path: Path,
) -> None:
    conflicts: list[str] = []
    if Path(str(value.get("config_path"))).resolve(strict=False) != config_path:
        conflicts.append("config")
    if Path(str(value.get("data_dir"))).resolve(strict=False) != data_dir:
        conflicts.append("data-dir")
    if conflicts:
        raise BridgeError(
            "daemon_config_conflict",
            f"Daemon at {socket_path} is already running with different {', '.join(conflicts)} parameters",
        )


def _remove_stale_socket(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError, OSError):
        if path.is_socket():
            path.unlink()


__all__ = [
    "BridgeDaemon", "bridge_request", "daemon_status", "default_socket_path",
    "ensure_daemon", "run_daemon", "runtime_directory", "stop_daemon",
]
