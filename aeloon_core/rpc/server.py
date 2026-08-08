"""Electron-owned length-framed aeloon-rpc-v1 server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aeloon_core.rpc.adapter import AeloonRpcAdapter
from aeloon_core.rpc.protocol import MAX_FRAME_BYTES, RpcError
from aeloon_core.runtime.service import RuntimeService

MAX_CLIENTS = 8
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


def pack_frame(value: Any) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise RpcError("invalid_argument", "RPC frame exceeds 12 MiB")
    return struct.pack("!I", len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader, *, timeout: float | None = None) -> Any:
    try:
        header = await asyncio.wait_for(reader.readexactly(4), timeout)
        (length,) = struct.unpack("!I", header)
        if length > MAX_FRAME_BYTES:
            raise RpcError("invalid_argument", "RPC frame exceeds 12 MiB")
        payload = await asyncio.wait_for(reader.readexactly(length), timeout)
        return json.loads(payload)
    except asyncio.IncompleteReadError:
        raise EOFError from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RpcError("invalid_argument", "RPC frame contains invalid JSON") from None


class RpcConnection:
    def __init__(
        self,
        server: AeloonRpcServer,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.server = server
        self.reader = reader
        self.writer = writer
        self.attachment_roots: tuple[Path, ...] = ()
        self.session_ids: set[str] = set()
        self.subscribed = False
        self._write_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        try:
            while True:
                try:
                    request = await read_frame(self.reader)
                except EOFError:
                    break
                task = asyncio.create_task(self._handle(request))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self.server.connections.discard(self)
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self.subscribed:
            return
        if event.get("session_id") is not None and event.get("session_id") not in self.session_ids:
            return
        await self._send({"method": "event", "params": event})

    async def _handle(self, request: Any) -> None:
        request_id: Any = request.get("id") if isinstance(request, Mapping) else None
        try:
            if not isinstance(request, Mapping):
                raise RpcError("invalid_argument", "RPC request must be an object")
            if not isinstance(request_id, str | int):
                raise RpcError("invalid_argument", "RPC request id must be a string or integer")
            method = request.get("method")
            params = request.get("params") or {}
            if not isinstance(method, str) or not isinstance(params, Mapping):
                raise RpcError("invalid_argument", "RPC method or params are invalid")
            result = await self.server.adapter.dispatch(
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
                for event in result.pop("events", []):
                    await self._send({"method": "event", "params": event})
            await self._send({"id": request_id, "result": result})
        except RpcError as exc:
            await self._send({"id": request_id, "error": exc.to_rpc()})
        except Exception:
            error = RpcError("internal_error", "Aeloon Core could not complete the request")
            await self._send({"id": request_id, "error": error.to_rpc()})

    async def _send(self, value: dict[str, Any]) -> None:
        encoded = pack_frame(value)
        async with self._write_lock:
            self.writer.write(encoded)
            await self.writer.drain()


class AeloonRpcServer:
    def __init__(self, adapter: AeloonRpcAdapter, socket_path: Path | str) -> None:
        self.adapter = adapter
        self.socket_path = Path(socket_path).expanduser().resolve(strict=False)
        self.connections: set[RpcConnection] = set()
        self.server: asyncio.AbstractServer | None = None
        self._remove_listener = adapter.add_event_listener(self._broadcast)

    async def run(self) -> None:
        if len(os.fsencode(self.socket_path)) >= 104:
            raise RpcError("invalid_argument", "Unix socket path is too long")
        parent_existed = self.socket_path.parent.exists()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.socket_path.parent.chmod(0o700)
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        self.server = await asyncio.start_unix_server(self._accept, path=str(self.socket_path))
        self.socket_path.chmod(0o600)
        loop = asyncio.get_running_loop()
        for name in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(name, self.adapter.request_shutdown)
        try:
            await self.adapter.shutdown_signal.wait()
        finally:
            self.server.close()
            for connection in tuple(self.connections):
                connection.writer.close()
            await asyncio.gather(
                *(connection.writer.wait_closed() for connection in tuple(self.connections)),
                return_exceptions=True,
            )
            await self.server.wait_closed()
            await self.adapter.close()
            self._remove_listener()
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self.connections) >= MAX_CLIENTS:
            writer.close()
            await writer.wait_closed()
            return
        connection = RpcConnection(self, reader, writer)
        self.connections.add(connection)
        await connection.run()

    async def _broadcast(self, event: dict[str, Any]) -> None:
        await asyncio.gather(
            *(connection.send_event(event) for connection in tuple(self.connections)),
            return_exceptions=True,
        )


async def run_rpc_server(runtime: RuntimeService, *, socket_path: Path | str) -> None:
    await AeloonRpcServer(AeloonRpcAdapter(runtime), socket_path).run()


async def rpc_request(
    socket_path: Path | str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(Path(socket_path))), timeout
        )
    except (OSError, TimeoutError) as exc:
        raise RpcError("invalid_state", f"Aeloon RPC is unavailable: {exc}") from None
    try:
        writer.write(pack_frame({"id": "client", "method": method, "params": params or {}}))
        await writer.drain()
        response = await read_frame(reader, timeout=timeout)
        if not isinstance(response, Mapping):
            raise RpcError("internal_error", "Aeloon RPC returned an invalid response")
        if response.get("error"):
            error = response["error"]
            raise RpcError(
                str((error.get("data") or {}).get("code") or "internal_error"),
                str(error.get("message") or "Aeloon RPC request failed"),
            )
        return dict(response.get("result") or {})
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


__all__ = [
    "AeloonRpcServer",
    "RpcConnection",
    "pack_frame",
    "read_frame",
    "rpc_request",
    "run_rpc_server",
]
