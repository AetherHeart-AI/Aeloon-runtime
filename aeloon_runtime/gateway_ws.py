"""WebSocket transport for the v3 gateway.

``RuntimeV3Connection`` reads and writes through six byte-stream methods —
``readexactly`` on the reader, ``write``/``drain``/``is_closing``/``close``/
``wait_closed`` on the writer. That is the whole transport surface, so a
WebSocket becomes another transport by supplying those six methods rather than
by teaching the connection layer a second protocol.

The length-prefixed framing is kept inside the WebSocket payload even though
WebSocket already delimits messages. One framing everywhere means ``read_frame``
and ``pack_frame`` stay transport-agnostic and the wire is identical on both
paths; four bytes per message is not worth a second set of framing rules.

Binding is restricted to loopback until the pairing and token work lands. An
unauthenticated listener on a routable address would be an open door, and this
module has no way to tell an owner from anyone else yet.
"""

from __future__ import annotations

import asyncio
import ipaddress
import ssl
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets
from websockets.asyncio.server import ServerConnection, serve as ws_serve

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from aeloon_runtime.runtime_server_v3 import RuntimeV3Server


class WebSocketByteStream:
    """Adapts one WebSocket connection to the reader/writer pair the gateway uses.

    The same object is handed in as both reader and writer: the connection layer
    only ever calls the six methods below, and splitting them into two wrappers
    would add a class without adding a boundary.
    """

    def __init__(self, connection: ServerConnection) -> None:
        self._connection = connection
        self._buffer = bytearray()
        self._pending = bytearray()
        self._eof = False
        self._closed = False

    # -- reader ---------------------------------------------------------------

    async def readexactly(self, count: int) -> bytes:
        while len(self._buffer) < count:
            if self._eof:
                # Mirrors StreamReader: the caller turns this into EOFError and
                # shuts the connection down cleanly.
                raise asyncio.IncompleteReadError(bytes(self._buffer), count)
            try:
                message = await self._connection.recv()
            except websockets.ConnectionClosed:
                self._eof = True
                continue
            if isinstance(message, str):
                # The gateway speaks binary frames only. A text frame means the
                # peer is not talking aeloon-rpc, so stop rather than guess.
                self._eof = True
                continue
            self._buffer.extend(message)
        chunk = bytes(self._buffer[:count])
        del self._buffer[:count]
        return chunk

    # -- writer ---------------------------------------------------------------

    def write(self, data: bytes) -> None:
        # write() is synchronous in the StreamWriter contract while sending is
        # not, so bytes accumulate here and drain() emits them. The gateway pairs
        # every write with a drain, so one RPC frame is one WebSocket message.
        self._pending.extend(data)

    async def drain(self) -> None:
        if not self._pending:
            return
        payload = bytes(self._pending)
        self._pending.clear()
        try:
            await self._connection.send(payload)
        except websockets.ConnectionClosed as exc:
            raise ConnectionResetError("WebSocket connection is closed") from exc

    def is_closing(self) -> bool:
        return self._closed or self._connection.close_code is not None

    def close(self) -> None:
        self._closed = True
        self._pending.clear()

    async def wait_closed(self) -> None:
        await self._connection.close()


def _require_loopback(host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host in {"localhost", ""}:
            return
        raise ValueError(
            f"--listen currently accepts loopback only; {host!r} would expose an "
            "unauthenticated Runtime. Device pairing unlocks other addresses."
        ) from None
    if not address.is_loopback:
        raise ValueError(
            f"--listen currently accepts loopback only; {host} would expose an "
            "unauthenticated Runtime. Device pairing unlocks other addresses."
        )


def parse_listen(value: str) -> tuple[str, int]:
    """Parse ``HOST:PORT`` — including bracketed IPv6 — and reject routable hosts."""
    text = value.strip()
    if text.startswith("["):
        closing = text.find("]")
        if closing == -1 or not text[closing + 1 :].startswith(":"):
            raise ValueError("--listen must look like [::1]:PORT for IPv6")
        host, port_text = text[1:closing], text[closing + 2 :]
    else:
        host, separator, port_text = text.rpartition(":")
        if not separator:
            raise ValueError("--listen must look like HOST:PORT")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"--listen port is not a number: {port_text}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"--listen port is out of range: {port}")
    _require_loopback(host)
    return host or "127.0.0.1", port


def build_tls_context(certificate: Path | None, key: Path | None) -> ssl.SSLContext | None:
    """TLS is opt-in for now: R1 serves loopback, where the socket is the boundary.

    Certificate generation belongs with pairing, because the fingerprint only
    means something once it travels with the pairing string.
    """
    if certificate is None and key is None:
        return None
    if certificate is None or key is None:
        raise ValueError("--tls-cert and --tls-key must be provided together")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(certificate), keyfile=str(key))
    return context


async def serve_websocket(
    server: RuntimeV3Server,
    *,
    host: str,
    port: int,
    tls: ssl.SSLContext | None = None,
) -> Any:
    """Start the WebSocket listener and return it, already serving."""
    from aeloon_runtime.runtime_server_v3 import MAX_CLIENTS, RuntimeV3Connection

    async def handler(connection: ServerConnection) -> None:
        if len(server.connections) >= MAX_CLIENTS:
            await connection.close(code=1013, reason="Runtime connection limit reached")
            return
        stream = WebSocketByteStream(connection)
        # Reader and writer are the same adapter; the gateway never distinguishes
        # them beyond the six methods.
        rpc_connection = RuntimeV3Connection(server, stream, stream)
        server.connections.add(rpc_connection)
        try:
            await rpc_connection.run()
        finally:
            server.connections.discard(rpc_connection)

    return await ws_serve(
        handler,
        host,
        port,
        ssl=tls,
        # Frames are already bounded by the gateway's own limit; letting the
        # library enforce a smaller one would truncate legitimate attachments.
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    )


__all__ = [
    "WebSocketByteStream",
    "build_tls_context",
    "parse_listen",
    "serve_websocket",
]
