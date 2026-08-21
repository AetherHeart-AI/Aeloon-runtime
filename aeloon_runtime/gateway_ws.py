"""WebSocket transport for the v4 gateway.

``RuntimeConnection`` reads and writes through six byte-stream methods —
``readexactly`` on the reader, ``write``/``drain``/``is_closing``/``close``/
``wait_closed`` on the writer. That is the whole transport surface, so a
WebSocket becomes another transport by supplying those six methods rather than
by teaching the connection layer a second protocol.

The length-prefixed framing is kept inside the WebSocket payload even though
WebSocket already delimits messages. One framing everywhere means ``read_frame``
and ``pack_frame`` stay transport-agnostic and the wire is identical on both
paths; four bytes per message is not worth a second set of framing rules.

A routable bind is allowed only when TLS is ready *and* at least one device is
already paired. An empty pairing store on a LAN address would let anyone walk
the enrollment flow.
"""

from __future__ import annotations

import asyncio
import ipaddress
import ssl
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve

from aeloon_runtime.pairing import is_loopback_host
from aeloon_runtime.rpc.protocol import RpcError

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from aeloon_runtime.runtime_server import RuntimeServer


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
            max_bytes = 40 * 1024 * 1024 + 4
            if len(message) > max_bytes or len(self._buffer) + len(message) > max_bytes:
                self._eof = True
                raise RpcError("payload_too_large", "RPC frame exceeds 40 MiB")
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


def _require_loopback(host: str, *, tls_ready: bool = False, paired: bool = False) -> None:
    """Reject a routable bind unless TLS and a non-empty pairing store are both ready."""

    if is_loopback_host(host):
        return
    if tls_ready and paired:
        return
    try:
        ipaddress.ip_address(host)
        shown = host
    except ValueError:
        shown = repr(host)
    raise ValueError(
        f"--listen currently accepts loopback only; {shown} would expose an "
        "unauthenticated Runtime. Device pairing unlocks other addresses."
    )


def parse_listen(value: str) -> tuple[str, int]:
    """Parse ``HOST:PORT`` — including bracketed IPv6.

    Bind policy lives in ``_require_loopback`` so a routable address can be
    accepted only after pairing and TLS are actually ready.
    """
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
    return host or "127.0.0.1", port


def build_tls_context(certificate: Path | None, key: Path | None) -> ssl.SSLContext | None:
    """Load an explicit certificate pair. Auto-generation lives in pairing."""

    if certificate is None and key is None:
        return None
    if certificate is None or key is None:
        raise ValueError("--tls-cert and --tls-key must be provided together")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(certificate), keyfile=str(key))
    return context


WS_PING_INTERVAL_S = 15
WS_PING_TIMEOUT_S = 15
WS_WRITE_LIMIT_BYTES = 4 * 1024 * 1024


async def serve_websocket(
    server: RuntimeServer,
    *,
    host: str,
    port: int,
    tls: ssl.SSLContext | None = None,
) -> Any:
    """Start the WebSocket listener and return it, already serving."""
    from aeloon_runtime.runtime_server import (
        MAX_FRAME_BYTES,
        MAX_PENDING_AUTH,
        RuntimeConnection,
    )

    async def handler(connection: ServerConnection) -> None:
        if len(server.pending_connections) >= MAX_PENDING_AUTH:
            await connection.close(code=1013, reason="Runtime connection limit reached")
            return
        stream = WebSocketByteStream(connection)
        remote = connection.remote_address
        # Rate limiting is intentionally keyed to the peer IP, never its
        # ephemeral source port. A caller must not bypass backoff by opening a
        # fresh TCP connection for every guess.
        source = str(remote[0]) if remote else "websocket"
        # Reader and writer are the same adapter; the gateway never distinguishes
        # them beyond the six methods. WebSocket always requires a device token
        # (or a one-time enrollment code); the Unix socket is the local boundary.
        rpc_connection = RuntimeConnection(
            server, stream, stream, requires_auth=True, auth_source=source
        )
        server.pending_connections.add(rpc_connection)
        try:
            await rpc_connection.run()
        finally:
            server.connections.discard(rpc_connection)
            server.pending_connections.discard(rpc_connection)

    return await ws_serve(
        handler,
        host,
        port,
        ssl=tls,
        # Frames are already bounded by the gateway's own limit; letting the
        # library enforce a smaller one would truncate legitimate attachments.
        max_size=MAX_FRAME_BYTES + 4,
        ping_interval=WS_PING_INTERVAL_S,
        ping_timeout=WS_PING_TIMEOUT_S,
        # Permessage-deflate and the 32 KiB write buffer turn 25 MiB uploads
        # over SSH into CPU- and RTT-bound transfers. Keep the socket raw and
        # give the tunnel a window that can fill a long-fat pipe.
        compression=None,
        write_limit=WS_WRITE_LIMIT_BYTES,
    )


__all__ = [
    "WS_PING_INTERVAL_S",
    "WS_PING_TIMEOUT_S",
    "WebSocketByteStream",
    "build_tls_context",
    "parse_listen",
    "require_listen_host",
    "serve_websocket",
]


def require_listen_host(host: str, *, tls_ready: bool, paired: bool) -> None:
    _require_loopback(host, tls_ready=tls_ready, paired=paired)
