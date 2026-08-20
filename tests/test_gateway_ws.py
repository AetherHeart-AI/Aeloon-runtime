from __future__ import annotations

import asyncio
import json
import os
import struct
from pathlib import Path

import pytest
import websockets

from aeloon_runtime.gateway_ws import WebSocketByteStream, build_tls_context, parse_listen
from aeloon_runtime.runtime_server_v3 import serve_v3


def test_parse_listen_accepts_loopback_forms() -> None:
    assert parse_listen("127.0.0.1:7420") == ("127.0.0.1", 7420)
    assert parse_listen("localhost:80") == ("localhost", 80)
    assert parse_listen("[::1]:9000") == ("::1", 9000)


@pytest.mark.parametrize("value", ["0.0.0.0:7420", "192.168.1.5:7420", "example.com:443"])
def test_parse_listen_refuses_routable_addresses(value: str) -> None:
    # Until device pairing exists there is nothing to authenticate against, so a
    # routable bind would publish an unauthenticated Runtime.
    with pytest.raises(ValueError, match="loopback only"):
        parse_listen(value)


@pytest.mark.parametrize("value", ["127.0.0.1", "127.0.0.1:abc", "127.0.0.1:0", "[::1]7420"])
def test_parse_listen_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_listen(value)


def test_build_tls_context_requires_both_halves(tmp_path: Path) -> None:
    assert build_tls_context(None, None) is None
    with pytest.raises(ValueError, match="must be provided together"):
        build_tls_context(tmp_path / "cert.pem", None)
    with pytest.raises(ValueError, match="must be provided together"):
        build_tls_context(None, tmp_path / "key.pem")


class _FakeConnection:
    """Minimal stand-in for a websockets ServerConnection."""

    def __init__(self, messages: list[bytes | str]) -> None:
        self._messages = list(messages)
        self.sent: list[bytes] = []
        self.close_code: int | None = None

    async def recv(self) -> bytes | str:
        if not self._messages:
            raise websockets.ConnectionClosed(None, None)
        return self._messages.pop(0)

    async def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.close_code = 1000


@pytest.mark.asyncio
async def test_byte_stream_reassembles_frames_across_messages() -> None:
    # WebSocket message boundaries are not frame boundaries: the length prefix
    # decides, so a frame split across two messages must still read back whole.
    stream = WebSocketByteStream(_FakeConnection([b"abc", b"defgh"]))  # type: ignore[arg-type]
    assert await stream.readexactly(4) == b"abcd"
    assert await stream.readexactly(4) == b"efgh"


@pytest.mark.asyncio
async def test_byte_stream_reports_close_as_incomplete_read() -> None:
    stream = WebSocketByteStream(_FakeConnection([b"ab"]))  # type: ignore[arg-type]
    with pytest.raises(asyncio.IncompleteReadError):
        await stream.readexactly(4)


@pytest.mark.asyncio
async def test_byte_stream_sends_one_message_per_drain() -> None:
    connection = _FakeConnection([])
    stream = WebSocketByteStream(connection)  # type: ignore[arg-type]
    stream.write(b"one")
    stream.write(b"-two")
    await stream.drain()
    await stream.drain()
    assert connection.sent == [b"one-two"]


@pytest.mark.asyncio
async def test_byte_stream_ignores_text_frames() -> None:
    # A text frame means the peer is not speaking aeloon-rpc; stop instead of
    # trying to interpret it as protocol bytes.
    stream = WebSocketByteStream(_FakeConnection(["hello"]))  # type: ignore[arg-type]
    with pytest.raises(asyncio.IncompleteReadError):
        await stream.readexactly(1)


async def _ws_request(url: str, method: str, params: dict) -> dict:
    async with websockets.connect(url, max_size=None) as connection:
        await connection.send(_frame({
            "id": "1",
            "method": "system.handshake",
            "params": {
                "protocol": {"min": "3.0.0", "max": "3.0.0"},
                "client": {"name": "pytest", "version": "0", "platform": "test"},
            },
        }))
        await _read(connection)
        await connection.send(_frame({"id": "2", "method": method, "params": params}))
        return await _read(connection)


def _frame(value: dict) -> bytes:
    payload = json.dumps(value, separators=(",", ":")).encode()
    return struct.pack("!I", len(payload)) + payload


async def _read(connection) -> dict:
    buffer = bytearray()
    while True:
        buffer.extend(await connection.recv())
        if len(buffer) >= 4:
            (size,) = struct.unpack("!I", buffer[:4])
            if len(buffer) >= 4 + size:
                return json.loads(bytes(buffer[4 : 4 + size]))


@pytest.mark.asyncio
async def test_websocket_transport_serves_the_same_method_table(tmp_path: Path) -> None:
    # AF_UNIX paths are capped near 104 bytes, and pytest's tmp_path is longer.
    socket_path = Path("/tmp") / f"aeloon-ws-{os.getpid()}.sock"
    port = 45_501
    task = asyncio.create_task(
        serve_v3(
            socket_path=socket_path,
            data_dir=tmp_path / "data",
            workspace_roots=(tmp_path,),
            listen=("127.0.0.1", port),
        )
    )
    try:
        for _ in range(200):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        # The listener comes up after the Unix socket; retry until it accepts.
        for _ in range(200):
            try:
                health = await _ws_request(f"ws://127.0.0.1:{port}", "system.health", {})
                break
            except OSError:
                await asyncio.sleep(0.02)
        else:  # pragma: no cover - only on a wedged listener
            raise AssertionError("WebSocket listener never accepted a connection")
        assert health["result"]["ok"] is True
        # The Unix socket keeps serving while the WebSocket listener is up.
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(_frame({"id": "1", "method": "system.handshake", "params": {
            "protocol": {"min": "3.0.0", "max": "3.0.0"},
            "client": {"name": "pytest", "version": "0", "platform": "test"},
        }}))
        await writer.drain()
        size = struct.unpack("!I", await reader.readexactly(4))[0]
        assert json.loads(await reader.readexactly(size))["result"]["protocol"] == "3.0.0"
        writer.close()
        await writer.wait_closed()
    finally:
        await _ws_request(f"ws://127.0.0.1:{port}", "system.shutdown", {})
        await asyncio.wait_for(task, timeout=5)
        socket_path.unlink(missing_ok=True)
