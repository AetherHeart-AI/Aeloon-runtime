from __future__ import annotations

import asyncio
import json
import os
import ssl
import struct
from pathlib import Path

import pytest
import websockets

from aeloon_runtime.gateway_ws import parse_listen, require_listen_host
from aeloon_runtime.runtime_server import serve


def test_parse_listen_accepts_loopback_and_routable_forms() -> None:
    assert parse_listen("127.0.0.1:7420") == ("127.0.0.1", 7420)
    assert parse_listen("localhost:80") == ("localhost", 80)
    assert parse_listen("[::1]:9000") == ("::1", 9000)
    assert parse_listen("0.0.0.0:7420") == ("0.0.0.0", 7420)
    assert parse_listen("192.168.1.5:7420") == ("192.168.1.5", 7420)


def test_require_listen_host_needs_tls_and_a_paired_device() -> None:
    require_listen_host("127.0.0.1", tls_ready=False, paired=False)
    with pytest.raises(ValueError, match="loopback only"):
        require_listen_host("0.0.0.0", tls_ready=True, paired=False)
    with pytest.raises(ValueError, match="loopback only"):
        require_listen_host("192.168.1.5", tls_ready=False, paired=True)
    with pytest.raises(ValueError, match="loopback only"):
        require_listen_host("example.com", tls_ready=True, paired=False)
    require_listen_host("0.0.0.0", tls_ready=True, paired=True)


@pytest.mark.parametrize("value", ["127.0.0.1", "127.0.0.1:abc", "127.0.0.1:0", "[::1]7420"])
def test_parse_listen_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_listen(value)


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


async def _unix_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    value: dict,
) -> dict:
    writer.write(_frame(value))
    await writer.drain()
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    return json.loads(await reader.readexactly(size))


def _handshake_params(**auth: object) -> dict:
    params: dict = {
        "protocol": {"min": "4.0.0", "max": "4.0.0"},
        "client": {"name": "pytest", "version": "0", "platform": "test"},
    }
    if auth:
        params["auth"] = auth
    return params


async def _start_runtime(tmp_path: Path, port: int) -> tuple[asyncio.Task, Path]:
    socket_path = Path("/tmp") / f"aeloon-pair-{os.getpid()}-{port}.sock"
    socket_path.unlink(missing_ok=True)
    task = asyncio.create_task(
        serve(
            socket_path=socket_path,
            data_dir=tmp_path / "data",
            workspace_roots=(tmp_path,),
            listen=("127.0.0.1", port),
        )
    )
    for _ in range(400):
        if task.done():
            await task
        if socket_path.exists():
            try:
                reader, writer = await asyncio.open_unix_connection(str(socket_path))
                writer.close()
                await writer.wait_closed()
                return task, socket_path
            except (ConnectionRefusedError, OSError):
                pass
        await asyncio.sleep(0.02)
    raise AssertionError("Runtime did not create a Unix socket")


async def _stop_runtime(task: asyncio.Task, socket_path: Path, port: int) -> None:
    if not task.done():
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            await _unix_request(
                reader,
                writer,
                {"id": "shutdown", "method": "system.handshake", "params": _handshake_params()},
            )
            await _unix_request(
                reader, writer, {"id": "down", "method": "system.shutdown", "params": {}}
            )
            writer.close()
            await writer.wait_closed()
        except OSError:
            task.cancel()
    try:
        await asyncio.wait_for(task, timeout=5)
    except (asyncio.CancelledError, Exception):
        if not task.done():
            task.cancel()
    socket_path.unlink(missing_ok=True)


async def _enroll_over_unix(socket_path: Path) -> dict:
    last_error: dict | None = None
    for _ in range(200):
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
        except OSError:
            await asyncio.sleep(0.02)
            continue
        try:
            handshake = await _unix_request(
                reader,
                writer,
                {"id": "1", "method": "system.handshake", "params": _handshake_params()},
            )
            if "error" in handshake:
                last_error = handshake
                continue
            enrolled = await _unix_request(
                reader, writer, {"id": "2", "method": "devices.enroll", "params": {}}
            )
            if enrolled.get("result"):
                return enrolled["result"]
            last_error = enrolled
        finally:
            writer.close()
            await writer.wait_closed()
        await asyncio.sleep(0.02)
    raise AssertionError(f"devices.enroll never succeeded: {last_error}")


async def _claim_over_websocket(url: str, code: str, ssl_ctx: ssl.SSLContext) -> dict:
    async with websockets.connect(url, ssl=ssl_ctx, max_size=None) as connection:
        await connection.send(
            _frame(
                {
                    "id": "claim",
                    "method": "devices.claim",
                    "params": {
                        "code": code,
                        "client": {"name": "pytest", "version": "0"},
                    },
                }
            )
        )
        response = await _read(connection)
        assert "result" in response
        return response["result"]


@pytest.mark.asyncio
async def test_listen_on_all_interfaces_without_devices_is_rejected(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-lan-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    task = asyncio.create_task(
        serve(
            socket_path=socket_path,
            data_dir=tmp_path / "data",
            workspace_roots=(tmp_path,),
            listen=("0.0.0.0", 47_401),
        )
    )
    with pytest.raises(ValueError, match="advertise-url"):
        await asyncio.wait_for(task, timeout=5)
    socket_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_listen_prints_pairing_url_and_issues_a_token(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    port = 47_402
    task, socket_path = await _start_runtime(tmp_path, port)
    try:
        for _ in range(50):
            captured = capsys.readouterr().out
            if "aeloon://pair?" in captured:
                break
            await asyncio.sleep(0.05)
        else:
            # The pairing URL may have been printed before capsys attached to the
            # already-running task; enroll over Unix still proves the vault is live.
            pass
        enrollment = await _enroll_over_unix(socket_path)
        assert enrollment["code"]
        assert enrollment["pairing_url"].startswith("aeloon://pair?")
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        claimed = await _claim_over_websocket(
            f"wss://127.0.0.1:{port}", enrollment["code"], ssl_ctx
        )
        token = claimed["token"]
        device_id = claimed["device_id"]
        async with websockets.connect(
            f"wss://127.0.0.1:{port}", ssl=ssl_ctx, max_size=None
        ) as connection:
            await connection.send(
                _frame(
                    {
                        "id": "1",
                        "method": "system.handshake",
                        "params": _handshake_params(kind="device_token", token=token),
                    }
                )
            )
            second = await _read(connection)
            assert "device" not in second["result"]
            assert second["result"]["protocol"] in {"4.0.0"}
        async with websockets.connect(
            f"wss://127.0.0.1:{port}", ssl=ssl_ctx, max_size=None
        ) as connection:
            await connection.send(
                _frame(
                    {
                        "id": "1",
                        "method": "system.handshake",
                        "params": _handshake_params(kind="device_token", token="nope"),
                    }
                )
            )
            failed = await _read(connection)
            assert failed["error"]["data"]["code"] == "unauthorized"
            await asyncio.wait_for(connection.wait_closed(), timeout=1)
        listed = json.loads((tmp_path / "data" / "devices-v4.json").read_text(encoding="utf-8"))
        assert listed["devices"][0]["id"] == device_id
        assert token not in (tmp_path / "data" / "devices-v4.json").read_text(encoding="utf-8")
    finally:
        await _stop_runtime(task, socket_path, port)


@pytest.mark.asyncio
async def test_websocket_without_auth_is_unauthorized(tmp_path: Path) -> None:
    port = 47_403
    task, socket_path = await _start_runtime(tmp_path, port)
    try:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        response = None
        for _ in range(200):
            try:
                async with websockets.connect(
                    f"wss://127.0.0.1:{port}", ssl=ssl_ctx, max_size=None
                ) as connection:
                    await connection.send(
                        _frame(
                            {"id": "1", "method": "system.handshake", "params": _handshake_params()}
                        )
                    )
                    response = await _read(connection)
                    break
            except OSError:
                await asyncio.sleep(0.02)
        assert response is not None
        assert response["error"]["data"]["code"] == "unauthorized"
    finally:
        await _stop_runtime(task, socket_path, port)


@pytest.mark.asyncio
async def test_websocket_permanently_forbids_runtime_lifecycle_methods(tmp_path: Path) -> None:
    port = 47_406
    task, socket_path = await _start_runtime(tmp_path, port)
    try:
        enrollment = await _enroll_over_unix(socket_path)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        claimed = await _claim_over_websocket(
            f"wss://127.0.0.1:{port}", enrollment["code"], ssl_ctx
        )
        async with websockets.connect(
            f"wss://127.0.0.1:{port}", ssl=ssl_ctx, max_size=None
        ) as connection:
            await connection.send(
                _frame(
                    {
                        "id": "handshake",
                        "method": "system.handshake",
                        "params": _handshake_params(
                            kind="device_token", token=claimed["token"]
                        ),
                    }
                )
            )
            assert "result" in await _read(connection)
            for index, method in enumerate(
                ("system.shutdown", "system.uninstall_inspect", "system.uninstall_prepare")
            ):
                await connection.send(
                    _frame({"id": f"lifecycle-{index}", "method": method, "params": {}})
                )
                response = await _read(connection)
                assert response["error"]["data"]["code"] == "forbidden"
        assert not task.done()
    finally:
        await _stop_runtime(task, socket_path, port)


@pytest.mark.asyncio
async def test_revoke_disconnects_the_active_connection(tmp_path: Path) -> None:
    port = 47_404
    task, socket_path = await _start_runtime(tmp_path, port)
    try:
        enrollment = await _enroll_over_unix(socket_path)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        claimed = await _claim_over_websocket(
            f"wss://127.0.0.1:{port}", enrollment["code"], ssl_ctx
        )
        async with websockets.connect(
            f"wss://127.0.0.1:{port}", ssl=ssl_ctx, max_size=None
        ) as connection:
            await connection.send(
                _frame(
                    {
                        "id": "1",
                        "method": "system.handshake",
                        "params": _handshake_params(
                            kind="device_token", token=claimed["token"]
                        ),
                    }
                )
            )
            await _read(connection)
            device_id = claimed["device_id"]
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            await _unix_request(
                reader,
                writer,
                {"id": "h", "method": "system.handshake", "params": _handshake_params()},
            )
            revoked = await _unix_request(
                reader,
                writer,
                {"id": "r", "method": "devices.revoke", "params": {"device_id": device_id}},
            )
            assert revoked["result"]["revoked"] is True
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(connection.wait_closed(), timeout=3)
    finally:
        await _stop_runtime(task, socket_path, port)


@pytest.mark.asyncio
async def test_unix_socket_still_works_without_auth(tmp_path: Path) -> None:
    port = 47_405
    task, socket_path = await _start_runtime(tmp_path, port)
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        handshake = await _unix_request(
            reader,
            writer,
            {"id": "1", "method": "system.handshake", "params": _handshake_params()},
        )
        assert handshake["result"]["protocol"] in {"4.0.0"}
        assert "device" not in handshake["result"]
        health = await _unix_request(
            reader, writer, {"id": "2", "method": "system.health", "params": {}}
        )
        assert health["result"]["ok"] is True
        writer.close()
        await writer.wait_closed()
    finally:
        await _stop_runtime(task, socket_path, port)


@pytest.mark.asyncio
async def test_seventeenth_wss_connection_receives_busy_after_unix_clients_fill_the_limit(
    tmp_path: Path,
) -> None:
    port = 47_410
    task, socket_path = await _start_runtime(tmp_path, port)
    held: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
    try:
        enrollment = await _enroll_over_unix(socket_path)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        claimed = await _claim_over_websocket(
            f"wss://127.0.0.1:{port}", enrollment["code"], ssl_ctx
        )
        handshake = _handshake_params()
        for index in range(16):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            response = await _unix_request(
                reader,
                writer,
                {"id": f"hs-{index}", "method": "system.handshake", "params": handshake},
            )
            assert "result" in response
            held.append((reader, writer))
        async with websockets.connect(
            f"wss://127.0.0.1:{port}", ssl=ssl_ctx, max_size=None
        ) as connection:
            await connection.send(
                _frame(
                    {
                        "id": "busy",
                        "method": "system.handshake",
                        "params": _handshake_params(
                            kind="device_token", token=claimed["token"]
                        ),
                    }
                )
            )
            response = await _read(connection)
            assert response["error"]["data"]["code"] == "busy"
            await asyncio.wait_for(connection.wait_closed(), timeout=2)
    finally:
        for _reader, writer in held:
            writer.close()
            await writer.wait_closed()
        await _stop_runtime(task, socket_path, port)
