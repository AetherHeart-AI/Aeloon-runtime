from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import os
import struct
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import aeloon_runtime.runtime_server as runtime_server
from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.rpc.protocol import RpcError
from aeloon_runtime.runtime.session import SessionError
from aeloon_runtime.runtime_server import (
    MAX_FRAME_BYTES,
    RuntimeConnection,
    RuntimeServer,
    _root_id,
    serve,
)


def _project_params(root: Path, target: Path | None = None) -> dict[str, str]:
    root = root.resolve()
    target = (target or root).resolve()
    relative = target.relative_to(root).as_posix() or "."
    return {"root_id": _root_id(root), "relative_path": relative}


async def _request(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, value: object
) -> dict:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack("!I", len(payload)) + payload)
    await writer.drain()
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    return json.loads((await reader.readexactly(size)).decode("utf-8"))


async def _handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> dict:
    return await _request(
        reader,
        writer,
        {
            "id": "handshake",
            "method": "system.handshake",
            "params": {
                "protocol": {"min": "4.0.0", "max": "4.0.0"},
                "client": {"name": "pytest", "version": "0", "platform": "test"},
            },
        },
    )


async def _request_with_events(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, value: object
) -> tuple[dict, list[dict]]:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack("!I", len(payload)) + payload)
    await writer.drain()
    events: list[dict] = []
    while True:
        size = struct.unpack("!I", await reader.readexactly(4))[0]
        frame = json.loads((await reader.readexactly(size)).decode("utf-8"))
        if frame.get("method") == "event":
            if isinstance(frame.get("params"), dict):
                events.append(frame["params"])
            continue
        if isinstance(value, dict) and frame.get("id") == value.get("id"):
            return frame, events


async def _read_response(reader: asyncio.StreamReader) -> dict:
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    return json.loads((await reader.readexactly(size)).decode("utf-8"))


@pytest.mark.asyncio
async def test_v4_handshake_health_and_shutdown(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-{os.getpid()}.sock"
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    incompatible = await _request(
        reader,
        writer,
        {
            "id": "0",
            "method": "system.handshake",
            "params": {
                # A client that only speaks the removed major cannot be served.
                "protocol": {"min": "3.0.0", "max": "3.0.0"},
                "client": {"name": "pytest", "version": "0", "platform": "test"},
            },
        },
    )
    assert incompatible["error"]["data"]["code"] == "protocol_incompatible"
    before_handshake = await _request(
        reader, writer, {"id": "0a", "method": "system.health", "params": {}}
    )
    assert before_handshake["error"]["data"]["code"] == "protocol_incompatible"
    handshake = await _request(
        reader,
        writer,
        {
            "id": "1",
            "method": "system.handshake",
            "params": {
                "protocol": {"min": "4.0.0", "max": "4.0.0"},
                "client": {"name": "pytest", "version": "0", "platform": "test"},
            },
        },
    )
    assert handshake["result"]["limits"]["request_bytes"] == MAX_FRAME_BYTES
    assert handshake["result"]["runtime"]["id"]
    assert handshake["result"]["host"]["platform"]
    health = await _request(reader, writer, {"id": "2", "method": "system.health", "params": {}})
    assert health["result"]["ok"] is True
    capabilities = await _request(
        reader, writer, {"id": "2a", "method": "system.capabilities", "params": {}}
    )
    core_capability = next(
        item for item in capabilities["result"]["capabilities"] if item["id"] == "core"
    )
    assert "system.snapshot" in core_capability["methods"]
    assert "diagnostics.logs" in core_capability["methods"]
    assert "plugins.configure" not in core_capability["methods"]
    cloud_capability = next(
        item for item in capabilities["result"]["capabilities"] if item["id"] == "cloud"
    )
    assert "plugin.cloud.account_status" in cloud_capability["methods"]
    catalog = await _request(reader, writer, {"id": "2b", "method": "catalog.get", "params": {}})
    assert "default_model_id" in catalog["result"]
    cloud = await _request(
        reader,
        writer,
        {"id": "2c", "method": "plugin.cloud.account_status", "params": {}},
    )
    assert "authenticated" in cloud["result"]
    plugins = await _request(reader, writer, {"id": "2d", "method": "plugins.list", "params": {}})
    assert plugins["error"]["data"]["code"] == "capability_unavailable"
    shutdown = await _request(
        reader, writer, {"id": "3", "method": "system.shutdown", "params": {}}
    )
    assert shutdown["result"]["accepted"] is True
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)
    runtime_log = tmp_path / "data" / "runtime.log"
    assert runtime_log.stat().st_mode & 0o777 == 0o600
    lifecycle = [
        json.loads(line)["event"]
        for line in runtime_log.read_text(encoding="utf-8").splitlines()
    ]
    assert lifecycle[0] == "started"
    assert lifecycle[-1] == "stopped"


@pytest.mark.asyncio
async def test_v4_subscribe_returns_replay_in_result_without_duplicate_event_frames(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    socket_path = Path("/tmp") / f"aeloon-v4-replay-{os.getpid()}.sock"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, socket_path, (tmp_path,), data_dir)
    task = asyncio.create_task(server.run())
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    await server._on_runtime_event(
        {"name": "system.shutdown", "payload": {"intentional": False, "reason": "replay-test"}}
    )
    writer.write(
        struct.pack(
            "!I",
            len(
                payload := json.dumps(
                    {
                        "id": "subscribe",
                        "method": "events.subscribe",
                        "params": {"thread_ids": [], "after_seq": 0},
                    },
                    separators=(",", ":"),
                ).encode()
            ),
        )
        + payload
    )
    await writer.drain()
    frames: list[dict[str, Any]] = []
    while True:
        frame = await _read_response(reader)
        frames.append(frame)
        if frame.get("id") == "subscribe":
            break
    assert [frame for frame in frames if frame.get("method") == "event"] == []
    response = next(frame for frame in frames if frame.get("id") == "subscribe")
    assert [event["name"] for event in response["result"]["events"]] == ["system.shutdown"]
    server.stop_event.set()
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_control_requests_are_not_blocked_by_a_long_request(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-concurrent-{os.getpid()}.sock"
    runtime = create_runtime_service(
        config_path=tmp_path / "data" / "config.json", data_dir=tmp_path / "data"
    )
    server = RuntimeServer(runtime, socket_path, (tmp_path,), tmp_path / "data")
    task = asyncio.create_task(server.run())
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_dispatch = server.dispatch

    async def delayed_dispatch(
        method: str, params: Mapping[str, Any], connection: Any = None
    ) -> Any:
        if method == "catalog.get":
            entered.set()
            await release.wait()
        return await original_dispatch(method, params, connection)

    server.dispatch = delayed_dispatch  # type: ignore[method-assign]
    slow = {"id": "slow", "method": "catalog.get", "params": {}}
    encoded = json.dumps(slow, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack("!I", len(encoded)) + encoded)
    await writer.drain()
    await asyncio.wait_for(entered.wait(), timeout=1)

    health = asyncio.create_task(
        _request(reader, writer, {"id": "health", "method": "system.health", "params": {}})
    )
    health_result = await asyncio.wait_for(health, timeout=1)
    assert health_result["result"]["ok"] is True

    release.set()
    slow_result = await asyncio.wait_for(_read_response(reader), timeout=1)
    assert slow_result["id"] == "slow"
    assert "result" in slow_result
    await _request(reader, writer, {"id": "shutdown", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_trace_recording_is_explicit_and_captures_boundary(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-trace-{os.getpid()}.sock"
    trace_directory = tmp_path / "traces"
    task = asyncio.create_task(
        serve(
            socket_path=socket_path,
            data_dir=tmp_path / "data",
            workspace_roots=(tmp_path,),
            record_trace=trace_directory,
        )
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    health = await _request(reader, writer, {"id": "1", "method": "system.health", "params": {}})
    assert health["result"]["ok"] is True
    await _request(reader, writer, {"id": "2", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)

    trace_files = sorted(trace_directory.glob("*.jsonl"))
    assert len(trace_files) == 1
    records = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    assert [record["kind"] for record in records] == [
        "checkpoint",
        "request",
        "response",
        "request",
        "response",
        "request",
        "event",
        "response",
    ]


@pytest.mark.asyncio
async def test_v4_does_not_unlink_a_non_socket_path(tmp_path: Path) -> None:
    socket_path = tmp_path / "runtime.sock"
    socket_path.write_text("keep me", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a Unix socket"):
        await serve(
            socket_path=socket_path,
            data_dir=tmp_path / "data",
            workspace_roots=(tmp_path,),
        )
    assert socket_path.read_text(encoding="utf-8") == "keep me"


@pytest.mark.asyncio
async def test_v4_data_lock_rejects_concurrent_runtime(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-lock-{os.getpid()}.sock"
    data_dir = tmp_path / "data"
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=data_dir, workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    second_runtime = create_runtime_service(
        config_path=data_dir / "config.json",
        data_dir=data_dir,
    )
    second = RuntimeServer(
        second_runtime,
        Path("/tmp") / f"aeloon-v4-lock-second-{os.getpid()}.sock",
        (tmp_path,),
        data_dir,
    )
    with pytest.raises(RuntimeError, match="already in use"):
        await second.run()
    await second_runtime.close()
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    await _request(reader, writer, {"id": "shutdown", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_composition_locks_data_before_runtime_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-compose-lock-{os.getpid()}.sock"
    data_dir = tmp_path / "data"
    observed = False
    original = runtime_server.create_runtime_service

    def factory(*args: Any, **kwargs: Any) -> Any:
        nonlocal observed
        fd = os.open(data_dir / ".runtime.lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                observed = True
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_server, "create_runtime_service", factory)
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=data_dir, workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    await _request(reader, writer, {"id": "shutdown", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)
    assert observed is True


@pytest.mark.asyncio
async def test_v4_gateway_rejects_unknown_parameters(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-invalid-{os.getpid()}.sock"
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    invalid = await _request(
        reader,
        writer,
        {"id": "1", "method": "system.health", "params": {"unexpected": True}},
    )
    assert invalid["error"]["data"]["code"] == "invalid_argument"
    await _request(reader, writer, {"id": "2", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_frame_limit_returns_error_before_closing_connection(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-frame-limit-{os.getpid()}.sock"
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    # The header alone is enough to exercise the length guard; the Runtime must
    # not attempt to allocate or read a 40 MiB body before rejecting it.
    writer.write(struct.pack("!I", MAX_FRAME_BYTES + 1))
    await writer.drain()
    response = await asyncio.wait_for(_read_response(reader), timeout=1)
    assert response["id"] is None
    assert response["error"]["data"]["code"] == "payload_too_large"
    assert await reader.read() == b""
    writer.close()
    await writer.wait_closed()
    # A frame violation only closes that client; the Runtime remains available
    # for other clients, so stop it through a clean replacement connection.
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    await _request(reader, writer, {"id": "shutdown", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_projects_threads_and_workspace_boundary(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-projects-{os.getpid()}.sock"
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    added = await _request(
        reader,
        writer,
        {"id": "1", "method": "project.add", "params": _project_params(tmp_path)},
    )
    project = added["result"]["project"]
    created = await _request(
        reader,
        writer,
        {
            "id": "2",
            "method": "thread.create",
            "params": {"project_id": project["id"], "title": "Test thread", "kind": "standard"},
        },
    )
    thread = created["result"]["thread"]
    listed = await _request(
        reader, writer, {"id": "3", "method": "thread.list", "params": {"filter": "active"}}
    )
    assert listed["result"]["threads"][0]["id"] == thread["id"]
    reordered = await _request(
        reader,
        writer,
        {
            "id": "3a",
            "method": "thread.reorder",
            "params": {"project_id": project["id"], "thread_ids": [thread["id"]]},
        },
    )
    assert reordered["result"]["threads"][0]["id"] == thread["id"]
    read = await _request(
        reader,
        writer,
        {"id": "4", "method": "fs.read", "params": {"thread_id": thread["id"], "path": "note.txt"}},
    )
    assert read["result"]["content"] == "hello"
    artifacts = await _request(
        reader,
        writer,
        {
            "id": "4a",
            "method": "artifact.download",
            "params": {"thread_id": thread["id"], "path": "note.txt"},
        },
    )
    assert base64.b64decode(artifacts["result"]["data_base64"]) == b"hello"
    opened = await _request(
        reader,
        writer,
        {"id": "4b", "method": "terminal.open", "params": {"thread_id": thread["id"]}},
    )
    assert opened["result"]["opened"] is True
    closed = await _request(
        reader,
        writer,
        {"id": "4c", "method": "terminal.close", "params": {"thread_id": thread["id"]}},
    )
    assert closed["result"]["closed"] is True
    forbidden = await _request(
        reader,
        writer,
        {
            "id": "5",
            "method": "fs.read",
            "params": {"thread_id": thread["id"], "path": "../secret"},
        },
    )
    assert forbidden["error"]["data"]["code"] == "forbidden"
    uploaded = await _request(
        reader,
        writer,
        {
            "id": "6",
            "method": "attachment.upload",
            "params": {
                "name": "hello.txt",
                "mime_type": "text/plain",
                "data_base64": base64.b64encode(b"attachment").decode("ascii"),
            },
        },
    )
    attachment_id = uploaded["result"]["attachment_id"]
    downloaded = await _request(
        reader,
        writer,
        {"id": "7", "method": "attachment.download", "params": {"attachment_id": attachment_id}},
    )
    assert base64.b64decode(downloaded["result"]["data_base64"]) == b"attachment"
    await _request(reader, writer, {"id": "8", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_turn_created_precedes_operation_events(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-turn-{os.getpid()}.sock"
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)

    async def finish_without_provider(
        _session: object, _session_runtime: object, operation: object
    ) -> None:
        operation.status = "completed"  # type: ignore[attr-defined]

    runtime._execute_turn = finish_without_provider  # type: ignore[method-assign]
    server = RuntimeServer(runtime, socket_path, (tmp_path,), data_dir)
    task = asyncio.create_task(server.run())
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _handshake(reader, writer)
    added = await _request(
        reader,
        writer,
        {"id": "project", "method": "project.add", "params": _project_params(tmp_path)},
    )
    project_id = added["result"]["project"]["id"]
    created = await _request(
        reader,
        writer,
        {
            "id": "thread",
            "method": "thread.create",
            "params": {"project_id": project_id, "title": "Turn events", "kind": "standard"},
        },
    )
    thread_id = created["result"]["thread"]["id"]
    await _request(
        reader,
        writer,
        {
            "id": "subscribe",
            "method": "events.subscribe",
            "params": {"thread_ids": [thread_id], "after_seq": 0},
        },
    )
    response, events = await _request_with_events(
        reader,
        writer,
        {
            "id": "turn",
            "method": "turn.start",
            "params": {"thread_id": thread_id, "text": "queued turn", "attachment_ids": []},
        },
    )
    assert "result" in response
    names = [event.get("name") for event in events]
    assert "turn.created" in names
    assert "operation.queued" in names
    assert names.index("turn.created") < names.index("operation.queued")
    turn_event = next(event for event in events if event.get("name") == "turn.created")
    assert turn_event["payload"]["turn"]["core_turn_id"] == response["result"]["operation_id"]
    thread = await _request(
        reader,
        writer,
        {"id": "thread-after-turn", "method": "thread.get", "params": {"thread_id": thread_id}},
    )
    assert any(
        item.get("core_turn_id") == response["result"]["operation_id"]
        for item in thread["result"]["turns"]
    )
    assert any(item.get("name") == "turn.created" for item in thread["result"]["events"])
    await _request(reader, writer, {"id": "shutdown", "method": "system.shutdown", "params": {}})
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_v4_thread_create_rolls_back_workspace_only_project_on_session_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)

    async def fail_create_session(**_: object) -> object:
        raise RuntimeError("session storage unavailable")

    monkeypatch.setattr(runtime, "create_session", fail_create_session)
    try:
        with pytest.raises(RuntimeError, match="session storage unavailable"):
            await server.dispatch(
                "thread.create",
                {"workspace": str(tmp_path), "kind": "standard", "title": "orphan"},
            )
        assert await server.dispatch("project.list", {}) == {"projects": []}
        assert await server.dispatch("thread.list", {"filter": "all"}) == {"threads": []}
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_event_delivery_only_enqueues_and_does_not_wait_for_socket_drain() -> None:
    class Writer:
        def is_closing(self) -> bool:
            return False

    connection = RuntimeConnection.__new__(RuntimeConnection)
    connection.writer = Writer()
    connection.thread_ids = set()
    connection.subscribed = True
    connection._closed = False
    connection._write_sequence = 0
    connection._pending_event_frames = 0
    connection._pending_event_bytes = 0
    connection._event_queue_limit = runtime_server.EVENT_QUEUE_LIMIT
    connection._event_queue_bytes = runtime_server.EVENT_QUEUE_BYTES
    connection._writes = asyncio.PriorityQueue()

    await connection.send_event({"name": "operation.started", "thread_id": "thread-1"})
    priority, _sequence, value, future = await connection._writes.get()
    assert priority == 1
    assert value["method"] == "event"
    assert value["params"]["thread_id"] == "thread-1"
    future.set_result(None)
    await asyncio.sleep(0)
    assert connection._pending_event_frames == 0
    assert connection._pending_event_bytes == 0


@pytest.mark.asyncio
async def test_event_delivery_closes_a_slow_client_at_the_bound() -> None:
    class Writer:
        def __init__(self) -> None:
            self.closed = False

        def is_closing(self) -> bool:
            return self.closed

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

    connection = RuntimeConnection.__new__(RuntimeConnection)
    connection.writer = Writer()
    connection.thread_ids = set()
    connection.subscribed = True
    connection._closed = False
    connection._write_sequence = 0
    connection._pending_event_frames = 0
    connection._pending_event_bytes = 0
    connection._event_queue_limit = runtime_server.EVENT_QUEUE_LIMIT
    connection._event_queue_bytes = runtime_server.EVENT_QUEUE_BYTES
    connection._writes = asyncio.PriorityQueue()
    connection._request_tasks = set()
    connection._writer_task = None
    connection.handshaken = False
    connection.server = None

    for index in range(runtime_server.EVENT_QUEUE_LIMIT + 1):
        await connection.send_event({"name": "terminal.output", "thread_id": str(index)})

    assert connection._closed is True
    await asyncio.sleep(0)
    assert connection._pending_event_frames == 0


@pytest.mark.asyncio
async def test_v4_worktree_create_rolls_back_implicit_project_on_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)

    def fail_worktree(*_: object, **__: object) -> object:
        raise RuntimeError("git worktree unavailable")

    monkeypatch.setattr("aeloon_runtime.runtime_server.git_create_worktree", fail_worktree)
    try:
        with pytest.raises(RpcError, match="git worktree unavailable"):
            await server.dispatch(
                "thread.create",
                {"workspace": str(tmp_path), "kind": "worktree", "title": "orphan"},
            )
        assert await server.dispatch("project.list", {}) == {"projects": []}
        assert await server.dispatch("thread.list", {"filter": "all"}) == {"threads": []}
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_project_remove_deletes_runtime_sessions_and_projection(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        project = await server.dispatch("project.add", _project_params(tmp_path))
        project_id = project["project"]["id"]
        created = await server.dispatch(
            "thread.create", {"project_id": project_id, "kind": "standard"}
        )
        thread_id = created["thread"]["id"]
        assert await runtime.get_session(thread_id)

        assert await server.dispatch("project.remove", {"project_id": project_id}) == {
            "removed": True
        }
        assert await server.dispatch("project.list", {}) == {"projects": []}
        assert await server.dispatch("thread.list", {"filter": "all"}) == {"threads": []}
        with pytest.raises(SessionError, match="not found"):
            await runtime.get_session(thread_id)
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_gateway_maps_internal_session_events_to_thread_events(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        project = await server.dispatch("project.add", _project_params(tmp_path))
        created = await server.dispatch(
            "thread.create", {"project_id": project["project"]["id"], "kind": "standard"}
        )
        thread_id = created["thread"]["id"]
        await server._on_runtime_event(
            {
                "name": "session.renamed",
                "session_id": thread_id,
                "operation_id": None,
                "payload": {"title": "Renamed"},
            }
        )
        assert server.events[-1]["name"] == "thread.renamed"
        assert server.events[-1]["thread_id"] == thread_id
        assert (await server.dispatch("thread.get", {"thread_id": thread_id}))[
            "events"
        ][-1]["name"] == "thread.renamed"
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_worktree_lifecycle_is_runtime_owned(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    subprocess.run(["git", "-C", str(project_path), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(project_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_path), "config", "user.name", "Runtime Test"],
        check=True,
    )
    (project_path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(project_path), "commit", "-qm", "base"], check=True)

    data_dir = tmp_path / "runtime-data"
    runtime = create_runtime_service(
        config_path=data_dir / "config.json",
        data_dir=data_dir,
    )
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        project = await server.dispatch("project.add", _project_params(tmp_path, project_path))
        created = await server.dispatch(
            "thread.create",
            {
                "project_id": project["project"]["id"],
                "kind": "worktree",
                "title": "Worktree test",
            },
        )
        thread = created["thread"]
        workspace = Path(thread["workspace"])
        assert workspace.is_dir()
        assert (workspace / ".git").exists()
        (workspace / "artifact.txt").write_text("worktree artifact\n", encoding="utf-8")
        read = await server.dispatch(
            "fs.read", {"thread_id": thread["id"], "path": "artifact.txt"}
        )
        assert read["content"] == "worktree artifact\n"
        resolved = await server.dispatch(
            "artifact.resolve", {"thread_id": thread["id"], "paths": ["artifact.txt"]}
        )
        assert resolved["artifacts"][0]["exists"] is True
        (workspace / "artifact.txt").unlink()
        inspected = await server.dispatch("system.uninstall_inspect", {})
        assert inspected["worktrees"][0]["workspace"] == str(workspace)
        deleted = await server.dispatch("thread.delete", {"thread_id": thread["id"]})
        assert deleted["deleted"] is True
        assert not workspace.exists()
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_uninstall_prepare_validates_complete_response_before_first_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "runtime-data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    removed: list[Path] = []

    async def inspection() -> dict[str, Any]:
        return {
            "active_operations": 0,
            "worktrees": [{
                "project_path": str(tmp_path / "project"),
                "workspace": str(tmp_path / "workspace"),
                "exists": True,
                "dirty": False,
            }],
        }

    def reject_result(_method: str, _value: Any) -> None:
        raise RpcError("internal_error", "invalid uninstall response")

    monkeypatch.setattr(server, "_uninstall_inspect", inspection)
    monkeypatch.setattr(server, "validate_result", reject_result)
    monkeypatch.setattr(
        runtime_server,
        "git_remove_worktree",
        lambda _project, workspace: removed.append(workspace),
    )
    try:
        with pytest.raises(RpcError, match="invalid uninstall response"):
            await server.dispatch("system.uninstall_prepare", {})
        assert removed == []
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_uninstall_prepare_reports_partial_progress_and_retries_missing_worktrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "runtime-data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    project = tmp_path / "project"
    first = tmp_path / "first"
    second = tmp_path / "second"
    states = {first: True, second: True}
    fail_second_once = True
    calls: list[Path] = []

    async def inspection() -> dict[str, Any]:
        return {
            "active_operations": 0,
            "worktrees": [
                {
                    "project_path": str(project),
                    "workspace": str(workspace),
                    "exists": exists,
                    "dirty": False,
                }
                for workspace, exists in states.items()
            ],
        }

    def remove(_project: Path, workspace: Path) -> None:
        nonlocal fail_second_once
        calls.append(workspace)
        if workspace == second and fail_second_once:
            fail_second_once = False
            raise RuntimeError("second worktree is busy")
        states[workspace] = False

    monkeypatch.setattr(server, "_uninstall_inspect", inspection)
    monkeypatch.setattr(runtime_server, "git_remove_worktree", remove)
    try:
        with pytest.raises(RpcError, match="second worktree is busy") as raised:
            await server.dispatch("system.uninstall_prepare", {})
        assert raised.value.data == {
            "removed_worktrees": [{"project_path": str(project), "workspace": str(first)}]
        }
        assert calls == [first, second]

        retried = await server.dispatch("system.uninstall_prepare", {})
        assert retried == {
            "prepared": True,
            "removed_worktrees": [
                {"project_path": str(project), "workspace": str(first)},
                {"project_path": str(project), "workspace": str(second)},
            ],
        }
        # The missing first worktree is idempotent; only the failed second
        # worktree needs another destructive call.
        assert calls == [first, second, second]
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_empty_workspace_roots_do_not_authorize_cwd(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    secret = tmp_path / "secret"
    secret.mkdir()
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (), data_dir)
    try:
        assert server.workspace_roots == ()
        snapshot = await server.dispatch("system.snapshot", {})
        assert snapshot["default_workspace"] == ""
        with pytest.raises(RpcError, match="Unknown workspace root"):
            await server.dispatch("project.add", {"root_id": _root_id(secret), "relative_path": "."})
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v4_project_refresh_persists_git_detection(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        added = await server.dispatch("project.add", _project_params(tmp_path, project_path))
        assert added["project"]["is_git"] is False
        subprocess.run(["git", "init", "-q", str(project_path)], check=True)
        refreshed = await server.dispatch(
            "project.refresh", {"project_id": added["project"]["id"]}
        )
        assert refreshed["project"]["is_git"] is True
        assert refreshed["projects"][0]["is_git"] is True
        snapshot = await server.dispatch("system.snapshot", {})
        assert snapshot["projects"][0]["is_git"] is True
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_diagnostics_logs_returns_newest_filtered_records(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        from aeloon_runtime.runtime_log import RuntimeLog

        server.log = RuntimeLog(data_dir)
        server.log.write(
            "connected", transport="unix", device_id="dev-1", source="unix", token="secret"
        )
        server.log.write(
            "resync_served",
            after_seq=1,
            current_seq=4,
            replay_complete=True,
            replayed_events=3,
        )
        (data_dir / "runtime.log").write_bytes(
            (data_dir / "runtime.log").read_bytes() + b"{not json\n"
        )
        logs = await server.dispatch("diagnostics.logs", {"limit": 10})
        assert logs["truncated"] is False
        assert [item["event"] for item in logs["entries"]] == ["resync_served", "connected"]
        assert "token" not in logs["entries"][1]["fields"]
        assert logs["entries"][1]["fields"]["transport"] == "unix"
        with pytest.raises(RpcError, match="limit"):
            await server.dispatch("diagnostics.logs", {"limit": 0})
        with pytest.raises(RpcError, match="Invalid parameters"):
            await server.dispatch("diagnostics.logs", {"path": "/etc/passwd"})
    finally:
        if server.log is not None:
            server.log.close()
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_event_delivery_closes_when_pending_bytes_exceed_the_bound() -> None:
    class Writer:
        def __init__(self) -> None:
            self.closed = False

        def is_closing(self) -> bool:
            return self.closed

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

    connection = RuntimeConnection.__new__(RuntimeConnection)
    connection.writer = Writer()
    connection.thread_ids = set()
    connection.subscribed = True
    connection._closed = False
    connection._write_sequence = 0
    connection._pending_event_frames = 0
    connection._pending_event_bytes = 0
    connection._event_queue_limit = runtime_server.EVENT_QUEUE_LIMIT
    connection._event_queue_bytes = 64
    connection._writes = asyncio.PriorityQueue()
    connection._request_tasks = set()
    connection._writer_task = None
    connection.handshaken = False
    connection.server = None
    await connection.send_event(
        {"name": "terminal.output", "thread_id": "thread-1", "payload": {"data": "x" * 200}}
    )
    assert connection._closed is True


@pytest.mark.asyncio
async def test_overflow_close_cancels_a_stuck_writer() -> None:
    class StuckWriter:
        def __init__(self) -> None:
            self.closed = False
            self.aborted = False

        def is_closing(self) -> bool:
            return self.closed

        def write(self, _data: bytes) -> None:
            return None

        async def drain(self) -> None:
            await asyncio.sleep(60)

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

        @property
        def transport(self) -> object:
            writer = self

            class Transport:
                def abort(self) -> None:
                    writer.aborted = True

            return Transport()

    connection = RuntimeConnection.__new__(RuntimeConnection)
    connection.writer = StuckWriter()
    connection.thread_ids = set()
    connection.subscribed = True
    connection._closed = False
    connection._close_reason = "peer"
    connection._write_sequence = 0
    connection._pending_event_frames = 0
    connection._pending_event_bytes = 0
    connection._event_queue_limit = 2
    connection._event_queue_bytes = 1024 * 1024
    connection._writes = asyncio.PriorityQueue()
    connection._request_tasks = set()
    connection.write_lock = asyncio.Lock()
    connection.handshaken = True
    connection.connected_at = 0.0
    connection.server = None
    connection.device_id = None
    connection.requires_auth = False
    connection._disconnected_logged = False
    connection._writer_task = asyncio.create_task(connection._writer_loop())
    await connection.send_event({"name": "log.entry", "payload": {"category": "r3", "action": "a"}})
    await connection.send_event({"name": "log.entry", "payload": {"category": "r3", "action": "b"}})
    await asyncio.wait_for(
        connection.send_event({"name": "log.entry", "payload": {"category": "r3", "action": "c"}}),
        timeout=2,
    )
    assert connection._closed is True
    assert connection.writer.aborted is True


@pytest.mark.asyncio
async def test_unix_seventeenth_connection_receives_busy(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v4-limit-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    task = asyncio.create_task(
        serve(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
    )
    try:
        for _ in range(200):
            if socket_path.exists():
                break
            await asyncio.sleep(0.01)
        handshake = {
            "protocol": {"min": "4.0.0", "max": "4.0.0"},
            "client": {"name": "pytest", "version": "0", "platform": "test"},
        }
        held: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        for index in range(16):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            response = await _request(
                reader,
                writer,
                {"id": f"hs-{index}", "method": "system.handshake", "params": handshake},
            )
            assert "result" in response
            held.append((reader, writer))
        extra_reader, extra_writer = await asyncio.open_unix_connection(str(socket_path))
        busy = await _request(
            extra_reader,
            extra_writer,
            {"id": "busy", "method": "system.handshake", "params": handshake},
        )
        assert busy["error"]["data"]["code"] == "busy"
        extra_writer.close()
        await extra_writer.wait_closed()
        for _reader, writer in held:
            writer.close()
            await writer.wait_closed()
        logs = await _connect_and_read_logs(socket_path)
        events = {item["event"] for item in logs["entries"]}
        assert "connected" in events
        assert "disconnected" in events
        assert "resync_served" in events
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        await _request(
            reader,
            writer,
            {
                "id": "down-hs",
                "method": "system.handshake",
                "params": handshake,
            },
        )
        await _request(reader, writer, {"id": "down", "method": "system.shutdown", "params": {}})
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        socket_path.unlink(missing_ok=True)


async def _connect_and_read_logs(socket_path: Path) -> dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    await _request(
        reader,
        writer,
        {
            "id": "log-hs",
            "method": "system.handshake",
            "params": {
                "protocol": {"min": "4.0.0", "max": "4.0.0"},
                "client": {"name": "pytest", "version": "0", "platform": "test"},
            },
        },
    )
    await _request(
        reader,
        writer,
        {"id": "sub", "method": "events.subscribe", "params": {"thread_ids": []}},
    )
    response = await _request(
        reader, writer, {"id": "logs", "method": "diagnostics.logs", "params": {"limit": 200}}
    )
    writer.close()
    await writer.wait_closed()
    return response["result"]

