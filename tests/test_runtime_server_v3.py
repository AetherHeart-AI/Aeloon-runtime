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

import aeloon_runtime.runtime_server_v3 as runtime_server_v3
from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.rpc.protocol import RpcError
from aeloon_runtime.runtime.session import SessionError
from aeloon_runtime.runtime_server_v3 import (
    MAX_FRAME_BYTES,
    RuntimeV3Connection,
    RuntimeV3Server,
    serve_v3,
)


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
                "protocol": {"min": "3.0.0", "max": "3.0.0"},
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
async def test_v3_handshake_health_and_shutdown(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-{os.getpid()}.sock"
    task = asyncio.create_task(
        serve_v3(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
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
                # A client that only speaks the next major cannot be served.
                "protocol": {"min": "4.0.0", "max": "4.0.0"},
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
                "protocol": {"min": "3.0.0", "max": "3.0.0"},
                "client": {"name": "pytest", "version": "0", "platform": "test"},
            },
        },
    )
    assert handshake["result"]["limits"]["request_bytes"] == MAX_FRAME_BYTES
    assert handshake["result"]["workspace_roots"] == [str(tmp_path)]
    health = await _request(reader, writer, {"id": "2", "method": "system.health", "params": {}})
    assert health["result"]["ok"] is True
    capabilities = await _request(
        reader, writer, {"id": "2a", "method": "system.capabilities", "params": {}}
    )
    core_capability = next(
        item for item in capabilities["result"]["capabilities"] if item["id"] == "core"
    )
    assert "system.snapshot" in core_capability["methods"]
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
async def test_v3_subscribe_returns_replay_in_result_without_duplicate_event_frames(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    socket_path = Path("/tmp") / f"aeloon-v3-replay-{os.getpid()}.sock"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, socket_path, (tmp_path,), data_dir)
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
async def test_v3_control_requests_are_not_blocked_by_a_long_request(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-concurrent-{os.getpid()}.sock"
    runtime = create_runtime_service(
        config_path=tmp_path / "data" / "config.json", data_dir=tmp_path / "data"
    )
    server = RuntimeV3Server(runtime, socket_path, (tmp_path,), tmp_path / "data")
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

    async def delayed_dispatch(method: str, params: Mapping[str, Any]) -> Any:
        if method == "catalog.get":
            entered.set()
            await release.wait()
        return await original_dispatch(method, params)

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
async def test_v3_trace_recording_is_explicit_and_captures_boundary(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-trace-{os.getpid()}.sock"
    trace_directory = tmp_path / "traces"
    task = asyncio.create_task(
        serve_v3(
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
async def test_v3_does_not_unlink_a_non_socket_path(tmp_path: Path) -> None:
    socket_path = tmp_path / "runtime.sock"
    socket_path.write_text("keep me", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a Unix socket"):
        await serve_v3(
            socket_path=socket_path,
            data_dir=tmp_path / "data",
            workspace_roots=(tmp_path,),
        )
    assert socket_path.read_text(encoding="utf-8") == "keep me"


@pytest.mark.asyncio
async def test_v3_data_lock_rejects_concurrent_runtime(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-lock-{os.getpid()}.sock"
    data_dir = tmp_path / "data"
    task = asyncio.create_task(
        serve_v3(socket_path=socket_path, data_dir=data_dir, workspace_roots=(tmp_path,))
    )
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    second_runtime = create_runtime_service(
        config_path=data_dir / "config.json",
        data_dir=data_dir,
    )
    second = RuntimeV3Server(
        second_runtime,
        Path("/tmp") / f"aeloon-v3-lock-second-{os.getpid()}.sock",
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
async def test_v3_composition_locks_data_before_runtime_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-compose-lock-{os.getpid()}.sock"
    data_dir = tmp_path / "data"
    observed = False
    original = runtime_server_v3.create_runtime_service

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

    monkeypatch.setattr(runtime_server_v3, "create_runtime_service", factory)
    task = asyncio.create_task(
        serve_v3(socket_path=socket_path, data_dir=data_dir, workspace_roots=(tmp_path,))
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
async def test_v3_gateway_rejects_unknown_parameters(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-invalid-{os.getpid()}.sock"
    task = asyncio.create_task(
        serve_v3(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
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
async def test_v3_frame_limit_returns_error_before_closing_connection(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-frame-limit-{os.getpid()}.sock"
    task = asyncio.create_task(
        serve_v3(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
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
async def test_v3_projects_threads_and_workspace_boundary(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-projects-{os.getpid()}.sock"
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    task = asyncio.create_task(
        serve_v3(socket_path=socket_path, data_dir=tmp_path / "data", workspace_roots=(tmp_path,))
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
        {"id": "1", "method": "project.add", "params": {"path": str(tmp_path)}},
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
async def test_v3_turn_created_precedes_operation_events(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-v3-turn-{os.getpid()}.sock"
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)

    async def finish_without_provider(
        _session: object, _session_runtime: object, operation: object
    ) -> None:
        operation.status = "completed"  # type: ignore[attr-defined]

    runtime._execute_turn = finish_without_provider  # type: ignore[method-assign]
    server = RuntimeV3Server(runtime, socket_path, (tmp_path,), data_dir)
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
        {"id": "project", "method": "project.add", "params": {"path": str(tmp_path)}},
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
async def test_v3_thread_create_rolls_back_workspace_only_project_on_session_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)

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

    connection = RuntimeV3Connection.__new__(RuntimeV3Connection)
    connection.writer = Writer()
    connection.thread_ids = set()
    connection.subscribed = True
    connection._closed = False
    connection._write_sequence = 0
    connection._pending_event_frames = 0
    connection._writes = asyncio.PriorityQueue()

    await connection.send_event({"name": "operation.started", "thread_id": "thread-1"})
    priority, _sequence, value, future = await connection._writes.get()
    assert priority == 1
    assert value["method"] == "event"
    assert value["params"]["thread_id"] == "thread-1"
    future.set_result(None)
    await asyncio.sleep(0)
    assert connection._pending_event_frames == 0


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

    connection = RuntimeV3Connection.__new__(RuntimeV3Connection)
    connection.writer = Writer()
    connection.thread_ids = set()
    connection.subscribed = True
    connection._closed = False
    connection._write_sequence = 0
    connection._pending_event_frames = 0
    connection._writes = asyncio.PriorityQueue()
    connection._request_tasks = set()
    connection._writer_task = None

    for index in range(runtime_server_v3.EVENT_QUEUE_LIMIT + 1):
        await connection.send_event({"name": "terminal.output", "thread_id": str(index)})

    assert connection._closed is True
    await asyncio.sleep(0)
    assert connection._pending_event_frames == 0


@pytest.mark.asyncio
async def test_v3_worktree_create_rolls_back_implicit_project_on_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)

    def fail_worktree(*_: object, **__: object) -> object:
        raise RuntimeError("git worktree unavailable")

    monkeypatch.setattr("aeloon_runtime.runtime_server_v3.git_create_worktree", fail_worktree)
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
async def test_v3_project_remove_deletes_runtime_sessions_and_projection(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        project = await server.dispatch("project.add", {"path": str(tmp_path)})
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
async def test_v3_gateway_maps_internal_session_events_to_thread_events(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        project = await server.dispatch("project.add", {"path": str(tmp_path)})
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
async def test_v3_worktree_lifecycle_is_runtime_owned(tmp_path: Path) -> None:
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
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        project = await server.dispatch("project.add", {"path": str(project_path)})
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
async def test_v3_empty_workspace_roots_do_not_authorize_cwd(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    secret = tmp_path / "secret"
    secret.mkdir()
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (), data_dir)
    try:
        assert server.workspace_roots == ()
        snapshot = await server.dispatch("system.snapshot", {})
        assert snapshot["default_workspace"] == ""
        with pytest.raises(RpcError, match="outside"):
            await server.dispatch("project.add", {"path": str(secret)})
    finally:
        await runtime.close()
        server.store.close()


@pytest.mark.asyncio
async def test_v3_project_refresh_persists_git_detection(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        added = await server.dispatch("project.add", {"path": str(project_path)})
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
