from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.bridge.daemon import (
    BridgeDaemon,
    bridge_request,
    daemon_status,
    stop_daemon,
)
from aeloon_core.bridge.protocol import BridgeError
from aeloon_core.config import Config, save_config
from aeloon_core.harness import (
    AgentHarness,
    AssistantMessage,
    AssistantStreamEvent,
    ScriptedProvider,
    TextContent,
    Usage,
    get_deepseek_model,
)
from aeloon_core.service import CoreService


def scripted_service(tmp_path: Path, text: str = "bridge answer") -> CoreService:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)

    def factory(config: Config, session: Any) -> AgentHarness:
        message = AssistantMessage(
            (TextContent(text),),
            "deepseek",
            config.agent.model,
            usage=Usage(input=2, output=3, total_tokens=5),
        )
        return AgentHarness(
            provider=ScriptedProvider([message]),
            model=get_deepseek_model(config.agent.model),
            cwd=str(config.workspace),
            session=session,
        )

    return CoreService(config_path=config_path, harness_factory=factory)


@pytest.mark.asyncio
async def test_stable_turn_projection_and_replay(tmp_path: Path) -> None:
    service = scripted_service(tmp_path)
    metadata = await service.session_create({"workspace": str(tmp_path), "title": "Bridge"})
    started = await service.turn_start(
        {
            "session_id": metadata["session_id"],
            "input": {"kind": "prompt", "text": "hello"},
        },
        attachment_roots=(),
    )
    operation = service._operation({"operation_id": started["operation_id"]})
    assert operation.task is not None
    await operation.task

    snapshot = await service.session_get({"session_id": metadata["session_id"]})
    turn = snapshot["timeline"][0]
    assert turn["turn_id"] == started["operation_id"]
    assert turn["status"] == "completed"
    assert turn["final_content"] == "bridge answer"
    assert turn["usage"]["totalTokens"] == 5
    assert snapshot["active_operations"] == []

    replay = await service.events_subscribe(
        {"session_ids": [metadata["session_id"]], "after_seq": 0}
    )
    assert replay["replay_complete"] is True
    assert replay["events"][0]["name"] == "operation.queued"
    assert replay["events"][-2]["name"] == "operation.completed"
    assert all("traceback" not in json.dumps(event).lower() for event in replay["events"])


@pytest.mark.asyncio
async def test_session_serialization_and_cross_session_concurrency(tmp_path: Path) -> None:
    tracker = TrackingProvider()
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)

    def factory(config: Config, session: Any) -> AgentHarness:
        return AgentHarness(
            provider=tracker,
            model=get_deepseek_model(config.agent.model),
            cwd=str(config.workspace),
            session=session,
        )

    service = CoreService(config_path=config_path, harness_factory=factory)
    first = await service.session_create({"workspace": str(tmp_path)})
    second = await service.session_create({"workspace": str(tmp_path)})
    operations = [
        await service.turn_start(
            {"session_id": first["session_id"], "input": {"kind": "prompt", "text": text}},
            attachment_roots=(),
        )
        for text in ("one", "two")
    ]
    operations.append(
        await service.turn_start(
            {"session_id": second["session_id"], "input": {"kind": "prompt", "text": "three"}},
            attachment_roots=(),
        )
    )
    await asyncio.gather(
        *(service._operation({"operation_id": item["operation_id"]}).task for item in operations)
    )
    assert tracker.max_by_session[first["session_id"]] == 1
    assert tracker.max_global >= 2


@pytest.mark.asyncio
async def test_attachment_roots_copy_limits_and_cleanup(tmp_path: Path) -> None:
    service = scripted_service(tmp_path)
    metadata = await service.session_create({"workspace": str(tmp_path)})
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "notes.txt"
    source.write_text("stable source", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(BridgeError, match="declared roots"):
        await service.turn_start(
            {
                "session_id": metadata["session_id"],
                "input": {
                    "kind": "prompt",
                    "text": "bad",
                    "attachments": [{"type": "file", "name": "outside.txt", "path": str(outside)}],
                },
            },
            attachment_roots=(allowed,),
        )

    started = await service.turn_start(
        {
            "session_id": metadata["session_id"],
            "input": {
                "kind": "prompt",
                "text": "good",
                "attachments": [{"type": "file", "name": "notes.txt", "path": str(source)}],
            },
        },
        attachment_roots=(allowed,),
    )
    operation = service._operation({"operation_id": started["operation_id"]})
    managed = Path(operation.input["attachments"][0]["managed_path"])
    assert managed.read_text(encoding="utf-8") == "stable source"
    source.write_text("changed", encoding="utf-8")
    assert managed.read_text(encoding="utf-8") == "stable source"
    assert operation.task is not None
    await operation.task
    await service.session_delete({"session_id": metadata["session_id"]})
    assert not managed.exists()


@pytest.mark.asyncio
async def test_revisioned_settings_never_return_secret(tmp_path: Path) -> None:
    service = scripted_service(tmp_path)
    initial = await service.settings_get({})
    updated = await service.settings_update(
        {
            "revision": initial["revision"],
            "patch": {"default_model_id": "deepseek-v4-pro"},
            "secret_actions": [
                {"path": "deepseek.api_key", "action": "set", "value": "very-secret"}
            ],
        }
    )
    assert updated["default_model_id"] == "deepseek-v4-pro"
    assert updated["deepseek"]["credential_configured"] is True
    assert "very-secret" not in json.dumps(updated)
    assert (tmp_path / "config.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(BridgeError, match="refresh"):
        await service.settings_update({"revision": initial["revision"], "patch": {}})


@pytest.mark.asyncio
async def test_unfinished_persisted_run_is_interrupted_after_service_restart(
    tmp_path: Path,
) -> None:
    service = scripted_service(tmp_path)
    metadata = await service.session_create({"workspace": str(tmp_path)})
    session = await service.repository.open(metadata["session_id"])
    await session.append_run_start(
        run_id="crashed-turn",
        input={"kind": "prompt", "text": "unfinished", "attachments": []},
        model_id="deepseek-v4-flash",
        thinking_level="off",
    )

    restarted = scripted_service(tmp_path)
    snapshot = await restarted.session_get({"session_id": metadata["session_id"]})
    assert snapshot["timeline"][0]["turn_id"] == "crashed-turn"
    assert snapshot["timeline"][0]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_daemon_socket_permissions_and_multi_client_handshake(tmp_path: Path) -> None:
    service = scripted_service(tmp_path)
    runtime = Path(tempfile.mkdtemp(prefix="aeloon-bridge-", dir="/tmp"))
    socket_path = runtime / "bridge.sock"
    daemon = BridgeDaemon(service, socket_path)
    task = asyncio.create_task(daemon.run())
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    assert socket_path.stat().st_mode & 0o777 == 0o600
    assert socket_path.parent.stat().st_mode & 0o777 == 0o700
    first, second = await asyncio.gather(
        bridge_request(socket_path, "system.handshake"),
        bridge_request(socket_path, "system.handshake"),
    )
    assert first["server_instance_id"] == second["server_instance_id"]
    assert first["protocol_version"] == 2
    reader, writer = await asyncio.open_unix_connection(socket_path)
    writer.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "subscribe",
                "method": "events.subscribe",
                "params": {"session_ids": [], "after_seq": 0},
            }
        ).encode()
        + b"\n"
    )
    await writer.drain()
    subscribed = json.loads(await asyncio.wait_for(reader.readline(), 2))
    assert subscribed["result"]["replay_complete"] is True

    stopping = asyncio.create_task(stop_daemon(socket_path))
    notification = json.loads(await asyncio.wait_for(reader.readline(), 2))
    assert notification["method"] == "event"
    assert notification["params"]["name"] == "system.shutdown"
    assert notification["params"]["payload"]["intentional"] is True
    assert await asyncio.wait_for(stopping, 6) == {
        "status": "stopped",
        "socket_path": str(socket_path.resolve()),
    }
    await asyncio.wait_for(task, 2)

    missing = runtime / "missing.sock"
    assert await daemon_status(missing) == {
        "status": "stopped",
        "socket_path": str(missing.resolve()),
    }
    assert await stop_daemon(missing) == {
        "status": "stopped",
        "socket_path": str(missing.resolve()),
    }

    stale = runtime / "stale.sock"
    stale_socket = socket.socket(socket.AF_UNIX)
    stale_socket.bind(str(stale))
    stale_socket.close()
    assert (await daemon_status(stale))["status"] == "stopped"
    assert (await stop_daemon(stale))["status"] == "stopped"
    assert not stale.exists()
    writer.close()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(writer.wait_closed(), 1)
    runtime.rmdir()


class TrackingProvider:
    def __init__(self) -> None:
        self.active_global = 0
        self.max_global = 0
        self.active_by_session: defaultdict[str, int] = defaultdict(int)
        self.max_by_session: defaultdict[str, int] = defaultdict(int)

    async def stream(self, model: Any, context: Any, _options: Any):
        session_id = context.session_id
        self.active_global += 1
        self.active_by_session[session_id] += 1
        self.max_global = max(self.max_global, self.active_global)
        self.max_by_session[session_id] = max(
            self.max_by_session[session_id], self.active_by_session[session_id]
        )
        try:
            yield AssistantStreamEvent("start")
            await asyncio.sleep(0.04)
            message = AssistantMessage(
                (TextContent("done"),), "deepseek", model.id, usage=Usage(total_tokens=1)
            )
            yield AssistantStreamEvent("text_delta", delta="done", content_index=0)
            yield AssistantStreamEvent("done", message=message)
        finally:
            self.active_global -= 1
            self.active_by_session[session_id] -= 1
