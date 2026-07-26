from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.web.bridge import WebBridge, _history_turn_view
from aeloon_core.web.launcher import (
    WEB_HOST_ENV,
    WEB_PORT_ENV,
    WEB_TOKEN_ENV,
    build_web_environment,
)


class _Sessions:
    def __init__(self) -> None:
        self.current = "session-1"

    def new_session(self) -> str:
        return self.current

    def history(self, session_id: str) -> list[dict[str, Any]]:
        assert session_id == self.current
        return [
            {
                "turn_id": "turn-1",
                "request_id": "request-1",
                "created_at": "2026-07-25T10:00:00+00:00",
                "user_prompt": "Inspect the repository",
                "final_content": "Done",
                "tools_used": ["read"],
                "blocks": [{"id": "answer-1", "type": "text", "content": "Done"}],
                "usage": {"total_tokens": 12},
            }
        ]

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": self.current,
                "title": "Inspect the repository",
                "updated_at": "2026-07-25T10:00:00+00:00",
                "turns": 1,
            }
        ]


class _Orchestrator:
    def __init__(self) -> None:
        self.sessions = _Sessions()


@pytest.mark.asyncio
async def test_web_bridge_ready_snapshot_contains_conversation_only(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = []

    async def sink(record: dict[str, Any]) -> None:
        records.append(record)

    bridge = WebBridge(
        Config(workspace=tmp_path, data_dir=tmp_path / "data"),
        orchestrator=_Orchestrator(),
        sink=sink,
    )
    await bridge.emit_ready()

    ready = records[0]
    assert ready["type"] == "ready"
    assert ready["payload"]["session_id"] == "session-1"
    assert ready["payload"]["history"][0]["turn_id"] == "turn-1"
    assert ready["payload"]["history"][0]["request_id"] == "request-1"
    assert ready["payload"]["history"][0]["blocks"][0]["content"] == "Done"
    assert "workers" not in ready["payload"]
    assert "flows" not in ready["payload"]
    await bridge.close()


@pytest.mark.asyncio
async def test_web_bridge_rejects_removed_durable_commands(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = []

    async def sink(record: dict[str, Any]) -> None:
        records.append(record)

    bridge = WebBridge(
        Config(workspace=tmp_path, data_dir=tmp_path / "data"),
        orchestrator=_Orchestrator(),
        sink=sink,
    )
    await bridge.dispatch(
        {
            "type": "command",
            "command": "inspect_flow",
            "request_id": "request-1",
            "payload": {"flow_id": "flow-1"},
        }
    )

    assert records[0]["type"] == "response"
    assert records[0]["command"] == "inspect_flow"
    assert records[0]["ok"] is False
    assert records[0]["error"]["code"] == "unknown_command"
    await bridge.close()


def test_web_environment_keeps_token_and_config_out_of_argv(tmp_path: Path) -> None:
    config = Config(workspace=tmp_path, data_dir=tmp_path / "data")
    environment = build_web_environment(
        config,
        host="127.0.0.1",
        port=0,
        token="one-time-token",
        environ={"PATH": "/usr/bin", "PYTHONPATH": "/existing"},
    )

    assert environment[WEB_HOST_ENV] == "127.0.0.1"
    assert environment[WEB_PORT_ENV] == "0"
    assert environment[WEB_TOKEN_ENV] == "one-time-token"
    assert json.loads(environment["AELOON_CORE_WEB_CONFIG_JSON"])["workspace"] == str(
        tmp_path
    )
    assert environment["PYTHONPATH"].split(":", 1)[0] == str(
        Path(__file__).resolve().parents[1]
    )
    assert environment["PYTHONPATH"].endswith(":/existing")


def test_history_snapshot_bounds_tool_results_like_the_live_protocol() -> None:
    view = _history_turn_view(
        {
            "blocks": [
                {
                    "id": "tool-1",
                    "type": "tool_call",
                    "result": "x" * 40_000,
                }
            ]
        }
    )

    assert len(view["blocks"][0]["result"]) <= 16_000
    assert view["blocks"][0]["result_truncated"] is True
