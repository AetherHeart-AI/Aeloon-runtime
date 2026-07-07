from __future__ import annotations

import json

import pytest
from rich.console import Console

from aeloon_core.__main__ import (
    _load_with_path_overrides,
    _looks_like_implicit_chat,
    _looks_like_legacy_run,
    build_parser,
)
from aeloon_core.terminal_cli import TerminalEventRenderer


def test_chat_commands_are_registered() -> None:
    args = build_parser().parse_args(["chat", "--hide-gateway-logs", "hello"])

    assert args.command == "chat"
    assert args.prompt == ["hello"]
    assert args.hide_gateway_logs is True
    assert _looks_like_legacy_run(["chat"]) is False
    assert _looks_like_legacy_run(["tui"]) is False


def test_implicit_chat_invocations() -> None:
    assert _looks_like_implicit_chat(["--data-dir", "/tmp/aeloon"]) is True
    assert _looks_like_implicit_chat(["--gateway-log-level", "DEBUG"]) is True
    assert _looks_like_implicit_chat(["--gateway-log-level", "DEBUG", "hello"]) is True
    assert _looks_like_implicit_chat(["--help"]) is False
    assert _looks_like_implicit_chat(["webui"]) is False
    assert _looks_like_implicit_chat(["hello"]) is False


def test_runtime_workspace_defaults_to_invocation_cwd(tmp_path, monkeypatch) -> None:
    configured_workspace = tmp_path / "configured"
    run_dir = tmp_path / "run"
    configured_workspace.mkdir()
    run_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"workspace": str(configured_workspace)}),
        encoding="utf-8",
    )

    monkeypatch.chdir(run_dir)

    config = _load_with_path_overrides(config_path, workspace=None, data_dir=None)

    assert config.workspace == run_dir.resolve()


def test_runtime_workspace_flag_overrides_invocation_cwd(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    explicit_workspace = tmp_path / "explicit"
    run_dir.mkdir()
    explicit_workspace.mkdir()
    monkeypatch.chdir(run_dir)

    config = _load_with_path_overrides(None, workspace=explicit_workspace, data_dir=None)

    assert config.workspace == explicit_workspace.resolve()


@pytest.mark.asyncio
async def test_terminal_renderer_keeps_core_info_and_gateway_logs() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console, show_gateway_logs=True)

    await renderer.emit("chat.turn.start", {"turn_id": "turn-1"})
    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "text-1", "type": "text", "role": "assistant"}},
    )
    await renderer.emit("chat.block.delta", {"block_id": "text-1", "delta": "hello"})
    await renderer.emit(
        "chat.block.add",
        {
            "block": {
                "id": "tool-1",
                "type": "tool_call",
                "name": "read",
                "arguments": {"path": "README.md"},
            }
        },
    )
    await renderer.emit(
        "chat.block.update",
        {
            "block_id": "tool-1",
            "patch": {"status": "done", "result": "file contents", "duration_ms": 3},
        },
    )
    await renderer.emit(
        "log.entry",
        {
            "level": "INFO",
            "source": "tool.result",
            "message": "tool read -> done",
            "ts": "2026-07-07T10:00:00+00:00",
            "detail": {},
        },
    )

    output = console.export_text()
    assert "turn turn-1" in output
    assert "Assistant" in output
    assert "hello" in output
    assert "Tool read" in output
    assert "read -> done" in output
    assert "gateway INFO" in output
    assert "tool.result" in output


@pytest.mark.asyncio
async def test_terminal_renderer_filters_gateway_logs_by_level() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console, show_gateway_logs=True, gateway_log_level="WARNING")

    await renderer.emit(
        "log.entry",
        {
            "level": "INFO",
            "source": "kernel.status",
            "message": "Thinking",
            "ts": "2026-07-07T10:00:00+00:00",
        },
    )
    await renderer.emit(
        "log.entry",
        {
            "level": "ERROR",
            "source": "tool.result",
            "message": "failed",
            "ts": "2026-07-07T10:00:01+00:00",
        },
    )

    output = console.export_text()
    assert "Thinking" not in output
    assert "failed" in output
    assert len(renderer.gateway_logs) == 2
