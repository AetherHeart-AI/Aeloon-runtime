from __future__ import annotations

import json

import pytest
from rich.console import Console

from aeloon_core.__main__ import (
    CONFIG_SETTERS,
    _coerce_config_value,
    _load_with_path_overrides,
    _looks_like_implicit_chat,
    _looks_like_legacy_run,
    build_parser,
)
from aeloon_core.context import SYSTEM_PROMPT
from aeloon_core.orchestrator import TurnResult
from aeloon_core.providers.base import LLMResponse
from aeloon_core.terminal_cli import TerminalEventRenderer
from aeloon_core.turn_events import TurnEventProgress


def test_chat_commands_are_registered() -> None:
    args = build_parser().parse_args(["chat", "--hide-gateway-logs", "hello"])

    assert args.command == "chat"
    assert args.prompt == ["hello"]
    assert args.hide_gateway_logs is True
    assert _looks_like_legacy_run(["chat"]) is False
    assert _looks_like_legacy_run(["tui"]) is False
    assert _looks_like_legacy_run(["webui"]) is False


def test_context_compaction_config_setters_are_registered() -> None:
    args = build_parser().parse_args(
        ["config", "set", "context-compaction-trigger-ratio", "0.85"]
    )

    assert args.config_command == "set"
    assert args.key == "context-compaction-trigger-ratio"
    assert CONFIG_SETTERS["context-compaction-enabled"] == (
        "agents",
        "defaults",
        "context_compaction",
        "enabled",
    )
    assert _coerce_config_value("context-compaction-enabled", "false") is False
    assert _coerce_config_value("context-compaction-trigger-ratio", "0.85") == 0.85
    assert _coerce_config_value("context-compaction-preserve-recent-tokens", "none") is None
    assert "max-tokens" not in CONFIG_SETTERS


def test_uasm_config_setters_are_registered() -> None:
    args = build_parser().parse_args(["config", "set", "uasm-enabled", "true"])

    assert args.config_command == "set"
    assert args.key == "uasm-enabled"
    assert CONFIG_SETTERS["uasm-temporary-guard-enabled"] == (
        "agents",
        "defaults",
        "uasm",
        "temporary_guard_enabled",
    )
    assert _coerce_config_value("uasm-enabled", "true") is True
    assert _coerce_config_value("uasm-minimal-context-recent-turns", "3") == 3
    assert _coerce_config_value("uasm-guard-decision-mode", "binary") == "binary"


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
            "patch": {
                "status": "done",
                "result": "1| secret file contents\n\n(End of file - 1 lines total)",
                "duration_ms": 3,
            },
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
    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "text-1", "type": "text", "role": "assistant"}},
    )
    await renderer.emit("chat.block.delta", {"block_id": "text-1", "delta": "hello"})
    await renderer.emit("chat.turn.end", {"duration_ms": 2500})

    output = console.export_text()
    assert "turn turn-1" in output
    assert "Assistant" in output
    assert "hello" in output
    assert "Tool read" in output
    assert "read -> done" in output
    assert "read 20 chars/1 lines" in output
    assert "secret file contents" not in output
    assert "gateway INFO" in output
    assert "tool.result" in output
    assert renderer.last_turn_duration_ms == 2500


@pytest.mark.asyncio
async def test_turn_progress_aggregates_usage_across_model_calls() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)
    await progress.on_llm_response(
        LLMResponse(
            content="first",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
    )
    await progress.on_llm_response(
        LLMResponse(
            content="second",
            usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        )
    )

    assert progress.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }
    response_payloads = [payload for event, payload in events if event == "chat.llm.response"]
    assert response_payloads[-1]["usage"]["total_tokens"] == 23
    assert response_payloads[-1]["call_usage"]["total_tokens"] == 23
    assert response_payloads[-1]["aggregate_usage"]["total_tokens"] == 35


@pytest.mark.asyncio
async def test_turn_progress_attributes_context_and_harness_usage() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)
    await progress.on_usage({"total_tokens": 4}, node_kind="context_processing")
    await progress.on_usage({"total_tokens": 3}, node_kind="harness")

    assert progress.usage["total_tokens"] == 7
    assert progress.usage_by_node_kind == {
        "context_processing": {"total_tokens": 4},
        "harness": {"total_tokens": 3},
    }
    assert [event for event, _payload in events] == ["chat.usage", "chat.usage"]


@pytest.mark.asyncio
async def test_terminal_renderer_updates_summary_from_usage_event() -> None:
    renderer = TerminalEventRenderer(
        Console(record=True, width=100),
        show_gateway_logs=False,
    )

    await renderer.emit("chat.usage", {"usage": {"total_tokens": 9}})

    assert renderer.last_usage == {"total_tokens": 9}


def test_terminal_renderer_turn_summary_includes_duration_seconds() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    renderer.last_turn_duration_ms = 1234

    renderer.print_turn_summary(
        TurnResult(
            session_id="session-1",
            final_content="done",
            tools_used=["read"],
            messages=[],
            blocks=[],
        )
    )

    output = console.export_text()
    assert "session session-1 | tools read | duration 1.2s" in output


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


@pytest.mark.asyncio
async def test_terminal_renderer_streams_assistant_delta_before_turn_end() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)

    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "text-1", "type": "text", "role": "assistant"}},
    )
    await renderer.emit("chat.block.delta", {"block_id": "text-1", "delta": "hello"})

    output = console.export_text()
    assert "Assistant" in output
    assert "hello" in output


def test_terminal_renderer_prints_user_without_panel() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)

    renderer.print_user("hello")

    output = console.export_text()
    assert "You: hello" in output
    assert "╭" not in output


@pytest.mark.asyncio
async def test_terminal_renderer_streams_full_assistant_and_summarizes_tool_outputs() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    long_output = "\n".join(f"line {index}" for index in range(1, 8))
    tool_output = "\n".join(f"secret tool line {index}" for index in range(1, 8))

    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "text-1", "type": "text", "role": "assistant"}},
    )
    await renderer.emit("chat.block.delta", {"block_id": "text-1", "delta": long_output})
    await renderer.emit("chat.turn.end", {})
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
            "patch": {"status": "done", "result": tool_output, "duration_ms": 3},
        },
    )

    output = console.export_text()
    assert "Assistant" in output
    assert "read -> done" in output
    assert "line 1" in output
    assert "line 7" in output
    assert "Assistant Summary" not in output
    assert "collapsed" not in output
    assert "line 4" in output
    assert "read 132 chars" in output
    assert "secret tool line" not in output


@pytest.mark.asyncio
async def test_terminal_renderer_summarizes_write_without_content() -> None:
    console = Console(record=True, width=160)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    content = "private written content"

    await renderer.emit(
        "chat.block.add",
        {
            "block": {
                "id": "tool-1",
                "type": "tool_call",
                "name": "write",
                "arguments": {"path": "notes.txt", "content": content},
            }
        },
    )
    await renderer.emit(
        "chat.block.update",
        {
            "block_id": "tool-1",
            "patch": {
                "status": "done",
                "result": "Successfully wrote 23 bytes to /tmp/notes.txt",
                "duration_ms": 4,
            },
        },
    )

    output = console.export_text()
    assert "Tool write -> writing ~" in output
    assert "Tool write -> done" in output
    assert "notes.txt" in output
    assert "wrote 23 chars" in output
    assert content not in output


@pytest.mark.asyncio
async def test_terminal_renderer_shows_public_thinking_summary_only() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    reasoning = "\n".join(
        [
            (
                "2026-07-07T10:00:00+00:00 [thought] "
                "I need to verify who this person is before answering."
            ),
            (
                '2026-07-07T10:00:01+00:00 [tool_call] {"summary": "Call read", '
                '"tool_name": "read"}'
            ),
            (
                '2026-07-07T10:00:02+00:00 [tool_result] {"summary": "read '
                'returned 120 characters", "tool_name": "read", "duration_ms": 5}'
            ),
            "2026-07-07T10:00:03+00:00 [status] Reviewing all gathered context",
            "model raw thinking text",
        ]
    )

    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "reasoning-1", "type": "reasoning", "role": "assistant"}},
    )
    await renderer.emit(
        "chat.block.update",
        {"block_id": "reasoning-1", "patch": {"content": reasoning}},
    )
    await renderer.emit(
        "chat.block.update",
        {"block_id": "reasoning-1", "patch": {"status": "done"}},
    )

    output = console.export_text()
    assert "Thinking" in output
    assert "model raw thinking text" in output
    assert "tool call: Call read" not in output
    assert "tool result: read returned 120 characters" not in output
    assert "status: Reviewing all gathered context" not in output


@pytest.mark.asyncio
async def test_terminal_renderer_streams_raw_reasoning_delta() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)

    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "reasoning-1", "type": "reasoning", "role": "assistant"}},
    )
    await renderer.emit(
        "chat.block.delta",
        {"block_id": "reasoning-1", "delta": "private chain fragment"},
    )

    output = console.export_text()
    assert "Thinking" in output
    assert "private chain fragment" in output


@pytest.mark.asyncio
async def test_terminal_renderer_separates_thinking_from_assistant_stream() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)

    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "reasoning-1", "type": "reasoning", "role": "assistant"}},
    )
    await renderer.emit(
        "chat.block.delta",
        {"block_id": "reasoning-1", "delta": "thinking text"},
    )
    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "text-1", "type": "text", "role": "assistant"}},
    )
    await renderer.emit("chat.block.delta", {"block_id": "text-1", "delta": "answer"})

    output = console.export_text()
    assert "thinking text\nAssistant\nanswer" in output


@pytest.mark.asyncio
async def test_turn_progress_status_events_are_log_only_in_tui() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    progress = TurnEventProgress(session_id="session-1", emit=renderer.emit)

    await progress("Thinking...")
    await progress("I need to search because this may refer to a real person.")

    output = console.export_text()
    assert "status: Thinking..." not in output
    assert "I need to search because this may refer to a real person." not in output


def test_system_prompt_requests_public_tool_thoughts() -> None:
    assert "public thinking note" in SYSTEM_PROMPT
    assert "before the tool call" in SYSTEM_PROMPT
    assert "reasoning/thinking field" in SYSTEM_PROMPT
