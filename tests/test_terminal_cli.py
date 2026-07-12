from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from aeloon_core.__main__ import (
    CONFIG_SETTERS,
    _coerce_config_value,
    _load_with_path_overrides,
    _looks_like_implicit_chat,
    _looks_like_legacy_run,
    _run_chat,
    build_parser,
    main,
)
from aeloon_core.config import Config
from aeloon_core.context import SYSTEM_PROMPT
from aeloon_core.loop_guard import LoopGuardAction, LoopGuardDecision
from aeloon_core.orchestrator import TurnResult
from aeloon_core.providers.base import LLMResponse, ToolCallRequest
from aeloon_core.task_graph import TaskNode
from aeloon_core.temporary_guard import GuardResolution
from aeloon_core.terminal_cli import TerminalEventRenderer
from aeloon_core.turn_events import TurnEventProgress


def test_chat_commands_are_registered() -> None:
    args = build_parser().parse_args(["chat", "--hide-gateway-logs", "hello"])

    assert args.command == "chat"
    assert args.prompt == ["hello"]
    assert args.show_gateway_logs is False
    assert args.hide_gateway_logs is True
    assert args.gateway_log_level is None
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


def test_uasm_config_setters_keep_runtime_trace_and_context_controls() -> None:
    args = build_parser().parse_args(
        ["config", "set", "uasm-minimal-context-recent-turns", "3"]
    )

    assert args.config_command == "set"
    assert args.key == "uasm-minimal-context-recent-turns"
    assert _coerce_config_value("uasm-minimal-context-recent-turns", "3") == 3
    assert "uasm-rule-engine-enabled" not in CONFIG_SETTERS


def test_implicit_chat_invocations() -> None:
    assert _looks_like_implicit_chat(["--data-dir", "/tmp/aeloon"]) is True
    assert _looks_like_implicit_chat(["--show-gateway-logs"]) is True
    assert _looks_like_implicit_chat(["--gateway-log-level", "DEBUG"]) is True
    assert _looks_like_implicit_chat(["--gateway-log-level", "DEBUG", "hello"]) is True
    assert _looks_like_implicit_chat(["chat", "--show-gateway-logs"]) is False
    assert _looks_like_implicit_chat(["tui", "--gateway-log-level", "DEBUG"]) is False
    assert _looks_like_implicit_chat(["--help"]) is False
    assert _looks_like_implicit_chat(["webui"]) is False
    assert _looks_like_implicit_chat(["hello"]) is False


def test_main_keeps_explicit_chat_subcommands_with_log_options(monkeypatch) -> None:
    calls = []

    async def fake_run_chat(args) -> None:
        calls.append(args)

    monkeypatch.setattr("aeloon_core.__main__._run_chat", fake_run_chat)

    main(["chat", "--show-gateway-logs"])
    main(["tui", "--gateway-log-level", "DEBUG"])

    assert [args.command for args in calls] == ["chat", "tui"]
    assert calls[0].show_gateway_logs is True
    assert calls[1].gateway_log_level == "DEBUG"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "expected_show", "expected_level", "expected_detail"),
    [
        ([], False, "INFO", False),
        (["--show-gateway-logs"], True, "INFO", False),
        (["--gateway-log-level", "DEBUG"], True, "DEBUG", False),
        (["--gateway-log-detail"], True, "INFO", True),
        (
            ["--gateway-log-level", "DEBUG", "--hide-gateway-logs"],
            False,
            "DEBUG",
            False,
        ),
    ],
)
async def test_chat_gateway_log_options_are_explicit_opt_in(
    options: list[str],
    expected_show: bool,
    expected_level: str,
    expected_detail: bool,
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_terminal_cli(_config, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("aeloon_core.__main__.run_terminal_cli", fake_run_terminal_cli)
    args = build_parser().parse_args(
        ["chat", "--config", str(tmp_path / "missing.json"), *options]
    )

    await _run_chat(args)

    assert captured["show_gateway_logs"] is expected_show
    assert captured["gateway_log_level"] == expected_level
    assert captured["gateway_log_detail"] is expected_detail


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
async def test_terminal_renderer_keeps_core_info_when_gateway_logs_are_enabled() -> None:
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
    assert "turn turn-1" not in output
    assert "Aeloon" in output
    assert "hello" in output
    assert "◇ 工具 read" in output
    assert "✓ 工具 read" in output
    assert "read 20 chars/1 lines" in output
    assert "secret file contents" not in output
    assert "gateway INFO" in output
    assert "tool.result" in output
    assert renderer.last_turn_duration_ms == 2500


@pytest.mark.asyncio
async def test_terminal_renderer_defaults_to_quiet_allowlist() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console)

    await renderer.emit("chat.turn.start", {"turn_id": "hidden-turn"})
    await renderer.emit("chat.status", {"text": "hidden status"})
    await renderer.emit(
        "chat.profile.pinned",
        {"profile": {"profile_id": "hidden-profile", "artifact_id": "secret-artifact"}},
    )
    await renderer.emit(
        "log.entry",
        {
            "level": "INFO",
            "source": "kernel.status",
            "message": "hidden gateway log",
        },
    )
    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "reasoning-1", "type": "reasoning"}},
    )
    await renderer.emit(
        "chat.block.delta",
        {"block_id": "reasoning-1", "delta": "hidden reasoning"},
    )
    await renderer.emit(
        "chat.block.add",
        {"block": {"id": "text-1", "type": "text", "role": "assistant"}},
    )
    await renderer.emit("chat.block.delta", {"block_id": "text-1", "delta": "answer"})

    assert renderer.show_gateway_logs is False
    assert console.export_text() == "Aeloon\nanswer"
    assert len(renderer.gateway_logs) == 1


def test_terminal_renderer_header_is_compact() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console)
    config = Config(
        workspace=Path("/tmp/workspace"),
        data_dir=Path("/tmp/private-data-dir"),
    )

    renderer.print_header(config, session_id="session-1")

    output = console.export_text()
    assert "Aeloon Core" in output
    assert "default" in output
    assert str(config.workspace) in output
    assert "session session-1" in output
    assert str(config.data_dir) not in output
    assert "gateway logs" not in output


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
    await progress.on_usage(
        {"total_tokens": 4},
        node_kind="context_processing",
        component="minimal_context",
    )
    await progress.on_usage(
        {"total_tokens": 3},
        node_kind="harness",
        component="profile_master",
    )

    assert progress.usage["total_tokens"] == 7
    assert progress.usage_by_node_kind == {
        "context_processing": {"total_tokens": 4},
        "harness": {"total_tokens": 3},
    }
    assert progress.usage_by_component == {
        "minimal_context": {"total_tokens": 4},
        "profile_master": {"total_tokens": 3},
    }
    assert [event for event, _payload in events] == ["chat.usage", "chat.usage"]


@pytest.mark.asyncio
async def test_turn_progress_emits_sanitized_guard_decision_and_usage() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)
    decision = LoopGuardDecision(
        LoopGuardAction.STOP_OFF_TRACK,
        reason="repeated tool failures",
        final_content="private final content",
        prompt_message={"role": "system", "content": "private recovery prompt"},
    )

    resolution = GuardResolution(
        decision=decision,
        source="rule_fallback",
        usage={"total_tokens": 3},
        fallback_used=True,
    )
    await progress.on_guard_decision(resolution)
    await progress.on_loop_guard_decision(
        decision,
        event="tool_result_failed",
        source="rule_fallback",
        fallback_used=True,
    )

    assert [event for event, _payload in events] == [
        "chat.usage",
        "chat.guard.decision",
    ]
    payload = events[-1][1]
    assert set(payload) == {
        "session_id",
        "turn_id",
        "ts",
        "source",
        "event",
        "action",
        "reason",
        "budget_grant",
        "fallback_used",
    }
    assert payload["source"] == "rule_fallback"
    assert payload["event"] == "tool_result_failed"
    assert payload["action"] == "stop_off_track"
    assert payload["fallback_used"] is True
    assert "private" not in json.dumps(payload)
    assert progress.usage == {"total_tokens": 3}


@pytest.mark.asyncio
async def test_delegated_guard_uses_separate_branch_correlated_event() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)
    await progress.on_profile_delegate_guard_decision(
        "delegate-2-3",
        "fact_checker#1",
        LoopGuardDecision(
            LoopGuardAction.RETURN_TO_MODEL,
            reason="retry one branch",
            final_content="private final",
            prompt_message={"role": "system", "content": "private prompt"},
        ),
        event="tool_result_failed",
        source="rule_engine",
    )

    assert [event for event, _payload in events] == ["chat.profile.delegate.guard"]
    payload = events[0][1]
    assert payload["branch_id"] == "delegate-2-3"
    assert payload["subagent_label"] == "fact_checker#1"
    assert payload["reason"] == "retry one branch"
    assert "private" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_turn_progress_tool_result_includes_duration_and_guard_aligned_error() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)
    tool_call = ToolCallRequest(id="call-1", name="exec", arguments={"command": "false"})
    await progress.on_tool_calls([tool_call])
    await progress.on_tool_result(
        TaskNode(
            index=0,
            call_id="call-1",
            tool_name="exec",
            arguments={"command": "false"},
            mode="exclusive",
            result="  error: command failed",
        )
    )

    patch = next(
        payload["patch"]
        for event, payload in events
        if event == "chat.block.update"
        and payload.get("block_id") == "call-1"
        and "result" in payload.get("patch", {})
    )
    assert patch["status"] == "error"
    assert isinstance(patch["duration_ms"], int)
    assert patch["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_terminal_renderer_updates_summary_from_usage_event() -> None:
    renderer = TerminalEventRenderer(
        Console(record=True, width=100),
        show_gateway_logs=False,
    )

    await renderer.emit("chat.usage", {"usage": {"total_tokens": 9}})

    assert renderer.last_usage == {"total_tokens": 9}


@pytest.mark.asyncio
async def test_profile_events_hide_provenance_and_keep_subagent_lifecycle() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    progress = TurnEventProgress(session_id="session-1", emit=renderer.emit)

    await progress.on_profile_pinned(
        {
            "profile_id": "coding-team",
            "revision": 2,
            "artifact_id": "abcdef1234567890",
            "generation": 4,
        }
    )
    await progress.on_profile_route(
        "planner",
        source="profile_master",
        fallback_used=False,
    )
    await progress.on_profile_handoff(
        "planner",
        "implementer",
        "plan ready",
        handoff_count=1,
        handoff_limit=8,
    )
    await progress.on_profile_completion("implementer", "done")

    output = console.export_text()
    assert "coding-team" not in output
    assert "abcdef123456" not in output
    assert "◆ 子agent planner · 启动" in output
    assert "↳ 子agent planner → implementer · 交接 1/8 · plan ready" in output
    assert "✓ 子agent implementer · 完成" in output


@pytest.mark.asyncio
async def test_parallel_subagent_events_and_tools_are_labeled() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    progress = TurnEventProgress(session_id="session-1", emit=renderer.emit)

    await progress.on_profile_delegate_branch_start(
        "delegate-1-1",
        "source_scout#1",
        "source_scout",
        "find primary biography sources",
    )
    await progress.on_tool_calls(
        [
            ToolCallRequest(
                id="delegate-1-1:search-1",
                name="websearch",
                arguments={"query": "primary biography"},
            )
        ],
        subagent_label="source_scout#1",
    )
    await progress.on_tool_result(
        TaskNode(
            index=0,
            call_id="delegate-1-1:search-1",
            tool_name="websearch",
            arguments={"query": "primary biography"},
            mode="read_only",
            result="1. source",
        ),
        subagent_label="source_scout#1",
    )
    await progress.on_profile_delegate_branch_complete(
        "delegate-1-1",
        "source_scout#1",
        "source_scout",
        status="completed",
        summary="verified biography sources",
        duration_ms=123,
        tools_used=["websearch"],
    )
    await progress.on_profile_delegate_join(
        "research_lead",
        delegation_round=1,
        branch_count=1,
        succeeded=1,
        duration_ms=125,
    )
    await progress.on_profile_delegate_guard_decision(
        "delegate-1-1",
        "source_scout#1",
        LoopGuardDecision(
            LoopGuardAction.RETURN_TO_MODEL,
            reason="retry failed source",
        ),
        event="tool_result_failed",
        source="rule_engine",
    )

    output = console.export_text()
    assert "◆ 子agent source_scout#1 · 并行启动" in output
    assert "◇ 工具 websearch [source_scout#1]" in output
    assert "✓ 工具 websearch [source_scout#1]" in output
    assert "✓ 子agent source_scout#1 · 完成 · 123 ms" in output
    assert "↳ 并行子agent · 汇总 1/1 · 125 ms" in output
    assert "Guard [rule_engine/source_scout#1] · 重试 · retry failed source" in output


@pytest.mark.asyncio
async def test_subagent_task_update_is_single_line_and_bounded() -> None:
    console = Console(record=True, width=240)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)
    progress = TurnEventProgress(session_id="session-1", emit=renderer.emit)
    summary = "first line\n" + ("detail " * 40)

    await progress.on_profile_handoff(
        "planner",
        "implementer",
        summary,
        handoff_count=1,
        handoff_limit=8,
    )

    output = console.export_text()
    update = next(line for line in output.splitlines() if line.startswith("↳ 子agent"))
    assert "first line detail" in update
    assert "chars]" in update


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "action", "source", "fallback_used", "expected"),
    [
        ("empty_response", "continue", "rule_engine", False, "· 重试"),
        ("duplicate_tool_calls", "continue", "temporary_guard", False, "· 继续"),
        ("tool_result_failed", "return_to_model", "rule_engine", False, "· 重试"),
        ("iteration_budget", "extend_budget", "temporary_guard", False, "· 扩容 +2"),
        ("output_exhausted", "finalize", "rule_engine", False, "· 收尾"),
        ("output_exhausted", "final_response", "rule_engine", False, "· 停止"),
        ("tool_result_failed", "stop_off_track", "rule_fallback", True, "· 停止 · 回退"),
    ],
)
async def test_terminal_renderer_shows_key_guard_decisions(
    event: str,
    action: str,
    source: str,
    fallback_used: bool,
    expected: str,
) -> None:
    console = Console(record=True, width=80)
    renderer = TerminalEventRenderer(console)

    await renderer.emit(
        "chat.guard.decision",
        {
            "source": source,
            "event": event,
            "action": action,
            "reason": "bounded guard reason " * 20,
            "budget_grant": 2,
            "fallback_used": fallback_used,
            "evidence": "private evidence must not render",
        },
    )

    output = console.export_text()
    assert f"Guard [{source}]" in output
    assert expected in output
    assert "private evidence" not in output
    assert len(output.splitlines()) == 1


@pytest.mark.asyncio
async def test_tool_calls_remain_paired_when_concurrent_results_finish_out_of_order() -> None:
    console = Console(record=True, width=120)
    renderer = TerminalEventRenderer(console)

    for call_id, path in (("call-1", "first.txt"), ("call-2", "second.txt")):
        await renderer.emit(
            "chat.block.add",
            {
                "block": {
                    "id": call_id,
                    "type": "tool_call",
                    "name": "read",
                    "arguments": {"path": path},
                }
            },
        )
    for call_id, body in (("call-2", "2| second"), ("call-1", "1| first")):
        await renderer.emit(
            "chat.block.update",
            {
                "block_id": call_id,
                "patch": {"status": "done", "result": body, "duration_ms": 5},
            },
        )

    lines = console.export_text().splitlines()
    assert lines == [
        "◇ 工具 read · path=first.txt",
        "◇ 工具 read · path=second.txt",
        "✓ 工具 read · path=second.txt · read 6 chars/1 lines · 5 ms",
        "✓ 工具 read · path=first.txt · read 5 chars/1 lines · 5 ms",
    ]


@pytest.mark.asyncio
async def test_transcript_activity_is_single_line_at_eighty_columns() -> None:
    console = Console(record=True, width=80)
    renderer = TerminalEventRenderer(console)
    progress = TurnEventProgress(session_id="session-1", emit=renderer.emit)

    await renderer.emit(
        "chat.block.add",
        {
            "block": {
                "id": "tool-1",
                "type": "tool_call",
                "name": "exec",
                "arguments": {"command": "very-long-command " * 20},
            }
        },
    )
    await progress.on_profile_route(
        "planner-with-a-very-long-name",
        source="profile_master",
        fallback_used=True,
    )
    await progress.on_profile_handoff(
        "planner-with-a-very-long-name",
        "implementer-with-a-very-long-name",
        "long handoff detail " * 20,
        handoff_count=1,
        handoff_limit=8,
    )

    lines = console.export_text().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("◇ 工具 exec")
    assert lines[1].startswith("◆ 子agent")
    assert "回退选择" in lines[1]
    assert lines[2].startswith("↳ 子agent")


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
    assert output.strip() == "完成 · 耗时 1.2s"


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
    assert "Aeloon" in output
    assert "hello" in output


def test_terminal_renderer_prints_user_without_panel() -> None:
    console = Console(record=True, width=100)
    renderer = TerminalEventRenderer(console, show_gateway_logs=False)

    renderer.print_user("hello")

    output = console.export_text()
    assert "› hello" in output
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
    assert "Aeloon" in output
    assert "✓ 工具 read" in output
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
    assert "◇ 工具 write" in output
    assert "✓ 工具 write" in output
    assert "notes.txt" in output
    assert "wrote 23 chars" in output
    assert content not in output


@pytest.mark.asyncio
async def test_terminal_renderer_hides_reasoning_content_updates() -> None:
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
    assert output == ""


@pytest.mark.asyncio
async def test_terminal_renderer_hides_raw_reasoning_delta() -> None:
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
    assert output == ""


@pytest.mark.asyncio
async def test_hidden_reasoning_does_not_interrupt_assistant_stream() -> None:
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
    assert output == "Aeloon\nanswer"


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
