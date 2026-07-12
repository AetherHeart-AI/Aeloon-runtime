"""Rich terminal chat UI for Aeloon Core."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from aeloon_core.config import Config
from aeloon_core.loop_guard import tool_result_failed
from aeloon_core.orchestrator import AeloonCoreOrchestrator, TurnResult
from aeloon_core.turn_events import TurnEventProgress

LOG_LEVELS = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
LOG_STYLES = {
    "TRACE": "dim",
    "DEBUG": "dim",
    "INFO": "blue",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}
SEMANTIC_STYLES = {
    "user": "bold magenta",
    "assistant": "bold cyan",
    "tool": "cyan",
    "agent": "bold bright_blue",
    "guard": "bold yellow",
    "success": "bold green",
    "error": "bold red",
    "muted": "dim",
}
GUARD_ACTION_LABELS = {
    "continue": "继续",
    "return_to_model": "重试",
    "extend_budget": "扩容",
    "finalize": "收尾",
    "final_response": "停止",
    "stop_off_track": "停止",
}
TRANSCRIPT_DETAIL_CHARS = 160
HISTORY_PREVIEW_CHARS = 160
PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})


class TerminalEventRenderer:
    """Render structured progress events as a compact terminal transcript."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_gateway_logs: bool = False,
        gateway_log_level: str = "INFO",
        gateway_log_detail: bool = False,
    ) -> None:
        self.console = console or Console()
        self.show_gateway_logs = show_gateway_logs
        self.gateway_log_level = gateway_log_level.upper()
        self.gateway_log_detail = gateway_log_detail
        self.gateway_logs: list[dict[str, Any]] = []
        self.block_types: dict[str, str] = {}
        self.block_names: dict[str, str] = {}
        self.block_arguments: dict[str, Any] = {}
        self.block_agents: dict[str, str] = {}
        self.block_content_lengths: dict[str, int] = {}
        self.block_contents: dict[str, str] = {}
        self.last_usage: dict[str, Any] = {}
        self.last_turn_duration_ms: int | None = None
        self._assistant_streaming = False
        self._lock = asyncio.Lock()

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        """Receive one bridge event."""

        async with self._lock:
            self._render_event(event, payload)

    def print_header(self, config: Config, *, session_id: str) -> None:
        """Print the persistent CLI context."""

        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(justify="right")
        table.add_row(
            Text("Aeloon Core", style=SEMANTIC_STYLES["assistant"]),
            Text(str(config.agents.defaults.model), style=SEMANTIC_STYLES["tool"]),
        )
        table.add_row(
            Text(str(config.workspace), style=SEMANTIC_STYLES["muted"]),
            Text(f"session {session_id}", style=SEMANTIC_STYLES["muted"]),
        )
        self.console.print(
            Panel(
                table,
                border_style=SEMANTIC_STYLES["tool"],
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def print_help(self) -> None:
        """Print interactive commands."""

        table = Table(box=box.SIMPLE, show_header=False, expand=False)
        table.add_column("command", style="cyan", no_wrap=True)
        table.add_column("description")
        table.add_row("/help", "show commands")
        table.add_row("/new", "start a new session")
        table.add_row("/sessions", "list saved sessions")
        table.add_row("/resume <id>", "continue a saved session")
        table.add_row("/history", "show the current session history")
        table.add_row("/logs [on|off|info|debug|warning|error|detail]", "control gateway logs")
        table.add_row("/clear", "clear the terminal")
        table.add_row("/quit", "exit")
        self.console.print(Panel(table, title="Commands", border_style="cyan"))

    def print_user(self, prompt: str) -> None:
        """Print a user turn."""

        self._finish_stream_line()
        self.console.print(Text(f"› {prompt}", style=SEMANTIC_STYLES["user"]))

    def print_turn_summary(self, result: TurnResult) -> None:
        """Print final session metadata for a turn."""

        self._finish_stream_line()
        usage = _format_usage(self.last_usage)
        duration = _format_duration(self.last_turn_duration_ms)
        parts = ["完成"]
        if duration:
            parts.append(f"耗时 {duration}")
        if usage:
            parts.append(usage)
        self.console.print(Text(" · ".join(parts), style=SEMANTIC_STYLES["muted"]))
        self.last_usage = {}
        self.last_turn_duration_ms = None

    def print_error(self, message: str) -> None:
        """Print an error without treating it as assistant output."""

        self._finish_stream_line()
        self.console.print(Panel(message, title="Error", border_style=SEMANTIC_STYLES["error"]))

    def print_sessions(self, sessions: list[Any], *, current_session_id: str) -> None:
        """Print saved sessions."""

        self._finish_stream_line()
        if not sessions:
            self.console.print("[dim]No saved sessions yet.[/]")
            return

        table = Table(title="Sessions", box=box.SIMPLE)
        table.add_column("session", style="cyan", no_wrap=True)
        table.add_column("turns", justify="right")
        table.add_column("updated")
        table.add_column("title")
        for item in sessions:
            marker = "*" if item.session_id == current_session_id else " "
            table.add_row(
                f"{marker} {item.session_id}",
                str(item.turns),
                _short_time(item.updated_at),
                item.title,
            )
        self.console.print(table)

    def print_history(self, records: list[dict[str, Any]], *, session_id: str) -> None:
        """Print a compact current-session history."""

        self._finish_stream_line()
        if not records:
            self.console.print(f"[dim]No history for session {session_id}.[/]")
            return

        table = Table(title=f"History {session_id}", box=box.SIMPLE)
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("created")
        table.add_column("user")
        table.add_column("assistant")
        for index, record in enumerate(records, start=1):
            table.add_row(
                str(index),
                _short_time(str(record.get("created_at") or "")),
                _preview(record.get("user_prompt"), limit=HISTORY_PREVIEW_CHARS),
                _preview(record.get("final_content"), limit=HISTORY_PREVIEW_CHARS),
            )
        self.console.print(table)

    def print_log_settings(self) -> None:
        """Print current gateway log settings."""

        detail = "detail" if self.gateway_log_detail else "compact"
        state = "on" if self.show_gateway_logs else "off"
        self.console.print(
            f"[dim]Gateway logs: {state}, level {self.gateway_log_level}, {detail}.[/]"
        )

    def _render_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "log.entry":
            self._render_log(payload)
            return
        if event == "chat.turn.start":
            self._finish_stream_line()
            return
        if event == "chat.status":
            return
        if event == "chat.guard.decision":
            self._render_guard_decision(payload)
            return
        if event == "chat.profile.delegate.guard":
            self._render_guard_decision(payload)
            return
        if event == "chat.block.add":
            self._render_block_add(payload)
            return
        if event == "chat.block.delta":
            self._render_block_delta(payload)
            return
        if event == "chat.block.update":
            self._render_block_update(payload)
            return
        if event == "chat.llm.response":
            usage = payload.get("aggregate_usage", payload.get("usage"))
            self.last_usage = usage if isinstance(usage, dict) else {}
            return
        if event == "chat.usage":
            usage = payload.get("usage")
            if isinstance(usage, dict):
                self.last_usage = usage
            return
        if event.startswith("chat.profile."):
            self._render_profile_event(event, payload)
            return
        if event == "chat.turn.end":
            duration = payload.get("duration_ms")
            self.last_turn_duration_ms = duration if isinstance(duration, int) else None
            self._flush_text_blocks()
            self._finish_stream_line()

    def _render_profile_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "chat.profile.pinned":
            return
        self._finish_stream_line()
        text = Text(no_wrap=True, overflow="ellipsis")
        if event == "chat.profile.route":
            text.append("◆ ", style=SEMANTIC_STYLES["agent"])
            text.append("子agent ", style=SEMANTIC_STYLES["muted"])
            text.append(
                str(payload.get("agent_id") or "unknown"),
                style=SEMANTIC_STYLES["agent"],
            )
            text.append(" · 启动", style=SEMANTIC_STYLES["muted"])
            if payload.get("fallback_used"):
                text.append(" · 回退选择", style=SEMANTIC_STYLES["guard"])
        elif event == "chat.profile.delegate.start":
            label = str(payload.get("label") or payload.get("agent_id") or "unknown")
            task = _one_line(
                str(payload.get("task") or "并行任务"),
                limit=TRANSCRIPT_DETAIL_CHARS,
            )
            text.append("◆ ", style=SEMANTIC_STYLES["agent"])
            text.append("子agent ", style=SEMANTIC_STYLES["muted"])
            text.append(label, style=SEMANTIC_STYLES["agent"])
            text.append(" · 并行启动", style=SEMANTIC_STYLES["muted"])
            if task:
                text.append(" · ", style=SEMANTIC_STYLES["muted"])
                text.append(task)
        elif event == "chat.profile.delegate.complete":
            label = str(payload.get("label") or payload.get("agent_id") or "unknown")
            status = str(payload.get("status") or "failed")
            succeeded = status == "completed"
            style = SEMANTIC_STYLES["success"] if succeeded else SEMANTIC_STYLES["error"]
            text.append("✓ " if succeeded else "✕ ", style=style)
            text.append("子agent ", style=SEMANTIC_STYLES["muted"])
            text.append(label, style=style)
            text.append(" · 完成" if succeeded else " · 失败", style=SEMANTIC_STYLES["muted"])
            duration = _compact_duration(payload.get("duration_ms"))
            if duration:
                text.append(f" · {duration}", style=SEMANTIC_STYLES["muted"])
            summary = _one_line(
                str(payload.get("summary") or ""),
                limit=TRANSCRIPT_DETAIL_CHARS,
            )
            if summary:
                text.append(" · ", style=SEMANTIC_STYLES["muted"])
                text.append(summary)
        elif event == "chat.profile.delegate.join":
            count = payload.get("branch_count", "?")
            succeeded = payload.get("succeeded", "?")
            text.append("↳ ", style=SEMANTIC_STYLES["agent"])
            text.append("并行子agent", style=SEMANTIC_STYLES["agent"])
            text.append(f" · 汇总 {succeeded}/{count}", style=SEMANTIC_STYLES["muted"])
            duration = _compact_duration(payload.get("duration_ms"))
            if duration:
                text.append(f" · {duration}", style=SEMANTIC_STYLES["muted"])
        elif event == "chat.profile.handoff":
            target = str(payload.get("recommended_agent_id") or "profile master")
            source = str(payload.get("from_agent_id") or "unknown")
            summary = _one_line(
                str(payload.get("summary") or "任务已交接"),
                limit=TRANSCRIPT_DETAIL_CHARS,
            )
            text.append("↳ ", style=SEMANTIC_STYLES["agent"])
            text.append("子agent ", style=SEMANTIC_STYLES["muted"])
            text.append(source, style=SEMANTIC_STYLES["agent"])
            text.append(" → ", style=SEMANTIC_STYLES["muted"])
            text.append(target, style=SEMANTIC_STYLES["agent"])
            text.append(
                f" · 交接 {payload.get('handoff_count', '?')}/{payload.get('handoff_limit', '?')}",
                style=SEMANTIC_STYLES["muted"],
            )
            if summary:
                text.append(" · ", style=SEMANTIC_STYLES["muted"])
                text.append(summary)
        elif event == "chat.profile.completion":
            text.append("✓ ", style=SEMANTIC_STYLES["success"])
            text.append("子agent ", style=SEMANTIC_STYLES["muted"])
            text.append(
                str(payload.get("agent_id") or "unknown"),
                style=SEMANTIC_STYLES["success"],
            )
            text.append(" · 完成", style=SEMANTIC_STYLES["muted"])
        else:
            return
        self._print_transcript_line(text)

    def _render_guard_decision(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action") or "").lower()
        self._finish_stream_line()
        stopped = action in {"final_response", "stop_off_track"}
        style = SEMANTIC_STYLES["error"] if stopped else SEMANTIC_STYLES["guard"]
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append("✕ " if stopped else "⚠ ", style=style)
        source = str(payload.get("source") or "rule_engine")
        subagent_label = str(payload.get("subagent_label") or "")
        if subagent_label:
            source = f"{source}/{subagent_label}"
        text.append("Guard", style=style)
        text.append(f" [{source}]", style=SEMANTIC_STYLES["muted"])
        guard_event = str(payload.get("event") or "")
        if action == "continue" and guard_event in {"empty_response", "output_exhausted"}:
            label = "重试"
        else:
            label = GUARD_ACTION_LABELS.get(action, action or "决策")
        grant = payload.get("budget_grant")
        if action == "extend_budget" and isinstance(grant, int) and grant > 0:
            label = f"{label} +{grant}"
        text.append(f" · {label}", style=style)
        if payload.get("fallback_used"):
            text.append(" · 回退", style=SEMANTIC_STYLES["guard"])
        reason = _one_line(
            str(payload.get("reason") or payload.get("event") or ""),
            limit=TRANSCRIPT_DETAIL_CHARS,
        )
        if reason:
            text.append(" · ", style=SEMANTIC_STYLES["muted"])
            text.append(reason)
        self._print_transcript_line(text)

    def _render_block_add(self, payload: dict[str, Any]) -> None:
        block = payload.get("block") if isinstance(payload.get("block"), dict) else {}
        block_id = str(block.get("id") or "")
        block_type = str(block.get("type") or "")
        if block_id:
            self.block_types[block_id] = block_type
            if block.get("name"):
                self.block_names[block_id] = str(block.get("name"))
            if "arguments" in block:
                self.block_arguments[block_id] = block.get("arguments")
            if block.get("subagent_label"):
                self.block_agents[block_id] = str(block.get("subagent_label"))
            if "content" in block:
                self.block_contents[block_id] = str(block.get("content") or "")
        if block_type == "tool_call":
            self._consume_pending_text_blocks()
            self._finish_stream_line()
            name = str(block.get("name") or "tool")
            self._print_transcript_line(
                _tool_line(
                    name=name,
                    status="running",
                    detail=_tool_call_detail_text(name, block.get("arguments")),
                    subagent_label=self.block_agents.get(block_id),
                )
            )

    def _render_block_delta(self, payload: dict[str, Any]) -> None:
        block_id = str(payload.get("block_id") or "")
        delta = str(payload.get("delta") or "")
        if not delta:
            return
        self.block_contents[block_id] = self.block_contents.get(block_id, "") + delta
        block_type = self.block_types.get(block_id)
        if block_type == "text":
            self._render_text_stream(block_id)
        elif block_type == "reasoning":
            self.block_content_lengths[block_id] = len(self.block_contents[block_id])

    def _render_block_update(self, payload: dict[str, Any]) -> None:
        block_id = str(payload.get("block_id") or "")
        block_type = self.block_types.get(block_id)
        patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
        if block_type == "text" and "content" in patch:
            self._render_text_content(block_id, str(patch.get("content") or ""))
            return
        if block_type == "reasoning":
            if "content" in patch:
                self.block_contents[block_id] = str(patch.get("content") or "")
            self.block_content_lengths[block_id] = len(
                self.block_contents.get(block_id, "")
            )
            return
        if block_type == "tool_call":
            self._render_tool_result(block_id, patch)

    def _render_text_content(self, block_id: str, content: str) -> None:
        self.block_contents[block_id] = content
        self._render_text_stream(block_id)

    def _render_tool_result(self, block_id: str, patch: dict[str, Any]) -> None:
        if "result" not in patch and "status" not in patch:
            return
        self._finish_stream_line()
        status = str(patch.get("status") or "done")
        name = self.block_names.get(block_id, "tool")
        raw_result = patch.get("result")
        failed = status.lower() in {"error", "failed", "failure"} or tool_result_failed(
            "" if raw_result is None else str(raw_result)
        )
        rendered_status = "error" if failed else "done"
        duration = patch.get("duration_ms")
        detail = _tool_result_detail_text(
            name,
            raw_result,
            arguments=self.block_arguments.get(block_id),
            status=rendered_status,
            duration_ms=duration,
        )
        self._print_transcript_line(
            _tool_line(
                name=name,
                status=rendered_status,
                detail=detail,
                subagent_label=self.block_agents.get(block_id),
            )
        )

    def _render_log(self, payload: dict[str, Any]) -> None:
        self.gateway_logs.append(payload)
        if not self.show_gateway_logs:
            return
        level = str(payload.get("level") or "INFO").upper()
        if LOG_LEVELS.get(level, 20) < LOG_LEVELS.get(self.gateway_log_level, 20):
            return
        self._finish_stream_line()
        source = str(payload.get("source") or "gateway")
        message = _preview(payload.get("message"), limit=280)
        ts = _short_time(str(payload.get("ts") or ""))
        style = LOG_STYLES.get(level, "blue")
        text = Text()
        text.append("gateway ", style="dim")
        text.append(level.ljust(7), style=style)
        text.append(f" {ts} ", style="dim")
        text.append(source, style="cyan")
        text.append(f" {message}")
        self.console.print(text)
        if self.gateway_log_detail:
            detail = payload.get("detail")
            if detail is not None:
                self.console.print(
                    Syntax(
                        json.dumps(detail, ensure_ascii=False, indent=2, default=str),
                        "json",
                        word_wrap=True,
                    )
                )

    def _append_assistant(self, text: str) -> None:
        if not self._assistant_streaming:
            self.console.print(Text("Aeloon", style=SEMANTIC_STYLES["assistant"]))
            self._assistant_streaming = True
        self.console.print(Text(text), end="")
        self._flush_console()

    def _flush_text_blocks(self) -> None:
        self._finish_stream_line()
        for block_id, block_type in self.block_types.items():
            if block_type == "text":
                self._render_text_block(block_id)

    def _consume_pending_text_blocks(self) -> None:
        for block_id, block_type in self.block_types.items():
            if block_type == "text":
                self.block_content_lengths[block_id] = len(
                    self.block_contents.get(block_id, "")
                )

    def _unprinted_suffix(self, block_id: str, content: str) -> str:
        printed = self.block_content_lengths.get(block_id, 0)
        return content[printed:] if 0 < printed <= len(content) else content

    def _render_text_block(self, block_id: str) -> None:
        content = self.block_contents.get(block_id, "")
        new_text = self._unprinted_suffix(block_id, content)
        if not new_text.strip():
            self.block_content_lengths[block_id] = len(content)
            return
        self._append_assistant(new_text)
        self.block_content_lengths[block_id] = len(content)

    def _render_text_stream(self, block_id: str) -> None:
        content = self.block_contents.get(block_id, "")
        new_text = self._unprinted_suffix(block_id, content)
        if not new_text:
            return
        self._append_assistant(new_text)
        self.block_content_lengths[block_id] = len(content)

    def _finish_stream_line(self) -> None:
        if not self._assistant_streaming:
            return
        self.console.print()
        self._assistant_streaming = False

    def _print_transcript_line(self, text: Text) -> None:
        text.truncate(max(1, self.console.size.width), overflow="ellipsis")
        self.console.print(text, soft_wrap=True)

    def _flush_console(self) -> None:
        flush = getattr(self.console.file, "flush", None)
        if callable(flush):
            flush()


class TerminalChatCli:
    """Interactive terminal runner."""

    def __init__(
        self,
        config: Config,
        *,
        session_id: str | None = None,
        show_gateway_logs: bool = False,
        gateway_log_level: str = "INFO",
        gateway_log_detail: bool = False,
        console: Console | None = None,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.orchestrator = AeloonCoreOrchestrator(config)
        self.session_id = session_id or self.orchestrator.sessions.new_session()
        self.prompt_session = PromptSession(style=PROMPT_STYLE) if _is_tty() else None
        self.renderer = TerminalEventRenderer(
            self.console,
            show_gateway_logs=show_gateway_logs,
            gateway_log_level=gateway_log_level,
            gateway_log_detail=gateway_log_detail,
        )

    async def run(self, initial_prompt: str | None = None) -> None:
        """Run either one rich-rendered prompt or an interactive prompt loop."""

        self.renderer.print_header(self.config, session_id=self.session_id)
        sink_id = self._install_log_sink()
        try:
            if initial_prompt:
                await self._run_turn(initial_prompt)
                return

            self.console.print("[dim]Type /help for commands. Ctrl-D or /quit exits.[/]")
            while True:
                try:
                    prompt = await self._read_prompt()
                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    return

                prompt = prompt.strip()
                if not prompt:
                    continue
                if prompt.startswith("/"):
                    keep_running = self._handle_command(prompt)
                    if not keep_running:
                        return
                    continue
                await self._run_turn(prompt)
        finally:
            logger.remove(sink_id)

    async def _run_turn(self, prompt: str) -> TurnResult | None:
        self.renderer.print_user(prompt)
        progress = TurnEventProgress(session_id=self.session_id, emit=self.renderer.emit)
        try:
            result = await self.orchestrator.run_turn(
                prompt,
                session_id=self.session_id,
                on_progress=progress,
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.renderer.print_error("Turn interrupted.")
            return None
        except Exception as exc:
            self.renderer.print_error(str(exc))
            return None

        self.session_id = result.session_id
        self.renderer.print_turn_summary(result)
        return result

    async def _read_prompt(self) -> str:
        if self.prompt_session is not None:
            return await self.prompt_session.prompt_async(
                [("class:prompt", "aeloon: ")],
                enable_suspend=True,
            )

        self.console.print(Text("aeloon: ", style="bold cyan"), end="")
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":
            raise EOFError
        return line.rstrip("\n")

    def _handle_command(self, command: str) -> bool:
        parts = command.split()
        name = parts[0].lower()
        args = parts[1:]
        if name in {"/quit", "/exit", "/q"}:
            return False
        if name == "/help":
            self.renderer.print_help()
            return True
        if name == "/clear":
            self.console.clear()
            self.renderer.print_header(self.config, session_id=self.session_id)
            return True
        if name == "/new":
            self.session_id = self.orchestrator.sessions.new_session()
            self.renderer.print_header(self.config, session_id=self.session_id)
            return True
        if name == "/sessions":
            self.renderer.print_sessions(
                self.orchestrator.sessions.list_sessions(),
                current_session_id=self.session_id,
            )
            return True
        if name == "/resume":
            self._resume_session(args)
            return True
        if name == "/history":
            self.renderer.print_history(
                self.orchestrator.sessions.history(self.session_id),
                session_id=self.session_id,
            )
            return True
        if name == "/logs":
            self._configure_logs(args)
            return True

        self.console.print(f"[red]Unknown command:[/] {command}")
        self.console.print("[dim]Type /help for commands.[/]")
        return True

    def _resume_session(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]Usage:[/] /resume <session-id>")
            return
        session_id = args[0]
        records = self.orchestrator.sessions.history(session_id)
        if not records:
            self.console.print(
                f"[yellow]No saved history for session {session_id}; using it anyway.[/]"
            )
        self.session_id = session_id
        self.renderer.print_header(self.config, session_id=self.session_id)

    def _configure_logs(self, args: list[str]) -> None:
        if not args:
            self.renderer.print_log_settings()
            return
        for arg in args:
            value = arg.lower()
            if value == "on":
                self.renderer.show_gateway_logs = True
            elif value == "off":
                self.renderer.show_gateway_logs = False
            elif value == "detail":
                self.renderer.gateway_log_detail = not self.renderer.gateway_log_detail
            elif value.upper() in LOG_LEVELS:
                self.renderer.gateway_log_level = value.upper()
                self.renderer.show_gateway_logs = True
            else:
                self.console.print(f"[red]Unknown /logs option:[/] {arg}")
        self.renderer.print_log_settings()

    def _install_log_sink(self) -> int:
        loop = asyncio.get_running_loop()

        def sink(message: Any) -> None:
            record = message.record
            payload = {
                "level": record["level"].name,
                "message": record["message"],
                "source": "loguru",
                "ts": record["time"].isoformat(),
                "detail": {
                    "logger": {
                        "name": record["name"],
                        "module": record["module"],
                        "function": record["function"],
                        "line": record["line"],
                    }
                },
            }
            loop.create_task(self.renderer.emit("log.entry", payload))

        logger.remove()
        return logger.add(sink, level="TRACE")


async def run_terminal_cli(
    config: Config,
    *,
    prompt: str | None = None,
    session_id: str | None = None,
    show_gateway_logs: bool = False,
    gateway_log_level: str = "INFO",
    gateway_log_detail: bool = False,
) -> None:
    """Run the terminal chat CLI."""

    cli = TerminalChatCli(
        config,
        session_id=session_id,
        show_gateway_logs=show_gateway_logs,
        gateway_log_level=gateway_log_level,
        gateway_log_detail=gateway_log_detail,
    )
    await cli.run(initial_prompt=prompt)


def _format_usage(usage: dict[str, Any]) -> str:
    if not usage:
        return ""
    keys = (
        ("prompt_tokens", "in"),
        ("completion_tokens", "out"),
        ("total_tokens", "total"),
    )
    parts = [f"{label} {usage[key]}" for key, label in keys if key in usage]
    if parts:
        return "tokens " + ", ".join(parts)
    return "usage " + ", ".join(f"{key} {value}" for key, value in usage.items())


def _format_duration(duration_ms: int | None) -> str:
    if duration_ms is None:
        return ""
    return f"{duration_ms / 1000:.1f}s"


def _preview(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} chars]"


def _tool_line(
    *,
    name: str,
    status: str,
    detail: str,
    subagent_label: str | None = None,
) -> Text:
    failed = status == "error"
    completed = status == "done"
    icon = "✕" if failed else "✓" if completed else "◇"
    icon_style = (
        SEMANTIC_STYLES["error"]
        if failed
        else SEMANTIC_STYLES["success"]
        if completed
        else SEMANTIC_STYLES["guard"]
    )
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"{icon} ", style=icon_style)
    text.append("工具 ", style=SEMANTIC_STYLES["muted"])
    text.append(name, style=SEMANTIC_STYLES["tool"])
    if subagent_label:
        text.append(f" [{subagent_label}]", style=SEMANTIC_STYLES["agent"])
    if detail:
        text.append(" · ", style=SEMANTIC_STYLES["muted"])
        text.append(_one_line(detail, limit=TRANSCRIPT_DETAIL_CHARS))
    return text


def _compact_duration(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    milliseconds = max(0, int(value))
    if milliseconds < 1_000:
        return f"{milliseconds} ms"
    return f"{milliseconds / 1_000:.1f}s"


def _tool_call_detail_text(name: str, arguments: Any) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    if name == "read":
        return _join_parts(
            _path_detail(args),
            _number_arg(args, "offset"),
            _number_arg(args, "limit"),
        )
    if name == "write":
        return _join_parts(_path_detail(args), f"{_string_arg_len(args, 'content')} chars")
    if name == "edit":
        return _join_parts(
            _path_detail(args),
            f"old {_string_arg_len(args, 'old_text')} chars",
            f"new {_string_arg_len(args, 'new_text')} chars",
            "replace_all" if args.get("replace_all") else "",
        )
    if name == "exec":
        return _join_parts(
            _preview_arg(args, "command", limit=120),
            _preview_arg(args, "working_dir", label="cwd"),
            _number_arg(args, "timeout"),
        )
    if name == "glob":
        return _join_parts(
            _preview_arg(args, "pattern"),
            _preview_arg(args, "root"),
            _number_arg(args, "limit"),
        )
    if name == "grep":
        return _join_parts(
            _preview_arg(args, "pattern"),
            _preview_arg(args, "path"),
            _preview_arg(args, "include"),
            _number_arg(args, "limit"),
        )
    if name == "webfetch":
        return _join_parts(_preview_arg(args, "url", limit=120), _number_arg(args, "max_chars"))
    if name == "websearch":
        return _join_parts(_preview_arg(args, "query", limit=120), _number_arg(args, "max_results"))
    if name == "todowrite":
        todos = args.get("todos")
        if isinstance(todos, list):
            return f"{len(todos)} todos"
    return _generic_arg_summary(arguments)


# Tools whose result summary is just "<n> <noun>" plus size/duration.
_RESULT_COUNT_NOUNS = {"glob": "matches", "grep": "matches", "websearch": "results"}


def _tool_result_detail_text(
    name: str,
    result: Any,
    *,
    arguments: Any,
    status: str,
    duration_ms: Any,
) -> str:
    text = "" if result is None else str(result)
    args = arguments if isinstance(arguments, dict) else {}
    duration = f"{duration_ms} ms" if duration_ms is not None else ""
    if status == "error" or text.startswith("Error"):
        return _join_parts("error", _one_line(text, limit=120), duration)

    count_noun = _RESULT_COUNT_NOUNS.get(name)
    if count_noun is not None:
        return _join_parts(
            _result_count_summary(text, noun=count_noun),
            _text_size_summary(text),
            duration,
        )
    if name == "read":
        chars, lines = _read_result_size(text)
        return _join_parts(_path_detail(args), f"read {chars} chars/{lines} lines", duration)
    if name == "write":
        wrote = f"wrote {_string_arg_len(args, 'content')} chars"
        return _join_parts(_path_detail(args), wrote, duration)
    if name == "edit":
        old_len = _string_arg_len(args, "old_text")
        new_len = _string_arg_len(args, "new_text")
        return _join_parts(
            _path_detail(args),
            f"edited {old_len} -> {new_len} chars",
            duration,
        )
    if name == "exec":
        return _join_parts(_exit_code_summary(text), _text_size_summary(text), duration)
    if name == "webfetch":
        return _join_parts(_web_status_summary(text), _text_size_summary(text), duration)
    return _join_parts(_text_size_summary(text), duration)


def _generic_arg_summary(arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return _text_size_summary("" if arguments is None else str(arguments))
    parts: list[str] = []
    hidden_keys = {"content", "old_text", "new_text", "text", "body", "input", "prompt"}
    for key, value in arguments.items():
        if isinstance(value, str):
            if key in hidden_keys:
                parts.append(f"{key} {len(value)} chars")
            else:
                parts.append(f"{key}={_one_line(value, limit=60)}")
        elif isinstance(value, list):
            parts.append(f"{key} {len(value)} items")
        elif isinstance(value, dict):
            parts.append(f"{key} {len(value)} keys")
        else:
            parts.append(f"{key}={value}")
    return _join_parts(*parts)


def _path_detail(args: dict[str, Any]) -> str:
    return _preview_arg(args, "path", limit=120)


def _preview_arg(
    args: dict[str, Any],
    key: str,
    *,
    label: str | None = None,
    limit: int = 80,
) -> str:
    value = args.get(key)
    if value in (None, ""):
        return ""
    prefix = f"{label or key}="
    return f"{prefix}{_one_line(str(value), limit=limit)}"


def _number_arg(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    return f"{key}={value}" if value is not None else ""


def _string_arg_len(args: dict[str, Any], key: str) -> int:
    value = args.get(key)
    return len(value) if isinstance(value, str) else 0


def _text_size_summary(text: str) -> str:
    return f"returned {len(text)} chars/{_line_count(text)} lines"


def _read_result_size(text: str) -> tuple[int, int]:
    content_lines: list[str] = []
    for line in text.splitlines():
        prefix, separator, rest = line.partition("| ")
        if separator and prefix.isdigit():
            content_lines.append(rest)
    if not content_lines:
        if text.startswith("(Empty file"):
            return 0, 0
        return len(text), _line_count(text)
    return sum(len(line) for line in content_lines), len(content_lines)


def _result_count_summary(text: str, *, noun: str) -> str:
    clean = text.strip()
    if not clean or clean == "(no matches)" or clean == "(no results)" or clean == "(no output)":
        return f"0 {noun}"
    return f"{_line_count(clean)} {noun}"


def _exit_code_summary(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("Exit code:"):
            return f"exit {stripped.removeprefix('Exit code:').strip()}"
    return "exit unknown"


def _web_status_summary(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Status:"):
            return f"status {stripped.removeprefix('Status:').strip()}"
    return ""


def _line_count(text: str) -> int:
    return len(text.splitlines()) if text else 0


def _join_parts(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


def _one_line(value: str, *, limit: int) -> str:
    text = " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} chars]"


def _short_time(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone().strftime("%H:%M:%S")


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
