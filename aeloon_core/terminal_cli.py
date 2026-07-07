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
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from aeloon_core.config import Config
from aeloon_core.orchestrator import AeloonCoreOrchestrator, TurnResult
from server.bridge import WebUITurnProgress

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
RESULT_PREVIEW_CHARS = 1_200
HISTORY_PREVIEW_CHARS = 160
PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})


class TerminalEventRenderer:
    """Render Web UI progress events as a compact terminal transcript."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_gateway_logs: bool = True,
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
        self.block_content_lengths: dict[str, int] = {}
        self.last_usage: dict[str, Any] = {}
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
        table.add_row("[bold cyan]Aeloon Core[/]", f"[cyan]{config.agents.defaults.model}[/]")
        table.add_row(str(config.workspace), f"session [bold]{session_id}[/]")
        table.add_row(str(config.data_dir), "gateway logs on" if self.show_gateway_logs else "")
        self.console.print(Panel(table, title="CLI", border_style="cyan", box=box.ROUNDED))

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
        self.console.print(Panel(prompt, title="You", border_style="magenta", expand=False))

    def print_turn_summary(self, result: TurnResult) -> None:
        """Print final session metadata for a turn."""

        self._finish_stream_line()
        tools = ", ".join(result.tools_used) if result.tools_used else "none"
        usage = _format_usage(self.last_usage)
        summary = f"session {result.session_id} | tools {tools}"
        if usage:
            summary = f"{summary} | {usage}"
        self.console.print(Text(summary, style="dim"))
        self.last_usage = {}

    def print_error(self, message: str) -> None:
        """Print an error without treating it as assistant output."""

        self._finish_stream_line()
        self.console.print(Panel(message, title="Error", border_style="red"))

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
            self.console.print(Rule(f"turn {payload.get('turn_id', '')}", style="cyan"))
            return
        if event == "chat.status":
            self._finish_stream_line()
            label = "tools" if payload.get("kind") == "tool_hint" else "status"
            self.console.print(f"[dim]{label}[/] {_preview(payload.get('text'), limit=240)}")
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
            usage = payload.get("usage")
            self.last_usage = usage if isinstance(usage, dict) else {}
            return
        if event == "chat.turn.end":
            self._finish_stream_line()

    def _render_block_add(self, payload: dict[str, Any]) -> None:
        block = payload.get("block") if isinstance(payload.get("block"), dict) else {}
        block_id = str(block.get("id") or "")
        block_type = str(block.get("type") or "")
        if block_id:
            self.block_types[block_id] = block_type
            if block.get("name"):
                self.block_names[block_id] = str(block.get("name"))
        if block_type == "tool_call":
            self._finish_stream_line()
            title = f"Tool {block.get('name') or 'call'}"
            body = _format_arguments(block.get("arguments"))
            self.console.print(Panel(body, title=title, border_style="yellow", expand=False))

    def _render_block_delta(self, payload: dict[str, Any]) -> None:
        block_id = str(payload.get("block_id") or "")
        if self.block_types.get(block_id) != "text":
            return
        delta = str(payload.get("delta") or "")
        if not delta:
            return
        self._append_assistant(delta)
        self.block_content_lengths[block_id] = self.block_content_lengths.get(block_id, 0) + len(
            delta
        )

    def _render_block_update(self, payload: dict[str, Any]) -> None:
        block_id = str(payload.get("block_id") or "")
        block_type = self.block_types.get(block_id)
        patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
        if block_type == "text" and "content" in patch:
            self._render_text_content(block_id, str(patch.get("content") or ""))
            return
        if block_type == "tool_call":
            self._render_tool_result(block_id, patch)

    def _render_text_content(self, block_id: str, content: str) -> None:
        printed = self.block_content_lengths.get(block_id, 0)
        if 0 < printed <= len(content):
            new_text = content[printed:]
        else:
            new_text = content
        if not new_text:
            return
        self._append_assistant(new_text)
        self.block_content_lengths[block_id] = len(content)

    def _render_tool_result(self, block_id: str, patch: dict[str, Any]) -> None:
        if "result" not in patch and "status" not in patch:
            return
        self._finish_stream_line()
        status = str(patch.get("status") or "done")
        name = self.block_names.get(block_id, "tool")
        style = "red" if status == "error" else "green"
        result = _preview(patch.get("result"), limit=RESULT_PREVIEW_CHARS)
        duration = patch.get("duration_ms")
        title = f"{name} -> {status}"
        if duration is not None:
            title = f"{title} ({duration} ms)"
        self.console.print(Panel(result or "(empty)", title=title, border_style=style))

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
            self.console.print("[bold cyan]Assistant[/]")
            self._assistant_streaming = True
        self.console.print(Text(text), end="")

    def _finish_stream_line(self) -> None:
        if not self._assistant_streaming:
            return
        self.console.print()
        self._assistant_streaming = False


class TerminalChatCli:
    """Interactive terminal runner."""

    def __init__(
        self,
        config: Config,
        *,
        session_id: str | None = None,
        show_gateway_logs: bool = True,
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
        progress = WebUITurnProgress(session_id=self.session_id, emit=self.renderer.emit)
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
    show_gateway_logs: bool = True,
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


def _format_arguments(value: Any) -> str | Syntax:
    if value is None:
        return "{}"
    if isinstance(value, dict | list):
        return Syntax(json.dumps(value, ensure_ascii=False, indent=2), "json", word_wrap=True)
    return str(value)


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


def _preview(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} chars]"


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
