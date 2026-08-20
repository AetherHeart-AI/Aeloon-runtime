"""Command-line interface for the Aeloon runtime."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from aeloon_runtime.bootstrap import CloudAccountGateway, create_runtime_service
from aeloon_runtime.config import (
    CloudProviderConfig,
    Config,
    DeepSeekProviderConfig,
    load_config,
    public_config,
    resolve_config_path,
    save_config,
)
from aeloon_runtime.core import (
    RunError,
    RunEvent,
    message_to_dict,
)
from aeloon_runtime.migration import migrate_workbench
from aeloon_runtime.rpc.protocol import RpcError
from aeloon_runtime.rpc.server import run_rpc_server
from aeloon_runtime.runtime import (
    JsonlSessionRepository,
    SessionError,
)
from aeloon_runtime.runtime.agent import SessionAgent
from aeloon_runtime.runtime.providers import (
    DeepSeekProvider,
    ProviderManager,
    qualify_model_id,
    resolve_model_id,
)
from aeloon_runtime.runtime.skill_runtime import run_bundled_skill
from aeloon_runtime.gateway_ws import build_tls_context, parse_listen
from aeloon_runtime.runtime_server_v3 import serve_v3
from aeloon_runtime.version import __version__

CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    "workspace": ("workspace",),
    "data-dir": ("data_dir",),
    "api-key": ("providers", "deepseek", "api_key"),
    "endpoint": ("providers", "deepseek", "endpoint"),
    "proxy": ("providers", "deepseek", "proxy"),
    "model": ("agent", "model"),
    "thinking": ("agent", "thinking_level"),
    "max-tokens": ("agent", "max_tokens"),
    "temperature": ("agent", "temperature"),
    "timeout-ms": ("agent", "timeout_ms"),
    "steering-mode": ("agent", "steering_mode"),
    "follow-up-mode": ("agent", "follow_up_mode"),
    "retry-enabled": ("agent", "retry", "enabled"),
    "max-retries": ("agent", "retry", "max_retries"),
    "base-delay-ms": ("agent", "retry", "base_delay_ms"),
    "max-retry-delay-ms": ("agent", "retry", "max_retry_delay_ms"),
    "compaction-enabled": ("agent", "compaction", "enabled"),
    "reserve-tokens": ("agent", "compaction", "reserve_tokens"),
    "keep-recent-tokens": ("agent", "compaction", "keep_recent_tokens"),
    "shell-path": ("tools", "shell_path"),
    "auto-resize-images": ("tools", "auto_resize_images"),
    "resource-roots": ("resources", "roots"),
    "no-skills": ("resources", "no_skills"),
    "no-prompt-templates": ("resources", "no_prompt_templates"),
    "no-context-files": ("resources", "no_context_files"),
}

_TOOL_SUMMARY_MAX_CHARS = 240
_TASK_COMMAND = "__task__"

_KNOWN_COMMANDS = {
    "resume",
    "history",
    "login",
    "logout",
    "whoami",
    "models",
    "doctor",
    "completion",
    "config",
    "provider",
    "system",
    "rpc",
    "serve",
    "migrate",
}


def _add_run_arguments(
    command: argparse.ArgumentParser,
    *,
    allow_session: bool,
) -> None:
    command.add_argument("prompt", nargs="*", help="Task to give the coding agent.")
    source = command.add_mutually_exclusive_group()
    source.add_argument("--prompt-file", "--file", type=Path, help="Read a UTF-8 task file.")
    source.add_argument("--stdin", action="store_true", help=argparse.SUPPRESS)
    command.add_argument(
        "--output",
        choices=("text", "json", "stream-json"),
        default="text",
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output",
        help="Print one machine-readable JSON result.",
    )
    command.add_argument(
        "--stream",
        action="store_const",
        const="stream-json",
        dest="output",
        help=argparse.SUPPRESS,
    )
    detail = command.add_mutually_exclusive_group()
    detail.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show concise tool activity on stderr; repeat for lifecycle events.",
    )
    detail.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show only the final response.",
    )
    if allow_session:
        command.add_argument("--session", help=argparse.SUPPRESS)
        command.add_argument(
            "--no-session",
            "--ephemeral",
            action="store_true",
            help="Do not save this task to history.",
        )
    else:
        command.add_argument("--session", help="Resume this session instead of the latest one.")
        command.set_defaults(no_session=False)
    command.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    command.add_argument("-C", "--workspace", type=Path, help="Run in this workspace.")
    command.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    command.add_argument("--session-dir", type=Path, help=argparse.SUPPRESS)
    command.add_argument("-m", "--model", help="Use this model for the task.")
    command.add_argument(
        "--effort",
        choices=("off", "minimal", "low", "medium", "high", "max"),
        help="Set the model reasoning effort for this task.",
    )


def _add_account_arguments(command: argparse.ArgumentParser, *, login: bool = False) -> None:
    if login:
        command.add_argument("username", nargs="?", help="Account username or email.")
    command.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    command.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    command.add_argument(
        "--json", action="store_const", const="json", dest="output", help="Print JSON output."
    )
    command.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help=argparse.SUPPRESS,
    )


def _add_provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("provider_id", help="Short name used in provider/model ids.")
    command.add_argument("--name", help="Human-readable provider name.")
    command.add_argument("--endpoint", required=True, help="Custom API base URL.")
    command.add_argument(
        "--backend",
        choices=("openai", "llamacpp", "ollama", "vllm"),
        default="openai",
        help="Backend metadata dialect; inference remains OpenAI-compatible.",
    )
    command.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Optional discovered-model allowlist; repeat for multiple models.",
    )
    command.add_argument("--api-key", help="Optional API key.")
    command.add_argument("--proxy", help="Optional HTTP proxy.")
    command.add_argument(
        "--header",
        action="append",
        dest="headers",
        metavar="NAME=VALUE",
        help="Additional HTTP header; repeat for multiple headers.",
    )


def _add_provider_runtime_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    command.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    command.add_argument(
        "--json", action="store_const", const="json", dest="output", help="Print JSON output."
    )
    command.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help=argparse.SUPPRESS,
    )


def _add_rpc_commands(parent: argparse.ArgumentParser) -> None:
    commands = parent.add_subparsers(dest="rpc_command", required=True, metavar="COMMAND")
    serve = commands.add_parser("serve", help="Run the Electron-owned Core RPC process.")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--data-dir", type=Path)
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument(
        "--listen",
        help=(
            "Also serve aeloon-rpc over WebSocket at HOST:PORT. Loopback only "
            "until device pairing lands; the Unix socket is always served."
        ),
    )
    serve.add_argument("--tls-cert", type=Path, help="TLS certificate for --listen.")
    serve.add_argument("--tls-key", type=Path, help="TLS private key for --listen.")
    serve.add_argument("--max-concurrent-operations", type=int, default=4)


def _add_runtime_serve_command(commands: Any) -> None:
    serve = commands.add_parser("serve", help="Run the standalone aeloon Runtime.")
    serve.add_argument("--unix", type=Path, required=True, help="Unix socket path.")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--data-dir", type=Path)
    serve.add_argument("--workspace-root", type=Path, action="append")
    serve.add_argument(
        "--no-workspace-root",
        action="store_true",
        help="Start with no authorized workspace roots (desktop before folder pick).",
    )
    serve.add_argument(
        "--record-trace",
        type=Path,
        help="Opt in to a redacted JSONL boundary trace under DIRECTORY.",
    )
    serve.add_argument(
        "--listen",
        help=(
            "Also serve aeloon-rpc over WebSocket at HOST:PORT. Loopback only "
            "until device pairing lands; the Unix socket is always served."
        ),
    )
    serve.add_argument("--tls-cert", type=Path, help="TLS certificate for --listen.")
    serve.add_argument("--tls-key", type=Path, help="TLS private key for --listen.")
    serve.add_argument("--max-concurrent-operations", type=int, default=4)


def _add_runtime_migrate_command(commands: Any) -> None:
    migrate = commands.add_parser(
        "migrate", help="Migrate Workbench/Core data into Runtime storage."
    )
    migrate.add_argument("--from-workbench", type=Path, required=True)
    migrate.add_argument("--from-core", type=Path, required=True)
    migrate.add_argument("--data-dir", type=Path, required=True)
    migrate.add_argument("--roots-output", type=Path, required=True)


def _hide_internal_subcommands(commands: Any) -> None:
    """Hide implementation-only commands from top-level help."""

    commands._choices_actions = [
        action for action in commands._choices_actions if action.help != argparse.SUPPRESS
    ]


def build_parser() -> argparse.ArgumentParser:
    runtime_entrypoint = Path(sys.argv[0]).name == "aeloon-runtime" or os.environ.get(
        "AELOON_RUNTIME_MODE"
    ) == "1"
    program_name = "aeloon-runtime" if runtime_entrypoint else "aeloon-runtime"
    parser = argparse.ArgumentParser(
        prog=program_name,
        description="A coding agent for the current workspace.",
        epilog=(
            "examples:\n"
            f'  {program_name} "fix the failing tests"\n'
            f"  {program_name} resume \"continue with the implementation\"\n"
            f"  {program_name} provider add studio --endpoint http://127.0.0.1:8000\n"
            f"  {program_name} login\n"
            f"  {program_name} doctor\n\n"
            "task options: -C PATH, -m MODEL, --json, -v, --ephemeral\n"
            f"Use `{program_name} -- TASK` when TASK starts with a command name.\n"
            f"Run `{program_name} COMMAND --help` for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {cli_version()}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = commands.add_parser(_TASK_COMMAND, help=argparse.SUPPRESS)
    _add_run_arguments(run, allow_session=True)

    resume = commands.add_parser("resume", help="Continue the latest task in this workspace.")
    _add_run_arguments(resume, allow_session=False)

    history = commands.add_parser("history", help="Show recent tasks and sessions.")
    history.add_argument("session_id", nargs="?", help="Show one session in detail.")
    history.add_argument("--all", action="store_true", help="Include every workspace.")
    history.add_argument("-C", "--workspace", type=Path, help="Filter by workspace.")
    history.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    history.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    history.add_argument("--json", action="store_true", help="Print JSON output.")

    login = commands.add_parser("login", help="Sign in to Aeloon Cloud.")
    _add_account_arguments(login, login=True)
    logout = commands.add_parser("logout", help=argparse.SUPPRESS)
    _add_account_arguments(logout)
    whoami = commands.add_parser("whoami", help=argparse.SUPPRESS)
    _add_account_arguments(whoami)

    models = commands.add_parser(
        "models",
        help="List models or choose the default.",
        description="Run without a subcommand to list connected models.",
    )
    models.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    models.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    models.add_argument("--json", action="store_true", help="Print JSON output.")
    models.set_defaults(models_command="list")
    model_commands = models.add_subparsers(dest="models_command", metavar="COMMAND")
    model_use = model_commands.add_parser("use", help="Set the default model.")
    model_use.add_argument(
        "model_id", help="Provider-qualified model id from `aeloon-runtime models`."
    )
    model_use.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    model_use.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    model_use.add_argument("--json", action="store_true", help="Print JSON output.")

    doctor = commands.add_parser("doctor", help="Check configuration and show suggested fixes.")
    doctor.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    doctor.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    doctor.add_argument("--json", action="store_true", help="Print JSON output.")

    completion = commands.add_parser("completion", help=argparse.SUPPRESS)
    completion.add_argument("shell", choices=("bash", "zsh", "fish"))

    config = commands.add_parser("config", help="Manage persistent configuration.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_path = config_commands.add_parser("path", help="Print the config path.")
    config_path.add_argument("--config", type=Path)
    config_show = config_commands.add_parser("show", help="Print the effective config.")
    config_show.add_argument("--config", type=Path)
    config_show.add_argument("--show-secrets", action="store_true")
    config_init = config_commands.add_parser("init", help="Create a config file.")
    config_init.add_argument("--config", type=Path)
    config_init.add_argument("--workspace", type=Path)
    config_init.add_argument("--data-dir", type=Path)
    config_init.add_argument("--force", action="store_true")
    config_set = config_commands.add_parser("set", help="Set one config value.")
    config_set.add_argument("key", choices=sorted(CONFIG_PATHS))
    config_set.add_argument("value")
    config_set.add_argument("--config", type=Path)

    rpc = commands.add_parser("rpc", help=argparse.SUPPRESS)
    _add_rpc_commands(rpc)
    _add_runtime_serve_command(commands)
    _add_runtime_migrate_command(commands)

    system = commands.add_parser("system", help=argparse.SUPPRESS)
    system_commands = system.add_subparsers(dest="system_command", required=True, metavar="COMMAND")
    system_skill = system_commands.add_parser(
        "skill", help="Run one trusted bundled Skill entry point."
    )
    system_skill.add_argument("skill_id")
    system_skill.add_argument("skill_action")
    system_skill.add_argument("skill_arguments", nargs=argparse.REMAINDER)

    provider = commands.add_parser("provider", help="Manage inference Providers.")
    provider_commands = provider.add_subparsers(dest="provider_command", required=True)
    provider_add = provider_commands.add_parser("add", help="Add an inference Provider.")
    _add_provider_arguments(provider_add)
    provider_remove = provider_commands.add_parser("remove", help="Remove an inference Provider.")
    provider_remove.add_argument("provider_id")
    for command_name in ("list", "add", "remove"):
        if command_name == "add":
            command = provider_add
        elif command_name == "remove":
            command = provider_remove
        else:
            command = provider_commands.add_parser(
                command_name,
                help={
                    "list": "List configured API providers.",
                }[command_name],
            )
        _add_provider_runtime_arguments(command)
    _hide_internal_subcommands(commands)
    # Set this after child parsers are created so their own usage remains concise.
    parser.usage = f"{program_name} [TASK...] | {program_name} COMMAND ..."
    return parser


class RunRenderer:
    """Render events while collecting a stable final JSON result."""

    def __init__(self, output: str, *, verbose: bool | int = False, quiet: bool = False) -> None:
        self.output = output
        self.verbose = int(verbose)
        self.quiet = quiet
        self.tools_used: list[str] = []
        self._tool_started_at: dict[str, float] = {}
        self._tool_args: dict[str, dict[str, Any]] = {}
        self._status_visible = False

    async def __call__(self, event: RunEvent) -> None:
        if (
            self.verbose >= 2
            and self.output == "text"
            and event.type
            in {
                "agent_start",
                "agent_end",
                "auto_retry_start",
                "auto_retry_end",
                "compaction_start",
                "compaction_end",
            }
        ):
            print(f"[debug] {event.type}", file=sys.stderr)
        if event.type == "agent_start":
            self._render_status("Working…")
        elif event.type == "tool_execution_start":
            name = str(event.data.get("toolName") or "")
            call_id = str(event.data.get("toolCallId") or "")
            if name and name not in self.tools_used:
                self.tools_used.append(name)
            if call_id and self.verbose and self.output == "text":
                self._tool_started_at[call_id] = time.monotonic()
                self._tool_args[call_id] = (
                    dict(event.data["args"]) if isinstance(event.data.get("args"), dict) else {}
                )
            if self.verbose and self.output == "text":
                self._clear_status()
                self._render_tool_start(name, event.data.get("args"))
                if self.verbose >= 2 and call_id:
                    print(f"[debug] tool call {call_id}", file=sys.stderr)
            elif self.output == "text":
                action, _ = _tool_invocation_summary(
                    name,
                    event.data.get("args") if isinstance(event.data.get("args"), dict) else {},
                )
                statuses = {
                    "run": "Running a command…",
                    "read": "Reading files…",
                    "write": "Updating files…",
                    "search": "Searching the workspace…",
                }
                self._render_status(statuses.get(action, "Working…"))
        elif event.type == "tool_execution_end" and self.verbose and self.output == "text":
            self._render_tool_end(event)
        elif event.type in {"agent_end", "abort", "settled"}:
            self._clear_status()
        if self.output == "stream-json":
            print(_json(event.to_dict()), flush=True)
            return
        # Text output is intentionally buffered until the complete assistant message is
        # available. Rendering partial Markdown produces noisy, malformed terminal output.

    def _render_tool_start(self, name: str, raw_args: Any) -> None:
        args = raw_args if isinstance(raw_args, dict) else {}
        action, summary = _tool_invocation_summary(name, args)
        print(f"[{action}] {summary}", file=sys.stderr)

    def _render_status(self, value: str) -> None:
        if self.output != "text" or self.quiet or self.verbose or not _is_interactive_stderr():
            return
        summary = _one_line_summary(value, max_chars=100)
        print(f"\r\x1b[2K{summary}", end="", file=sys.stderr, flush=True)
        self._status_visible = True

    def _clear_status(self) -> None:
        if not self._status_visible:
            return
        print("\r\x1b[2K", end="", file=sys.stderr, flush=True)
        self._status_visible = False

    def _render_tool_end(self, event: RunEvent) -> None:
        name = str(event.data.get("toolName") or "unknown")
        call_id = str(event.data.get("toolCallId") or "")
        raw_result = event.data.get("result")
        result = raw_result if isinstance(raw_result, dict) else {}
        details_value = result.get("details")
        details = details_value if isinstance(details_value, dict) else {}
        args = self._tool_args.pop(call_id, {})
        exit_code = details.get("exitCode")
        is_error = bool(
            event.data.get("isError")
            or result.get("isError")
            or (name == "bash" and isinstance(exit_code, int) and exit_code != 0)
        )
        qualifiers = _tool_result_summary(name, args, result, details, is_error=is_error)
        started_at = self._tool_started_at.pop(call_id, None)
        if started_at is not None:
            qualifiers.append(_format_duration(time.monotonic() - started_at))
        truncation = details.get("truncation")
        if details.get("truncated") or (
            isinstance(truncation, dict) and truncation.get("truncated")
        ):
            qualifiers.append("truncated")
        label = "failed" if is_error else "ok"
        suffix = f" · {' · '.join(qualifiers)}" if qualifiers else ""
        print(f"[{label}] {name}{suffix}", file=sys.stderr)

    def finish_text(self, final_text: str) -> None:
        self._clear_status()
        if self.output != "text":
            return
        if not final_text:
            print()
            return
        if _is_interactive_terminal():
            Console(file=sys.stdout, highlight=False, soft_wrap=True).print(Markdown(final_text))
        else:
            print(final_text)


def _tool_invocation_summary(name: str, args: dict[str, Any]) -> tuple[str, str]:
    if name == "bash":
        command = _one_line_summary(str(args.get("command") or ""))
        summary = f"$ {command or '(empty command)'}"
        if args.get("timeout") is not None:
            summary += f" · timeout {args['timeout']}s"
        return "run", summary
    if name == "write":
        content = str(args.get("content") or "")
        return "write", f"{args.get('path', '')} · {_format_bytes(len(content.encode('utf-8')))}"
    if name == "edit":
        edits = args.get("edits")
        count = len(edits) if isinstance(edits, list) else 0
        return "write", f"{args.get('path', '')} · {_count_label(count, 'replacement')}"
    if name == "read":
        line_range = ""
        if args.get("offset") is not None or args.get("limit") is not None:
            offset = int(args.get("offset") or 1)
            limit = args.get("limit")
            line_range = (
                f" · lines {offset}-{offset + int(limit) - 1}"
                if limit
                else f" · from line {offset}"
            )
        return "read", f"{args.get('path', '')}{line_range}"
    if name == "grep":
        return "search", _search_summary(args, noun="matches")
    if name == "find":
        return "search", _search_summary(args, noun="files")
    if name == "ls":
        return "read", str(args.get("path") or ".")
    return "run", name or "unknown"


def _search_summary(args: dict[str, Any], *, noun: str) -> str:
    pattern = repr(str(args.get("pattern") or ""))
    path = str(args.get("path") or ".")
    return _one_line_summary(f"{pattern} in {path} ({noun})")


def _tool_result_summary(
    name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    details: dict[str, Any],
    *,
    is_error: bool,
) -> list[str]:
    if name == "bash" and isinstance(details.get("exitCode"), int):
        summary: list[str] = []
        summary.append(f"exit {details['exitCode']}")
        output_bytes = details.get("outputBytes")
        if isinstance(output_bytes, int):
            summary.append(f"{_format_bytes(output_bytes)} output")
        return summary
    if is_error:
        error = _one_line_summary(_tool_result_text(result))
        return [error] if error else []
    if name == "read":
        summary = []
        size = details.get("sizeBytes")
        if isinstance(size, int):
            summary.append(_format_bytes(size))
        selected_lines = details.get("selectedLines")
        if isinstance(selected_lines, int):
            summary.append(_count_label(selected_lines, "line"))
        mime = details.get("mimeType")
        if mime:
            summary.append(str(mime))
        return summary
    if name == "write":
        size = details.get("sizeBytes")
        if not isinstance(size, int):
            size = len(str(args.get("content") or "").encode("utf-8"))
        return [_format_bytes(size)]
    if name == "edit":
        count = details.get("replacements")
        if not isinstance(count, int):
            edits = args.get("edits")
            count = len(edits) if isinstance(edits, list) else 0
        summary = [_count_label(count, "replacement")]
        size = details.get("sizeAfterBytes")
        if isinstance(size, int):
            summary.append(_format_bytes(size))
        return summary
    if name in {"grep", "find", "ls"}:
        count = details.get("resultCount")
        noun = {"grep": "result", "find": "file", "ls": "entry"}[name]
        return [_count_label(count, noun)] if isinstance(count, int) else []
    return []


def _tool_result_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    rendered: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            rendered.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            rendered.append(f"[image attachment: {block.get('mimeType') or 'unknown type'}]")
    return "\n".join(rendered)


def _one_line_summary(value: str, *, max_chars: int = _TOOL_SUMMARY_MAX_CHARS) -> str:
    rendered = " ".join(value.split())
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 1].rstrip() + "…"


def _format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(max(0, size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _is_interactive_terminal() -> bool:
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


def _is_interactive_stderr() -> bool:
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def _stdin_is_interactive() -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    return bool(isatty and isatty())


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1_000)}ms"
    return f"{seconds:.2f}s"


async def run_command(args: argparse.Namespace) -> int:
    if args.session and args.no_session:
        raise RunError("invalid_argument", "--session and --no-session cannot be combined")
    prompt = _read_prompt(args)
    config = _with_run_overrides(load_config(args.config), args)
    session_dir = args.session_dir or config.data_dir
    repository = None if args.no_session else JsonlSessionRepository(session_dir)
    session = None
    if args.session:
        assert repository is not None
        session = await repository.open(args.session)
        if args.workspace is None:
            config = config.model_copy(
                update={"workspace": Path(session.metadata.cwd)}
            ).normalized()
        restored_model = (await session.build_context()).model
        if restored_model is not None and args.model is None:
            restored_id = restored_model[1]
            if "/" not in restored_id:
                restored_id = qualify_model_id(restored_model[0], restored_id)
            config = config.model_copy(
                update={"agent": config.agent.model_copy(update={"model": restored_id})}
            )
    cloud_account = CloudAccountGateway(
        _cloud_provider_config(config),
        data_dir=config.data_dir,
    )
    manager = _provider_manager(config, cloud_account)
    try:
        if config.agent.model.strip():
            model = await manager.model(config.agent.model)
        else:
            available = _available_cli_models(config, await manager.models())
            if not available:
                raise RunError(
                    "model_not_configured",
                    "No connected model is available",
                )
            model = available[0]
            config = config.model_copy(
                update={"agent": config.agent.model_copy(update={"model": model.id})}
            )
    except PermissionError as exc:
        await manager.close()
        await cloud_account.close()
        raise RunError("auth", str(exc)) from None
    except RunError:
        await manager.close()
        await cloud_account.close()
        raise
    except (KeyError, RuntimeError) as exc:
        await manager.close()
        await cloud_account.close()
        raise RunError("model_not_found", str(exc)) from None
    if session is None and not args.no_session:
        assert repository is not None
        session = await repository.create(cwd=config.workspace)

    agent = SessionAgent(
        config=config,
        session=session,
        provider_manager=manager,
    )
    renderer = RunRenderer(
        args.output,
        verbose=args.verbose,
        quiet=bool(getattr(args, "quiet", False)),
    )
    agent.subscribe(renderer)
    started = time.monotonic()
    try:
        run_result = await agent.prompt(prompt, run_id=f"cli-{time.time_ns()}")
        message = run_result.final_message
    finally:
        renderer._clear_status()
        await agent.close()
        await cloud_account.close()
    result = {
        "type": "result",
        "status": _status(message.stop_reason),
        "session_id": session.id if session else None,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "final_content": message.text,
        "message": message_to_dict(message),
        "usage": message.usage.to_dict(),
        "tools_used": renderer.tools_used,
        "model": message.model,
        "workspace": str(config.workspace),
    }
    renderer.finish_text(message.text)
    if args.output in {"json", "stream-json"}:
        print(_json(result), flush=True)
    if message.error_message:
        print(message.error_message, file=sys.stderr)
    return 0 if message.stop_reason not in {"error", "aborted"} else 1


async def resume_command(args: argparse.Namespace) -> int:
    """Resume an explicit session or the newest session for the active workspace."""

    if not args.prompt and not args.prompt_file and not args.stdin:
        if _stdin_is_interactive():
            args.prompt = ["Continue the previous task."]
        else:
            args.stdin = True
    if args.session is None:
        config = _with_run_overrides(load_config(args.config), args)
        repository = JsonlSessionRepository(args.session_dir or config.data_dir)
        sessions = await repository.list(cwd=config.workspace)
        if not sessions:
            raise SessionError(
                "not_found",
                f"No saved task was found for {config.workspace}",
            )
        args.session = max(sessions, key=_session_mtime).id
    return await run_command(args)


async def history_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    data_dir = args.data_dir or config.data_dir
    repository = JsonlSessionRepository(data_dir)
    if args.session_id:
        session_id = await _resolve_session_id(repository, args.session_id)
        session = await repository.open(session_id)
        context = await session.build_context()
        result = {
            "id": session.id,
            "created_at": session.metadata.created_at,
            "workspace": session.metadata.cwd,
            "name": await session.get_name(),
            "messages": [message_to_dict(message) for message in context.messages],
            "stats": await session.stats(),
        }
        if args.json:
            print(_json(result))
        else:
            title = result["name"] or _session_summary(context.messages) or "Untitled task"
            stats = result["stats"]
            print(title)
            print(f"Session: {session.id}")
            print(f"Workspace: {session.metadata.cwd}")
            print(f"Messages: {stats['messageCount']} · Tokens: {stats['totalTokens']}")
        return 0

    workspace = None if args.all else (args.workspace or config.workspace)
    sessions = await repository.list(cwd=workspace)
    sessions.sort(key=_session_mtime, reverse=True)
    payload: list[dict[str, Any]] = []
    for metadata in sessions:
        session = await repository.open(metadata.id)
        context = await session.build_context()
        payload.append(
            {
                "id": metadata.id,
                "created_at": metadata.created_at,
                "updated_at": datetime.fromtimestamp(_session_mtime(metadata))
                .astimezone()
                .isoformat(),
                "workspace": metadata.cwd,
                "summary": await session.get_name()
                or _session_summary(context.messages)
                or "Untitled task",
            }
        )
    if args.json:
        print(_json(payload))
        return 0
    if not payload:
        scope = "any workspace" if args.all else str(workspace)
        print(f"No saved tasks found for {scope}.")
        return 0
    _print_table(
        ("UPDATED", "WORKSPACE", "SUMMARY", "SESSION"),
        [
            (
                _format_history_time(str(item["updated_at"])),
                Path(str(item["workspace"])).name or str(item["workspace"]),
                _one_line_summary(str(item["summary"]), max_chars=48),
                str(item["id"])[:8],
            )
            for item in payload
        ],
    )
    return 0


async def _resolve_session_id(repository: JsonlSessionRepository, value: str) -> str:
    sessions = await repository.list()
    matches = [item.id for item in sessions if item.id.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SessionError("not_found", f"Session {value} not found")
    raise SessionError("invalid_argument", f"Session prefix {value} is ambiguous")


def _format_history_time(value: str) -> str:
    return value[:16].replace("T", " ") if len(value) >= 16 else value


def _session_mtime(metadata: Any) -> float:
    try:
        return metadata.path.stat().st_mtime
    except OSError:
        return 0.0


def _session_summary(messages: Any) -> str:
    for message in messages:
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return _one_line_summary(content, max_chars=80)
        text = " ".join(str(getattr(part, "text", "")) for part in content)
        if text.strip():
            return _one_line_summary(text, max_chars=80)
    return ""


async def models_command(args: argparse.Namespace) -> int:
    if args.models_command == "use":
        return await model_use_command(args)
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir}).normalized()
    account = CloudAccountGateway(_cloud_provider_config(config), data_dir=config.data_dir)
    manager = _provider_manager(config, account)
    try:
        models = _available_cli_models(config, await manager.models())
    finally:
        await manager.close()
        await account.close()
    effective_id = models[0].id if models else ""
    if config.agent.model:
        try:
            effective_id = resolve_model_id(config.agent.model, (model.id for model in models))
        except KeyError:
            effective_id = config.agent.model
    payload = [
        {
            "id": model.id,
            "name": model.name,
            "provider": model.provider,
            "context_window": model.context_window,
            "selected": model.id == effective_id,
            "automatic": not config.agent.model and model.id == effective_id,
        }
        for model in models
    ]
    if args.json:
        print(_json(payload))
        return 0
    if not payload:
        print("No models are connected. Run `aeloon-runtime provider add ...` or `aeloon-runtime login`.")
        return 0
    _print_table(
        ("", "MODEL", "PROVIDER", "CONTEXT"),
        [
            (
                "*" if item["selected"] else "",
                str(item["id"]),
                str(item["provider"]),
                f"{int(item['context_window']):,}",
            )
            for item in payload
        ],
    )
    if payload[0]["automatic"]:
        print(
            "* automatically used when no default is set; pin one with "
            "`aeloon-runtime models use MODEL`."
        )
    return 0


def _available_cli_models(config: Config, models: dict[str, Any]) -> list[Any]:
    deepseek = config.providers["deepseek"]
    assert isinstance(deepseek, DeepSeekProviderConfig)
    return [
        model for model in models.values() if model.provider != "deepseek" or bool(deepseek.api_key)
    ]


def _cloud_provider_config(config: Config) -> CloudProviderConfig:
    provider = config.providers["aeloon-cloud"]
    assert isinstance(provider, CloudProviderConfig)
    return provider


def _provider_manager(config: Config, account: Any) -> ProviderManager:
    def create_deepseek(_provider_id: str, configured: Any, _account: Any):
        return DeepSeekProvider(
            name=configured.name,
            endpoint=configured.endpoint,
            api_key=configured.api_key,
            proxy=configured.proxy,
            headers=configured.headers,
            enabled=configured.enabled,
        )

    return ProviderManager(
        config,
        account=account,
        driver_factories={"deepseek": create_deepseek},
    )


async def model_use_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir}).normalized()
    runtime = create_runtime_service(
        config_path=args.config,
        data_dir=args.data_dir,
    )
    try:
        catalog = await runtime.catalog_get({})
        available = [
            str(item.get("id"))
            for item in catalog.get("models") or []
            if isinstance(item, dict)
            and item.get("id")
            and not (
                item.get("provider_id") == "deepseek"
                and not getattr(config.providers["deepseek"], "api_key", None)
            )
        ]
        requested = args.model_id.strip()
        try:
            candidate = resolve_model_id(requested, available)
        except KeyError:
            raise RunError(
                "model_not_found",
                f"Model is not available: {requested}",
            ) from None
        settings = await runtime.settings_get({})
        result = await runtime.settings_update(
            {
                "revision": settings["revision"],
                "patch": {"default_model_id": candidate},
            }
        )
    finally:
        await runtime.close()
    if args.json:
        print(_json(result))
    else:
        print(f"Default model set to {candidate}.")
    return 0


async def doctor_command(args: argparse.Namespace) -> int:
    path = resolve_config_path(args.config)
    config = load_config(path)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir}).normalized()
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, message: str, fix: str = "") -> None:
        checks.append({"name": name, "status": status, "message": message, "fix": fix})

    if path.is_file():
        add("config", "ok", str(path))
    else:
        add(
            "config",
            "warning",
            "Using built-in defaults",
            "Connect an API with `aeloon-runtime provider add ...` or run `aeloon-runtime login`.",
        )
    if config.workspace.is_dir():
        add("workspace", "ok", str(config.workspace))
    else:
        add(
            "workspace",
            "error",
            f"Directory does not exist: {config.workspace}",
            "Run Aeloon in a project directory or pass `-C PATH`.",
        )

    account = CloudAccountGateway(_cloud_provider_config(config), data_dir=config.data_dir)
    manager = _provider_manager(config, account)
    try:
        if not config.agent.model.strip():
            available = _available_cli_models(config, await manager.models())
            if available:
                add(
                    "model",
                    "warning",
                    f"No default is set; runs automatically use {available[0].id}",
                    "Pin it with `aeloon-runtime models use MODEL` if desired.",
                )
                add("credential", "ok", "Credentials are configured")
            else:
                add(
                    "model",
                    "error",
                    "No connected model is available",
                    "Connect with `aeloon-runtime provider add ...` or `aeloon-runtime login`.",
                )
        else:
            try:
                model = await manager.model(config.agent.model)
            except (KeyError, RuntimeError, PermissionError) as exc:
                add("model", "error", str(exc), "Run `aeloon-runtime models` to choose another model.")
            else:
                add("model", "ok", f"{model.id} ({model.name})")
                if model.provider == "deepseek" and not getattr(
                    config.providers["deepseek"], "api_key", None
                ):
                    add(
                        "credential",
                        "error",
                        "DeepSeek API key is not configured",
                        "Add an API with `aeloon-runtime provider add ...` or log in with "
                        "`aeloon-runtime login`.",
                    )
                else:
                    add("credential", "ok", "Credentials are configured")
    finally:
        await manager.close()
        await account.close()

    add(
        "desktop-runtime",
        "ok",
        "Core is started and supervised by Electron when the desktop app runs",
    )
    if args.json:
        print(
            _json(
                {
                    "ok": not any(item["status"] == "error" for item in checks),
                    "checks": checks,
                }
            )
        )
    else:
        symbols = {"ok": "✓", "warning": "!", "error": "✗"}
        for item in checks:
            print(f"{symbols[item['status']]} {item['name']}: {item['message']}")
            if item["fix"]:
                print(f"  {item['fix']}")
    return 1 if any(item["status"] == "error" for item in checks) else 0


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    if _is_interactive_terminal():
        table = Table(show_header=True, header_style="bold", box=None)
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        Console(file=sys.stdout, highlight=False).print(table)
        return
    print("\t".join(headers))
    for row in rows:
        print("\t".join(row))


def completion_command(args: argparse.Namespace) -> int:
    commands = "resume history login logout whoami models doctor config provider system"
    scripts = {
        "bash": (
            "_aeloon_runtime_complete() {\n"
            "  provider current\n"
            '  current="${COMP_WORDS[COMP_CWORD]}"\n'
            f'  COMPREPLY=($(compgen -W "{commands}" -- "$current"))\n'
            "}\n"
            "complete -F _aeloon_runtime_complete aeloon-runtime"
        ),
        "zsh": (f"#compdef aeloon-runtime\n_arguments '1:command:({commands})' '*::argument:->args'"),
        "fish": "\n".join(
            f"complete -c aeloon-runtime -f -n '__fish_use_subcommand' -a {command}"
            for command in commands.split()
        ),
    }
    print(scripts[args.shell])
    return 0


def config_command(args: argparse.Namespace) -> int:
    path = resolve_config_path(args.config)
    if args.config_command == "path":
        print(path)
        return 0
    if args.config_command == "show":
        print(_json(public_config(load_config(path), show_secrets=args.show_secrets)))
        return 0
    if args.config_command == "init":
        config = Config()
        updates: dict[str, Any] = {}
        if args.workspace is not None:
            updates["workspace"] = args.workspace
        if args.data_dir is not None:
            updates["data_dir"] = args.data_dir
        if updates:
            config = config.model_copy(update=updates)
        print(save_config(config.normalized(), path, force=args.force))
        return 0
    config = load_config(path)
    raw = config.model_dump(mode="json")
    _set_nested(raw, CONFIG_PATHS[args.key], _parse_value(args.value))
    validated = Config.model_validate(raw).normalized()
    print(save_config(validated, path))
    return 0


async def rpc_command(args: argparse.Namespace) -> int:
    runtime = create_runtime_service(
        config_path=args.config,
        data_dir=args.data_dir,
        max_concurrent_operations=args.max_concurrent_operations,
    )
    await run_rpc_server(runtime, socket_path=args.socket)
    return 0


async def cloud_command(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.cloud_command == "login":
        username = (args.username or _read_cloud_username()).strip()
        if not username:
            raise ValueError("Aeloon Cloud username is required")
        password = _read_cloud_password()
        params = {"username": username, "password": password}
    runtime = create_runtime_service(
        config_path=args.config,
        data_dir=args.data_dir,
    )
    try:
        handlers = {
            "status": runtime.account_status,
            "login": runtime.account_login,
            "logout": runtime.account_logout,
        }
        result = await handlers[args.cloud_command](params)
    finally:
        await runtime.close()
    _print_cloud_result(args.cloud_command, result, output=args.output)
    return 0


def _read_cloud_username() -> str:
    print("Aeloon Cloud username: ", end="", file=sys.stderr, flush=True)
    value = sys.stdin.readline()
    if not value:
        raise ValueError("Aeloon Cloud username is required") from None
    return value


def _read_cloud_password() -> str:
    try:
        password = getpass.getpass("Aeloon Cloud password: ")
    except EOFError:
        raise ValueError("Aeloon Cloud password must be entered from a terminal") from None
    if not password:
        raise ValueError("Aeloon Cloud password is required")
    return password


def _print_cloud_result(command: str, result: dict[str, Any], *, output: str) -> None:
    if output == "json":
        print(_json(result))
        return
    if command == "logout":
        print("Signed out of Aeloon Cloud.")
        return
    user = result.get("user")
    if not result.get("authenticated") or not isinstance(user, dict):
        print("Not signed in to Aeloon Cloud.")
        return
    username = str(user.get("username") or "").strip()
    display_name = str(user.get("display_name") or username or "Aeloon user").strip()
    identity = f" (@{username})" if username and username != display_name else ""
    print(f"Signed in to Aeloon Cloud as {display_name}{identity}.")


async def provider_command(args: argparse.Namespace) -> int:
    command = args.provider_command
    params: dict[str, Any] = {}
    if command == "list":
        method = "list"
    elif command == "add":
        params = {
            "provider_id": args.provider_id,
            "driver": "custom",
            "backend": args.backend,
            "name": args.name or args.provider_id,
            "endpoint": args.endpoint,
        }
        if args.models:
            params["models"] = args.models
        if args.api_key:
            params["api_key"] = args.api_key
        if args.proxy:
            params["proxy"] = args.proxy
        if args.headers:
            params["headers"] = _parse_headers(args.headers)
        method = "add"
    else:
        params = {"provider_id": args.provider_id}
        method = "remove"

    runtime = create_runtime_service(
        config_path=args.config,
        data_dir=args.data_dir,
    )
    try:
        handlers = {
            "list": runtime.provider_list,
            "add": runtime.provider_add,
            "remove": runtime.provider_remove,
        }
        result = await handlers[method](params)
    finally:
        await runtime.close()
    _print_provider_result(
        command,
        result,
        output=args.output,
    )
    return 0


def _parse_headers(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, header_value = value.partition("=")
        if not separator or not name.strip():
            raise ValueError("--header must use NAME=VALUE")
        result[name.strip()] = header_value
    return result


def _print_provider_result(
    command: str,
    result: dict[str, Any],
    *,
    output: str,
) -> None:
    if output == "json":
        print(_json(result))
        return
    if command == "list":
        providers = result.get("providers") or []
        if not providers:
            print("No providers configured.")
            return
        for provider in providers:
            status = "signed in" if provider.get("authenticated") else provider.get("kind")
            print(f"{provider['id']}\t{provider['name']}\t{status}")
        return
    if command == "add":
        provider = result["provider"]
        backend = provider.get("backend") or provider["driver"]
        print(f"Added provider {provider['id']} [{backend}] ({provider['endpoint']}).")
        if provider.get("model_ids"):
            print("The first model in `aeloon-runtime models` is used automatically.")
            print(f"Optional pin: aeloon-runtime models use {provider['model_ids'][0]}")
        return
    print(f"Removed provider {result['provider_id']}.")


def _with_run_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates: dict[str, Any] = {}
    if args.workspace is not None:
        updates["workspace"] = args.workspace
    if args.data_dir is not None:
        updates["data_dir"] = args.data_dir
    agent_updates: dict[str, Any] = {}
    if args.model is not None:
        agent_updates["model"] = args.model
    if args.effort is not None:
        agent_updates["thinking_level"] = args.effort
    if agent_updates:
        updates["agent"] = config.agent.model_copy(update=agent_updates)
    return config.model_copy(update=updates).normalized()


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt and (args.prompt_file or args.stdin):
        raise RunError("invalid_argument", "Use exactly one prompt source")
    if args.prompt_file:
        value = args.prompt_file.read_text(encoding="utf-8")
    elif args.stdin:
        value = sys.stdin.read()
    else:
        value = " ".join(args.prompt)
    if not value.strip():
        raise RunError("invalid_argument", "Prompt must not be empty")
    return value


def _status(stop_reason: str) -> str:
    if stop_reason == "error":
        return "error"
    if stop_reason == "aborted":
        return "aborted"
    return "completed"


def _set_nested(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def _parse_value(value: str) -> Any:
    if value.lower() == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda item: asdict(item) if is_dataclass(item) else str(item),
    )


async def async_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized = _normalize_argv(raw)
    parser = build_parser()
    args = parser.parse_args(normalized)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == _TASK_COMMAND:
        return await run_command(args)
    if args.command == "resume":
        return await resume_command(args)
    if args.command == "history":
        return await history_command(args)
    if args.command in {"login", "logout", "whoami"}:
        args.cloud_command = "status" if args.command == "whoami" else args.command
        return await cloud_command(args)
    if args.command == "models":
        return await models_command(args)
    if args.command == "doctor":
        return await doctor_command(args)
    if args.command == "completion":
        return completion_command(args)
    if args.command == "rpc":
        return await rpc_command(args)
    if args.command == "serve":
        if args.no_workspace_root and args.workspace_root:
            parser.error("--no-workspace-root cannot be combined with --workspace-root")
        if args.no_workspace_root:
            workspace_roots: tuple[Path, ...] | None = ()
        elif args.workspace_root:
            workspace_roots = tuple(args.workspace_root)
        else:
            workspace_roots = None
        await serve_v3(
            socket_path=args.unix,
            data_dir=args.data_dir,
            config_path=args.config,
            workspace_roots=workspace_roots,
            max_concurrent_operations=args.max_concurrent_operations,
            record_trace=args.record_trace,
            listen=parse_listen(args.listen) if args.listen else None,
            tls_context=build_tls_context(args.tls_cert, args.tls_key),
        )
        return 0
    if args.command == "migrate":
        result = migrate_workbench(
            from_workbench=args.from_workbench,
            from_core=args.from_core,
            data_dir=args.data_dir,
            roots_output=args.roots_output,
        )
        print(_json(result))
        return 0
    if args.command == "system":
        return run_bundled_skill(
            args.skill_id,
            args.skill_action,
            list(args.skill_arguments),
        )
    if args.command == "provider":
        return await provider_command(args)
    return config_command(args)


def _normalize_argv(argv: list[str]) -> list[str]:
    """Route bare arguments to the task command."""

    if not argv:
        return [] if _stdin_is_interactive() else [_TASK_COMMAND, "--stdin"]
    if argv[0] == "--":
        return [_TASK_COMMAND, *argv[1:]]
    if argv[0] in {"-h", "--help", "--version"} or argv[0] in _KNOWN_COMMANDS:
        return argv
    return [_TASK_COMMAND, *argv]


def cli_version() -> str:
    """Expose the standalone Runtime release independently of legacy Core CLI."""

    return (
        "0.1.0"
        if Path(sys.argv[0]).name == "aeloon-runtime"
        or os.environ.get("AELOON_RUNTIME_MODE") == "1"
        else __version__
    )


def _json_errors_for(argv: list[str]) -> bool:
    if "--json" in argv or "--stream" in argv:
        return True
    for index, value in enumerate(argv[:-1]):
        if value == "--output" and argv[index + 1] in {"json", "stream-json"}:
            return True
    return bool(argv and argv[0] in {"config", "rpc", "provider"})


def _print_cli_error(code: str, message: str, *, as_json: bool) -> None:
    if as_json:
        print(_json({"error": code, "message": message}), file=sys.stderr)
        return
    print(f"Error: {message}", file=sys.stderr)
    suggestions = {
        "auth": (
            "Run `aeloon-runtime provider add ...` or `aeloon-runtime login` to configure a model source."
        ),
        "model_not_configured": (
            "Add an API with `aeloon-runtime provider add ...` or sign in with `aeloon-runtime login`, "
            "then Aeloon will use the first available model automatically."
        ),
        "model_not_found": "Run `aeloon-runtime models` to see available models.",
        "not_found": "Run `aeloon-runtime history` to see saved tasks.",
    }
    if code in suggestions:
        print(suggestions[code], file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    json_errors = _json_errors_for(raw)
    try:
        return asyncio.run(async_main(raw))
    except (RpcError, RunError, SessionError) as exc:
        _print_cli_error(exc.code, str(exc), as_json=json_errors)
        return 2
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        _print_cli_error("invalid_argument", str(exc), as_json=json_errors)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
