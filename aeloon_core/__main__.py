"""Command-line interface for the pure-Python Aeloon harness."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from aeloon_core.bridge.daemon import (
    bridge_request,
    daemon_status,
    default_socket_path,
    ensure_daemon,
    run_daemon,
    stop_daemon,
)
from aeloon_core.bridge.protocol import BridgeError, load_schema
from aeloon_core.cloud import CloudAccountService, CloudError
from aeloon_core.config import (
    Config,
    load_config,
    public_config,
    resolve_config_path,
    save_config,
)
from aeloon_core.harness import (
    AgentHarness,
    CompactionSettings,
    DeepSeekProvider,
    HarnessError,
    HarnessEvent,
    JsonlSessionRepository,
    ResourceLoader,
    SessionError,
    StreamOptions,
    message_to_dict,
)
from aeloon_core.providers import UnifiedProviderRegistry, qualify_model_id

CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    "workspace": ("workspace",),
    "data-dir": ("data_dir",),
    "api-key": ("deepseek", "api_key"),
    "base-url": ("deepseek", "base_url"),
    "proxy": ("deepseek", "proxy"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Aeloon coding-agent harness.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a prompt through the harness.")
    run.add_argument("prompt", nargs="*", help="Prompt text.")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--prompt-file", type=Path, help="Read a UTF-8 prompt file.")
    source.add_argument("--stdin", action="store_true", help="Read the prompt from stdin.")
    run.add_argument("--output", choices=("text", "json", "stream-json"), default="text")
    run.add_argument(
        "--verbose",
        action="store_true",
        help="Show tool execution details on stderr (default: quiet).",
    )
    run.add_argument("--session", help="Continue an existing session id.")
    run.add_argument("--no-session", action="store_true", help="Do not persist this run.")
    run.add_argument("--config", type=Path)
    run.add_argument("--workspace", type=Path)
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--model")

    session = commands.add_parser("session", help="Inspect harness sessions.")
    session_commands = session.add_subparsers(dest="session_command", required=True)
    session_list = session_commands.add_parser("list", help="List saved sessions.")
    session_list.add_argument("--config", type=Path)
    session_list.add_argument("--data-dir", type=Path)
    session_list.add_argument("--workspace", type=Path)
    session_show = session_commands.add_parser("show", help="Show one saved session.")
    session_show.add_argument("session_id")
    session_show.add_argument("--config", type=Path)
    session_show.add_argument("--data-dir", type=Path)

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

    bridge = commands.add_parser("bridge", help="Manage the local Bridge v2 daemon.")
    bridge_commands = bridge.add_subparsers(dest="bridge_command", required=True)
    for name in ("serve", "ensure", "status", "stop"):
        command = bridge_commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--socket", type=Path)
        if name in {"serve", "ensure"}:
            command.add_argument("--max-concurrent-operations", type=int, default=4)
        if name in {"ensure", "status", "stop"}:
            command.add_argument("--output", choices=("text", "json"), default="text")
    bridge_commands.add_parser("schema")

    cloud = commands.add_parser("cloud", help="Manage the Aeloon Cloud account.")
    cloud_commands = cloud.add_subparsers(dest="cloud_command", required=True)
    cloud_login = cloud_commands.add_parser("login", help="Sign in to Aeloon Cloud.")
    cloud_login.add_argument("username", nargs="?", help="Account username or email.")
    for command_name in ("login", "status", "logout"):
        command = cloud_login if command_name == "login" else cloud_commands.add_parser(
            command_name,
            help={
                "status": "Show the current Aeloon Cloud account.",
                "logout": "Sign out of Aeloon Cloud.",
            }[command_name],
        )
        command.add_argument("--config", type=Path)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--socket", type=Path)
        command.add_argument("--output", choices=("text", "json"), default="text")

    provider = commands.add_parser("provider", help="Manage cloud and local API providers.")
    provider_commands = provider.add_subparsers(dest="provider_command", required=True)
    provider_login = provider_commands.add_parser("login", help="Sign in to Aeloon Cloud.")
    provider_login.add_argument("username", nargs="?", help="Account username or email.")
    provider_add = provider_commands.add_parser("add", help="Add a local API provider.")
    provider_add.add_argument("provider_id", help="Stable prefix used in provider/model ids.")
    provider_add.add_argument("--name")
    provider_add.add_argument("--base-url", required=True)
    provider_add.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model id (repeatable); omit to discover models from GET /models.",
    )
    provider_add.add_argument(
        "--no-api-key", action="store_true", help="The local endpoint does not require a key."
    )
    provider_remove = provider_commands.add_parser("remove", help="Remove a local API provider.")
    provider_remove.add_argument("provider_id")
    for command_name in ("login", "status", "logout", "list", "add", "remove"):
        if command_name == "login":
            command = provider_login
        elif command_name == "add":
            command = provider_add
        elif command_name == "remove":
            command = provider_remove
        else:
            command = provider_commands.add_parser(command_name)
        command.add_argument("--config", type=Path)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--socket", type=Path)
        command.add_argument("--output", choices=("text", "json"), default="text")
    return parser


class RunRenderer:
    """Render events while collecting a stable final JSON result."""

    def __init__(self, output: str, *, verbose: bool = False) -> None:
        self.output = output
        self.verbose = verbose
        self.tools_used: list[str] = []
        self._tool_started_at: dict[str, float] = {}
        self._tool_args: dict[str, dict[str, Any]] = {}

    async def __call__(self, event: HarnessEvent) -> None:
        if event.type == "tool_execution_start":
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
                self._render_tool_start(name, event.data.get("args"))
        elif event.type == "tool_execution_end" and self.verbose and self.output == "text":
            self._render_tool_end(event)
        if self.output == "stream-json":
            print(_json(event.to_dict()), flush=True)
            return
        # Text output is intentionally buffered until the complete assistant message is
        # available. Rendering partial Markdown produces noisy, malformed terminal output.

    def _render_tool_start(self, name: str, raw_args: Any) -> None:
        args = raw_args if isinstance(raw_args, dict) else {}
        action, summary = _tool_invocation_summary(name, args)
        print(f"[{action}] {summary}", file=sys.stderr)

    def _render_tool_end(self, event: HarnessEvent) -> None:
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


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1_000)}ms"
    return f"{seconds:.2f}s"


async def run_command(args: argparse.Namespace) -> int:
    if args.session and args.no_session:
        raise HarnessError("invalid_argument", "--session and --no-session cannot be combined")
    prompt = _read_prompt(args)
    config = _with_run_overrides(load_config(args.config), args)
    repository = None if args.no_session else JsonlSessionRepository(config.data_dir)
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
    cloud_account = CloudAccountService(config.cloud, data_dir=config.data_dir)
    registry = UnifiedProviderRegistry(
        config,
        cloud_account,
        local_provider_factory=DeepSeekProvider,
    )
    try:
        model = await registry.model(config.agent.model)
    except PermissionError as exc:
        await cloud_account.close()
        raise HarnessError("auth", str(exc)) from None
    except (KeyError, RuntimeError, CloudError) as exc:
        await cloud_account.close()
        raise HarnessError("model_not_found", str(exc)) from None
    if session is None and not args.no_session:
        assert repository is not None
        session = await repository.create(cwd=config.workspace)

    provider = registry.provider(model)
    resources = ResourceLoader(
        cwd=config.workspace,
        agent_dir=Path("~/.aeloon-core"),
        additional_roots=tuple(config.resources.roots),
        no_skills=config.resources.no_skills,
        no_prompt_templates=config.resources.no_prompt_templates,
        no_context_files=config.resources.no_context_files,
    )
    harness = AgentHarness(
        provider=provider,
        model=model,
        cwd=str(config.workspace),
        session=session,
        resource_loader=resources,
        thinking_level=config.agent.thinking_level,
        stream_options=StreamOptions(
            timeout_ms=config.agent.timeout_ms,
            max_tokens=config.agent.max_tokens,
            temperature=config.agent.temperature,
            thinking_level=config.agent.thinking_level,
            max_retries=config.agent.retry.max_retries if config.agent.retry.enabled else 0,
            base_delay_ms=config.agent.retry.base_delay_ms,
            max_retry_delay_ms=config.agent.retry.max_retry_delay_ms,
        ),
        steering_mode=config.agent.steering_mode,
        follow_up_mode=config.agent.follow_up_mode,
        compaction=CompactionSettings(
            enabled=config.agent.compaction.enabled,
            reserve_tokens=config.agent.compaction.reserve_tokens,
            keep_recent_tokens=config.agent.compaction.keep_recent_tokens,
        ),
        shell_path=config.tools.shell_path,
        auto_resize_images=config.tools.auto_resize_images,
    )
    renderer = RunRenderer(args.output, verbose=args.verbose)
    harness.subscribe(renderer)
    started = time.monotonic()
    try:
        message = await harness.prompt(prompt)
    finally:
        await harness.close()
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


async def session_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    data_dir = args.data_dir or config.data_dir
    repository = JsonlSessionRepository(data_dir)
    if args.session_command == "list":
        metadata = await repository.list(cwd=args.workspace)
        print(
            _json(
                [
                    {
                        "id": item.id,
                        "created_at": item.created_at,
                        "cwd": item.cwd,
                        "path": str(item.path),
                        "parent_session_path": item.parent_session_path,
                        "metadata": item.metadata,
                    }
                    for item in metadata
                ]
            )
        )
        return 0
    session = await repository.open(args.session_id)
    context = await session.build_context()
    print(
        _json(
            {
                "id": session.id,
                "created_at": session.metadata.created_at,
                "cwd": session.metadata.cwd,
                "path": str(session.path),
                "leaf_id": await session.get_leaf_id(),
                "name": await session.get_name(),
                "entries": await session.get_entries(),
                "context": [message_to_dict(message) for message in context.messages],
                "stats": await session.stats(),
            }
        )
    )
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
    config = load_config(path, use_environment=False)
    raw = config.model_dump(mode="json")
    _set_nested(raw, CONFIG_PATHS[args.key], _parse_value(args.value))
    validated = Config.model_validate(raw).normalized()
    print(save_config(validated, path))
    return 0


async def bridge_command(args: argparse.Namespace) -> int:
    if args.bridge_command == "schema":
        print(json.dumps(load_schema(), ensure_ascii=False, separators=(",", ":")))
        return 0
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir}).normalized()
    socket_path = args.socket or default_socket_path(config.data_dir)
    if args.bridge_command == "serve":
        await run_daemon(
            config_path=args.config,
            data_dir=args.data_dir,
            socket_path=socket_path,
            max_concurrent_operations=args.max_concurrent_operations,
        )
        return 0
    if args.bridge_command == "ensure":
        result = await ensure_daemon(
            config_path=args.config,
            data_dir=args.data_dir,
            socket_path=socket_path,
            max_concurrent_operations=args.max_concurrent_operations,
        )
    elif args.bridge_command == "status":
        result = await daemon_status(socket_path)
    else:
        result = await stop_daemon(socket_path)
    if getattr(args, "output", "text") == "json":
        print(_json(result))
    else:
        print(f"Aeloon Core bridge: {result.get('status', 'ok')} ({socket_path})")
    return 0


async def cloud_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir}).normalized()
    socket_path = args.socket or default_socket_path(config.data_dir)

    params: dict[str, Any] = {}
    method = f"cloud.account.{args.cloud_command}"
    timeout = 3.0
    if args.cloud_command == "login":
        username = (args.username or _read_cloud_username()).strip()
        if not username:
            raise ValueError("Aeloon Cloud username is required")
        password = _read_cloud_password()
        params = {"username": username, "password": password}
        timeout = 60.0

    daemon = await ensure_daemon(
        config_path=args.config,
        data_dir=args.data_dir,
        socket_path=socket_path,
    )
    result = await bridge_request(
        Path(str(daemon["socket_path"])),
        method,
        params,
        timeout=timeout,
    )
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
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir}).normalized()
    socket_path = args.socket or default_socket_path(config.data_dir)

    command = args.provider_command
    params: dict[str, Any] = {}
    timeout = 3.0
    if command == "login":
        username = (args.username or _read_cloud_username()).strip()
        if not username:
            raise ValueError("Aeloon Cloud username is required")
        params = {"username": username, "password": _read_cloud_password()}
        method = "cloud.account.login"
        timeout = 60.0
    elif command in {"status", "logout"}:
        method = f"cloud.account.{command}"
    elif command == "list":
        method = "provider.list"
    elif command == "add":
        api_key = "no-key" if args.no_api_key else _read_local_api_key()
        params = {
            "provider_id": args.provider_id,
            "name": args.name or args.provider_id,
            "base_url": args.base_url,
            "api_key": api_key,
        }
        if args.models:
            params["models"] = args.models
        method = "provider.local.add"
    else:
        params = {"provider_id": args.provider_id}
        method = "provider.local.remove"

    required_methods = () if method.startswith("cloud.account.") else (method,)
    daemon = await ensure_daemon(
        config_path=args.config,
        data_dir=args.data_dir,
        socket_path=socket_path,
        required_methods=required_methods,
    )
    result = await bridge_request(
        Path(str(daemon["socket_path"])), method, params, timeout=timeout
    )
    _print_provider_result(command, result, output=args.output)
    return 0


def _read_local_api_key() -> str:
    try:
        value = getpass.getpass("Local API key (leave empty for none): ")
    except EOFError:
        raise ValueError("Local API key must be entered from a terminal") from None
    return value or "no-key"


def _print_provider_result(command: str, result: dict[str, Any], *, output: str) -> None:
    if output == "json":
        print(_json(result))
        return
    if command in {"login", "status", "logout"}:
        _print_cloud_result(command, result, output="text")
        return
    if command == "list":
        for provider in result.get("providers") or []:
            status = "signed in" if provider.get("authenticated") else provider.get("kind")
            print(f"{provider['id']}\t{provider['name']}\t{status}")
        return
    if command == "add":
        provider = result["provider"]
        print(f"Added local provider {provider['id']} ({provider['base_url']}).")
        return
    print(f"Removed local provider {result['provider_id']}.")


def _with_run_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates: dict[str, Any] = {}
    if args.workspace is not None:
        updates["workspace"] = args.workspace
    if args.data_dir is not None:
        updates["data_dir"] = args.data_dir
    if args.model is not None:
        updates["agent"] = config.agent.model_copy(update={"model": args.model})
    return config.model_copy(update=updates).normalized()


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt and (args.prompt_file or args.stdin):
        raise HarnessError("invalid_argument", "Use exactly one prompt source")
    if args.prompt_file:
        value = args.prompt_file.read_text(encoding="utf-8")
    elif args.stdin:
        value = sys.stdin.read()
    else:
        value = " ".join(args.prompt)
    if not value.strip():
        raise HarnessError("invalid_argument", "Prompt must not be empty")
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
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return await run_command(args)
    if args.command == "session":
        return await session_command(args)
    if args.command == "bridge":
        return await bridge_command(args)
    if args.command == "cloud":
        return await cloud_command(args)
    if args.command == "provider":
        return await provider_command(args)
    return config_command(args)


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except (BridgeError, HarnessError, SessionError) as exc:
        print(_json({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(_json({"error": "invalid_argument", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
