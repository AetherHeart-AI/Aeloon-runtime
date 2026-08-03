"""Command-line interface for the pure-Python Aeloon harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    get_deepseek_model,
    message_to_dict,
)

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

_TOOL_COMMAND_MAX_LINES = 20
_TOOL_COMMAND_MAX_CHARS = 2_000
_TOOL_RESULT_MAX_LINES = 80
_TOOL_RESULT_MAX_CHARS = 8_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Aeloon coding-agent harness.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a prompt through the harness.")
    run.add_argument("prompt", nargs="*", help="Prompt text.")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--prompt-file", type=Path, help="Read a UTF-8 prompt file.")
    source.add_argument("--stdin", action="store_true", help="Read the prompt from stdin.")
    run.add_argument("--output", choices=("text", "json", "stream-json"), default="text")
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
    return parser


class RunRenderer:
    """Render events while collecting a stable final JSON result."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.tools_used: list[str] = []
        self._printed_text = False
        self._tool_started_at: dict[str, float] = {}

    async def __call__(self, event: HarnessEvent) -> None:
        if event.type == "tool_execution_start":
            name = str(event.data.get("toolName") or "")
            call_id = str(event.data.get("toolCallId") or "")
            if name and name not in self.tools_used:
                self.tools_used.append(name)
            if call_id:
                self._tool_started_at[call_id] = time.monotonic()
            if self.output == "text":
                self._render_tool_start(name, event.data.get("args"))
        elif event.type == "tool_execution_end" and self.output == "text":
            self._render_tool_end(event)
        if self.output == "stream-json":
            print(_json(event.to_dict()), flush=True)
            return
        if self.output != "text" or event.type != "message_update":
            return
        update = event.data.get("assistantMessageEvent") or {}
        if update.get("type") == "text_delta":
            print(str(update.get("delta") or ""), end="", flush=True)
            self._printed_text = True

    def _render_tool_start(self, name: str, raw_args: Any) -> None:
        args = raw_args if isinstance(raw_args, dict) else {}
        print(f"[tool] {name or 'unknown'}", file=sys.stderr)
        for line in _tool_invocation_lines(name, args):
            print(f"  {line}", file=sys.stderr)

    def _render_tool_end(self, event: HarnessEvent) -> None:
        name = str(event.data.get("toolName") or "unknown")
        call_id = str(event.data.get("toolCallId") or "")
        raw_result = event.data.get("result")
        result = raw_result if isinstance(raw_result, dict) else {}
        details_value = result.get("details")
        details = details_value if isinstance(details_value, dict) else {}
        is_error = bool(event.data.get("isError") or result.get("isError"))
        qualifiers: list[str] = []
        if name == "bash" and details.get("exitCode") is not None:
            qualifiers.append(f"exit {details['exitCode']}")
        started_at = self._tool_started_at.pop(call_id, None)
        if started_at is not None:
            qualifiers.append(_format_duration(time.monotonic() - started_at))
        if details.get("truncated"):
            qualifiers.append("truncated")
        suffix = f" · {' · '.join(qualifiers)}" if qualifiers else ""
        label = "tool error" if is_error else "tool result"
        print(f"[{label}] {name}{suffix}", file=sys.stderr)

        preview = _tool_result_text(result)
        preview, truncated = _truncate_terminal_text(
            preview,
            max_lines=_TOOL_RESULT_MAX_LINES,
            max_chars=_TOOL_RESULT_MAX_CHARS,
            keep_tail=name == "bash",
        )
        if truncated:
            marker = (
                "[... terminal preview truncated; showing tail ...]"
                if name == "bash"
                else ("[... terminal preview truncated ...]")
            )
            preview = f"{marker}\n{preview}" if name == "bash" else f"{preview}\n{marker}"
        for line in (preview or "(no output)").splitlines():
            print(f"  {line}", file=sys.stderr)

    def finish_text(self, final_text: str) -> None:
        if self.output != "text":
            return
        if not self._printed_text:
            print(final_text)
        elif final_text:
            print()


def _tool_invocation_lines(name: str, args: dict[str, Any]) -> list[str]:
    if name == "bash":
        command, truncated = _truncate_terminal_text(
            str(args.get("command") or ""),
            max_lines=_TOOL_COMMAND_MAX_LINES,
            max_chars=_TOOL_COMMAND_MAX_CHARS,
        )
        lines = command.splitlines() or [""]
        rendered = [f"$ {lines[0]}", *(f"> {line}" for line in lines[1:])]
        if truncated:
            rendered.append("[... command preview truncated ...]")
        if args.get("timeout") is not None:
            rendered.append(f"timeout={args['timeout']}s")
        return rendered
    if name == "write":
        content = str(args.get("content") or "")
        return [f"path={args.get('path', '')} ({len(content.encode('utf-8'))} bytes)"]
    if name == "edit":
        edits = args.get("edits")
        count = len(edits) if isinstance(edits, list) else 0
        return [f"path={args.get('path', '')} ({count} replacement(s))"]
    if name == "read":
        return [_format_selected_args(args, ("path", "offset", "limit"))]
    if name == "grep":
        return [_format_selected_args(args, ("pattern", "path", "glob", "limit"))]
    if name == "find":
        return [_format_selected_args(args, ("pattern", "path", "limit"))]
    if name == "ls":
        return [_format_selected_args(args, ("path", "limit")) or "path=."]
    rendered, truncated = _truncate_terminal_text(
        _json(args),
        max_lines=_TOOL_COMMAND_MAX_LINES,
        max_chars=_TOOL_COMMAND_MAX_CHARS,
    )
    if truncated:
        rendered += "\n[... arguments preview truncated ...]"
    return rendered.splitlines() or ["{}"]


def _format_selected_args(args: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(f"{key}={args[key]!r}" for key in keys if args.get(key) is not None)


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


def _truncate_terminal_text(
    value: str,
    *,
    max_lines: int,
    max_chars: int,
    keep_tail: bool = False,
) -> tuple[str, bool]:
    lines = value.splitlines()
    too_many_lines = len(lines) > max_lines
    selected = lines[-max_lines:] if keep_tail else lines[:max_lines]
    rendered = "\n".join(selected)
    too_many_chars = len(rendered) > max_chars
    if too_many_chars:
        rendered = rendered[-max_chars:] if keep_tail else rendered[:max_chars]
        if keep_tail:
            rendered = rendered.split("\n", 1)[-1]
        else:
            rendered = rendered.rsplit("\n", 1)[0] or rendered
    return rendered, too_many_lines or too_many_chars


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
        if restored_model is not None and restored_model[0] == "deepseek" and args.model is None:
            config = config.model_copy(
                update={"agent": config.agent.model_copy(update={"model": restored_model[1]})}
            )
    model = replace(get_deepseek_model(config.agent.model), base_url=config.deepseek.base_url)
    if session is None and not args.no_session:
        assert repository is not None
        session = await repository.create(cwd=config.workspace)

    provider = DeepSeekProvider(
        api_key=config.deepseek.api_key,
        base_url=config.deepseek.base_url,
        proxy=config.deepseek.proxy,
        headers=config.deepseek.extra_headers,
    )
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
    renderer = RunRenderer(args.output)
    harness.subscribe(renderer)
    started = time.monotonic()
    try:
        message = await harness.prompt(prompt)
    finally:
        await harness.close()
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
    return config_command(args)


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except (HarnessError, SessionError) as exc:
        print(_json({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(_json({"error": "invalid_argument", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
