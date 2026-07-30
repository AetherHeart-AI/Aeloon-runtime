"""CLI entry point for Aeloon Core."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aeloon_core.config import Config, load_config, resolve_config_path, save_config
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.web.events import TurnEventProgress
from aeloon_core.web.launcher import WebLaunchError, run_web_ui

LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
COMMANDS = {"run", "serve", "web", "config"}
REMOVED_COMMANDS = {"profile", "webui", "chat", "tui"}
WEB_ONLY_OPTIONS = {
    "--no-open",
    "--port",
    "--gateway-log-level",
}
CONFIG_SETTERS = {
    "mode": ("mode",),
    "workspace": ("workspace",),
    "data-dir": ("data_dir",),
    "provider": ("agents", "defaults", "provider"),
    "model": ("agents", "defaults", "model"),
    "master-model": ("agents", "routing", "master"),
    "research-expert-model": (
        "agents",
        "routing",
        "experts",
        "builtin:research",
    ),
    "coding-expert-model": (
        "agents",
        "routing",
        "experts",
        "builtin:coding",
    ),
    "reasoning-effort": ("agents", "defaults", "reasoning_effort"),
    "max-output-tokens": ("agents", "defaults", "max_output_tokens"),
    "max-iterations": ("agents", "defaults", "max_iterations"),
    "experts-enabled": ("experts", "enabled"),
    "experts-max-calls-per-turn": ("experts", "max_calls_per_turn"),
    "experts-max-concurrency": ("experts", "max_concurrency"),
    "experts-stage-request-limit": ("experts", "stage_request_limit"),
    "experts-timeout-seconds": ("experts", "timeout_seconds"),
    "experts-max-upstream-chars": ("experts", "max_upstream_chars"),
    "mcp-config-path": ("mcp", "config_path"),
    "mcp-master-allowlist": ("mcp", "master_allowlist"),
    "tools-master-capabilities": ("tools", "master_capabilities"),
    "context-compaction-enabled": ("agents", "defaults", "context_compaction", "enabled"),
    "context-compaction-trigger-ratio": (
        "agents",
        "defaults",
        "context_compaction",
        "trigger_ratio",
    ),
    "context-compaction-preserve-recent-tokens": (
        "agents",
        "defaults",
        "context_compaction",
        "preserve_recent_tokens",
    ),
    "runtime-transition-trace-enabled": (
        "agents",
        "defaults",
        "runtime",
        "transition_trace_enabled",
    ),
    "runtime-stuck-detection-enabled": (
        "agents",
        "defaults",
        "runtime",
        "stuck_detection_enabled",
    ),
    "runtime-stuck-detection-threshold": (
        "agents",
        "defaults",
        "runtime",
        "stuck_detection_threshold",
    ),
    "runtime-max-retries": (
        "agents",
        "defaults",
        "runtime",
        "max_retries",
    ),
}
DYNAMIC_PROVIDER_SETTERS = {
    "api-key": "api_key",
}
CONFIG_KEYS = {*CONFIG_SETTERS, *DYNAMIC_PROVIDER_SETTERS}
SECRET_KEYS = {"api_key"}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Run Aeloon Core.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one agent turn.")
    _add_run_args(run_parser)

    serve_parser = subparsers.add_parser("serve", help="Start the local Web UI.")
    _add_web_args(serve_parser)

    web_parser = subparsers.add_parser("web", help="Alias for serve.")
    _add_web_args(web_parser)

    config_parser = subparsers.add_parser("config", help="Manage persistent config.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    config_path_parser = config_subparsers.add_parser("path", help="Print config path.")
    config_path_parser.add_argument(
        "--config", type=Path, default=None, help="Optional config JSON path."
    )

    config_show_parser = config_subparsers.add_parser("show", help="Print effective config.")
    config_show_parser.add_argument(
        "--config", type=Path, default=None, help="Optional config JSON path."
    )
    config_show_parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Print secrets instead of masking them.",
    )

    config_init_parser = config_subparsers.add_parser(
        "init",
        help="Write a persistent config file.",
    )
    _add_config_write_args(config_init_parser)
    config_init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file.",
    )

    config_set_parser = config_subparsers.add_parser("set", help="Set one config value.")
    config_set_parser.add_argument(
        "--config", type=Path, default=None, help="Optional config JSON path."
    )
    config_set_parser.add_argument("key", choices=sorted(CONFIG_KEYS))
    config_set_parser.add_argument("value")

    return parser


def build_legacy_run_parser() -> argparse.ArgumentParser:
    """Build the legacy prompt-first parser."""

    parser = argparse.ArgumentParser(description="Run one Aeloon Core agent turn.")
    _add_run_args(parser)
    return parser


def _add_path_args(parser: argparse.ArgumentParser, *, session: bool = False) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    if session:
        parser.add_argument("--session", default=None, help="Existing session id to continue.")
    parser.add_argument("--workspace", type=Path, default=None, help="Override workspace.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data dir.")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text to send to the agent (omit with --prompt-file or --stdin).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read the prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the prompt from standard input.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format. JSON suppresses streaming progress.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model for this run.",
    )
    _add_path_args(parser, session=True)


def _add_web_args(parser: argparse.ArgumentParser) -> None:
    _add_path_args(parser, session=True)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    parser.add_argument("--port", type=int, default=7331, help="Local port (default: 7331).")
    parser.add_argument(
        "--gateway-log-level",
        choices=sorted(LOG_LEVELS),
        default="INFO",
        help="Minimum level retained by the Web diagnostics view.",
    )


def _add_config_write_args(parser: argparse.ArgumentParser) -> None:
    _add_path_args(parser)
    parser.add_argument(
        "--mode",
        choices=("normal", "expert"),
        default=None,
        help="Capability policy mode (default: normal).",
    )
    parser.add_argument(
        "--provider",
        choices=("deepseek",),
        default=None,
        help="Pydantic AI model provider (default: deepseek).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Default model name.",
    )


class _PlainTextProgressSink:
    """Render TurnEventProgress events as plain stdout lines for the `run` command."""

    def __init__(self) -> None:
        self._block_types: dict[str, str] = {}
        self._block_names: dict[str, str] = {}
        self._streaming = False

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event == "chat.status":
            self._break_stream()
            prefix = "tools" if payload.get("kind") == "tool_hint" else "status"
            print(f"[{prefix}] {payload.get('text', '')}")
        elif event == "chat.block.add":
            block = payload.get("block", {})
            block_id = block.get("id")
            self._block_types[block_id] = block.get("type")
            self._block_names[block_id] = block.get("name")
            if block.get("type") == "tool_call":
                self._break_stream()
                print(f"[tool call] {block.get('name')}")
        elif event == "chat.block.delta":
            if self._block_types.get(payload.get("block_id")) == "text":
                print(payload.get("delta", ""), end="", flush=True)
                self._streaming = True
        elif event == "chat.block.update":
            block_id = payload.get("block_id")
            patch = payload.get("patch", {})
            if "result" in patch and self._block_types.get(block_id) == "tool_call":
                self._break_stream()
                result = str(patch.get("result") or "")
                preview = result[:500] + ("..." if len(result) > 500 else "")
                print(f"[tool result] {self._block_names.get(block_id)}: {preview}")
        elif event == "chat.turn.end":
            self._break_stream()
            print(f"\n[final]\n{payload.get('final', '')}")

    def _break_stream(self) -> None:
        if self._streaming:
            print()
            self._streaming = False


class _SilentProgressSink:
    """Consume progress events without mixing them into machine-readable stdout."""

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        del event, payload


async def _run_prompt(args: argparse.Namespace) -> None:
    prompt = _resolve_run_prompt(args)
    config = _load_with_path_overrides(
        args.config,
        workspace=getattr(args, "workspace", None),
        data_dir=getattr(args, "data_dir", None),
        model=getattr(args, "model", None),
    )
    if not config.workspace.exists():
        raise SystemExit(f"Workspace does not exist: {config.workspace}")
    if not config.workspace.is_dir():
        raise SystemExit(f"Workspace is not a directory: {config.workspace}")

    orchestrator = AeloonCoreOrchestrator(config)
    try:
        session_id = args.session or orchestrator.sessions.new_session()
        output_format = getattr(args, "output", "text")
        sink = _SilentProgressSink() if output_format == "json" else _PlainTextProgressSink()
        progress = TurnEventProgress(session_id=session_id, emit=sink.emit)
        await progress.on_turn_start()
        result = await orchestrator.run_turn(
            prompt,
            session_id=session_id,
            on_progress=progress,
        )
        if output_format == "json":
            _print_json(_turn_result_payload(result, config=config))
        else:
            print(f"\n[session] {result.session_id}")
            if result.tools_used:
                print(f"[tools used] {', '.join(result.tools_used)}")
    finally:
        await orchestrator.close()


def _resolve_run_prompt(args: argparse.Namespace) -> str:
    positional = " ".join(getattr(args, "prompt", [])).strip()
    prompt_file = getattr(args, "prompt_file", None)
    use_stdin = bool(getattr(args, "stdin", False))
    source_count = int(bool(positional)) + int(prompt_file is not None) + int(use_stdin)
    if source_count != 1:
        raise SystemExit(
            "Provide exactly one prompt source: prompt text, --prompt-file, or --stdin."
        )

    if prompt_file is not None:
        path = Path(prompt_file).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Prompt file does not exist or is not a file: {path}")
        try:
            prompt = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SystemExit(f"Could not read prompt file {path}: {exc}") from None
    elif use_stdin:
        prompt = sys.stdin.read()
    else:
        prompt = positional

    if not prompt.strip():
        raise SystemExit("Prompt must not be empty.")
    return prompt


def _turn_result_payload(result: Any, *, config: Config) -> dict[str, Any]:
    default_model = config.agents.defaults.model_ref()
    return {
        "schema_version": 1,
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "status": result.status,
        "final_content": result.final_content,
        "tools_used": list(result.tools_used),
        "usage": dict(result.usage),
        "duration_ms": result.duration_ms,
        "transitions": list(result.transitions),
        "workspace": str(config.workspace),
        "models": {
            "default": default_model,
            "master": config.agents.routing.master or default_model,
            "experts": dict(config.agents.routing.experts),
        },
    }


async def _run_web(args: argparse.Namespace) -> None:
    config = _load_with_path_overrides(
        args.config,
        workspace=getattr(args, "workspace", None),
        data_dir=getattr(args, "data_dir", None),
    )
    try:
        await run_web_ui(
            config,
            session_id=args.session,
            port=args.port,
            open_browser=not args.no_open,
            gateway_log_level=args.gateway_log_level,
        )
    except WebLaunchError as exc:
        raise SystemExit(str(exc)) from None


def _run_config(args: argparse.Namespace) -> None:
    if args.config_command == "path":
        print(resolve_config_path(args.config))
        return
    if args.config_command == "show":
        config = load_config(args.config)
        print(json.dumps(_config_dump(config, show_secrets=args.show_secrets), indent=2))
        return
    if args.config_command == "init":
        path = resolve_config_path(args.config)
        if path.exists() and not args.force:
            raise SystemExit(f"Config already exists: {path}. Use --force to overwrite.")
        config = _config_with_write_args(load_config(args.config), args)
        written = save_config(config, path)
        print(f"Wrote config: {written}")
        return
    if args.config_command == "set":
        config = load_config(args.config)
        data = config.model_dump(mode="json")
        value = _coerce_config_value(args.key, args.value)
        if args.key == "model":
            value = _normalize_default_model_value(data, value)
        _set_nested_value(
            data,
            _config_setter_path(data, args.key),
            value,
        )
        written = save_config(Config.model_validate(data), args.config)
        print(f"Updated {args.key} in {written}")
        return
    raise SystemExit(f"Unknown config command: {args.config_command}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _load_with_path_overrides(
    config_path: Path | None,
    *,
    workspace: Path | None,
    data_dir: Path | None,
    model: str | None = None,
) -> Config:
    config = load_config(config_path)
    if model is not None:
        model = model.strip()
        if not model:
            raise SystemExit("Model name must not be empty.")
        data = config.model_dump(mode="json")
        normalized_model = _normalize_default_model_value(data, model)
        _set_nested_value(data, CONFIG_SETTERS["model"], normalized_model)
        _set_nested_value(data, CONFIG_SETTERS["master-model"], None)
        _set_nested_value(data, ("agents", "routing", "experts"), {})
        config = Config.model_validate(data)
    updates: dict[str, Any] = {
        "workspace": workspace if workspace is not None else Path.cwd(),
    }
    if data_dir is not None:
        updates["data_dir"] = data_dir
    return config.model_copy(update=updates).normalized()


def _config_with_write_args(config: Config, args: argparse.Namespace) -> Config:
    # Map each --init flag to the same nested config path the `set` command uses.
    data = config.model_dump(mode="json")
    if args.mode is not None:
        _set_nested_value(data, CONFIG_SETTERS["mode"], args.mode)
    if args.provider is not None:
        _set_nested_value(data, CONFIG_SETTERS["provider"], args.provider)
    write_arg_keys = {
        "workspace": "workspace",
        "data_dir": "data-dir",
        "api_key": "api-key",
        "model": "model",
    }
    for attr, key in write_arg_keys.items():
        raw = getattr(args, attr, None)
        if raw is None:
            continue
        value = _coerce_config_value(key, str(raw))
        if key == "model":
            value = _normalize_default_model_value(data, value)
        _set_nested_value(
            data,
            _config_setter_path(data, key),
            value,
        )
    return Config.model_validate(data)


def _config_setter_path(data: dict[str, Any], key: str) -> tuple[str, ...]:
    if field := DYNAMIC_PROVIDER_SETTERS.get(key):
        provider = _default_provider_from_data(data)
        return ("providers", provider, field)
    return CONFIG_SETTERS[key]


def _default_provider_from_data(data: dict[str, Any]) -> str:
    agents = data.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    provider = defaults.get("provider") if isinstance(defaults, dict) else None
    if provider not in {"deepseek"}:
        return "deepseek"
    return provider


def _normalize_default_model_value(data: dict[str, Any], value: Any) -> Any:
    """Accept `provider/model` for defaults and split it into provider + model."""

    if not isinstance(value, str) or "/" not in value:
        return value
    provider_candidate, _, model = value.partition("/")
    if provider_candidate not in {"deepseek"} or not model.strip():
        return value
    _set_nested_value(data, CONFIG_SETTERS["provider"], provider_candidate)
    return model.strip()


def _config_dump(config: Config, *, show_secrets: bool) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    if not show_secrets:
        _mask_secrets(data)
    return data


def _mask_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in SECRET_KEYS and isinstance(child, str) and child:
                value[key] = _mask_secret(child)
            else:
                _mask_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _mask_secrets(child)


def _mask_secret(value: str) -> str:
    if value == "no-key":
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _set_nested_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        nested = current.setdefault(key, {})
        if not isinstance(nested, dict):
            nested = {}
            current[key] = nested
        current = nested
    current[path[-1]] = value


def _coerce_config_value(key: str, value: str) -> Any:
    if key in {
        "experts-enabled",
        "mcp-master-allowlist",
        "tools-master-capabilities",
    }:
        return [item.strip() for item in value.split(",") if item.strip()]
    if key == "max-iterations" and value.strip().lower() in {
        "none",
        "null",
        "unlimited",
    }:
        return None
    if key == "context-compaction-preserve-recent-tokens":
        if value.strip().lower() in {"auto", "none", "null"}:
            return None
        return int(value)
    if key in {
        "max-iterations",
        "max-output-tokens",
        "experts-max-calls-per-turn",
        "experts-max-concurrency",
        "experts-stage-request-limit",
        "experts-max-upstream-chars",
        "runtime-stuck-detection-threshold",
        "runtime-max-retries",
    }:
        return int(value)
    if key in {"context-compaction-trigger-ratio", "experts-timeout-seconds"}:
        return float(value)
    if key in {
        "context-compaction-enabled",
        "runtime-transition-trace-enabled",
        "runtime-stuck-detection-enabled",
    }:
        return _parse_bool(value)
    return value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"Expected boolean value, got: {value}")


def _looks_like_legacy_run(argv: list[str]) -> bool:
    first = _first_positional(argv)
    return first is not None and first not in COMMANDS and first not in REMOVED_COMMANDS


def _first_positional(argv: list[str]) -> str | None:
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in {
            "--config",
            "--session",
            "--workspace",
            "--data-dir",
            "--prompt-file",
            "--output",
            "--model",
            "--gateway-log-level",
            "--port",
        }:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return None


def _looks_like_implicit_web(argv: list[str]) -> bool:
    if any(token in {"-h", "--help"} for token in argv):
        return False
    first = _first_positional(argv)
    if first is None:
        return True
    if first in COMMANDS or first in REMOVED_COMMANDS:
        return False
    return any(_is_web_only_option(token) for token in argv)


def _is_web_only_option(token: str) -> bool:
    return token in WEB_ONLY_OPTIONS or any(
        token.startswith(f"{option}=") for option in WEB_ONLY_OPTIONS
    )


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        _run_async(_run_web(build_parser().parse_args(["serve"])))
        return
    if _looks_like_implicit_web(raw_args):
        _run_async(_run_web(build_parser().parse_args(["serve", *raw_args])))
        return
    if _looks_like_legacy_run(raw_args):
        _run_async(_run_prompt(build_legacy_run_parser().parse_args(raw_args)))
        return

    args = build_parser().parse_args(raw_args)
    if args.command == "run":
        _run_async(_run_prompt(args))
        return
    if args.command in {"serve", "web"}:
        _run_async(_run_web(args))
        return
    if args.command == "config":
        _run_config(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


def _run_async(awaitable: Any) -> None:
    """Run one CLI coroutine without printing a traceback for Ctrl-C."""

    try:
        asyncio.run(awaitable)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
