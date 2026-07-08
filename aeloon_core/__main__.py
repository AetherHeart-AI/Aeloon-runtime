"""CLI entry point for Aeloon Core."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aeloon_core.config import Config, load_config, resolve_config_path, save_config
from aeloon_core.orchestrator import AeloonCoreOrchestrator, ConsoleProgress
from aeloon_core.terminal_cli import LOG_LEVELS, run_terminal_cli

COMMANDS = {"run", "chat", "tui", "config"}
REMOVED_COMMANDS = {"webui"}
CHAT_ONLY_OPTIONS = {"--hide-gateway-logs", "--gateway-log-detail", "--gateway-log-level"}
CONFIG_SETTERS = {
    "workspace": ("workspace",),
    "data-dir": ("data_dir",),
    "api-key": ("providers", "custom", "api_key"),
    "api-base": ("providers", "custom", "api_base"),
    "model": ("agents", "defaults", "model"),
    "max-tokens": ("agents", "defaults", "max_tokens"),
    "reasoning-effort": ("agents", "defaults", "reasoning_effort"),
    "max-iterations": ("agents", "defaults", "max_iterations"),
    "max-auto-continue-iterations": ("agents", "defaults", "max_auto_continue_iterations"),
    "max-finalization-iterations": ("agents", "defaults", "max_finalization_iterations"),
    "skills-enabled": ("skills", "enabled"),
    "skills-external": ("skills", "external"),
    "skills-claude-code": ("skills", "claude_code"),
    "skills-paths": ("skills", "paths"),
}
SECRET_KEYS = {"api_key"}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Run Aeloon Core.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one agent turn.")
    _add_run_args(run_parser)

    chat_parser = subparsers.add_parser("chat", help="Start the interactive terminal CLI.")
    _add_chat_args(chat_parser)

    tui_parser = subparsers.add_parser("tui", help="Alias for chat.")
    _add_chat_args(tui_parser)

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
    config_set_parser.add_argument("key", choices=sorted(CONFIG_SETTERS))
    config_set_parser.add_argument("value")

    return parser


def build_legacy_run_parser() -> argparse.ArgumentParser:
    """Build the legacy prompt-first parser."""

    parser = argparse.ArgumentParser(description="Run one Aeloon Core agent turn.")
    _add_run_args(parser)
    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt", nargs="+", help="Prompt text to send to the agent.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--session", default=None, help="Existing session id to continue.")
    parser.add_argument("--workspace", type=Path, default=None, help="Override workspace.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data dir.")


def _add_chat_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt to run once. Omit it for interactive chat.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--session", default=None, help="Existing session id to continue.")
    parser.add_argument("--workspace", type=Path, default=None, help="Override workspace.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data dir.")
    parser.add_argument(
        "--hide-gateway-logs",
        action="store_true",
        help="Hide compact gateway log lines.",
    )
    parser.add_argument(
        "--gateway-log-level",
        choices=sorted(LOG_LEVELS),
        default="INFO",
        help="Minimum gateway log level to display.",
    )
    parser.add_argument(
        "--gateway-log-detail",
        action="store_true",
        help="Print gateway log detail JSON.",
    )


def _add_config_write_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace to operate on.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Session data directory.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key.")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default=None, help="Default model.")
    parser.add_argument(
        "--max-tokens",
        default=None,
        help="Output token budget per model call. Use 'auto' for model-aware defaults.",
    )


async def _run_prompt(args: argparse.Namespace) -> None:
    config = _load_with_path_overrides(
        args.config,
        workspace=getattr(args, "workspace", None),
        data_dir=getattr(args, "data_dir", None),
    )
    orchestrator = AeloonCoreOrchestrator(config)
    prompt = " ".join(args.prompt)
    result = await orchestrator.run_turn(
        prompt,
        session_id=args.session,
        on_progress=ConsoleProgress(),
    )
    print(f"\n[session] {result.session_id}")
    if result.tools_used:
        print(f"[tools used] {', '.join(result.tools_used)}")


async def _run_chat(args: argparse.Namespace) -> None:
    config = _load_with_path_overrides(
        args.config,
        workspace=getattr(args, "workspace", None),
        data_dir=getattr(args, "data_dir", None),
    )
    prompt = " ".join(args.prompt).strip() or None
    await run_terminal_cli(
        config,
        prompt=prompt,
        session_id=args.session,
        show_gateway_logs=not args.hide_gateway_logs,
        gateway_log_level=args.gateway_log_level,
        gateway_log_detail=args.gateway_log_detail,
    )


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
        _set_nested_value(
            data, CONFIG_SETTERS[args.key], _coerce_config_value(args.key, args.value)
        )
        written = save_config(Config.model_validate(data), args.config)
        print(f"Updated {args.key} in {written}")
        return
    raise SystemExit(f"Unknown config command: {args.config_command}")


def _load_with_path_overrides(
    config_path: Path | None,
    *,
    workspace: Path | None,
    data_dir: Path | None,
) -> Config:
    config = load_config(config_path)
    updates: dict[str, Any] = {
        "workspace": workspace if workspace is not None else Path.cwd(),
    }
    if data_dir is not None:
        updates["data_dir"] = data_dir
    return config.model_copy(update=updates).normalized()


def _config_with_write_args(config: Config, args: argparse.Namespace) -> Config:
    data = config.model_dump(mode="json")
    if args.workspace is not None:
        data["workspace"] = str(args.workspace)
    if args.data_dir is not None:
        data["data_dir"] = str(args.data_dir)
    if args.api_key is not None:
        data.setdefault("providers", {}).setdefault("custom", {})["api_key"] = args.api_key
    if args.api_base is not None:
        data.setdefault("providers", {}).setdefault("custom", {})["api_base"] = args.api_base
    if args.model is not None:
        data.setdefault("agents", {}).setdefault("defaults", {})["model"] = args.model
    if args.max_tokens is not None:
        data.setdefault("agents", {}).setdefault("defaults", {})["max_tokens"] = (
            _coerce_config_value("max-tokens", args.max_tokens)
        )
    return Config.model_validate(data)


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
        "max-tokens",
        "max-iterations",
        "max-auto-continue-iterations",
        "max-finalization-iterations",
    }:
        if key == "max-tokens" and value.strip().lower() in {"auto", "none", "null"}:
            return None
        return int(value)
    if key in {"skills-enabled", "skills-external", "skills-claude-code"}:
        return _parse_bool(value)
    if key == "skills-paths":
        return [item.strip() for item in value.split(",") if item.strip()]
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
        if token in {"--config", "--session", "--workspace", "--data-dir", "--gateway-log-level"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return None


def _looks_like_implicit_chat(argv: list[str]) -> bool:
    if any(token in {"-h", "--help"} for token in argv):
        return False
    first = _first_positional(argv)
    if first is None:
        return True
    return any(_is_chat_only_option(token) for token in argv)


def _is_chat_only_option(token: str) -> bool:
    return token in CHAT_ONLY_OPTIONS or any(
        token.startswith(f"{option}=") for option in CHAT_ONLY_OPTIONS
    )


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        asyncio.run(_run_chat(build_parser().parse_args(["chat"])))
        return
    if _looks_like_implicit_chat(raw_args):
        asyncio.run(_run_chat(build_parser().parse_args(["chat", *raw_args])))
        return
    if _looks_like_legacy_run(raw_args):
        asyncio.run(_run_prompt(build_legacy_run_parser().parse_args(raw_args)))
        return

    args = build_parser().parse_args(raw_args)
    if args.command == "run":
        asyncio.run(_run_prompt(args))
        return
    if args.command in {"chat", "tui"}:
        asyncio.run(_run_chat(args))
        return
    if args.command == "config":
        _run_config(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
