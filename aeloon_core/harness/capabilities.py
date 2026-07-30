"""Shared Pydantic AI Harness capability construction."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.compaction import SlidingWindow
from pydantic_ai_harness.context import RepoContext
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

from aeloon_core.config import Config

HARNESS_PROTECTED_PATTERNS = (
    ".aeloon-core/*",
    ".git/*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "**/secrets*",
)
DENIED_ENV_PATTERNS = (
    *LLM_API_KEY_ENV_PATTERNS,
    "ARK_*",
    "AELOON_CORE_API_KEY",
    "*API_KEY*",
    "*CREDENTIAL*",
    "*PASSWORD*",
    "*SECRET*",
    "*TOKEN*",
    "DATABASE_URL",
    "SSH_AUTH_SOCK",
)


class CapabilityUnavailable(RuntimeError):
    """Raised when an optional Expert capability is not configured."""


WebCapabilityFactory = Callable[[], Any]
MASTER_CAPABILITY_NAMES = ("filesystem", "shell", "repo_context", "planning")


def history_capability(config: Config) -> SlidingWindow[Any] | None:
    """Translate Aeloon's context policy into Harness' zero-LLM compactor."""

    compaction = config.agents.defaults.context_compaction
    if not compaction.enabled:
        return None
    trigger_tokens = max(
        1,
        int(config.agents.defaults.context_window_tokens * compaction.trigger_ratio),
    )
    keep_tokens = compaction.preserve_recent_tokens or max(8_000, trigger_tokens // 2)
    keep_tokens = min(keep_tokens, max(0, trigger_tokens - 1))
    return SlidingWindow(
        max_tokens=trigger_tokens,
        keep_tokens=keep_tokens,
        preserve_first_user_message=True,
    )


def harness_capabilities(
    *,
    config: Config,
    names: Iterable[str],
    web_capability_factory: WebCapabilityFactory | None = None,
) -> list[Any]:
    """Build exactly the trusted capabilities declared for one agent."""

    requested = frozenset(names)
    capabilities: list[Any] = []
    if "filesystem" in requested:
        capabilities.append(
            FileSystem[Any](
                root_dir=config.workspace,
                protected_patterns=HARNESS_PROTECTED_PATTERNS,
            )
        )
    if "shell" in requested:
        capabilities.append(
            Shell[Any](
                cwd=config.workspace,
                default_timeout=float(config.tools.exec.timeout),
                denied_env_patterns=DENIED_ENV_PATTERNS,
            )
        )
    if "repo_context" in requested:
        capabilities.append(
            RepoContext[Any](
                workspace_dir=config.workspace,
                nested_traversal=True,
            )
        )
    if "planning" in requested:
        capabilities.append(Planning[Any]())
    if "web_search" in requested:
        capabilities.append(
            web_capability_factory()
            if web_capability_factory is not None
            else _default_exa_capability()
        )
    compaction = history_capability(config)
    if compaction is not None:
        capabilities.append(compaction)
    return capabilities


def master_capabilities(config: Config) -> list[Any]:
    """Build the mode-specific Master capability surface."""

    return harness_capabilities(
        config=config,
        names=master_capability_names(config),
    )


def master_capability_names(config: Config) -> tuple[str, ...]:
    """Expose the full normal surface or the configured expert-mode subset."""

    if config.mode == "normal":
        return MASTER_CAPABILITY_NAMES
    return tuple(config.tools.master_capabilities)


def _default_exa_capability() -> Any:
    if not os.environ.get("EXA_API_KEY"):
        raise CapabilityUnavailable(
            "research expert requires EXA_API_KEY for the default Exa web backend"
        )
    try:
        from pydantic_ai_harness.exa import ExaSearch
    except ImportError as exc:
        raise CapabilityUnavailable(
            "research expert requires the Exa dependency; run `uv sync` to install it"
        ) from exc
    return ExaSearch()


__all__ = [
    "CapabilityUnavailable",
    "DENIED_ENV_PATTERNS",
    "HARNESS_PROTECTED_PATTERNS",
    "MASTER_CAPABILITY_NAMES",
    "WebCapabilityFactory",
    "harness_capabilities",
    "history_capability",
    "master_capabilities",
    "master_capability_names",
]
