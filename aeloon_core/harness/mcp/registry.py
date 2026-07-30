"""Load explicitly configured MCP servers and resolve per-agent scopes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic_ai.toolsets import AbstractToolset

from aeloon_core.config import Config
from aeloon_core.harness.skill import ExpertSkillSnapshot
from aeloon_core.harness.skill.base import MCP_SERVER_NAME_PATTERN


class McpConfigError(ValueError):
    """Raised when configured MCP servers or scopes are invalid."""


class McpRegistry:
    """Immutable mapping of configured MCP server ids to prefixed toolsets."""

    def __init__(self, toolsets: Mapping[str, AbstractToolset[Any]] | None = None) -> None:
        self._toolsets = MappingProxyType(dict(toolsets or {}))

    @classmethod
    def discover(cls, config: Config) -> McpRegistry:
        """Load only the explicit Pydantic AI MCP configuration, when present."""

        path = config.mcp.config_path
        if path is None:
            return cls()
        if not path.is_file():
            raise McpConfigError(f"MCP config does not exist or is not a file: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise McpConfigError(f"could not read MCP config {path}: {exc}") from exc
        servers = raw.get("mcpServers") if isinstance(raw, dict) else None
        if not isinstance(servers, dict):
            raise McpConfigError(f"MCP config {path} requires an mcpServers object")
        names = list(servers)
        invalid = [
            name
            for name in names
            if not isinstance(name, str)
            or re.fullmatch(MCP_SERVER_NAME_PATTERN, name) is None
        ]
        if invalid:
            raise McpConfigError(
                "MCP server ids must match "
                f"{MCP_SERVER_NAME_PATTERN}: {', '.join(map(str, invalid))}"
            )
        try:
            from pydantic_ai.mcp import load_mcp_toolsets

            toolsets = load_mcp_toolsets(path)
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise McpConfigError(f"could not load MCP config {path}: {exc}") from exc
        if len(toolsets) != len(names):
            raise McpConfigError(
                f"MCP config {path} loaded {len(toolsets)} servers for {len(names)} ids"
            )
        return cls(dict(zip(names, toolsets, strict=True)))

    def ids(self) -> tuple[str, ...]:
        return tuple(self._toolsets)

    def master_toolsets(self, config: Config) -> tuple[AbstractToolset[Any], ...]:
        """Expose all configured servers in normal mode or the exact expert-mode scope."""

        if config.mode == "normal":
            return tuple(self._toolsets.values())
        return self._select(
            config.mcp.master_allowlist,
            owner="Master",
        )

    def expert_toolsets(
        self,
        expert: ExpertSkillSnapshot,
    ) -> tuple[AbstractToolset[Any], ...]:
        """Expose only MCP servers declared by one predefined ExpertSkill."""

        return self._select(expert.mcp_servers, owner=f"ExpertSkill {expert.id!r}")

    def _select(
        self,
        server_ids: list[str] | tuple[str, ...],
        *,
        owner: str,
    ) -> tuple[AbstractToolset[Any], ...]:
        unknown = [server_id for server_id in server_ids if server_id not in self._toolsets]
        if unknown:
            available = ", ".join(self.ids()) or "(none)"
            raise McpConfigError(
                f"{owner} references unknown MCP servers {', '.join(unknown)}; "
                f"configured: {available}"
            )
        return tuple(self._toolsets[server_id] for server_id in server_ids)


__all__ = ["McpConfigError", "McpRegistry"]
