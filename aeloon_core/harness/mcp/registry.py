"""Load explicitly configured MCP servers and resolve per-agent scopes."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict

from aeloon_core.config import Config
from aeloon_core.harness.skill import ExpertSkillSnapshot
from aeloon_core.harness.skill.base import MCP_SERVER_NAME_PATTERN
from aeloon_core.harness.tool import Tool


class McpConfigError(ValueError):
    """Raised when configured MCP servers or scopes are invalid."""


@dataclass(frozen=True, slots=True)
class McpServer:
    """One lazily connected standard MCP server configuration."""

    id: str
    config: Mapping[str, Any]


class McpRegistry:
    """Immutable mapping of configured MCP server ids to lazy server specs."""

    def __init__(self, toolsets: Mapping[str, Any] | None = None) -> None:
        self._toolsets = MappingProxyType(dict(toolsets or {}))

    @classmethod
    def discover(cls, config: Config) -> McpRegistry:
        """Load only the explicit MCP configuration without connecting."""

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
        invalid = [
            name
            for name in servers
            if not isinstance(name, str)
            or re.fullmatch(MCP_SERVER_NAME_PATTERN, name) is None
        ]
        if invalid:
            raise McpConfigError(
                "MCP server ids must match "
                f"{MCP_SERVER_NAME_PATTERN}: {', '.join(map(str, invalid))}"
            )
        parsed: dict[str, McpServer] = {}
        for name, server in servers.items():
            if not isinstance(server, dict):
                raise McpConfigError(f"MCP server {name!r} must be an object")
            if not isinstance(server.get("command") or server.get("url"), str):
                raise McpConfigError(
                    f"MCP server {name!r} requires either command or url"
                )
            parsed[name] = McpServer(name, MappingProxyType(dict(server)))
        return cls(parsed)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._toolsets)

    def master_toolsets(self, config: Config) -> tuple[Any, ...]:
        """Expose all servers in normal mode or the exact expert-mode scope."""

        if config.mode == "normal":
            return tuple(self._toolsets.values())
        return self._select(config.mcp.master_allowlist, owner="Master")

    def expert_toolsets(self, expert: ExpertSkillSnapshot) -> tuple[Any, ...]:
        """Expose only MCP servers declared by one predefined ExpertSkill."""

        return self._select(expert.mcp_servers, owner=f"ExpertSkill {expert.id!r}")

    def _select(
        self,
        server_ids: list[str] | tuple[str, ...],
        *,
        owner: str,
    ) -> tuple[Any, ...]:
        unknown = [server_id for server_id in server_ids if server_id not in self._toolsets]
        if unknown:
            available = ", ".join(self.ids()) or "(none)"
            raise McpConfigError(
                f"{owner} references unknown MCP servers {', '.join(unknown)}; "
                f"configured: {available}"
            )
        return tuple(self._toolsets[server_id] for server_id in server_ids)


class _McpArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class _McpTool(Tool):
    concurrency_mode = "exclusive"
    args_model = _McpArgs

    def __init__(self, *, name: str, description: str, schema: dict[str, Any], group: Any) -> None:
        self.name = name
        self.description = description
        self._schema = schema
        self._group = group

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._schema,
        }

    async def execute(self, **kwargs: Any) -> str:
        result = await self._group.call_tool(self.name, kwargs)
        parts: list[str] = []
        for content in result.content:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(json.dumps(content.model_dump(mode="json"), ensure_ascii=False))
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
        rendered = "\n".join(parts) or "(empty MCP result)"
        return f"Error: {rendered}" if bool(getattr(result, "isError", False)) else rendered


@asynccontextmanager
async def connect_mcp_toolsets(toolsets: tuple[Any, ...]) -> AsyncIterator[tuple[Tool, ...]]:
    """Connect selected MCP servers for exactly one Pi run."""

    from mcp.client.session_group import (
        ClientSessionGroup,
        SseServerParameters,
        StreamableHttpParameters,
    )
    from mcp.client.stdio import StdioServerParameters

    stack = AsyncExitStack()
    tools: list[Tool] = []
    try:
        await stack.__aenter__()
        for toolset in toolsets:
            if isinstance(toolset, Tool):
                tools.append(toolset)
                continue
            if not isinstance(toolset, McpServer):
                continue
            group = ClientSessionGroup()
            await stack.enter_async_context(group)
            config = dict(toolset.config)
            if "command" in config:
                params: Any = StdioServerParameters(
                    command=str(config["command"]),
                    args=[str(item) for item in config.get("args", [])],
                    env={str(key): str(value) for key, value in config.get("env", {}).items()}
                    or None,
                    cwd=config.get("cwd"),
                )
            elif config.get("transport") == "sse":
                params = SseServerParameters(
                    url=str(config["url"]),
                    headers=config.get("headers"),
                )
            else:
                params = StreamableHttpParameters(
                    url=str(config["url"]),
                    headers=config.get("headers"),
                    timeout=timedelta(seconds=float(config.get("timeout", 30))),
                    sse_read_timeout=timedelta(
                        seconds=float(config.get("sse_read_timeout", 300))
                    ),
                )
            await group.connect_to_server(params)
            for name, definition in group.tools.items():
                tools.append(
                    _McpTool(
                        name=name,
                        description=str(definition.description or ""),
                        schema=dict(definition.inputSchema or {"type": "object"}),
                        group=group,
                    )
                )
        yield tuple(tools)
    finally:
        await stack.aclose()


__all__ = [
    "McpConfigError",
    "McpRegistry",
    "McpServer",
    "connect_mcp_toolsets",
]
