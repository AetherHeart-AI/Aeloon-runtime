"""Tool registries for dynamic and role-scoped capability management."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from aeloon_core.tools.base import Tool


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""

        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""

        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name, validating params against its Pydantic model."""

        hint = "\n\n[Analyze the error above and try a different approach.]"
        tool = self._tools.get(name)
        if not tool:
            return _tool_not_found(name, self._tools)
        try:
            args = tool.args_model.model_validate(params)
        except ValidationError as exc:
            errors = "; ".join(_format_error(error) for error in exc.errors())
            return (
                f"Error [TOOL_ARGUMENTS_INVALID]: tool={name!r}; errors={errors!r}; "
                "next_action='retry with one JSON object matching the advertised tool schema'"
                f"{hint}"
            )
        try:
            result = await tool.execute(**args.model_dump(exclude_unset=True))
            if isinstance(result, str) and result.startswith("Error"):
                return result + hint
            return result
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return (
                f"Error [TOOL_EXECUTION_ERROR]: tool={name!r}; "
                f"actual={detail!r}; "
                "next_action='inspect the error and retry with a corrected call'"
                f"{hint}"
            )


class ScopedToolRegistry:
    """A read/execute-only view that can only narrow a host registry."""

    def __init__(self, registry: ToolRegistry, allowed_tools: Iterable[str]) -> None:
        self._registry = registry
        requested = frozenset(allowed_tools)
        self._allowed_names = tuple(
            name
            for definition in registry.get_definitions()
            if (name := _definition_name(definition)) is not None and name in requested
        )

    def get(self, name: str) -> Tool | None:
        """Return a tool only when it remains present in the allowed host scope."""

        if name not in self._allowed_names:
            return None
        return self._registry.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return schemas for currently present tools in the fixed allowed scope."""

        return [
            tool.to_schema()
            for name in self._allowed_names
            if (tool := self.get(name)) is not None
        ]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Re-check scope immediately before delegating an allowed execution."""

        if self.get(name) is None:
            return _tool_not_found(name, self._available_names())
        return await self._registry.execute(name, params)

    def _available_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._allowed_names if self.get(name) is not None)


def _definition_name(definition: dict[str, Any]) -> str | None:
    function = definition.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _tool_not_found(name: str, available: Iterable[str]) -> str:
    return (
        f"Error [TOOL_NOT_FOUND]: tool={name!r}; available={list(available)!r}; "
        "next_action='choose one of the advertised tools'"
    )


def _format_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = error.get("msg", "invalid value")
    return f"{location}: {message}" if location else message
