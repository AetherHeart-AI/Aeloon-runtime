"""Tool registry for dynamic tool management."""

from __future__ import annotations

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
            return f"Error: Tool '{name}' not found. Available: {', '.join(self._tools)}"
        try:
            args = tool.args_model.model_validate(params)
        except ValidationError as exc:
            errors = "; ".join(_format_error(error) for error in exc.errors())
            return f"Error: Invalid parameters for tool '{name}': {errors}{hint}"
        try:
            result = await tool.execute(**args.model_dump(exclude_unset=True))
            if isinstance(result, str) and result.startswith("Error"):
                return result + hint
            return result
        except Exception as exc:
            return f"Error executing {name}: {exc}" + hint


def _format_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = error.get("msg", "invalid value")
    return f"{location}: {message}" if location else message
