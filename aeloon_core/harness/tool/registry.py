"""Runtime tool registry."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ValidationError

from aeloon_core.harness.tool.base import Tool


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(
        self,
        *,
        execution_guard: Callable[[Tool], Any] | None = None,
        execution_started: Callable[[Tool], Any] | None = None,
        execution_finished: Callable[[Tool], Any] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._execution_guard = execution_guard
        self._execution_started = execution_started
        self._execution_finished = execution_finished

    def register(self, tool: Tool) -> None:
        """Register a tool."""

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""

        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in Anthropic format."""

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
        execution_started = False
        try:
            await self._invoke(self._execution_guard, tool)
            await self._invoke(self._execution_started, tool)
            execution_started = True
            try:
                result = await tool.execute(**args.model_dump(exclude_unset=True))
                await self._invoke(self._execution_guard, tool)
            finally:
                if execution_started:
                    await self._invoke(self._execution_finished, tool)
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

    @staticmethod
    async def _invoke(callback: Callable[[Tool], Any] | None, tool: Tool) -> None:
        if callback is None:
            return
        result = callback(tool)
        if inspect.isawaitable(result):
            await result


def _tool_not_found(name: str, available: Iterable[str]) -> str:
    return (
        f"Error [TOOL_NOT_FOUND]: tool={name!r}; available={list(available)!r}; "
        "next_action='choose one of the advertised tools'"
    )


def _format_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = error.get("msg", "invalid value")
    return f"{location}: {message}" if location else message
