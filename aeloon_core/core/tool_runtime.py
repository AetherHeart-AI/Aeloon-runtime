"""Tool registration, validation, execution, and cancellation for one run."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from aeloon_core.core.events import RunEventDispatcher
from aeloon_core.core.types import (
    RunError,
    TextContent,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    content_from_dict,
    content_to_dict,
)


class ToolRuntime:
    """Own the tool registry and all in-flight tool tasks."""

    def __init__(
        self,
        tools: Iterable[Tool],
        active_names: Sequence[str],
        events: RunEventDispatcher,
    ) -> None:
        self._events = events
        self._tools = self._unique_tools(tuple(tools))
        self._active_names = self._validate_active_names(active_names)
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @property
    def active_tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools[name] for name in self._active_names)

    @property
    def active_names(self) -> tuple[str, ...]:
        return self._active_names

    def cancel(self) -> None:
        for task in tuple(self._tasks):
            if not task.done():
                task.cancel()

    async def execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        *,
        is_aborted: Callable[[], bool],
    ) -> tuple[list[ToolResultMessage], bool]:
        sequential = any(
            self._tools.get(call.name) is not None
            and self._tools[call.name].execution_mode == "sequential"
            for call in calls
        )
        results: list[tuple[ToolResultMessage, bool]] = []
        if sequential:
            for index, call in enumerate(calls):
                task = self._start_call(call)
                try:
                    results.append(await task)
                finally:
                    self._tasks.discard(task)
                if is_aborted():
                    for skipped in calls[index + 1 :]:
                        results.append(await self._fail_unexecuted_call(skipped))
                    break
        else:
            tasks = [self._start_call(call) for call in calls]
            try:
                results = list(await asyncio.gather(*tasks))
            finally:
                self._tasks.difference_update(tasks)
        terminate = any(item[1] for item in results)
        return [item[0] for item in results], terminate

    async def _fail_unexecuted_call(
        self,
        call: ToolCall,
    ) -> tuple[ToolResultMessage, bool]:
        await self._events.emit(
            "tool_execution_start",
            {"toolCallId": call.id, "toolName": call.name, "args": call.arguments},
        )
        result = ToolResult.text("Operation aborted before execution", is_error=True)
        await self._events.emit(
            "tool_execution_end",
            {
                "toolCallId": call.id,
                "toolName": call.name,
                "result": tool_result_to_dict(result),
                "isError": True,
            },
        )
        return (
            ToolResultMessage(
                call.id,
                call.name,
                result.content,
                is_error=True,
            ),
            False,
        )

    async def fail_truncated_calls(self, calls: tuple[ToolCall, ...]) -> list[ToolResultMessage]:
        results: list[ToolResultMessage] = []
        for call in calls:
            await self._events.emit(
                "tool_execution_start",
                {"toolCallId": call.id, "toolName": call.name, "args": call.arguments},
            )
            text = (
                f'Tool call "{call.name}" was not executed: the response hit the output token '
                "limit, so its arguments may be truncated. Re-issue the tool call "
                "with complete arguments."
            )
            result = ToolResult.text(text, is_error=True)
            await self._events.emit(
                "tool_execution_end",
                {
                    "toolCallId": call.id,
                    "toolName": call.name,
                    "result": tool_result_to_dict(result),
                    "isError": True,
                },
            )
            results.append(ToolResultMessage(call.id, call.name, result.content, is_error=True))
        return results

    def _start_call(self, call: ToolCall) -> asyncio.Task[tuple[ToolResultMessage, bool]]:
        task = asyncio.create_task(self._execute_call(call))
        self._tasks.add(task)
        return task

    async def _execute_call(self, call: ToolCall) -> tuple[ToolResultMessage, bool]:
        await self._events.emit(
            "tool_execution_start",
            {"toolCallId": call.id, "toolName": call.name, "args": call.arguments},
        )
        tool = self._tools.get(call.name)
        args = dict(call.arguments)
        if tool is None or call.name not in self._active_names:
            result = ToolResult.text(f"Tool {call.name} not found", is_error=True)
        else:
            try:
                args = tool.prepare_arguments(args)
                Draft202012Validator(tool.parameters).validate(args)
                hook = await self._events.hook(
                    "tool_call",
                    {"toolCallId": call.id, "toolName": call.name, "input": args},
                )
                if hook.get("block"):
                    result = ToolResult.text(
                        str(hook.get("reason") or "Tool call blocked"),
                        is_error=True,
                    )
                else:
                    result = await tool.execute(call.id, args, self._update_callback(call))
            except asyncio.CancelledError:
                result = ToolResult.text("Operation aborted", is_error=True)
            except Exception as exc:
                result = ToolResult.text(f"{type(exc).__name__}: {exc}", is_error=True)
        result = await self._apply_result_hook(call, args, result)
        await self._events.emit(
            "tool_execution_end",
            {
                "toolCallId": call.id,
                "toolName": call.name,
                "result": tool_result_to_dict(result),
                "isError": result.is_error,
            },
        )
        message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            is_error=result.is_error,
            usage=result.usage,
        )
        return message, result.terminate

    def _update_callback(self, call: ToolCall) -> Callable[[ToolResult], Any]:
        async def on_update(update: ToolResult) -> None:
            await self._events.emit(
                "tool_execution_update",
                {
                    "toolCallId": call.id,
                    "toolName": call.name,
                    "partialResult": tool_result_to_dict(update),
                },
            )

        return on_update

    async def _apply_result_hook(
        self,
        call: ToolCall,
        args: dict[str, Any],
        result: ToolResult,
    ) -> ToolResult:
        hook = await self._events.hook(
            "tool_result",
            {
                "toolCallId": call.id,
                "toolName": call.name,
                "input": args,
                **tool_result_to_dict(result),
            },
        )
        if not hook:
            return result
        raw_content = hook.get("content", result.content)
        if isinstance(raw_content, str):
            raw_content = (TextContent(raw_content),)
        content = tuple(
            content_from_dict(part) if isinstance(part, Mapping) else part for part in raw_content
        )
        raw_usage = hook.get("usage", result.usage)
        return ToolResult(
            content=content,
            details=hook.get("details", result.details),
            is_error=bool(hook.get("isError", result.is_error)),
            terminate=bool(hook.get("terminate", result.terminate)),
            usage=Usage.from_dict(raw_usage) if isinstance(raw_usage, Mapping) else raw_usage,
        )

    @staticmethod
    def _unique_tools(tools: Sequence[Tool]) -> dict[str, Tool]:
        result: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in result:
                raise RunError("invalid_argument", f"Duplicate tool name: {tool.name}")
            result[tool.name] = tool
        return result

    def _validate_active_names(
        self,
        names: Sequence[str],
        tools: Mapping[str, Tool] | None = None,
    ) -> tuple[str, ...]:
        available = self._tools if tools is None else tools
        result = tuple(dict.fromkeys(names))
        unknown = [name for name in result if name not in available]
        if unknown:
            raise RunError(
                "invalid_argument",
                f"Unknown active tools: {', '.join(unknown)}",
            )
        return result


def tool_result_to_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "content": [content_to_dict(part) for part in result.content],
        "details": result.details,
        "isError": result.is_error,
        "terminate": result.terminate,
        "usage": result.usage.to_dict() if result.usage else None,
    }


__all__ = ["ToolRuntime", "tool_result_to_dict"]
