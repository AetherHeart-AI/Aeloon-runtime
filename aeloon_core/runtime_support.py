"""Shared services used by the state-machine runtime."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aeloon_core.providers.base import LLMProvider, ToolCallRequest
from aeloon_core.task_graph import TaskNode, TaskState, build_task_graph

if TYPE_CHECKING:
    from aeloon_core.tools.registry import ToolRegistry

Message = dict[str, Any]


def default_strip_think(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None


def _suffix_prefix_len(text: str, prefix: str) -> int:
    max_len = min(len(text), len(prefix) - 1)
    for size in range(max_len, 0, -1):
        if prefix.startswith(text[-size:]):
            return size
    return 0


class ThinkTagDeltaFilter:
    """Hide streamed <think>...</think> spans while preserving visible deltas."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        visible: list[str] = []
        while self._buffer:
            if self._inside:
                end = self._buffer.find(self._CLOSE)
                if end < 0:
                    keep = _suffix_prefix_len(self._buffer, self._CLOSE)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(visible)
                self._buffer = self._buffer[end + len(self._CLOSE) :]
                self._inside = False
                continue

            start = self._buffer.find(self._OPEN)
            if start < 0:
                keep = _suffix_prefix_len(self._buffer, self._OPEN)
                emit = self._buffer[:-keep] if keep else self._buffer
                if emit:
                    visible.append(emit)
                self._buffer = self._buffer[-keep:] if keep else ""
                return "".join(visible)

            if start > 0:
                visible.append(self._buffer[:start])
            self._buffer = self._buffer[start + len(self._OPEN) :]
            self._inside = True
        return "".join(visible)

    def flush(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        return tail


def provider_supports_streaming(provider: LLMProvider) -> bool:
    chat_stream = getattr(type(provider), "chat_stream", None)
    return chat_stream is not None and chat_stream is not LLMProvider.chat_stream


def default_tool_hint(tool_calls: list[ToolCallRequest]) -> str:
    def _fmt(tool_call: ToolCallRequest) -> str:
        args = tool_call.arguments or {}
        val = next(iter(args.values()), None) if isinstance(args, dict) else None
        if not isinstance(val, str):
            return tool_call.name
        if len(val) > 40:
            return f'{tool_call.name}("{val[:40]}...")'
        return f'{tool_call.name}("{val}")'

    return ", ".join(_fmt(tool_call) for tool_call in tool_calls)


def build_assistant_message(
    content: str | None,
    *,
    tool_uses: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
) -> Message:
    """Build one Anthropic assistant message from typed content blocks."""

    del reasoning_content
    blocks: list[dict[str, Any]] = []
    for block in thinking_blocks or []:
        if isinstance(block, dict) and block.get("type") in {
            "thinking",
            "redacted_thinking",
        }:
            blocks.append(dict(block))
    if content:
        blocks.append({"type": "text", "text": content})
    blocks.extend(dict(tool_use) for tool_use in tool_uses or [])
    if not blocks:
        blocks.append({"type": "text", "text": "(empty)"})
    return {"role": "assistant", "content": blocks}


def default_add_assistant_message(
    messages: list[Message],
    content: str | None,
    tool_uses: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
) -> list[Message]:
    messages.append(
        build_assistant_message(
            content,
            tool_uses=tool_uses,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    return messages


def default_add_tool_result(
    messages: list[Message],
    tool_call_id: str,
    tool_name: str,
    result: str,
) -> list[Message]:
    block = {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": result,
        "is_error": result.lstrip().lower().startswith("error"),
    }
    if messages and _is_tool_result_message(messages[-1]):
        messages[-1]["content"].append(block)
    else:
        messages.append({"role": "user", "content": [block]})
    return messages


def _is_tool_result_message(message: Message) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and all(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
    )


async def execute_tool_batch(
    *,
    tool_calls: list[ToolCallRequest],
    tools: ToolRegistry,
    on_node_complete: Callable[[TaskNode], Awaitable[None]] | None = None,
) -> list[TaskNode]:
    nodes = build_task_graph(tool_calls, tools)
    pending = {node.index: node for node in nodes}
    running: dict[int, asyncio.Task[str]] = {}

    async def _execute_node(node: TaskNode) -> str:
        return await tools.execute(node.tool_name, node.arguments)

    async def _notify(node: TaskNode) -> None:
        if on_node_complete is not None:
            await on_node_complete(node)

    try:
        while pending or running:
            ready = [node for node in pending.values() if not node.deps]
            for node in ready:
                node.state = TaskState.RUNNING
                running[node.index] = asyncio.create_task(_execute_node(node))
                pending.pop(node.index)

            if not running:
                raise RuntimeError("deadlock detected in tool task graph")

            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            finished_indexes = [index for index, task in running.items() if task in done]

            for index in finished_indexes:
                task = running.pop(index)
                node = nodes[index]
                try:
                    node.result = await task
                    node.state = TaskState.DONE
                except asyncio.CancelledError:
                    node.state = TaskState.CANCELLED
                    raise
                except Exception as exc:
                    node.state = TaskState.FAILED
                    node.error = str(exc)
                    node.result = f"Error executing {node.tool_name}: {exc}"

                await _notify(node)
                for dependent_index in node.dependents:
                    nodes[dependent_index].deps.discard(index)
    except BaseException:
        for task in running.values():
            task.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        for node in nodes:
            if node.state not in {TaskState.PENDING, TaskState.RUNNING}:
                continue
            node.state = TaskState.CANCELLED
            node.error = "cancelled"
            node.result = f"Error: Tool '{node.tool_name}' execution was cancelled."
            try:
                await _notify(node)
            except BaseException:
                pass
        raise
    return nodes


_PROVIDER_TOOL_CALL_ARGS_MAX_CHARS = 4096
_PROVIDER_TOOL_ARG_STRING_MAX_CHARS = 1024


def _shrink_oversized_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(arguments)
    for key, value in parsed.items():
        if isinstance(value, str) and len(value) > _PROVIDER_TOOL_ARG_STRING_MAX_CHARS:
            omitted = len(value) - _PROVIDER_TOOL_ARG_STRING_MAX_CHARS
            parsed[key] = (
                value[:_PROVIDER_TOOL_ARG_STRING_MAX_CHARS] + f"... [truncated {omitted} chars]"
            )
    if len(json.dumps(parsed, ensure_ascii=False)) <= _PROVIDER_TOOL_CALL_ARGS_MAX_CHARS:
        return parsed
    return {"_compacted_tool_arguments": True, "keys": list(parsed.keys())[:20]}


def shrink_answered_tool_args_for_provider(messages: list[Message]) -> list[Message]:
    answered = {
        block["tool_use_id"]
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_use_id"), str)
    }
    if not answered:
        return messages

    out: list[Message] = []
    changed = False
    for message in messages:
        blocks = message.get("content") if message.get("role") == "assistant" else None
        if not isinstance(blocks, list):
            out.append(message)
            continue
        new_blocks: list[Any] = []
        message_changed = False
        for block in blocks:
            arguments = block.get("input") if isinstance(block, dict) else None
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") in answered
                and isinstance(arguments, dict)
                and len(json.dumps(arguments, ensure_ascii=False))
                > _PROVIDER_TOOL_CALL_ARGS_MAX_CHARS
            ):
                shrunk = _shrink_oversized_tool_arguments(arguments)
                if shrunk != arguments:
                    new_blocks.append({**block, "input": shrunk})
                    message_changed = True
                    continue
            new_blocks.append(block)
        out.append({**message, "content": new_blocks} if message_changed else message)
        changed = changed or message_changed
    return out if changed else messages
