"""Deterministic forward context views for model calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from aeloon_core.context_compaction import is_compaction_message

if TYPE_CHECKING:
    from aeloon_core.state import LightweightState


LAZY_TOOL_RESULT_MARKER = "[aeloon-core:lazy-tool-result]"


@dataclass(frozen=True)
class ForwardContextResult:
    """A model-call view derived from canonical state without replacing it."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    lazy_references: tuple[str, ...] = ()
    original_message_count: int = 0


class ContextProcessor(Protocol):
    """Build a forward-only model context from canonical messages."""

    def process(
        self,
        *,
        state: LightweightState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        additional_messages: list[dict[str, Any]] | None = None,
    ) -> ForwardContextResult:
        """Return a fresh call view without modifying ``messages`` or ``state.messages``."""


class MinimalContextProcessor:
    """Select recent complete turns and isolate large tool results by lazy reference."""

    def __init__(
        self,
        *,
        preserve_recent_turns: int = 2,
        max_tool_result_chars: int = 8_000,
    ) -> None:
        if preserve_recent_turns < 1:
            raise ValueError("preserve_recent_turns must be at least 1")
        if max_tool_result_chars < 1:
            raise ValueError("max_tool_result_chars must be at least 1")
        self.preserve_recent_turns = preserve_recent_turns
        self.max_tool_result_chars = max_tool_result_chars

    def process(
        self,
        *,
        state: LightweightState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        additional_messages: list[dict[str, Any]] | None = None,
    ) -> ForwardContextResult:
        selected_indexes = _selected_message_indexes(messages, self.preserve_recent_turns)
        selected_indexes = _complete_tool_pairs(messages, selected_indexes)
        current_turn_start = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user"
            ),
            default=-1,
        )

        lazy_references: list[str] = []
        call_messages = []
        for index in sorted(selected_indexes):
            message = _copy_message(messages[index])
            reference = self._replace_large_tool_result(
                state,
                message,
                preserve_current_skill=index > current_turn_start,
            )
            if reference is not None and reference not in lazy_references:
                lazy_references.append(reference)
            call_messages.append(message)
        call_messages.extend(_copy_message(message) for message in additional_messages or [])

        return ForwardContextResult(
            messages=call_messages,
            tools=_filter_tools(tools, state.active_tools),
            lazy_references=tuple(lazy_references),
            original_message_count=len(messages),
        )

    def _replace_large_tool_result(
        self,
        state: LightweightState,
        message: dict[str, Any],
        *,
        preserve_current_skill: bool,
    ) -> str | None:
        if message.get("role") != "tool" or "content" not in message:
            return None
        if (
            preserve_current_skill
            and message.get("name") == "skill"
        ):
            return None
        content = message.get("content")
        serialized = _serialized_content(content)
        if len(serialized) <= self.max_tool_result_chars:
            return None

        reference = state.store_lazy(content, prefix="tool-result")
        preview = _bounded_preview(serialized, self.max_tool_result_chars)
        message["content"] = (
            f"{LAZY_TOOL_RESULT_MARKER}\n"
            f"ref: {reference}\n"
            f"characters: {len(serialized)}\n"
            f"bounded preview:\n{preview}"
        )
        return reference


def _selected_message_indexes(
    messages: list[dict[str, Any]],
    preserve_recent_turns: int,
) -> set[int]:
    if not messages:
        return set()

    leading_system = _leading_system_indexes(messages)
    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if len(user_indexes) <= preserve_recent_turns:
        return set(range(len(messages)))

    tail_start = user_indexes[-preserve_recent_turns]
    selected = {*leading_system, *range(tail_start, len(messages))}
    checkpoint = _latest_compaction_index(messages)
    if checkpoint is not None:
        selected.add(checkpoint)
    return selected


def _leading_system_indexes(messages: list[dict[str, Any]]) -> list[int]:
    indexes: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "system" or is_compaction_message(message):
            break
        indexes.append(index)
    return indexes


def _latest_compaction_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if is_compaction_message(messages[index]):
            return index
    return None


def _complete_tool_pairs(
    messages: list[dict[str, Any]],
    selected_indexes: set[int],
) -> set[int]:
    selected = set(selected_indexes)
    assistant_by_call_id: dict[str, int] = {}
    results_by_call_id: dict[str, list[int]] = {}

    for index, message in enumerate(messages):
        for call_id in _assistant_tool_call_ids(message):
            assistant_by_call_id[call_id] = index
        if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            results_by_call_id.setdefault(message["tool_call_id"], []).append(index)

    changed = True
    while changed:
        changed = False
        for index in tuple(selected):
            message = messages[index]
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                assistant_index = assistant_by_call_id.get(call_id)
                if assistant_index is not None and assistant_index not in selected:
                    selected.add(assistant_index)
                    changed = True
            for call_id in _assistant_tool_call_ids(message):
                for result_index in results_by_call_id.get(call_id, []):
                    if result_index not in selected:
                        selected.add(result_index)
                        changed = True
    return selected


def _assistant_tool_call_ids(message: dict[str, Any]) -> list[str]:
    if message.get("role") != "assistant" or not isinstance(message.get("tool_calls"), list):
        return []
    return [
        call_id
        for call in message["tool_calls"]
        if isinstance(call, dict) and isinstance((call_id := call.get("id")), str)
    ]


def _filter_tools(
    tools: list[dict[str, Any]],
    active_tools: list[str],
) -> list[dict[str, Any]]:
    allowed = set(active_tools)
    return [
        dict(tool)
        for tool in tools
        if isinstance(tool.get("function"), dict)
        and tool["function"].get("name") in allowed
    ]


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    return dict(message)


def _serialized_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _bounded_preview(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    head_size = max(1, (limit * 2) // 3)
    tail_size = max(1, limit - head_size)
    omitted = len(content) - head_size - tail_size
    return (
        f"{content[:head_size]}\n"
        f"... [{omitted} characters omitted; full value available at ref] ...\n"
        f"{content[-tail_size:]}"
    )
