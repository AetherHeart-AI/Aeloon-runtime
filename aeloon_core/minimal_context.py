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
                if _is_user_prompt(message)
            ),
            default=-1,
        )
        tool_names = _tool_names_by_id(messages)

        lazy_references: list[str] = []
        call_messages = []
        for index in sorted(selected_indexes):
            message = _copy_message(messages[index])
            references = self._replace_large_tool_results(
                state,
                message,
                preserve_current_skill=index > current_turn_start,
                tool_names=tool_names,
            )
            for reference in references:
                if reference not in lazy_references:
                    lazy_references.append(reference)
            call_messages.append(message)
        call_messages.extend(_copy_message(message) for message in additional_messages or [])

        return ForwardContextResult(
            messages=call_messages,
            tools=_filter_tools(tools, state.active_tools),
            lazy_references=tuple(lazy_references),
            original_message_count=len(messages),
        )

    def _replace_large_tool_results(
        self,
        state: LightweightState,
        message: dict[str, Any],
        *,
        preserve_current_skill: bool,
        tool_names: dict[str, str],
    ) -> list[str]:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            return []
        references: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if preserve_current_skill and tool_names.get(str(call_id)) == "skill":
                continue
            result = block.get("content")
            serialized = _serialized_content(result)
            if len(serialized) <= self.max_tool_result_chars:
                continue
            reference = state.store_lazy(result, prefix="tool-result")
            preview = _bounded_preview(serialized, self.max_tool_result_chars)
            block["content"] = (
                f"{LAZY_TOOL_RESULT_MARKER}\n"
                f"ref: {reference}\n"
                f"characters: {len(serialized)}\n"
                f"bounded preview:\n{preview}"
            )
            references.append(reference)
        return references


def _selected_message_indexes(
    messages: list[dict[str, Any]],
    preserve_recent_turns: int,
) -> set[int]:
    if not messages:
        return set()

    leading_system = _leading_system_indexes(messages)
    user_indexes = [
        index for index, message in enumerate(messages) if _is_user_prompt(message)
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
        for call_id in _tool_result_ids(message):
            results_by_call_id.setdefault(call_id, []).append(index)

    changed = True
    while changed:
        changed = False
        for index in tuple(selected):
            message = messages[index]
            for call_id in _tool_result_ids(message):
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
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, list):
        return []
    return [
        call_id
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance((call_id := block.get("id")), str)
    ]


def _tool_result_ids(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return []
    return [
        call_id
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance((call_id := block.get("tool_use_id")), str)
    ]


def _is_user_prompt(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and not _tool_result_ids(message)


def _tool_names_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_id = block.get("id")
            name = block.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def _filter_tools(
    tools: list[dict[str, Any]],
    active_tools: list[str],
) -> list[dict[str, Any]]:
    allowed = set(active_tools)
    return [
        dict(tool)
        for tool in tools
        if tool.get("name") in allowed
    ]


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    copied = dict(message)
    if isinstance(message.get("content"), list):
        copied["content"] = [
            dict(block) if isinstance(block, dict) else block
            for block in message["content"]
        ]
    return copied


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
