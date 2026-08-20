"""Session-aware context selection and persistence preparation."""

from __future__ import annotations

from typing import Any

from aeloon_runtime.core.context_stats import estimate_context_tokens, estimate_tokens
from aeloon_runtime.core.types import (
    AgentMessage,
    AssistantMessage,
    UserMessage,
    message_from_dict,
)
from aeloon_runtime.runtime.session import Session
from aeloon_runtime.runtime.summarization import (
    BRANCH_SUMMARY_PROMPT,
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CompactionPreparation,
    CompactionResult,
    CompactionSettings,
    compact_preparation,
    file_operations,
    serialize_conversation,
    summarize_branch,
)


async def prepare_compaction(
    session: Session,
    settings: CompactionSettings,
    *,
    force: bool = False,
) -> CompactionPreparation | None:
    entries = await session.get_branch()
    if not entries or entries[-1].get("type") == "compaction":
        return None
    context = await session.build_context()
    tokens_before = estimate_context_tokens(
        context.messages,
        usage_after_ms=context.compaction_boundary_ms,
        usage_after_index=context.compaction_boundary_index,
    )
    previous_summary: str | None = None
    start = 0
    for index in range(len(entries) - 1, -1, -1):
        if entries[index].get("type") == "compaction":
            previous_summary = str(entries[index].get("summary") or "")
            # Fold the previously retained messages into the next summary.
            start = 0
            break

    valid_cut_points = [
        index
        for index in range(start, len(entries))
        if isinstance(_entry_message(entries[index]), UserMessage | AssistantMessage)
    ]
    if not valid_cut_points:
        return None
    accumulated = 0
    cut = valid_cut_points[0]
    reached_budget = False
    for index in range(len(entries) - 1, start - 1, -1):
        message = _entry_message(entries[index])
        if message is None:
            continue
        accumulated += estimate_tokens(message)
        if accumulated >= settings.keep_recent_tokens:
            cut = next((point for point in valid_cut_points if point >= index), cut)
            reached_budget = True
            break
    if force and not reached_budget and accumulated > 0:
        forced_budget = max(1, accumulated // 2)
        accumulated = 0
        for index in range(len(entries) - 1, start - 1, -1):
            message = _entry_message(entries[index])
            if message is None:
                continue
            accumulated += estimate_tokens(message)
            if accumulated >= forced_budget:
                cut = next((point for point in valid_cut_points if point >= index), cut)
                break
    cut_message = _entry_message(entries[cut])
    is_split_turn = isinstance(cut_message, AssistantMessage)
    turn_start = -1
    if is_split_turn:
        for index in range(cut - 1, start - 1, -1):
            if isinstance(_entry_message(entries[index]), UserMessage):
                turn_start = index
                break
        is_split_turn = turn_start >= start
    first_kept = entries[cut] if cut < len(entries) else None
    if first_kept is None:
        return None
    history_end = turn_start if is_split_turn else cut
    to_summarize = tuple(
        message
        for entry in entries[start:history_end]
        if (message := _entry_message(entry)) is not None
    )
    turn_prefix = tuple(
        message
        for entry in entries[turn_start:cut]
        if is_split_turn and (message := _entry_message(entry)) is not None
    )
    retained = tuple(
        message for entry in entries[cut:] if (message := _entry_message(entry)) is not None
    )
    if not to_summarize and not turn_prefix:
        return None
    read_files, modified_files = file_operations((*to_summarize, *turn_prefix, *retained))
    return CompactionPreparation(
        first_kept_entry_id=str(first_kept["id"]),
        messages_to_summarize=to_summarize,
        turn_prefix_messages=turn_prefix,
        is_split_turn=is_split_turn,
        retained_tail=retained,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        read_files=tuple(sorted(read_files)),
        modified_files=tuple(sorted(modified_files)),
    )


def _entry_message(entry: dict[str, Any]) -> AgentMessage | None:
    entry_type = entry.get("type")
    if entry_type == "message":
        return message_from_dict(entry["message"])
    if entry_type == "custom_message":
        return UserMessage(str(entry.get("content") or ""))
    return None


__all__ = [
    "BRANCH_SUMMARY_PROMPT",
    "CompactionPreparation",
    "CompactionResult",
    "CompactionSettings",
    "SUMMARIZATION_PROMPT",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "compact_preparation",
    "estimate_context_tokens",
    "estimate_tokens",
    "prepare_compaction",
    "serialize_conversation",
    "summarize_branch",
]
