"""Session-aware context selection and persistence preparation."""

from __future__ import annotations

from typing import Any

from aeloon_core.core.context_stats import estimate_context_tokens, estimate_tokens
from aeloon_core.core.types import AgentMessage, UserMessage, message_from_dict
from aeloon_core.runtime.session import Session
from aeloon_core.runtime.summarization import (
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
) -> CompactionPreparation | None:
    entries = await session.get_branch()
    if not entries or entries[-1].get("type") == "compaction":
        return None
    context = await session.build_context()
    tokens_before = estimate_context_tokens(context.messages)
    previous_summary: str | None = None
    start = 0
    for index in range(len(entries) - 1, -1, -1):
        if entries[index].get("type") == "compaction":
            previous_summary = str(entries[index].get("summary") or "")
            # Fold the previously retained messages into the next summary.
            start = 0
            break

    accumulated = 0
    cut = start
    for index in range(len(entries) - 1, start - 1, -1):
        message = _entry_message(entries[index])
        if message is None:
            continue
        accumulated += estimate_tokens(message)
        cut = index
        if accumulated >= settings.keep_recent_tokens:
            break
    while cut > start:
        message = _entry_message(entries[cut])
        if isinstance(message, UserMessage):
            break
        cut -= 1
    if cut > start and entries[cut - 1].get("type") == "run_start":
        cut -= 1
    first_kept = entries[cut] if cut < len(entries) else None
    if first_kept is None:
        return None
    to_summarize = tuple(
        message for entry in entries[start:cut] if (message := _entry_message(entry)) is not None
    )
    retained = tuple(
        message for entry in entries[cut:] if (message := _entry_message(entry)) is not None
    )
    if not to_summarize:
        return None
    read_files, modified_files = file_operations((*to_summarize, *retained))
    return CompactionPreparation(
        first_kept_entry_id=str(first_kept["id"]),
        messages_to_summarize=to_summarize,
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
