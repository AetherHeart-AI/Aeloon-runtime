"""Runtime-owned semantic summarization for session coordinators."""

# Summarization prompts use readable, stable line wrapping.
# ruff: noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aeloon_core.core.context_stats import estimate_context_tokens, estimate_tokens
from aeloon_core.core.inference_runtime import collect_assistant
from aeloon_core.core.types import (
    AgentMessage,
    AssistantMessage,
    InferenceContext,
    InferencePort,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    UserMessage,
)

SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. Preserve existing goals, completed work, decisions, exact file paths, function names, and error messages. Add new progress and update next steps. Use the same structured format as the existing summary."""

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Constraints or "(none)"]

## Progress
### Done
- [x] [Completed work]
### In Progress
- [ ] [Unfinished work]
### Blocked
- [Blockers]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [What should happen next]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: tuple[AgentMessage, ...]
    retained_tail: tuple[AgentMessage, ...]
    tokens_before: int
    previous_summary: str | None
    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    retained_tail: tuple[AgentMessage, ...]
    usage: Usage
    details: dict[str, Any]


async def compact_preparation(
    preparation: CompactionPreparation,
    *,
    inference: InferencePort,
    model: Model,
    stream_options: StreamOptions,
    settings: CompactionSettings,
    custom_instructions: str | None = None,
) -> CompactionResult:
    conversation = serialize_conversation(preparation.messages_to_summarize)
    instructions = (
        UPDATE_SUMMARIZATION_PROMPT if preparation.previous_summary else SUMMARIZATION_PROMPT
    )
    if custom_instructions:
        instructions += f"\n\nAdditional focus: {custom_instructions}"
    prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
    if preparation.previous_summary:
        prompt += f"<previous-summary>\n{preparation.previous_summary}\n</previous-summary>\n\n"
    prompt += instructions
    options = StreamOptions(
        timeout_ms=stream_options.timeout_ms,
        max_tokens=min(int(settings.reserve_tokens * 0.8), model.max_tokens),
        thinking_level=stream_options.thinking_level,
        max_retries=stream_options.max_retries,
        base_delay_ms=stream_options.base_delay_ms,
        max_retry_delay_ms=stream_options.max_retry_delay_ms,
        headers=stream_options.headers,
        metadata=stream_options.metadata,
    )
    response = await collect_assistant(
        inference,
        model,
        InferenceContext(
            SUMMARIZATION_SYSTEM_PROMPT,
            (UserMessage(prompt),),
            (),
            "compaction",
        ),
        options,
    )
    if response.stop_reason in {"error", "aborted"}:
        raise RuntimeError(response.error_message or "Compaction summarization failed")
    summary = response.text
    if preparation.read_files:
        summary += "\n\nFiles read:\n" + "\n".join(f"- {path}" for path in preparation.read_files)
    if preparation.modified_files:
        summary += "\n\nFiles modified:\n" + "\n".join(
            f"- {path}" for path in preparation.modified_files
        )
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        retained_tail=preparation.retained_tail,
        usage=response.usage,
        details={
            "readFiles": list(preparation.read_files),
            "modifiedFiles": list(preparation.modified_files),
        },
    )


async def summarize_branch(
    messages: tuple[AgentMessage, ...],
    *,
    inference: InferencePort,
    model: Model,
    stream_options: StreamOptions,
    custom_instructions: str | None = None,
    replace_instructions: bool = False,
) -> tuple[str, Usage, dict[str, Any]]:
    instructions = (
        custom_instructions
        if replace_instructions and custom_instructions
        else BRANCH_SUMMARY_PROMPT
    )
    prompt = (
        f"<conversation>\n{serialize_conversation(messages)}\n</conversation>\n\n" + instructions
    )
    if custom_instructions and not replace_instructions:
        prompt += f"\n\nAdditional focus: {custom_instructions}"
    response = await collect_assistant(
        inference,
        model,
        InferenceContext(
            SUMMARIZATION_SYSTEM_PROMPT,
            (UserMessage(prompt),),
            (),
            "branch-summary",
        ),
        StreamOptions(
            timeout_ms=stream_options.timeout_ms,
            max_tokens=2_048,
            thinking_level=stream_options.thinking_level,
            max_retries=stream_options.max_retries,
            base_delay_ms=stream_options.base_delay_ms,
            max_retry_delay_ms=stream_options.max_retry_delay_ms,
            headers=stream_options.headers,
        ),
    )
    if response.stop_reason in {"error", "aborted"}:
        raise RuntimeError(response.error_message or "Branch summary failed")
    read_files, modified_files = file_operations(messages)
    summary = (
        "The user explored a different conversation branch before returning here.\n"
        "Summary of that exploration:\n\n" + response.text
    )
    return (
        summary,
        response.usage,
        {
            "readFiles": sorted(read_files),
            "modifiedFiles": sorted(modified_files),
        },
    )


def serialize_conversation(messages: tuple[AgentMessage, ...]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(
                    part.text for part in message.content if isinstance(part, TextContent)
                )
            )
            lines.append(f"User: {content}")
        elif isinstance(message, AssistantMessage):
            parts: list[str] = []
            for part in message.content:
                if isinstance(part, TextContent):
                    parts.append(part.text)
                elif isinstance(part, ThinkingContent):
                    parts.append(f"[thinking]\n{part.thinking}")
                elif isinstance(part, ToolCall):
                    parts.append(
                        f"[tool] {part.name} {json.dumps(part.arguments, ensure_ascii=False)}"
                    )
            lines.append("Assistant: " + "\n".join(parts))
        else:
            content = "\n".join(
                part.text for part in message.content if isinstance(part, TextContent)
            )
            lines.append(f"Tool ({message.tool_name}): {content}")
    return "\n\n".join(lines)


def file_operations(messages: tuple[AgentMessage, ...]) -> tuple[set[str], set[str]]:
    reads: set[str] = set()
    modified: set[str] = set()
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for part in message.tool_calls:
            path = part.arguments.get("path")
            if not isinstance(path, str):
                continue
            if part.name in {"read", "grep", "find", "ls"}:
                reads.add(path)
            elif part.name in {"edit", "write"}:
                modified.add(path)
    return reads, modified


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
    "file_operations",
    "serialize_conversation",
    "summarize_branch",
]
