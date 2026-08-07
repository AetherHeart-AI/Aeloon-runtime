"""Runtime-owned semantic summarization for session coordinators."""

# Summarization prompts use readable, stable line wrapping.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from aeloon_core.core.context_stats import estimate_context_tokens, estimate_tokens
from aeloon_core.core.events import RunEventDispatcher
from aeloon_core.core.inference_runtime import InferenceRuntime
from aeloon_core.core.types import (
    AgentMessage,
    AssistantMessage,
    InferencePort,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
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

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the kept suffix]

Be concise. Focus on what's needed to understand the kept suffix."""

TOOL_RESULT_MAX_CHARS = 2_000
HARD_TRUNCATION_MARKER = "\n\n[... content hard-truncated to fit summarization budget]"


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: tuple[AgentMessage, ...]
    turn_prefix_messages: tuple[AgentMessage, ...]
    is_split_turn: bool
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
    cancellation_event: asyncio.Event | None = None,
) -> CompactionResult:
    instructions = UPDATE_SUMMARIZATION_PROMPT if preparation.previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        instructions += f"\n\nAdditional focus: {custom_instructions}"
    summary = ""
    usage = Usage()
    if preparation.messages_to_summarize:
        summary, history_usage = await _rolling_summary(
            preparation.messages_to_summarize,
            inference=inference,
            model=model,
            stream_options=stream_options,
            reserve_tokens=settings.reserve_tokens,
            instructions=instructions,
            previous_summary=preparation.previous_summary,
            cancellation_event=cancellation_event,
        )
        usage = _combine_usage(usage, history_usage)
    elif preparation.previous_summary:
        summary = preparation.previous_summary
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        prefix_summary, prefix_usage = await _rolling_summary(
            preparation.turn_prefix_messages,
            inference=inference,
            model=model,
            stream_options=stream_options,
            reserve_tokens=settings.reserve_tokens,
            instructions=TURN_PREFIX_SUMMARIZATION_PROMPT,
            previous_summary=None,
            cancellation_event=cancellation_event,
        )
        history = summary or "No prior history."
        summary = f"{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_summary}"
        usage = _combine_usage(usage, prefix_usage)
    if not summary.strip():
        raise RuntimeError("Compaction summarization returned an empty summary")
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
        usage=usage,
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
    if custom_instructions and not replace_instructions:
        instructions += f"\n\nAdditional focus: {custom_instructions}"
    branch_settings = min(2_560, max(1, model.context_window // 4))
    branch_text, usage = await _rolling_summary(
        messages,
        inference=inference,
        model=model,
        stream_options=stream_options,
        reserve_tokens=branch_settings,
        instructions=instructions,
        previous_summary=None,
    )
    read_files, modified_files = file_operations(messages)
    summary = (
        "The user explored a different conversation branch before returning here.\n"
        "Summary of that exploration:\n\n" + branch_text
    )
    return (
        summary,
        usage,
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
            if message.stop_reason in {"error", "aborted"}:
                continue
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
        elif isinstance(message, ToolResultMessage):
            content = "\n".join(
                part.text for part in message.content if isinstance(part, TextContent)
            )
            content = _truncate_text(content, TOOL_RESULT_MAX_CHARS, tool_result=True)
            lines.append(f"Tool ({message.tool_name}): {content}")
    return "\n\n".join(lines)


async def _rolling_summary(
    messages: tuple[AgentMessage, ...],
    *,
    inference: InferencePort,
    model: Model,
    stream_options: StreamOptions,
    reserve_tokens: int,
    instructions: str,
    previous_summary: str | None,
    cancellation_event: asyncio.Event | None = None,
) -> tuple[str, Usage]:
    output_tokens = max(
        1,
        min(
            int(reserve_tokens * 0.8),
            model.max_tokens,
            max(1, int(model.context_window * 0.25)),
        ),
    )
    safety_tokens = min(4_096, max(64, int(model.context_window * 0.05)))
    input_tokens = max(1, model.context_window - output_tokens - safety_tokens)
    input_chars = max(4, input_tokens * 4)
    units = _conversation_units(messages, max_unit_chars=input_chars)
    if not units:
        raise RuntimeError("Summarization has no valid conversation content")

    current_summary = previous_summary or ""
    total_usage = Usage()
    position = 0
    request_index = 0
    while position < len(units):
        request_instructions = (
            instructions if request_index == 0 else UPDATE_SUMMARIZATION_PROMPT
        )
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT
        minimum_prompt_chars = len("<conversation>\n\n</conversation>\n\n") + 1
        if len(system_prompt) + len(request_instructions) + minimum_prompt_chars > input_chars:
            system_prompt = _truncate_text(system_prompt, max(0, input_chars // 4))
            instruction_budget = max(
                0,
                input_chars - len(system_prompt) - minimum_prompt_chars,
            )
            request_instructions = _truncate_text(
                request_instructions,
                instruction_budget,
            )
        fixed = (
            len(system_prompt)
            + len(request_instructions)
            + len("<conversation>\n\n</conversation>\n\n")
            + len("<previous-summary>\n\n</previous-summary>\n\n")
        )
        previous_budget = max(0, (input_chars - fixed) // 2)
        prompt_previous = _truncate_text(current_summary, previous_budget)
        conversation_budget = max(1, input_chars - fixed - len(prompt_previous))
        selected: list[str] = []
        selected_chars = 0
        while position < len(units):
            unit = units[position]
            if selected and selected_chars + len(unit) + 2 > conversation_budget:
                break
            if not selected and len(unit) > conversation_budget:
                unit = _truncate_text(unit, conversation_budget)
            selected.append(unit)
            selected_chars += len(unit) + 2
            position += 1
            if selected_chars >= conversation_budget:
                break
        conversation = "\n\n".join(selected)
        prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
        if prompt_previous:
            prompt += f"<previous-summary>\n{prompt_previous}\n</previous-summary>\n\n"
        prompt += request_instructions
        prompt = _truncate_text(prompt, max(0, input_chars - len(system_prompt)))
        response = await _summary_request(
            prompt,
            inference=inference,
            model=model,
            stream_options=stream_options,
            max_tokens=output_tokens,
            system_prompt=system_prompt,
            cancellation_event=cancellation_event,
        )
        current_summary = response.text.strip()
        if not current_summary:
            raise RuntimeError("Summarization returned an empty summary")
        total_usage = _combine_usage(total_usage, response.usage)
        request_index += 1
    return current_summary, total_usage


async def _summary_request(
    prompt: str,
    *,
    inference: InferencePort,
    model: Model,
    stream_options: StreamOptions,
    max_tokens: int,
    system_prompt: str,
    cancellation_event: asyncio.Event | None,
) -> AssistantMessage:
    options = StreamOptions(
        timeout_ms=stream_options.timeout_ms,
        max_tokens=max_tokens,
        thinking_level=stream_options.thinking_level,
        max_retries=stream_options.max_retries,
        base_delay_ms=stream_options.base_delay_ms,
        max_retry_delay_ms=stream_options.max_retry_delay_ms,
        headers=stream_options.headers,
        metadata=stream_options.metadata,
    )

    async def on_retry(data: dict[str, Any]) -> None:
        callback = stream_options.metadata.get("on_retry")
        if callable(callback):
            result = callback(data)
            if hasattr(result, "__await__"):
                await result

    runtime = InferenceRuntime(inference, RunEventDispatcher())
    request = asyncio.create_task(
        runtime.request(
            model=model,
            messages=(UserMessage(prompt),),
            system_prompt=system_prompt,
            tools=(),
            session_id=f"summary-{uuid.uuid4().hex}",
            stream_options=options,
            on_retry=on_retry,
        )
    )
    cancellation: asyncio.Task[bool] | None = None
    try:
        if cancellation_event is not None:
            cancellation = asyncio.create_task(cancellation_event.wait())
            done, _pending = await asyncio.wait(
                (request, cancellation),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation in done and cancellation_event.is_set() and not request.done():
                runtime.cancel()
        response = await request
    except asyncio.CancelledError:
        runtime.cancel()
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        raise
    finally:
        if cancellation is not None:
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
    if response.stop_reason in {"error", "aborted"}:
        raise RuntimeError(response.error_message or "Summarization failed")
    if not response.text.strip():
        raise RuntimeError("Summarization returned an empty summary")
    return response


def _conversation_units(
    messages: tuple[AgentMessage, ...],
    *,
    max_unit_chars: int,
) -> list[str]:
    turns: list[list[AgentMessage]] = []
    current: list[AgentMessage] = []
    for message in messages:
        if isinstance(message, UserMessage) and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    units: list[str] = []
    for turn in turns:
        serialized = serialize_conversation(tuple(turn))
        if serialized and len(serialized) <= max_unit_chars:
            units.append(serialized)
        elif serialized:
            units.extend(
                value
                for message in turn
                if (value := serialize_conversation((message,)))
            )
    return units


def _truncate_text(value: str, max_chars: int, *, tool_result: bool = False) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return ""
    if tool_result:
        marker = f"\n\n[... {len(value) - max_chars} more characters truncated]"
    else:
        marker = HARD_TRUNCATION_MARKER
    if len(marker) >= max_chars:
        return marker[:max_chars]
    return value[: max_chars - len(marker)] + marker


def _combine_usage(first: Usage, second: Usage) -> Usage:
    costs = {
        key: float(first.cost.get(key, 0)) + float(second.cost.get(key, 0))
        for key in set(first.cost) | set(second.cost)
    }
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        total_tokens=first.total_tokens + second.total_tokens,
        reasoning=(first.reasoning or 0) + (second.reasoning or 0) or None,
        cost=costs,
    )


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
