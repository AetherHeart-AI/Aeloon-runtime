"""Automatic model-context compaction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import tiktoken

from aeloon_core.config import ContextCompactionConfig
from aeloon_core.model_metadata import ModelLimits
from aeloon_core.providers.base import LLMProvider

COMPACTION_MARKER = "[aeloon-core:context-compaction]"
SUMMARY_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION.

Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done
- Any critical data, examples, files, commands, or references needed to continue

Be concise, structured, and focused on helping the next LLM continue without
re-reading the compacted transcript."""
SUMMARY_PREFIX = (
    "Earlier conversation summary. Use this to preserve continuity while avoiding "
    "re-reading compacted transcript messages."
)
_MIN_RECENT_TOKENS = 2_000
_MAX_RECENT_TOKENS = 8_000
_SUMMARY_SOURCE_MAX_TOKENS = 48_000
_SERIALIZED_MESSAGE_MAX_TOKENS = 2_000
_FALLBACK_SUMMARY_MAX_TOKENS = 2_000


@dataclass(frozen=True)
class CompactionResult:
    """Result of a context compaction pass."""

    messages: list[dict[str, Any]]
    compacted: bool
    original_tokens: int
    compacted_tokens: int
    trigger_tokens: int
    summary: str | None = None


async def maybe_compact_messages(
    *,
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    config: ContextCompactionConfig,
    context_window_tokens: int,
    output_tokens: int,
    model_limits: ModelLimits | None = None,
) -> CompactionResult:
    """Compact older history when the model-visible request is near the limit."""

    original_tokens = estimate_messages_tokens(messages, model=model)
    trigger_tokens = _trigger_tokens(
        config=config,
        context_window_tokens=model_limits.context_tokens
        if model_limits and model_limits.context_tokens
        else context_window_tokens,
        output_tokens=output_tokens,
    )
    if not config.enabled or original_tokens < trigger_tokens:
        return CompactionResult(
            messages=messages,
            compacted=False,
            original_tokens=original_tokens,
            compacted_tokens=original_tokens,
            trigger_tokens=trigger_tokens,
        )

    system_prefix = _runtime_system_prefix(messages)
    tail_start = _select_tail_start(
        messages,
        body_start=len(system_prefix),
        model=model,
        config=config,
        trigger_tokens=trigger_tokens,
    )
    if tail_start is None or tail_start <= len(system_prefix):
        return CompactionResult(
            messages=messages,
            compacted=False,
            original_tokens=original_tokens,
            compacted_tokens=original_tokens,
            trigger_tokens=trigger_tokens,
        )

    head = messages[len(system_prefix) : tail_start]
    tail = messages[tail_start:]
    if not _has_compactable_content(head):
        return CompactionResult(
            messages=messages,
            compacted=False,
            original_tokens=original_tokens,
            compacted_tokens=original_tokens,
            trigger_tokens=trigger_tokens,
        )

    source = _serialize_for_summary(head, model=model)
    source_budget = _summary_source_budget(
        trigger_tokens=trigger_tokens,
        tail_tokens=estimate_messages_tokens(tail, model=model),
        summary_max_tokens=config.summary_max_tokens,
    )
    source = truncate_middle_tokens(source, max_tokens=source_budget, model=model)
    summary = await _summarize(
        provider=provider,
        model=model,
        system_prefix=system_prefix,
        source=source,
        summary_max_tokens=config.summary_max_tokens,
    )
    compacted = [
        *system_prefix,
        _compaction_message(summary),
        *tail,
    ]
    compacted_tokens = estimate_messages_tokens(compacted, model=model)
    return CompactionResult(
        messages=compacted,
        compacted=True,
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        trigger_tokens=trigger_tokens,
        summary=summary,
    )


def estimate_messages_tokens(messages: list[dict[str, Any]], *, model: str) -> int:
    """Estimate chat message tokens with a stable tokenizer fallback."""

    encoding = _encoding_for_model(model)
    total = 0
    for message in messages:
        total += 4
        total += len(encoding.encode(str(message.get("role") or "")))
        total += len(
            encoding.encode(
                json.dumps(
                    {key: value for key, value in message.items() if key != "role"},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
        )
    return total + 2


def truncate_middle_tokens(text: str, *, max_tokens: int, model: str) -> str:
    """Truncate text by preserving the beginning and end."""

    if max_tokens <= 0 or not text:
        return ""
    encoding = _encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    marker = f"\n\n... [{len(tokens) - max_tokens} tokens compacted away] ...\n\n"
    marker_tokens = encoding.encode(marker)
    remaining = max_tokens - len(marker_tokens)
    if remaining <= 0:
        return marker.strip()
    left = remaining // 2
    right = remaining - left
    return encoding.decode(tokens[:left]) + marker + encoding.decode(tokens[-right:])


def is_compaction_message(message: dict[str, Any]) -> bool:
    """Whether a message is Aeloon's synthetic compaction summary."""

    return message.get("role") == "system" and str(message.get("content") or "").startswith(
        COMPACTION_MARKER
    )


def _trigger_tokens(
    *,
    config: ContextCompactionConfig,
    context_window_tokens: int,
    output_tokens: int,
) -> int:
    window = max(1, context_window_tokens)
    ratio_limit = int(window * config.trigger_ratio)
    reserved = min(config.buffer_tokens, max(1, output_tokens))
    return max(1, min(ratio_limit, window - reserved))


def _runtime_system_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system" or is_compaction_message(message):
            break
        prefix.append(message)
    return prefix


def _select_tail_start(
    messages: list[dict[str, Any]],
    *,
    body_start: int,
    model: str,
    config: ContextCompactionConfig,
    trigger_tokens: int,
) -> int | None:
    turn_starts = [
        index
        for index, message in enumerate(messages[body_start:], start=body_start)
        if message.get("role") == "user"
    ]
    if len(turn_starts) < 2:
        return None

    recent_budget = config.preserve_recent_tokens or min(
        _MAX_RECENT_TOKENS,
        max(_MIN_RECENT_TOKENS, trigger_tokens // 4),
    )
    starts = turn_starts[-config.preserve_recent_turns :]
    for start in starts:
        if estimate_messages_tokens(messages[start:], model=model) <= recent_budget:
            return start

    for start in reversed(turn_starts):
        if estimate_messages_tokens(messages[start:], model=model) <= recent_budget:
            return start
    return turn_starts[-1]


def _has_compactable_content(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") in {"user", "assistant", "tool", "system"} for message in messages
    )


async def _summarize(
    *,
    provider: LLMProvider,
    model: str,
    system_prefix: list[dict[str, Any]],
    source: str,
    summary_max_tokens: int,
) -> str:
    prompt = f"{SUMMARY_PROMPT}\n\nTranscript to compact:\n\n{source}"
    response = await provider.chat_with_retry(
        messages=[
            *system_prefix,
            {
                "role": "user",
                "content": prompt,
            },
        ],
        tools=[],
        model=model,
        max_tokens=summary_max_tokens,
        temperature=0.2,
    )
    summary = _strip_think(response.content)
    if response.finish_reason == "error" or not summary:
        return _fallback_summary(source, model=model)
    return summary


def _fallback_summary(source: str, *, model: str) -> str:
    excerpt = truncate_middle_tokens(
        source,
        max_tokens=_FALLBACK_SUMMARY_MAX_TOKENS,
        model=model,
    ).strip()
    if not excerpt:
        return "No compacted transcript content was available."
    return (
        "Automatic summary generation failed. Retained an extractive checkpoint "
        "from the compacted transcript:\n\n"
        f"{excerpt}"
    )


def _compaction_message(summary: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": f"{COMPACTION_MARKER}\n{SUMMARY_PREFIX}\n\n{summary.strip()}",
    }


def _serialize_for_summary(messages: list[dict[str, Any]], *, model: str) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages, start=1):
        text = _message_summary_text(message, model=model)
        if not text.strip():
            continue
        parts.append(f"## Message {index}: {message.get('role', 'unknown')}\n{text}")
    return "\n\n".join(parts)


def _message_summary_text(message: dict[str, Any], *, model: str) -> str:
    role = str(message.get("role") or "")
    if is_compaction_message(message):
        return str(message.get("content") or "")
    if role == "assistant":
        return _assistant_summary_text(message, model=model)
    if role == "tool":
        return _tool_summary_text(message, model=model)
    content = _content_to_text(message.get("content"))
    return truncate_middle_tokens(content, max_tokens=_SERIALIZED_MESSAGE_MAX_TOKENS, model=model)


def _assistant_summary_text(message: dict[str, Any], *, model: str) -> str:
    parts: list[str] = []
    content = _content_to_text(message.get("content"))
    if content.strip():
        parts.append(content)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        rendered = []
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else "unknown"
            arguments = function.get("arguments") if isinstance(function, dict) else None
            arg_text = (
                arguments if isinstance(arguments, str) else json.dumps(arguments, default=str)
            )
            rendered.append(f"- {name}: {arg_text}")
        parts.append("Tool calls:\n" + "\n".join(rendered))
    text = "\n\n".join(parts)
    return truncate_middle_tokens(text, max_tokens=_SERIALIZED_MESSAGE_MAX_TOKENS, model=model)


def _tool_summary_text(message: dict[str, Any], *, model: str) -> str:
    name = str(message.get("name") or "tool")
    content = _content_to_text(message.get("content"))
    text = f"Tool result from {name}:\n{content}"
    return truncate_middle_tokens(text, max_tokens=_SERIALIZED_MESSAGE_MAX_TOKENS, model=model)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2, default=str)


def _summary_source_budget(
    *,
    trigger_tokens: int,
    tail_tokens: int,
    summary_max_tokens: int,
) -> int:
    available = trigger_tokens - tail_tokens - summary_max_tokens
    return max(1_000, min(_SUMMARY_SOURCE_MAX_TOKENS, available))


def _strip_think(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
