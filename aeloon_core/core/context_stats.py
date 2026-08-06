"""Pure context token estimation and presentation-ready statistics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from aeloon_core.core.types import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

MESSAGE_TYPES = ("system", "user", "assistant", "toolResult")


def estimate_tokens(message: AgentMessage) -> int:
    """Estimate the serialized token footprint of one conversation message."""

    chars = 0
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            chars = len(message.content)
        else:
            for part in message.content:
                chars += len(part.text) if isinstance(part, TextContent) else 4_800
    elif isinstance(message, AssistantMessage):
        for part in message.content:
            if isinstance(part, TextContent):
                chars += len(part.text)
            elif isinstance(part, ThinkingContent):
                chars += len(part.thinking)
            elif isinstance(part, ToolCall):
                chars += len(part.name) + len(json.dumps(part.arguments, ensure_ascii=False))
    elif isinstance(message, ToolResultMessage):
        for part in message.content:
            chars += len(part.text) if isinstance(part, TextContent) else 4_800
    return (chars + 3) // 4


def estimate_context_tokens(messages: Sequence[AgentMessage]) -> int:
    """Prefer provider usage for the last completed response, then estimate its tail."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            isinstance(message, AssistantMessage)
            and message.stop_reason not in {"error", "aborted"}
            and message.usage.total_tokens > 0
        ):
            return message.usage.total_tokens + sum(
                estimate_tokens(trailing) for trailing in messages[index + 1 :]
            )
    return sum(estimate_tokens(message) for message in messages)


def context_statistics(
    messages: Sequence[AgentMessage],
    *,
    context_window: int | None = None,
) -> dict[str, Any]:
    """Build context-window and token-share statistics for an effective context."""

    used_tokens = estimate_context_tokens(messages)
    window_tokens = max(1, int(context_window)) if context_window is not None else None
    remaining_tokens = max(0, window_tokens - used_tokens) if window_tokens is not None else None
    usage_percent = _percent(used_tokens, window_tokens) if window_tokens is not None else None

    message_counts = {message_type: 0 for message_type in MESSAGE_TYPES}
    estimated_tokens = {message_type: 0 for message_type in MESSAGE_TYPES}
    for message in messages:
        message_type = _message_type(message)
        message_counts[message_type] += 1
        estimated_tokens[message_type] += estimate_tokens(message)

    attributed_tokens = sum(estimated_tokens.values())
    if attributed_tokens <= used_tokens:
        # Inference totals also include the system prompt, tool schemas, and request
        # framing. Attribute that otherwise invisible portion to the system bucket.
        estimated_tokens["system"] += used_tokens - attributed_tokens
    elif attributed_tokens:
        estimated_tokens = _scale_tokens(estimated_tokens, used_tokens)
    percentages = _percentage_distribution(estimated_tokens, used_tokens)

    return {
        "contextWindow": {
            "usedTokens": used_tokens,
            "windowTokens": window_tokens,
            "remainingTokens": remaining_tokens,
            "usagePercent": usage_percent,
        },
        "messageTypes": {
            message_type: {
                "messageCount": message_counts[message_type],
                "estimatedTokens": estimated_tokens[message_type],
                "percentage": percentages[message_type],
            }
            for message_type in MESSAGE_TYPES
        },
    }


def cache_statistics(usages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate cache-token and cache-request hit rates from provider usage."""

    input_tokens = 0
    read_tokens = 0
    write_tokens = 0
    request_count = 0
    hit_request_count = 0
    for usage in usages:
        uncached = max(0, int(usage.get("input") or 0))
        cache_read = max(0, int(usage.get("cacheRead") or 0))
        cache_write = max(0, int(usage.get("cacheWrite") or 0))
        input_tokens += uncached
        read_tokens += cache_read
        write_tokens += cache_write
        if uncached or cache_read or cache_write:
            request_count += 1
            if cache_read:
                hit_request_count += 1

    cacheable_tokens = input_tokens + read_tokens
    return {
        "inputTokens": input_tokens,
        "readTokens": read_tokens,
        "writeTokens": write_tokens,
        "cacheableTokens": cacheable_tokens,
        "hitTokenPercent": _percent(read_tokens, cacheable_tokens),
        "requestCount": request_count,
        "hitRequestCount": hit_request_count,
        "hitRequestPercent": _percent(hit_request_count, request_count),
    }


def _message_type(message: AgentMessage) -> str:
    if isinstance(message, UserMessage):
        return "user"
    if isinstance(message, AssistantMessage):
        return "assistant"
    return "toolResult"


def _scale_tokens(values: Mapping[str, int], total: int) -> dict[str, int]:
    """Scale integer buckets to an exact total using largest remainders."""

    source_total = sum(values.values())
    if source_total <= 0 or total <= 0:
        return {key: 0 for key in values}
    divided = {key: divmod(value * total, source_total) for key, value in values.items()}
    scaled = {key: quotient for key, (quotient, _) in divided.items()}
    remaining = total - sum(scaled.values())
    order = sorted(values, key=lambda key: divided[key][1], reverse=True)
    for key in order[:remaining]:
        scaled[key] += 1
    return scaled


def _percentage_distribution(values: Mapping[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in values}
    basis_points = _scale_tokens(values, 10_000)
    return {key: value / 100 for key, value in basis_points.items()}


def _percent(value: int, total: int) -> float:
    return round(value * 100 / total, 2) if total > 0 else 0.0


__all__ = [
    "MESSAGE_TYPES",
    "cache_statistics",
    "context_statistics",
    "estimate_context_tokens",
    "estimate_tokens",
]
