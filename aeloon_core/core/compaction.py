"""Context-budget contracts used by the stateless run engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from aeloon_core.core.types import AgentMessage, AssistantMessage, Usage

CompactionReason = Literal["threshold", "overflow", "explicit"]


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass(frozen=True, slots=True)
class ContextUpdate:
    """A replacement context produced without exposing a Session to core."""

    messages: tuple[AgentMessage, ...]
    summary: str
    tokens_before: int
    first_kept_id: str | None = None
    usage: Usage = field(default_factory=Usage)
    details: dict[str, Any] = field(default_factory=dict)
    compaction_boundary_ms: int | None = None
    compaction_boundary_index: int | None = None


class ContextCompactor(Protocol):
    async def compact(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        reason: CompactionReason,
    ) -> ContextUpdate: ...


def should_compact(context_tokens: int, context_window: int, policy: ContextPolicy) -> bool:
    return policy.enabled and context_tokens > context_window - policy.reserve_tokens


_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length"
        r"(?: of [\d,]+ tokens?|\s*\([\d,]+\))?",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length (?:is |of )?[\d,]+ tokens?",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"input \([\d,]+ tokens?\) is longer than the model'?s context length",
        r"exceeds the limit of [\d,]+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with [\d,]+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
    )
)
_NON_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:throttling error|service unavailable):",
        r"rate limit",
        r"too many requests",
        r"throttl(?:e|ed|ing)",
    )
)


def is_context_overflow(
    message: AssistantMessage,
    context_window: int | None = None,
) -> bool:
    """Detect provider-reported and silent context overflow."""

    error_message = message.error_message or ""
    if message.stop_reason == "error" and error_message:
        excluded = any(pattern.search(error_message) for pattern in _NON_OVERFLOW_PATTERNS)
        matched = any(pattern.search(error_message) for pattern in _OVERFLOW_PATTERNS)
        if not excluded and matched:
            return True
    if context_window and context_window > 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if message.stop_reason == "stop" and input_tokens > context_window:
            return True
        if (
            message.stop_reason == "length"
            and message.usage.output == 0
            and input_tokens >= context_window * 0.99
        ):
            return True
    return False


__all__ = [
    "CompactionReason",
    "ContextCompactor",
    "ContextPolicy",
    "ContextUpdate",
    "is_context_overflow",
    "should_compact",
]
