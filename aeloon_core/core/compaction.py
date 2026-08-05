"""Context-budget contracts used by the stateless run engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from aeloon_core.core.types import AgentMessage, Usage

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


class ContextCompactor(Protocol):
    async def compact(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        reason: CompactionReason,
    ) -> ContextUpdate: ...


def should_compact(context_tokens: int, context_window: int, policy: ContextPolicy) -> bool:
    return policy.enabled and context_tokens > context_window - policy.reserve_tokens


def is_context_overflow(error_message: str | None) -> bool:
    message = (error_message or "").lower()
    return any(
        marker in message
        for marker in (
            "context length",
            "context window",
            "maximum context",
            "max context",
            "too many tokens",
        )
    )


__all__ = [
    "CompactionReason",
    "ContextCompactor",
    "ContextPolicy",
    "ContextUpdate",
    "is_context_overflow",
    "should_compact",
]
