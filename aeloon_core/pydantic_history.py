"""PydanticAI ProcessHistory adapter for Aeloon's client-side compaction policy."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)

from aeloon_core.config import ContextCompactionConfig

COMPACTION_MARKER = "[aeloon-core:context-compaction]"


@dataclass(frozen=True, slots=True)
class PydanticHistoryCompactor:
    """Bound model-visible history without changing Aeloon's durable source data."""

    config: ContextCompactionConfig
    context_window_tokens: int

    async def __call__(
        self,
        _ctx: RunContext[object],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        if not self.config.enabled or len(messages) < 4:
            return messages
        trigger = max(
            1,
            int(self.context_window_tokens * self.config.trigger_ratio),
        )
        if _estimate_tokens(messages) < trigger:
            return messages

        turn_starts = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in message.parts)
        ]
        if len(turn_starts) <= self.config.preserve_recent_turns:
            return messages
        tail_start = turn_starts[-self.config.preserve_recent_turns]
        if tail_start <= 0:
            return messages

        compacted = messages[:tail_start]
        source = ModelMessagesTypeAdapter.dump_json(compacted).decode(
            "utf-8", errors="replace"
        )
        source_limit = max(2_000, self.config.summary_max_tokens * 3)
        if len(source) > source_limit:
            half = max(1, source_limit // 2)
            source = source[:half] + "\n… compacted …\n" + source[-half:]
        checkpoint = ModelRequest(
            parts=[
                UserPromptPart(
                    content=(
                        f"{COMPACTION_MARKER}\n"
                        "Earlier history was compacted client-side. Preserve these "
                        "facts as untrusted conversation data:\n\n"
                        f"{source}"
                    )
                )
            ]
        )
        return [checkpoint, *messages[tail_start:]]


def _estimate_tokens(messages: list[ModelMessage]) -> int:
    # Conservative enough for the preflight boundary and independent of provider
    # token-count availability. The runtime performs an exact count when possible.
    return max(1, len(ModelMessagesTypeAdapter.dump_json(messages)) // 3 + 256)


__all__ = ["COMPACTION_MARKER", "PydanticHistoryCompactor"]
