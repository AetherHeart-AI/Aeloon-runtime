from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from aeloon_core.config import ContextCompactionConfig
from aeloon_core.pydantic_history import COMPACTION_MARKER, PydanticHistoryCompactor


def _history(*, body_chars: int = 100) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for index in range(3):
        messages.extend(
            [
                ModelRequest(
                    parts=[UserPromptPart(f"request-{index} " + "x" * body_chars)]
                ),
                ModelResponse(parts=[TextPart(f"answer-{index} " + "y" * body_chars)]),
            ]
        )
    return messages


@pytest.mark.asyncio
async def test_history_below_trigger_is_unchanged() -> None:
    messages = _history(body_chars=5)
    processor = PydanticHistoryCompactor(
        ContextCompactionConfig(trigger_ratio=1.0),
        context_window_tokens=100_000,
    )

    result = await processor(cast(RunContext[Any], None), messages)

    assert result is messages


@pytest.mark.asyncio
async def test_history_compaction_is_client_side_and_preserves_recent_turn() -> None:
    messages = _history(body_chars=2_000)
    processor = PydanticHistoryCompactor(
        ContextCompactionConfig(
            trigger_ratio=0.1,
            preserve_recent_turns=1,
            summary_max_tokens=256,
        ),
        context_window_tokens=4_000,
    )

    result = await processor(cast(RunContext[Any], None), messages)

    assert len(result) < len(messages)
    assert COMPACTION_MARKER in str(result[0])
    assert "request-2" in str(result[-2])
    assert "answer-2" in str(result[-1])


@pytest.mark.asyncio
async def test_disabled_compaction_never_rewrites_history() -> None:
    messages = _history(body_chars=2_000)
    processor = PydanticHistoryCompactor(
        ContextCompactionConfig(enabled=False, trigger_ratio=0.1),
        context_window_tokens=4_000,
    )

    result = await processor(cast(RunContext[Any], None), messages)

    assert result is messages
