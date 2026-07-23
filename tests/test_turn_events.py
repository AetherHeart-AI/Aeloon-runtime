"""Regression tests for the live turn-event projection."""

from __future__ import annotations

from typing import Any

import pytest

from aeloon_core.turn_events import TurnEventProgress


@pytest.mark.asyncio
async def test_finish_turn_content_gets_a_distinct_canonical_text_block() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    progress = TurnEventProgress(session_id="master", emit=emit)
    await progress.on_turn_start()
    await progress.on_llm_delta("I will inspect and fix the problem.")
    await progress.on_final("The problem is fixed and verified.")

    text_blocks = [block for block in progress.blocks if block["type"] == "text"]
    assert [block["content"] for block in text_blocks] == [
        "I will inspect and fix the problem.",
        "The problem is fixed and verified.",
    ]
    turn_end = next(payload for name, payload in events if name == "chat.turn.end")
    assert turn_end["final"] == "The problem is fixed and verified."
    assert turn_end["blocks"][-1]["content"] == "The problem is fixed and verified."


@pytest.mark.asyncio
async def test_finish_turn_does_not_duplicate_matching_streamed_content() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    progress = TurnEventProgress(session_id="master", emit=emit)
    await progress.on_llm_delta("Already final.")
    await progress.on_final("Already final.")

    assert [block["content"] for block in progress.blocks if block["type"] == "text"] == [
        "Already final."
    ]
