"""Regression tests for the live turn-event projection."""

from __future__ import annotations

from typing import Any

import pytest

from aeloon_core.turn_events import (
    TurnEventProgress,
    _bounded_web_tool_result,
    _web_block_view,
)


@pytest.mark.asyncio
async def test_final_content_gets_a_distinct_canonical_text_block() -> None:
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
    assert [block["role"] for block in text_blocks] == ["narration", "final"]
    turn_end = next(payload for name, payload in events if name == "chat.turn.end")
    assert turn_end["final"] == "The problem is fixed and verified."
    assert turn_end["blocks"][-1]["content"] == "The problem is fixed and verified."


@pytest.mark.asyncio
async def test_final_does_not_duplicate_matching_streamed_content() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    progress = TurnEventProgress(session_id="master", emit=emit)
    await progress.on_llm_delta("Already final.")
    await progress.on_final("Already final.")

    assert [block["content"] for block in progress.blocks if block["type"] == "text"] == [
        "Already final."
    ]
    assert next(block for block in progress.blocks if block["type"] == "text")["role"] == (
        "final"
    )


@pytest.mark.asyncio
async def test_reasoning_content_stays_human_readable_and_metadata_moves_to_logs() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    progress = TurnEventProgress(session_id="master", emit=emit)
    await progress._append_reasoning_line(
        "Call read",
        kind="tool_call",
        data={"call_id": "call-1", "arguments": {"path": "README.md"}},
    )

    block = next(item for item in progress.blocks if item["type"] == "reasoning")
    assert block["content"] == "Call read"
    log = next(
        payload
        for name, payload in events
        if name == "log.entry" and payload["source"] == "reasoning.update"
    )
    assert log["detail"]["entry"]["call_id"] == "call-1"
    assert log["detail"]["entry"]["arguments"] == {"path": "README.md"}


@pytest.mark.asyncio
async def test_ephemeral_harness_worker_publishes_bounded_result_metadata() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    progress = TurnEventProgress(session_id="master", emit=emit)
    await progress.on_worker_lifecycle(
        event="completed",
        worker_id="dynamic-run",
        run_id="dynamic-run",
        worker_type_id="reviewer",
        status="completed",
        duration_ms=1_200,
        objective="Review the provider migration",
        summary="Verified the provider migration.",
        usage={"input_tokens": 700, "output_tokens": 300},
    )

    payload = events[0][1]
    assert payload["objective"] == "Review the provider migration"
    assert payload["summary"] == "Verified the provider migration."
    assert payload["usage"] == {"input_tokens": 700, "output_tokens": 300}


def test_web_tool_results_are_bounded_before_they_are_emitted() -> None:
    result = "x" * 40_000
    assert len(_bounded_web_tool_result(result)) <= 16_000
    block = _web_block_view({"type": "tool_call", "result": result})
    assert len(block["result"]) <= 16_000
    assert block["result_truncated"] is True
