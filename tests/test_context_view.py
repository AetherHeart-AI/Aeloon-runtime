from __future__ import annotations

from typing import Any

import pytest

from aeloon_core.config import ContextCompactionConfig
from aeloon_core.context_compaction import COMPACTION_MARKER
from aeloon_core.context_view import ContextViewPipeline
from aeloon_core.providers.base import LLMProvider, LLMResponse


class _SummaryProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature, reasoning_effort
        del tool_choice, response_format
        self.calls += 1
        return LLMResponse(
            content="Retained decisions and remaining work.",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


@pytest.mark.asyncio
async def test_model_round_view_keeps_full_append_only_history() -> None:
    messages = [{"role": "system", "content": "stable"}]
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"request {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )
    pipeline = ContextViewPipeline(provider=object(), model="test-model")

    view = await pipeline.render(messages=messages, tools=[])

    assert view.canonical_messages is messages
    assert view.messages == messages
    assert view.transformations == ()
    assert view.prefix_reset_reason is None


@pytest.mark.asyncio
async def test_provider_projection_is_stable_and_keeps_canonical_tool_arguments() -> None:
    body = "x" * 6_000
    messages = [
        {"role": "user", "content": "write it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "write-1",
                    "name": "write",
                    "input": {"path": "out.txt", "content": body},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "write-1",
                    "content": "written",
                }
            ],
        },
    ]
    pipeline = ContextViewPipeline(provider=object(), model="test-model")

    first = await pipeline.render(messages=messages, tools=[])
    extended = [*messages, {"role": "user", "content": "continue"}]
    second = await pipeline.render(messages=extended, tools=[])

    canonical_input = first.canonical_messages[1]["content"][0]["input"]
    projected_input = first.messages[1]["content"][0]["input"]
    assert canonical_input["content"] == body
    assert projected_input != canonical_input
    assert first.messages == second.messages[: len(first.messages)]
    assert first.transformations == ("answered_tool_arguments",)


@pytest.mark.asyncio
async def test_compaction_is_an_explicit_prefix_reset_in_the_same_pipeline() -> None:
    provider = _SummaryProvider()
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "old " + "x" * 3_000},
        {"role": "assistant", "content": "old answer " + "y" * 3_000},
        {"role": "user", "content": "current request"},
    ]
    pipeline = ContextViewPipeline(
        provider=provider,
        model="test-model",
        compaction=ContextCompactionConfig(
            trigger_ratio=0.1,
            preserve_recent_turns=1,
            summary_max_tokens=256,
        ),
        context_window_tokens=4_000,
    )

    view = await pipeline.render(messages=messages, tools=[])

    assert provider.calls == 1
    assert view.prefix_reset_reason == "compaction"
    assert "compaction" in view.transformations
    assert COMPACTION_MARKER in str(view.canonical_messages)
    assert view.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_answered_argument_projection_prevents_false_compaction() -> None:
    provider = _SummaryProvider()
    body = "x" * 108_000
    messages = [
        {"role": "user", "content": "write the generated artifact"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "write-1",
                    "name": "write",
                    "input": {"path": "artifact.txt", "content": body},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "write-1",
                    "content": "written",
                }
            ],
        },
        {"role": "user", "content": "continue with the next check"},
    ]
    pipeline = ContextViewPipeline(
        provider=provider,
        model="test-model",
        compaction=ContextCompactionConfig(
            trigger_ratio=0.1,
            preserve_recent_turns=1,
            summary_max_tokens=256,
        ),
        context_window_tokens=4_000,
    )

    view = await pipeline.render(messages=messages, tools=[])

    assert provider.calls == 0
    assert view.canonical_messages is messages
    assert view.prefix_reset_reason is None
    assert view.transformations == ("answered_tool_arguments",)
    assert view.messages[1]["content"][0]["input"]["content"] != body
    assert view.original_tokens > view.visible_tokens


@pytest.mark.asyncio
async def test_additional_finalization_message_is_view_only() -> None:
    messages = [{"role": "user", "content": "work"}]
    finalization = {"role": "user", "content": "wrap up"}
    pipeline = ContextViewPipeline(provider=object(), model="test-model")

    view = await pipeline.render(
        messages=messages,
        tools=[],
        additional_messages=[finalization],
    )

    assert view.canonical_messages is messages
    assert view.messages == [*messages, finalization]
    assert messages == [{"role": "user", "content": "work"}]
