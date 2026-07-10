from __future__ import annotations

from typing import Any

import pytest

from aeloon_core.config import ContextCompactionConfig
from aeloon_core.context_compaction import (
    COMPACTION_MARKER,
    estimate_request_tokens,
    is_compaction_message,
    maybe_compact_messages,
    truncate_middle_tokens,
)
from aeloon_core.providers.base import LLMProvider, LLMResponse


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

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
        del model, reasoning_effort, tool_choice, response_format
        self.calls.append(
            {
                "messages": messages,
                "tools": tools or [],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)


def compaction_config(**overrides: Any) -> ContextCompactionConfig:
    data: dict[str, Any] = {"summary_max_tokens": 256}
    data.update(overrides)
    return ContextCompactionConfig(**data)


@pytest.mark.asyncio
async def test_maybe_compact_messages_skips_below_trigger() -> None:
    provider = ScriptedProvider([])
    messages = [
        {"role": "system", "content": "runtime rules"},
        {"role": "user", "content": "small request"},
    ]

    result = await maybe_compact_messages(
        provider=provider,
        model="test-model",
        messages=messages,
        tools=[],
        config=compaction_config(),
        context_window_tokens=10_000,
    )

    assert result.compacted is False
    assert result.messages == messages
    assert result.trigger_tokens == 9_000
    assert provider.calls == []


@pytest.mark.asyncio
async def test_maybe_compact_messages_counts_tool_definitions() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="Summary from model",
                usage={"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            )
        ]
    )
    messages = [
        {"role": "system", "content": "runtime rules"},
        {"role": "user", "content": "old request " + ("alpha " * 200)},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "middle request"},
        {"role": "assistant", "content": "middle answer"},
        {"role": "user", "content": "current request"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "large_tool",
                "description": "schema " * 1_200,
            },
        }
    ]

    assert estimate_request_tokens(messages, tools=[], model="test-model") < 450
    assert estimate_request_tokens(messages, tools=tools, model="test-model") >= 450

    result = await maybe_compact_messages(
        provider=provider,
        model="test-model",
        messages=messages,
        tools=tools,
        config=compaction_config(preserve_recent_turns=1, preserve_recent_tokens=50),
        context_window_tokens=500,
    )

    assert result.compacted is True
    assert result.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }


@pytest.mark.asyncio
async def test_maybe_compact_messages_preserves_system_prefix_and_recent_tail() -> None:
    provider = ScriptedProvider([LLMResponse(content="Summary from model")])
    messages = [
        {"role": "system", "content": "runtime rules"},
        {"role": "system", "content": "skill guidance"},
        {"role": "user", "content": "old requirements " + ("alpha " * 500)},
        {
            "role": "assistant",
            "content": "I'll inspect the file.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"old.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read",
            "content": "old file content " + ("beta " * 500),
        },
        {"role": "assistant", "content": "old decision " + ("gamma " * 200)},
        {"role": "user", "content": "recent follow up"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "continue now"},
    ]

    result = await maybe_compact_messages(
        provider=provider,
        model="test-model",
        messages=messages,
        tools=[],
        config=compaction_config(preserve_recent_turns=2, preserve_recent_tokens=2_000),
        context_window_tokens=1_100,
    )

    assert result.compacted is True
    assert result.original_tokens > result.compacted_tokens
    assert result.messages[:2] == messages[:2]
    assert is_compaction_message(result.messages[2])
    assert "Summary from model" in str(result.messages[2]["content"])
    assert result.messages[3:] == messages[6:]
    assert messages[2] not in result.messages
    assert messages[4] not in result.messages

    assert len(provider.calls) == 1
    summary_call = provider.calls[0]
    assert summary_call["tools"] == []
    assert summary_call["max_tokens"] == 256
    assert summary_call["temperature"] == 0.2
    assert summary_call["messages"][:2] == messages[:2]
    assert "Transcript to compact" in summary_call["messages"][-1]["content"]
    assert "old requirements" in summary_call["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_maybe_compact_messages_replaces_prior_compaction_summary() -> None:
    provider = ScriptedProvider([LLMResponse(content="fresh summary")])
    prior_summary = {"role": "system", "content": f"{COMPACTION_MARKER}\nprior summary"}
    messages = [
        {"role": "system", "content": "runtime rules"},
        prior_summary,
        {"role": "user", "content": "old compacted follow up " + ("alpha " * 600)},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
    ]

    result = await maybe_compact_messages(
        provider=provider,
        model="test-model",
        messages=messages,
        tools=[],
        config=compaction_config(preserve_recent_tokens=120),
        context_window_tokens=500,
    )

    assert result.compacted is True
    assert sum(is_compaction_message(message) for message in result.messages) == 1
    assert "fresh summary" in str(result.messages[1]["content"])
    assert "prior summary" not in str(result.messages[1]["content"])
    assert result.messages[-1] == messages[-1]
    assert "prior summary" in provider.calls[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_maybe_compact_messages_uses_extract_fallback_when_summary_fails() -> None:
    provider = ScriptedProvider([LLMResponse(content="summary unavailable", finish_reason="error")])
    messages = [
        {"role": "system", "content": "runtime rules"},
        {"role": "user", "content": "important old context " + ("alpha " * 500)},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
    ]

    result = await maybe_compact_messages(
        provider=provider,
        model="test-model",
        messages=messages,
        tools=[],
        config=compaction_config(preserve_recent_tokens=120),
        context_window_tokens=500,
    )

    assert result.compacted is True
    assert result.summary is not None
    assert result.summary.startswith("Automatic summary generation failed.")
    assert "important old context" in result.summary


def test_truncate_middle_tokens_preserves_prefix_and_suffix() -> None:
    text = "start " + ("middle " * 500) + "end"

    truncated = truncate_middle_tokens(text, max_tokens=40, model="test-model")

    assert truncated.startswith("start ")
    assert truncated.endswith("end")
    assert "tokens compacted away" in truncated
