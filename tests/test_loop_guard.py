from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aeloon_core.loop_guard import (
    GuardAction,
    GuardEvent,
    GuardEvidence,
    GuardRequest,
    GuardReviewer,
    classify_malformed_tool_calls,
    suppress_successful_side_effect_duplicates,
)
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class Provider(LLMProvider):
    def __init__(self, response: LLMResponse | Exception) -> None:
        super().__init__()
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def request(*actions: GuardAction) -> GuardRequest:
    return GuardRequest(
        evidence=GuardEvidence(
            event=GuardEvent.TOOL_ERROR,
            cause="tool failed",
            goal="finish the task",
        ),
        allowed_actions=actions,
    )


@pytest.mark.asyncio
async def test_reviewer_returns_only_an_allowed_control_action() -> None:
    provider = Provider(
        LLMResponse(content='{"action":"retry"}', usage={"total_tokens": 7})
    )
    reviewer = GuardReviewer(provider=provider, model="same-model")

    resolution = await reviewer.decide(
        request(GuardAction.RETRY, GuardAction.FINALIZE)
    )

    assert resolution.action == GuardAction.RETRY
    assert resolution.source == "guard"
    assert resolution.usage == {"total_tokens": 7}
    assert provider.calls[0]["tools"] == []
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["max_tokens"] == 512
    assert provider.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(content='{"action":"retry","reason":"invented"}'),
        LLMResponse(content='{"action":"continue"}'),
        LLMResponse(content="not json"),
        LLMResponse(content='{"action":"retry"}', finish_reason="length"),
        LLMResponse(
            content='{"action":"retry"}',
            tool_calls=[ToolCallRequest(id="call", name="write", arguments={})],
        ),
        LLMResponse(
            content=None,
            reasoning_content='{"action":"retry"}',
            finish_reason="length",
        ),
        RuntimeError("guard unavailable"),
    ],
)
async def test_invalid_or_failed_review_falls_back_to_finalize(response) -> None:
    resolution = await GuardReviewer(provider=Provider(response), model="m").decide(
        request(GuardAction.RETRY, GuardAction.FINALIZE)
    )

    assert resolution.action == GuardAction.FINALIZE
    assert resolution.source == "fallback"


@pytest.mark.asyncio
async def test_guard_timeout_falls_back_but_external_cancellation_propagates() -> None:
    class SlowProvider(Provider):
        async def chat(self, messages, tools=None, **kwargs):
            del messages, tools, kwargs
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    resolution = await GuardReviewer(
        provider=SlowProvider(LLMResponse(content="unused")),
        model="m",
        timeout_seconds=0.01,
    ).decide(request(GuardAction.RETRY, GuardAction.FINALIZE))
    assert resolution.source == "fallback"

    task = asyncio.create_task(
        GuardReviewer(
            provider=SlowProvider(LLMResponse(content="unused")),
            model="m",
        ).decide(request(GuardAction.RETRY, GuardAction.FINALIZE))
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_evidence_is_bounded_and_redacts_secrets() -> None:
    evidence = GuardEvidence(
        event=GuardEvent.RUNTIME_ERROR,
        cause="x" * 10_000,
        failures=({"arguments": {"api_key": "secret", "value": "ok"}},) * 20,
        recent_outcomes=("y" * 2_000,) * 20,
    ).to_payload()

    encoded = json.dumps(evidence)
    assert len(encoded) <= 12_000
    assert "secret" not in encoded
    assert "[REDACTED]" in encoded


def test_local_validation_rejects_malformed_arguments() -> None:
    call = ToolCallRequest(id="bad", name="write", arguments="not-an-object")

    result = classify_malformed_tool_calls([call])

    assert result.executable_calls == ()
    assert result.rejected_calls == (call,)
    assert result.tool_results[0].call_id == "bad"


def test_only_successful_side_effect_duplicates_are_suppressed() -> None:
    prior = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "old-read",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"value":"x"}'},
                },
                {
                    "id": "old-write",
                    "type": "function",
                    "function": {"name": "write", "arguments": '{"value":"x"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "old-read", "content": "done"},
        {"role": "tool", "tool_call_id": "old-write", "content": "done"},
    ]
    calls = [
        ToolCallRequest(id="read", name="read", arguments={"value": "x"}),
        ToolCallRequest(id="write", name="write", arguments={"value": "x"}),
    ]

    result = suppress_successful_side_effect_duplicates(
        prior,
        calls,
        tool_modes={"read": "read_only", "write": "mutating"},
    )

    assert [call.id for call in result.executable_calls] == ["read"]
    assert [call.id for call in result.rejected_calls] == ["write"]


def test_failed_side_effect_can_be_retried_with_the_same_arguments() -> None:
    prior = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "failed-write",
                    "type": "function",
                    "function": {"name": "write", "arguments": '{"value":"x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "failed-write",
            "content": "Error: marker was not found",
        },
    ]
    repeated = ToolCallRequest(id="retry", name="write", arguments={"value": "x"})

    result = suppress_successful_side_effect_duplicates(
        prior,
        [repeated],
        tool_modes={"write": "mutating"},
    )

    assert result.executable_calls == (repeated,)
    assert result.rejected_calls == ()


def test_unknown_tool_mode_fails_closed_as_side_effecting() -> None:
    prior = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "custom", "arguments": '{"value":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": "done"},
    ]
    repeated = ToolCallRequest(id="repeat", name="custom", arguments={"value": "x"})

    result = suppress_successful_side_effect_duplicates(
        prior,
        [repeated],
        tool_modes={"custom": "unknown"},
    )

    assert result.executable_calls == ()
    assert result.rejected_calls == (repeated,)
