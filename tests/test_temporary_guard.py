from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from aeloon_core.loop_guard import LoopGuardAction, LoopGuardDecision
from aeloon_core.providers.base import LLMProvider, LLMResponse
from aeloon_core.temporary_guard import GuardEvidence, TemporaryGuard


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
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
        del reasoning_effort, tool_choice
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools or []),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": copy.deepcopy(response_format),
            }
        )
        if not self.responses:
            raise AssertionError("No scripted response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def evidence() -> GuardEvidence:
    return GuardEvidence(
        event="repeated_tool_failure",
        reason="all tool calls in the latest round failed",
        iteration=3,
        phase="running",
        state_digest="abc123",
        budgets={"remaining": 4},
        counters={"unproductive_rounds": 2},
        context={"message_count": 6, "lazy_reference_count": 1},
        failures=(
            {
                "tool_name": "write",
                "arguments": {"path": "game.html"},
                "result": "Error: file already exists",
            },
        ),
    )


def fallback_decision() -> LoopGuardDecision:
    return LoopGuardDecision(
        action=LoopGuardAction.STOP_OFF_TRACK,
        reason="deterministic fallback",
        final_content="original final content",
        prompt_message={"role": "system", "content": "original prompt"},
        progress_message="original progress",
        budget_grant=7,
    )


@pytest.mark.asyncio
async def test_binary_continue_uses_tool_free_json_call_and_local_recovery() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content='{"action":"continue"}',
                usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            )
        ]
    )
    guard = TemporaryGuard(
        provider=provider,
        model="guard-model",
        action_space="binary",
    )

    resolution = await guard.decide(evidence(), fallback_decision())

    assert resolution.source == "temporary_guard"
    assert resolution.fallback_used is False
    assert resolution.usage_category == "harness"
    assert resolution.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 3,
        "total_tokens": 23,
    }
    assert resolution.decision.action == LoopGuardAction.RETURN_TO_MODEL
    assert resolution.decision.prompt_message is not None
    assert "TEMPORARY GUARD RECOVERY" in resolution.decision.prompt_message["content"]
    assert "Error: file already exists" in resolution.decision.prompt_message["content"]

    call = provider.calls[0]
    assert call["tools"] == []
    assert call["model"] == "guard-model"
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_object"}
    assert '"continue", "terminate"' in call["messages"][0]["content"]
    user_payload = json.loads(call["messages"][1]["content"])
    assert set(user_payload) == {"evidence"}
    assert "messages" not in user_payload["evidence"]
    assert user_payload["evidence"]["state_digest"] == "abc123"
    assert user_payload["evidence"]["context"] == {
        "lazy_reference_count": 1,
        "message_count": 6,
    }


@pytest.mark.asyncio
async def test_binary_terminate_compiles_user_facing_text_locally() -> None:
    provider = ScriptedProvider([LLMResponse(content='{"action":"terminate"}')])
    guard = TemporaryGuard(provider=provider, model="guard-model", action_space="binary")

    resolution = await guard.decide(evidence(), fallback_decision())

    assert resolution.decision.action == LoopGuardAction.STOP_OFF_TRACK
    assert resolution.decision.final_content is not None
    assert "appears to be off track" in resolution.decision.final_content
    assert "all tool calls in the latest round failed" in resolution.decision.final_content


@pytest.mark.parametrize(
    ("action", "expected"),
    [(action.value, action) for action in LoopGuardAction],
)
@pytest.mark.asyncio
async def test_full_action_space_supports_every_loop_guard_action(
    action: str,
    expected: LoopGuardAction,
) -> None:
    provider = ScriptedProvider([LLMResponse(content=json.dumps({"action": action}))])
    guard = TemporaryGuard(provider=provider, model="guard-model", action_space="full")

    resolution = await guard.decide(evidence(), fallback_decision())

    assert resolution.source == "temporary_guard"
    assert resolution.decision.action == expected
    if expected == LoopGuardAction.RETURN_TO_MODEL:
        assert resolution.decision.prompt_message is not None
    elif expected == LoopGuardAction.EXTEND_BUDGET:
        assert resolution.decision.budget_grant == 1
    elif expected == LoopGuardAction.FINALIZE:
        assert resolution.decision.prompt_message == {
            "role": "user",
            "content": resolution.decision.prompt_message["content"],
        }
        assert "text only" in resolution.decision.prompt_message["content"]
    elif expected in {LoopGuardAction.FINAL_RESPONSE, LoopGuardAction.STOP_OFF_TRACK}:
        assert resolution.decision.final_content


@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(content="not json", usage={"total_tokens": 4}),
        LLMResponse(content='{"action":"invent_action"}', usage={"total_tokens": 5}),
        LLMResponse(
            content='{"action":"continue","reason":"model supplied text"}',
            usage={"total_tokens": 6},
        ),
        LLMResponse(
            content="Error calling guard",
            finish_reason="error",
            usage={"total_tokens": 7},
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_or_error_response_preserves_exact_fallback(
    response: LLMResponse,
) -> None:
    provider = ScriptedProvider([response])
    guard = TemporaryGuard(provider=provider, model="guard-model")
    fallback = fallback_decision()

    resolution = await guard.decide(evidence(), fallback)

    assert resolution.decision is fallback
    assert resolution.source == "rule_fallback"
    assert resolution.fallback_used is True
    assert resolution.usage == response.usage


@pytest.mark.asyncio
async def test_provider_exception_preserves_exact_fallback() -> None:
    provider = ScriptedProvider([RuntimeError("provider exploded")])
    guard = TemporaryGuard(provider=provider, model="guard-model")
    fallback = fallback_decision()

    resolution = await guard.decide(evidence(), fallback)

    assert resolution.decision is fallback
    assert resolution.source == "rule_fallback"
    assert resolution.fallback_used is True
    assert resolution.usage == {}


@pytest.mark.asyncio
async def test_guard_sends_only_bounded_structured_evidence() -> None:
    secret_tail = "SHOULD_NOT_REACH_PROVIDER"
    failures = tuple(
        {
            "tool_name": "exec" + "x" * 500,
            "arguments": {
                "command": "a" * 5_000 + secret_tail,
                "nested": {"payload": "b" * 5_000 + secret_tail},
            },
            "result": "c" * 5_000 + secret_tail,
        }
        for _ in range(10)
    )
    bounded_evidence = GuardEvidence(
        event="failure" + "e" * 1_000,
        reason="r" * 5_000 + secret_tail,
        iteration=2,
        budgets={f"budget-{index}": index for index in range(50)},
        counters={f"counter-{index}": index for index in range(50)},
        failures=failures,
    )
    provider = ScriptedProvider([LLMResponse(content='{"action":"continue"}')])
    guard = TemporaryGuard(provider=provider, model="guard-model", action_space="binary")

    await guard.decide(bounded_evidence, fallback_decision())

    raw_payload = provider.calls[0]["messages"][1]["content"]
    payload = json.loads(raw_payload)["evidence"]
    assert secret_tail not in raw_payload
    assert len(payload["failures"]) == 5
    assert len(payload["budgets"]) == 20
    assert len(payload["counters"]) == 20
    assert len(payload["event"]) < 200
    assert len(payload["reason"]) < 700
    assert len(payload["failures"][0]["result"]) < 1_300
    serialized_arguments = json.dumps(payload["failures"][0]["arguments"], ensure_ascii=False)
    assert len(serialized_arguments) < 700
    assert len(raw_payload) < 12_000
