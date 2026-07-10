from __future__ import annotations

from typing import Any

import pytest

from aeloon_core.providers.base import LLMProvider, LLMResponse


class StubProvider(LLMProvider):
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
        del (
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
            response_format,
        )
        raise NotImplementedError


def test_context_limit_400_is_not_transient() -> None:
    error = (
        "Error code: 400 - This endpoint's maximum context length is 500000 tokens. "
        "However, you requested about 506561 tokens, including 500000 in the output."
    )

    assert LLMProvider._is_transient_error(error) is False


def test_token_count_that_looks_like_status_is_not_transient() -> None:
    error = "Error code: 400 - maximum context length is 500 tokens"

    assert LLMProvider._is_transient_error(error) is False


def test_retryable_http_status_is_transient() -> None:
    assert LLMProvider._is_transient_error("Error code: 500 - internal server error") is True
    assert LLMProvider._is_transient_error("HTTP 429: rate limit exceeded") is True


@pytest.mark.asyncio
async def test_retry_accumulates_usage_from_every_attempt(monkeypatch) -> None:
    provider = StubProvider()
    responses = [
        LLMResponse(
            content="HTTP 429: rate limit exceeded",
            finish_reason="error",
            usage={"prompt_tokens": 2, "total_tokens": 2},
        ),
        LLMResponse(
            content="done",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        ),
    ]

    async def no_sleep(_delay: float) -> None:
        return None

    async def attempt(_attempt: int) -> LLMResponse:
        return responses.pop(0)

    monkeypatch.setattr("aeloon_core.providers.base.asyncio.sleep", no_sleep)

    response = await provider._retry(attempt, label="test")

    assert response.content == "done"
    assert response.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_tokens": 6,
    }
