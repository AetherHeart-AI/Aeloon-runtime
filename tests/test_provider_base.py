from __future__ import annotations

from aeloon_core.providers.base import LLMProvider


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
