from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai.types import responses
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from aeloon_core.config import AnthropicProviderConfig, VolcengineProviderConfig
from aeloon_core.pydantic_model import (
    VolcengineResponsesModel,
    _VolcengineResponsesEventStream,
    build_anthropic_model,
    build_volcengine_model,
    is_prompt_caching_unsupported_error,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "cache_key"),
    [
        ("https://api.anthropic.com", "anthropic_cache"),
        ("https://gateway.example/v1", "anthropic_cache_messages"),
    ],
)
async def test_builds_official_anthropic_model_and_endpoint_cache_mode(
    base_url: str,
    cache_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    class CapturingClient(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            # The transport is never used; keep the proxy assertion independent of
            # network configuration on the test host.
            kwargs.pop("proxy", None)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("aeloon_core.pydantic_model.httpx.AsyncClient", CapturingClient)
    bundle = build_anthropic_model(
        provider=AnthropicProviderConfig(
            api_key="sk-test",
            base_url=base_url,
            extra_headers={"x-routing-key": "tenant-a"},
            proxy="http://127.0.0.1:7890",
        ),
        model_name="claude-test",
        temperature=0.2,
        reasoning_effort="high",
        timeout=17,
    )
    try:
        assert isinstance(bundle.model, AnthropicModel)
        assert str(bundle.anthropic_client.base_url).rstrip("/") == base_url.rstrip("/")
        assert bundle.anthropic_client.default_headers["x-routing-key"] == "tenant-a"
        assert captured["proxy"] == "http://127.0.0.1:7890"
        assert bundle.settings["temperature"] == 0.2
        assert bundle.settings["timeout"] == 17
        assert bundle.settings["anthropic_effort"] == "high"
        assert bundle.settings[cache_key] is True
        other = {
            "anthropic_cache",
            "anthropic_cache_messages",
        } - {cache_key}
        assert all(key not in bundle.settings for key in other)
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_prompt_caching_can_be_disabled() -> None:
    bundle = build_anthropic_model(
        provider=AnthropicProviderConfig(api_key="sk-test", prompt_caching=False),
        model_name="claude-test",
        temperature=0.7,
        reasoning_effort=None,
        timeout=10,
    )
    try:
        assert "anthropic_cache" not in bundle.settings
        assert "anthropic_cache_messages" not in bundle.settings
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_kimi_context_suffix_maps_to_gateway_model_id() -> None:
    bundle = build_anthropic_model(
        provider=AnthropicProviderConfig(
            api_key="sk-test",
            base_url="https://api.kimi.com/coding/",
        ),
        model_name="k3[1m]",
        temperature=0.7,
        reasoning_effort=None,
        timeout=10,
    )
    try:
        assert bundle.model.model_name == "k3"
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_builds_volcengine_agent_plan_responses_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    class CapturingClient(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            kwargs.pop("proxy", None)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("aeloon_core.pydantic_model.httpx.AsyncClient", CapturingClient)
    bundle = build_volcengine_model(
        provider=VolcengineProviderConfig(
            api_key="ark-test-key",
            extra_headers={"x-routing-key": "tenant-a"},
            proxy="http://127.0.0.1:7890",
        ),
        model_name="ark-code-latest",
        temperature=0.2,
        reasoning_effort="high",
        timeout=17,
    )
    try:
        assert isinstance(bundle.model, VolcengineResponsesModel)
        assert isinstance(bundle.model, OpenAIResponsesModel)
        assert bundle.openai_client is not None
        assert (
            str(bundle.openai_client.base_url).rstrip("/")
            == "https://ark.cn-beijing.volces.com/api/plan/v3"
        )
        assert bundle.openai_client.default_headers["x-routing-key"] == "tenant-a"
        assert captured["proxy"] == "http://127.0.0.1:7890"
        assert bundle.settings["temperature"] == 0.2
        assert bundle.settings["timeout"] == 17
        assert bundle.settings["openai_reasoning_effort"] == "high"
        assert bundle.prompt_cache is None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_volcengine_stream_drops_empty_reasoning_frames() -> None:
    part_type = responses.ResponseReasoningSummaryPartAddedEvent.model_fields[
        "part"
    ].annotation
    empty_part = part_type.model_construct(text=None, type="summary_text")
    empty_added = responses.ResponseReasoningSummaryPartAddedEvent.model_construct(
        item_id="reasoning-1",
        output_index=0,
        part=empty_part,
        sequence_number=1,
        summary_index=0,
        type="response.reasoning_summary_part.added",
    )
    empty_delta = responses.ResponseReasoningSummaryTextDeltaEvent.model_construct(
        delta=None,
        item_id="reasoning-1",
        output_index=0,
        sequence_number=2,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    )
    valid_delta = responses.ResponseReasoningSummaryTextDeltaEvent.model_construct(
        delta="use the finish tool",
        item_id="reasoning-1",
        output_index=0,
        sequence_number=3,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    )

    class FakeStream:
        def __init__(self) -> None:
            self.events = iter((empty_added, empty_delta, valid_delta))
            self.closed = False

        async def __anext__(self) -> responses.ResponseStreamEvent:
            try:
                return next(self.events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def close(self) -> None:
            self.closed = True

    source = FakeStream()
    stream = _VolcengineResponsesEventStream(source)  # type: ignore[arg-type]

    assert await anext(stream) is valid_delta
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    await stream.close()
    assert source.closed is True


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": "cache_control is an unknown field"}, True),
        ({"error": "prompt caching not supported"}, True),
        ({"error": "max_tokens must be greater than zero"}, False),
    ],
)
def test_cache_compatibility_error_detection(body: dict[str, str], expected: bool) -> None:
    error = ModelHTTPError(status_code=400, model_name="gateway", body=body)
    assert is_prompt_caching_unsupported_error(error) is expected
