"""Tests for provider-specific Pydantic AI model construction."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.tools import ToolDefinition

from aeloon_core.config import DeepSeekProviderConfig
from aeloon_core.harness.provider import (
    build_deepseek_model,
    is_prompt_caching_unsupported_error,
)


@pytest.mark.asyncio
async def test_builds_direct_deepseek_model_with_transport_settings(
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

    monkeypatch.setattr(
        "aeloon_core.harness.provider.base.httpx.AsyncClient",
        CapturingClient,
    )
    bundle = build_deepseek_model(
        provider=DeepSeekProviderConfig(
            api_key="sk-test",
            extra_headers={"x-routing-key": "tenant-a"},
            proxy="http://127.0.0.1:7890",
        ),
        model_name="deepseek-v4-flash",
        temperature=0.2,
        reasoning_effort="high",
        timeout=17,
    )
    try:
        assert isinstance(bundle.model, OpenAIChatModel)
        assert isinstance(bundle.provider, DeepSeekProvider)
        assert (
            str(bundle.provider.client.base_url).rstrip("/")
            == "https://api.deepseek.com"
        )
        assert captured["proxy"] == "http://127.0.0.1:7890"
        assert bundle.settings["temperature"] == 0.2
        assert bundle.settings["timeout"] == 17
        assert bundle.settings["thinking"] == "high"
        assert bundle.settings["extra_headers"] == {"x-routing-key": "tenant-a"}
        assert bundle.prompt_cache is None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_deepseek_model_without_reasoning_effort() -> None:
    bundle = build_deepseek_model(
        provider=DeepSeekProviderConfig(api_key="sk-test"),
        model_name="deepseek-v4-flash",
        temperature=0.7,
        reasoning_effort=None,
        timeout=10,
    )
    try:
        assert "thinking" not in bundle.settings
        assert bundle.settings["temperature"] == 0.7
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_direct_deepseek_model_uses_thinking_tool_compatibility_profile() -> None:
    bundle = build_deepseek_model(
        provider=DeepSeekProviderConfig(api_key="sk-test"),
        model_name="deepseek-v4-flash",
        temperature=0.7,
        reasoning_effort=None,
        timeout=10,
    )
    try:
        assert isinstance(bundle.model, OpenAIChatModel)
        assert bundle.model.profile["openai_chat_thinking_field"] == "reasoning_content"
        assert bundle.model.profile["openai_chat_send_back_thinking_parts"] == "field"
        assert bundle.model.profile["openai_supports_tool_choice_required"] is False

        _, tool_choice = bundle.model._get_tool_choice(
            bundle.settings,
            ModelRequestParameters(
                output_mode="tool",
                output_tools=[ToolDefinition(name="final_result")],
                allow_text_output=False,
            ),
        )
        assert tool_choice == "auto"
    finally:
        await bundle.close()


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
