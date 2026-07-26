"""Volcengine Ark provider construction."""

from __future__ import annotations

from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from aeloon_core.config import VolcengineProviderConfig
from aeloon_core.harness.provider.base import (
    PydanticModelBundle,
    _base_settings,
    _http_client,
)


def build_volcengine_model(
    *,
    provider: VolcengineProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build Ark Agent Plan through Pydantic AI's OpenAI Responses provider."""

    http_client = _http_client(
        proxy=provider.proxy,
        timeout=timeout,
        extra_headers=provider.extra_headers,
    )
    pydantic_provider = OpenAIProvider(
        api_key=provider.api_key,
        base_url=provider.base_url,
        http_client=http_client,
    )
    model = OpenAIResponsesModel(model_name, provider=pydantic_provider)
    settings = _base_settings(
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        extra_headers=provider.extra_headers,
    )
    return PydanticModelBundle(
        model=model,
        provider=pydantic_provider,
        settings=settings,
        http_client=http_client,
    )


__all__ = ["build_volcengine_model"]
