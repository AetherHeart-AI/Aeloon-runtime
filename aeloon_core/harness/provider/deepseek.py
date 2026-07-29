"""Direct DeepSeek provider construction through Pydantic AI."""

from __future__ import annotations

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from aeloon_core.config import DeepSeekProviderConfig
from aeloon_core.harness.provider.base import (
    PydanticModelBundle,
    _base_settings,
    _http_client,
)


def build_deepseek_model(
    *,
    provider: DeepSeekProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build a DeepSeek Chat Completions model with its native Pydantic AI profile."""

    http_client = _http_client(
        proxy=provider.proxy,
        timeout=timeout,
        extra_headers=provider.extra_headers,
    )
    pydantic_provider = DeepSeekProvider(
        api_key=provider.api_key,
        http_client=http_client,
    )
    model = OpenAIChatModel(model_name, provider=pydantic_provider)
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


__all__ = ["build_deepseek_model"]
