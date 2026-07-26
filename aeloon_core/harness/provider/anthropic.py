"""Anthropic provider construction."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from aeloon_core.config import AnthropicProviderConfig
from aeloon_core.harness.provider.base import (
    PromptCacheState,
    PydanticModelBundle,
    _base_settings,
    _http_client,
)

_OFFICIAL_URLS = {
    "https://api.anthropic.com",
    "https://api.anthropic.com/v1",
}


def build_anthropic_model(
    *,
    provider: AnthropicProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build Anthropic Messages through Pydantic AI."""

    http_client = _http_client(
        proxy=provider.proxy,
        timeout=timeout,
        extra_headers=provider.extra_headers,
    )
    pydantic_provider = AnthropicProvider(
        api_key=provider.api_key,
        base_url=provider.base_url,
        http_client=http_client,
    )
    model = AnthropicModel(_api_model_id(model_name), provider=pydantic_provider)
    settings = _base_settings(
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        extra_headers=provider.extra_headers,
    )
    if provider.prompt_caching:
        normalized_url = provider.base_url.rstrip("/")
        if normalized_url in _OFFICIAL_URLS:
            settings["anthropic_cache"] = True  # type: ignore[typeddict-unknown-key]
        else:
            settings["anthropic_cache_messages"] = True  # type: ignore[typeddict-unknown-key]

    return PydanticModelBundle(
        model=model,
        provider=pydantic_provider,
        settings=settings,
        http_client=http_client,
        prompt_cache=PromptCacheState(),
    )


def _api_model_id(model_name: str) -> str:
    return "k3" if model_name == "k3[1m]" else model_name


__all__ = ["build_anthropic_model"]
