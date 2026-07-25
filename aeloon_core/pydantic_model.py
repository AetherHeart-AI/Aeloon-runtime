"""Pydantic AI-native provider and model construction for Aeloon Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from aeloon_core.config import AnthropicProviderConfig, VolcengineProviderConfig

_OFFICIAL_ANTHROPIC_URLS = {
    "https://api.anthropic.com",
    "https://api.anthropic.com/v1",
}


@dataclass(slots=True)
class PromptCacheState:
    """Process-local compatibility memory for one provider endpoint."""

    disabled: bool = False


@dataclass(slots=True)
class PydanticModelBundle:
    """A Pydantic AI model, its provider, and the transport owned by Aeloon."""

    model: Model
    provider: Provider[Any]
    settings: ModelSettings
    http_client: httpx.AsyncClient
    prompt_cache: PromptCacheState | None = None

    async def close(self) -> None:
        """Close the single transport shared by the Pydantic AI provider."""

        await self.http_client.aclose()


def _http_client(
    *,
    proxy: str | None,
    timeout: int,
    extra_headers: dict[str, str],
) -> httpx.AsyncClient:
    """Build the transport injected into a Pydantic AI provider."""

    return httpx.AsyncClient(
        proxy=proxy,
        timeout=httpx.Timeout(timeout),
        headers=extra_headers or None,
    )


def _base_settings(
    *,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
    extra_headers: dict[str, str],
) -> ModelSettings:
    """Use Pydantic AI's provider-neutral settings wherever possible."""

    settings: ModelSettings = {
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning_effort:
        settings["thinking"] = reasoning_effort  # type: ignore[typeddict-item]
    if extra_headers:
        settings["extra_headers"] = dict(extra_headers)
    return settings


def build_anthropic_model(
    *,
    provider: AnthropicProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build Anthropic Messages through Pydantic AI's provider abstraction."""

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
        if normalized_url in _OFFICIAL_ANTHROPIC_URLS:
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


def build_volcengine_model(
    *,
    provider: VolcengineProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build Ark Agent Plan with Pydantic AI's OpenAI Responses provider."""

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


def without_prompt_caching(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Return model settings with every Anthropic prompt-cache switch removed."""

    return {
        key: value
        for key, value in (settings or {}).items()
        if key
        not in {
            "anthropic_cache",
            "anthropic_cache_messages",
            "anthropic_cache_instructions",
            "anthropic_cache_tool_definitions",
        }
    }


def prompt_caching_enabled(settings: dict[str, Any] | None) -> bool:
    return any(
        bool((settings or {}).get(key))
        for key in (
            "anthropic_cache",
            "anthropic_cache_messages",
            "anthropic_cache_instructions",
            "anthropic_cache_tool_definitions",
        )
    )


def is_prompt_caching_unsupported_error(exc: Exception) -> bool:
    """Recognize only explicit cache-field compatibility failures."""

    body = getattr(exc, "body", None)
    text = f"{exc} {body}".lower()
    field_markers = ("cache_control", "cache control", "prompt cache", "prompt caching")
    rejection_markers = (
        "does not support",
        "extra_forbidden",
        "extra fields not permitted",
        "invalid field",
        "not allowed",
        "not supported",
        "unexpected argument",
        "unknown field",
        "unknown parameter",
        "unrecognized",
        "unsupported",
    )
    return any(marker in text for marker in field_markers) and any(
        marker in text for marker in rejection_markers
    )


def _api_model_id(model_name: str) -> str:
    """Preserve the existing Claude Code Kimi context-suffix compatibility."""

    return "k3" if model_name == "k3[1m]" else model_name


__all__ = [
    "PromptCacheState",
    "PydanticModelBundle",
    "build_anthropic_model",
    "build_volcengine_model",
    "is_prompt_caching_unsupported_error",
    "prompt_caching_enabled",
    "without_prompt_caching",
]
