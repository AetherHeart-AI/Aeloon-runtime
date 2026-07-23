"""PydanticAI production model construction for Aeloon Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, AsyncStream
from openai.types import responses
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import (
    OpenAIModelName,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
    OpenAIResponsesStreamedResponse,
)
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from aeloon_core.config import AnthropicProviderConfig, VolcengineProviderConfig

_OFFICIAL_ANTHROPIC_URLS = {
    "https://api.anthropic.com",
    "https://api.anthropic.com/v1",
}


class _VolcengineResponsesEventStream:
    """Drop empty reasoning frames emitted by Ark's Responses-compatible stream."""

    def __init__(self, source: AsyncStream[responses.ResponseStreamEvent]) -> None:
        self.source = source

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> responses.ResponseStreamEvent:
        while True:
            event = await self.source.__anext__()
            if not _is_empty_reasoning_event(event):
                return event

    async def close(self) -> None:
        await self.source.close()


def _is_empty_reasoning_event(event: responses.ResponseStreamEvent) -> bool:
    if isinstance(event, responses.ResponseReasoningSummaryPartAddedEvent):
        return not event.part.text
    if isinstance(event, responses.ResponseReasoningSummaryTextDeltaEvent):
        return not event.delta
    return False


class VolcengineResponsesModel(OpenAIResponsesModel):
    """OpenAI Responses model with Ark's empty reasoning frames normalized."""

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        expected_model_name: OpenAIModelName | None = None,
    ) -> OpenAIResponsesStreamedResponse:
        return await super()._process_streamed_response(
            _VolcengineResponsesEventStream(response),  # type: ignore[arg-type]
            model_settings,
            model_request_parameters,
            expected_model_name=expected_model_name,
        )


@dataclass(slots=True)
class PromptCacheState:
    """Process-local compatibility memory for one provider endpoint."""

    disabled: bool = False


@dataclass(slots=True)
class PydanticModelBundle:
    """A configured model plus the resources and settings it owns."""

    model: Model
    settings: ModelSettings
    http_client: httpx.AsyncClient
    prompt_cache: PromptCacheState | None = None
    anthropic_client: AsyncAnthropic | None = None
    openai_client: AsyncOpenAI | None = None

    async def close(self) -> None:
        if self.anthropic_client is not None:
            await self.anthropic_client.close()
        elif self.openai_client is not None:
            await self.openai_client.close()


def build_anthropic_model(
    *,
    provider: AnthropicProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build one reusable provider/model bundle for the process model router."""

    http_client = httpx.AsyncClient(
        proxy=provider.proxy,
        timeout=httpx.Timeout(timeout),
    )
    anthropic_client = AsyncAnthropic(
        api_key=provider.api_key,
        base_url=provider.base_url,
        default_headers=provider.extra_headers or None,
        http_client=http_client,
        timeout=timeout,
    )
    pydantic_provider = AnthropicProvider(anthropic_client=anthropic_client)
    model = AnthropicModel(_api_model_id(model_name), provider=pydantic_provider)

    settings: AnthropicModelSettings = {
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning_effort:
        settings["anthropic_effort"] = reasoning_effort
    if provider.prompt_caching:
        normalized_url = provider.base_url.rstrip("/")
        if normalized_url in _OFFICIAL_ANTHROPIC_URLS:
            settings["anthropic_cache"] = True
        else:
            settings["anthropic_cache_messages"] = True

    return PydanticModelBundle(
        model=model,
        settings=settings,
        http_client=http_client,
        prompt_cache=PromptCacheState(),
        anthropic_client=anthropic_client,
    )


def build_volcengine_model(
    *,
    provider: VolcengineProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PydanticModelBundle:
    """Build a Volcano Engine Ark Agent Plan model using the Responses API."""

    http_client = httpx.AsyncClient(
        proxy=provider.proxy,
        timeout=httpx.Timeout(timeout),
    )
    openai_client = AsyncOpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        default_headers=provider.extra_headers or None,
        http_client=http_client,
        timeout=timeout,
    )
    pydantic_provider = OpenAIProvider(openai_client=openai_client)
    model = VolcengineResponsesModel(model_name, provider=pydantic_provider)

    settings: OpenAIResponsesModelSettings = {
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning_effort:
        settings["openai_reasoning_effort"] = reasoning_effort  # type: ignore[typeddict-item]

    return PydanticModelBundle(
        model=model,
        settings=settings,
        http_client=http_client,
        openai_client=openai_client,
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
    "VolcengineResponsesModel",
    "build_anthropic_model",
    "build_volcengine_model",
    "is_prompt_caching_unsupported_error",
    "prompt_caching_enabled",
    "without_prompt_caching",
]
