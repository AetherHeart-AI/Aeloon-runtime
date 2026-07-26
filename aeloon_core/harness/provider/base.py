"""Provider-neutral model bundles, transports, and prompt-cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings


@dataclass(slots=True)
class PromptCacheState:
    """Process-local prompt-cache capability state for one endpoint."""

    disabled: bool = False


@dataclass(slots=True)
class PydanticModelBundle:
    """A model, provider, settings, and transport owned by Aeloon."""

    model: Model
    provider: Provider[Any]
    settings: ModelSettings
    http_client: httpx.AsyncClient
    prompt_cache: PromptCacheState | None = None

    async def close(self) -> None:
        await self.http_client.aclose()


def _http_client(
    *,
    proxy: str | None,
    timeout: int,
    extra_headers: dict[str, str],
) -> httpx.AsyncClient:
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
    settings: ModelSettings = {
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning_effort:
        settings["thinking"] = reasoning_effort  # type: ignore[typeddict-item]
    if extra_headers:
        settings["extra_headers"] = dict(extra_headers)
    return settings


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
    """Recognize explicit cache-field compatibility failures."""

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


__all__ = [
    "PromptCacheState",
    "PydanticModelBundle",
    "is_prompt_caching_unsupported_error",
    "prompt_caching_enabled",
    "without_prompt_caching",
]
