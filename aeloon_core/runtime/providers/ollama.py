"""Ollama's OpenAI-compatible runtime Provider."""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from aeloon_core.core import Model
from aeloon_core.runtime.providers.openai import OpenAICompatibleProvider

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/v1"


class OllamaProvider(OpenAICompatibleProvider):
    driver = "ollama"

    def __init__(
        self,
        *,
        provider_id: str,
        name: str,
        endpoint: str = OLLAMA_ENDPOINT,
        models: tuple[Model, ...] = (),
        enabled: bool = True,
        proxy: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            name=name,
            endpoint=endpoint,
            models=models,
            enabled=enabled,
            proxy=proxy,
            headers=headers,
            client=client,
            requires_api_key=False,
        )


__all__ = ["OLLAMA_ENDPOINT", "OllamaProvider"]
