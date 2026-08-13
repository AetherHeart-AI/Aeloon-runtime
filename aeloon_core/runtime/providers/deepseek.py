"""DeepSeek runtime Provider."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx

from aeloon_core.core import Model
from aeloon_core.runtime.providers.openai import OpenAICompatibleProvider

DEEPSEEK_PROVIDER_ID = "deepseek"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com"

DEEPSEEK_V4_FLASH = Model(
    id="deepseek/deepseek-v4-flash",
    name="DeepSeek V4 Flash",
    provider=DEEPSEEK_PROVIDER_ID,
    reasoning=True,
    context_window=1_000_000,
    max_output_tokens=384_000,
    cost={"input": 0.14, "output": 0.28, "cacheRead": 0.0028, "cacheWrite": 0.0},
)
DEEPSEEK_V4_PRO = replace(
    DEEPSEEK_V4_FLASH,
    id="deepseek/deepseek-v4-pro",
    name="DeepSeek V4 Pro",
    cost={"input": 0.435, "output": 0.87, "cacheRead": 0.003625, "cacheWrite": 0.0},
)
DEEPSEEK_MODELS = {
    DEEPSEEK_V4_FLASH.id: DEEPSEEK_V4_FLASH,
    DEEPSEEK_V4_PRO.id: DEEPSEEK_V4_PRO,
}


class DeepSeekProvider(OpenAICompatibleProvider):
    driver = "deepseek"

    def __init__(
        self,
        *,
        name: str = "DeepSeek",
        endpoint: str = DEEPSEEK_ENDPOINT,
        api_key: str | None = None,
        proxy: str | None = None,
        headers: dict[str, str] | None = None,
        models: tuple[Model, ...] | None = None,
        enabled: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            provider_id=DEEPSEEK_PROVIDER_ID,
            name=name,
            endpoint=endpoint,
            models=models if models is not None else tuple(DEEPSEEK_MODELS.values()),
            enabled=enabled,
            api_key=api_key,
            proxy=proxy,
            headers=headers,
            client=client,
            requires_api_key=True,
            thinking_level_map={"high": "high", "max": "max"},
            requires_reasoning_content=True,
        )

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "authenticated": bool(self.api_key),
        }


def get_deepseek_model(model_id: str) -> Model:
    canonical = model_id if model_id.startswith("deepseek/") else f"deepseek/{model_id}"
    return DEEPSEEK_MODELS[canonical]


__all__ = [
    "DEEPSEEK_ENDPOINT",
    "DEEPSEEK_MODELS",
    "DEEPSEEK_PROVIDER_ID",
    "DEEPSEEK_V4_FLASH",
    "DEEPSEEK_V4_PRO",
    "DeepSeekProvider",
    "get_deepseek_model",
]
