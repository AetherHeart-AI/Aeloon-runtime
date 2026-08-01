"""DeepSeek provider construction for pi-ai."""

from __future__ import annotations

from aeloon_core.config import DeepSeekProviderConfig
from aeloon_core.harness.provider.base import PiModel, PiModelBundle, _base_settings


def build_deepseek_model(
    *,
    provider: DeepSeekProviderConfig,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
) -> PiModelBundle:
    """Build a DeepSeek reference resolved by pi-ai inside the Bun bridge."""

    model = PiModel(
        provider="deepseek",
        model_id=model_name,
        api_key=provider.api_key,
        proxy=provider.proxy,
    )
    settings = _base_settings(
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        extra_headers=provider.extra_headers,
    )
    return PiModelBundle(model=model, settings=settings)


__all__ = ["build_deepseek_model"]
