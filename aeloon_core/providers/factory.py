"""Provider factory."""

from __future__ import annotations

from aeloon_core.config import Config
from aeloon_core.providers.base import GenerationSettings
from aeloon_core.providers.custom_provider import CustomProvider


def create_provider(config: Config) -> CustomProvider:
    """Create the single OpenAI-compatible provider."""

    provider_config = config.providers.custom
    defaults = config.agents.defaults
    provider = CustomProvider(
        api_key=provider_config.api_key,
        api_base=provider_config.api_base,
        default_model=defaults.model,
        extra_headers=provider_config.extra_headers,
        proxy=provider_config.proxy,
    )
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    provider._CHAT_TIMEOUT_S = defaults.chat_timeout
    return provider
