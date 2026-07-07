from __future__ import annotations

import pytest

from aeloon_core.model_metadata import (
    GENERIC_DEFAULT_MAX_TOKENS,
    ModelLimits,
    litellm_limits_from_table,
    openrouter_limits_from_payload,
    resolve_max_tokens_for_model,
    resolve_model_limits,
)


def test_openrouter_limits_use_top_provider_completion_cap() -> None:
    limits = openrouter_limits_from_payload(
        {
            "data": [
                {
                    "id": "deepseek/deepseek-v4-flash",
                    "context_length": 1_048_576,
                    "top_provider": {
                        "context_length": 1_048_576,
                        "max_completion_tokens": 16_384,
                    },
                }
            ]
        },
        "deepseek-v4-flash",
    )

    assert limits == ModelLimits(
        output_tokens=16_384,
        context_tokens=1_048_576,
        source="openrouter",
    )


def test_litellm_limits_use_max_output_tokens() -> None:
    limits = litellm_limits_from_table(
        {
            "azure_ai/deepseek-v4-flash": {
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 384_000,
            },
            "deepseek-v4-flash": {
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 8192,
                "max_tokens": 8192,
            }
        },
        "deepseek/deepseek-v4-flash",
    )

    assert limits == ModelLimits(
        output_tokens=8192,
        context_tokens=1_000_000,
        source="litellm",
    )


@pytest.mark.asyncio
async def test_explicit_max_tokens_skips_metadata_lookup() -> None:
    assert await resolve_max_tokens_for_model("deepseek-v4-flash", 1234) == 1234
    assert await resolve_max_tokens_for_model("deepseek-v4-flash", 0) == 1


@pytest.mark.asyncio
async def test_auto_max_tokens_falls_back_when_metadata_missing(monkeypatch) -> None:
    async def missing_limits(*args, **kwargs):
        del args, kwargs
        return None

    monkeypatch.setattr("aeloon_core.model_metadata.resolve_model_limits", missing_limits)

    assert await resolve_max_tokens_for_model("unknown-model", None) == GENERIC_DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
async def test_resolver_priority_tracks_api_base(monkeypatch) -> None:
    async def openrouter_limits(*args, **kwargs):
        del args, kwargs
        return ModelLimits(output_tokens=16_384, context_tokens=1_048_576, source="openrouter")

    async def litellm_limits(*args, **kwargs):
        del args, kwargs
        return ModelLimits(output_tokens=8192, context_tokens=1_000_000, source="litellm")

    monkeypatch.setattr("aeloon_core.model_metadata._lookup_openrouter_limits", openrouter_limits)
    monkeypatch.setattr("aeloon_core.model_metadata._lookup_litellm_limits", litellm_limits)

    assert (
        await resolve_model_limits(
            "deepseek-v4-flash",
            api_base="https://openrouter.ai/api/v1",
        )
    ) == ModelLimits(output_tokens=16_384, context_tokens=1_048_576, source="openrouter")
    assert (
        await resolve_model_limits(
            "deepseek-v4-flash",
            api_base="https://api.deepseek.com/v1",
        )
    ) == ModelLimits(output_tokens=8192, context_tokens=1_000_000, source="litellm")
