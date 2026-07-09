from __future__ import annotations

import pytest

from aeloon_core.model_metadata import (
    GENERIC_DEFAULT_MAX_TOKENS,
    ModelLimits,
    litellm_limits_from_table,
    resolve_max_tokens_for_model,
    resolve_model_limits,
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
async def test_resolve_model_limits_reads_litellm_table(monkeypatch) -> None:
    async def fake_table(*args, **kwargs):
        del args, kwargs
        return {
            "deepseek-v4-flash": {"max_input_tokens": 1_000_000, "max_output_tokens": 8192},
        }

    monkeypatch.setattr("aeloon_core.model_metadata._litellm_table", fake_table)

    assert (
        await resolve_model_limits("deepseek/deepseek-v4-flash")
    ) == ModelLimits(output_tokens=8192, context_tokens=1_000_000, source="litellm")
