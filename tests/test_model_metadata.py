from __future__ import annotations

import pytest

from aeloon_core.model_metadata import (
    litellm_context_window_from_table,
    resolve_context_window,
)


def test_litellm_metadata_uses_context_window_only() -> None:
    context_window = litellm_context_window_from_table(
        {
            "azure_ai/deepseek-v4-flash": {
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 384_000,
            },
            "deepseek-v4-flash": {
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 8192,
                "max_tokens": 8192,
            },
        },
        "deepseek/deepseek-v4-flash",
    )

    assert context_window == 1_000_000


@pytest.mark.asyncio
async def test_resolve_context_window_reads_litellm_table(monkeypatch) -> None:
    async def fake_table(*args, **kwargs):
        del args, kwargs
        return {
            "deepseek-v4-flash": {"max_input_tokens": 1_000_000, "max_output_tokens": 8192},
        }

    monkeypatch.setattr("aeloon_core.model_metadata._litellm_table", fake_table)

    assert (
        await resolve_context_window("deepseek/deepseek-v4-flash")
    ) == 1_000_000
