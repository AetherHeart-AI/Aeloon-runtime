from __future__ import annotations

from aeloon_core.providers.custom_provider import CustomProvider


def provider() -> CustomProvider:
    instance = object.__new__(CustomProvider)
    instance.default_model = "default-model"
    return instance


def test_unset_max_tokens_is_omitted_from_request() -> None:
    kwargs = provider()._build_kwargs(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=None,
    )

    assert "max_tokens" not in kwargs


def test_explicit_max_tokens_is_sent() -> None:
    kwargs = provider()._build_kwargs(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=12_345,
    )

    assert kwargs["max_tokens"] == 12_345
