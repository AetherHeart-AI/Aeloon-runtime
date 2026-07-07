from __future__ import annotations

from aeloon_core.config import load_config


def test_env_max_tokens_override(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "missing.json"
    monkeypatch.setenv("AELOON_CORE_MAX_TOKENS", "32768")

    config = load_config(config_path)

    assert config.agents.defaults.max_tokens == 32768


def test_env_max_tokens_auto(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "missing.json"
    monkeypatch.setenv("AELOON_CORE_MAX_TOKENS", "auto")

    config = load_config(config_path)

    assert config.agents.defaults.max_tokens is None
