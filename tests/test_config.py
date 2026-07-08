from __future__ import annotations

import os

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


def test_skill_env_overrides(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "missing.json"
    monkeypatch.setenv("AELOON_CORE_SKILLS_ENABLED", "false")
    monkeypatch.setenv("AELOON_CORE_DISABLE_EXTERNAL_SKILLS", "true")
    monkeypatch.setenv("AELOON_CORE_DISABLE_CLAUDE_CODE_SKILLS", "true")
    monkeypatch.setenv("AELOON_CORE_SKILL_PATHS", os.pathsep.join(["one", "two"]))

    config = load_config(config_path)

    assert config.skills.enabled is False
    assert config.skills.external is False
    assert config.skills.claude_code is False
    assert config.skills.paths == ["one", "two"]
