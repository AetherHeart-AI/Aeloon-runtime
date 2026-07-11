from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from aeloon_core.config import Config, load_config


def test_legacy_output_budget_settings_are_ignored(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "max_tokens": 32_768,
                        "context_compaction": {"buffer_tokens": 20_000},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELOON_CORE_MAX_TOKENS", "32768")

    config = load_config(config_path)

    assert "max_tokens" not in config.agents.defaults.model_dump()
    assert "buffer_tokens" not in config.agents.defaults.context_compaction.model_dump()


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


def test_state_machine_policy_defaults_enable_full_runtime() -> None:
    uasm = Config().agents.defaults.uasm

    assert not hasattr(uasm, "enabled")
    assert uasm.transition_trace_enabled is True


def test_profile_defaults_and_environment_selection(monkeypatch, tmp_path) -> None:
    defaults = Config().agents.defaults
    assert defaults.profile_id == "coding"
    assert defaults.max_handoffs == 8

    disabled = Config.model_validate({"agents": {"defaults": {"profile_id": None}}})
    assert disabled.agents.defaults.profile_id is None

    monkeypatch.setenv("AELOON_CORE_PROFILE_ID", "coding-team")
    config = load_config(tmp_path / "missing.json")
    assert config.agents.defaults.profile_id == "coding-team"

    monkeypatch.setenv("AELOON_CORE_PROFILE_ID", "none")
    config = load_config(tmp_path / "missing.json")
    assert config.agents.defaults.profile_id is None

    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        Config.model_validate({"agents": {"defaults": {"profile_id": "Coding.Team"}}})
