from __future__ import annotations

import json
import os

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
    assert uasm.rule_engine_enabled is True
    assert uasm.temporary_guard_enabled is True
    assert uasm.minimal_context_enabled is True
    assert uasm.transition_trace_enabled is True
    assert uasm.guard_decision_mode == "full"
