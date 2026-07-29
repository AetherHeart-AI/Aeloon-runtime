"""Tests for runtime config loading, migration, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aeloon_core.__main__ import main
from aeloon_core.config import Config, load_config, parse_model_ref, save_config


def _write_config(path: Path, defaults: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"agents": {"defaults": defaults}}),
        encoding="utf-8",
    )


def test_load_config_discards_removed_v1_profile_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(
        path,
        {
            "model": "test-model",
            "base_profile_id": "base",
            "profile_id": None,
            "max_handoffs": 8,
        },
    )

    config = load_config(path)

    assert config.agents.defaults.model == "test-model"
    assert config.agents.defaults.provider == "deepseek"
    dumped_defaults = config.model_dump(mode="json")["agents"]["defaults"]
    assert "base_profile_id" not in dumped_defaults
    assert "profile_id" not in dumped_defaults
    assert "max_handoffs" not in dumped_defaults

    cleaned_path = tmp_path / "cleaned.json"
    save_config(config, cleaned_path)
    cleaned_defaults = json.loads(cleaned_path.read_text(encoding="utf-8"))["agents"]["defaults"]
    assert "base_profile_id" not in cleaned_defaults
    assert "profile_id" not in cleaned_defaults
    assert "max_handoffs" not in cleaned_defaults


def test_model_routing_config_supports_master_and_expert_stage_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "routing": {
                        "master": "fast-model",
                        "experts": {
                            "builtin:research": "search-model",
                            "builtin:coding/review": "deepseek/deepseek-v4-pro",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.agents.routing.master == "fast-model"
    assert config.agents.routing.experts == {
        "builtin:research": "search-model",
        "builtin:coding/review": "deepseek/deepseek-v4-pro",
    }


def test_legacy_worker_routes_migrate_to_builtin_expert_stages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "routing": {
                        "workers": {
                            "builder": "build-model",
                            "reviewer": "review-model",
                            "unknown-custom-role": "discarded-model",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.agents.routing.experts == {
        "builtin:coding/build": "build-model",
        "builtin:coding/review": "review-model",
    }


def test_expert_limits_are_first_class_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "experts": {
                    "enabled": ["builtin:coding"],
                    "max_calls_per_turn": 12,
                    "stage_request_limit": 8,
                    "max_concurrency": 3,
                    "timeout_seconds": 42,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.experts.enabled == ["builtin:coding"]
    assert config.experts.max_calls_per_turn == 12
    assert config.experts.stage_request_limit == 8
    assert config.experts.max_concurrency == 3
    assert config.experts.timeout_seconds == 42


def test_skill_roots_are_explicit_and_normalized_against_workspace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "workspace": str(tmp_path / "repo"),
                "skills": {
                    "roots": [{"id": "team", "path": "../team-skills"}],
                    "master_allowlist": ["team:conventions"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.skills.roots[0].id == "team"
    assert config.skills.roots[0].path == (tmp_path / "team-skills").resolve()
    assert config.skills.master_allowlist == ["team:conventions"]


def test_removed_durable_settings_are_dropped_during_config_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "skills": {"enabled": True},
                "agents": {
                    "budgets": {"master": 12, "workers": {"reviewer": 30}},
                    "harness": {
                        "max_agent_calls": 12,
                        "sub_agent_request_limit": 7,
                        "dynamic_workflow_enabled": False,
                        "workflow_memory_mb": 512,
                    },
                    "templates": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    dumped = config.model_dump(mode="json")
    assert dumped["skills"] == {"roots": [], "master_allowlist": []}
    assert "budgets" not in dumped["agents"]
    assert "harness" not in dumped["agents"]
    assert "templates" not in dumped["agents"]
    assert dumped["experts"]["max_calls_per_turn"] == 12
    assert dumped["experts"]["stage_request_limit"] == 7


def test_config_set_writes_expert_model_and_runtime_limits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"

    main(["config", "set", "--config", str(path), "master-model", "fast-model"])
    main(["config", "set", "--config", str(path), "experts-max-calls-per-turn", "31"])
    main(["config", "set", "--config", str(path), "experts-max-concurrency", "3"])
    main(["config", "set", "--config", str(path), "experts-enabled", "builtin:coding"])
    main(
        [
            "config",
            "set",
            "--config",
            str(path),
            "coding-expert-model",
            "deepseek/deepseek-v4-pro",
        ]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["agents"]["routing"]["master"] == "fast-model"
    assert payload["agents"]["routing"]["experts"]["builtin:coding"] == ("deepseek/deepseek-v4-pro")
    assert payload["experts"]["max_calls_per_turn"] == 31
    assert payload["experts"]["max_concurrency"] == 3
    assert payload["experts"]["enabled"] == ["builtin:coding"]


def test_load_config_still_rejects_unknown_agent_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(
        path,
        {
            "base_profile_id": "base",
            "unexpected_setting": True,
        },
    )

    with pytest.raises(ValidationError, match="unexpected_setting"):
        load_config(path)


def test_removed_per_round_minimal_context_settings_are_not_persisted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    _write_config(
        path,
        {
            "uasm": {
                "minimal_context_recent_turns": 3,
                "minimal_context_tool_result_chars": 2_400,
                "tool_error_guard_threshold": 8,
                "budget_auto_continues": 9,
            }
        },
    )

    config = load_config(path)

    assert config.agents.defaults.runtime.stuck_detection_enabled is True
    assert config.agents.defaults.runtime.stuck_detection_threshold == 4
    cleaned_path = tmp_path / "cleaned.json"
    save_config(config, cleaned_path)
    cleaned_runtime = json.loads(cleaned_path.read_text(encoding="utf-8"))["agents"]["defaults"][
        "runtime"
    ]
    assert "minimal_context_recent_turns" not in cleaned_runtime
    assert "minimal_context_tool_result_chars" not in cleaned_runtime
    assert "tool_error_guard_threshold" not in cleaned_runtime
    assert "budget_auto_continues" not in cleaned_runtime


def test_legacy_compatibility_is_limited_to_persisted_config_loading() -> None:
    with pytest.raises(ValidationError, match="profile_id"):
        Config.model_validate({"agents": {"defaults": {"profile_id": "removed-profile"}}})


def test_non_object_config_uses_normal_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_deepseek_environment_selects_default_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELOON_CORE_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    config = load_config(tmp_path / "missing.json")

    assert config.agents.defaults.provider == "deepseek"
    assert config.providers.deepseek.api_key == "deepseek-test-key"
    assert config.agents.defaults.model == "deepseek-v4-pro"
    assert config.agents.defaults.model_ref() == "deepseek/deepseek-v4-pro"


def test_legacy_providers_active_migrates_to_defaults_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "providers": {"active": "deepseek"},
                "agents": {"defaults": {"model": "deepseek-v4-flash"}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.agents.defaults.provider == "deepseek"
    assert config.agents.defaults.model == "deepseek-v4-flash"
    cleaned_path = tmp_path / "cleaned.json"
    save_config(config, cleaned_path)
    cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
    assert "active" not in cleaned["providers"]
    assert cleaned["agents"]["defaults"]["provider"] == "deepseek"


def test_parse_model_ref_supports_bare_and_provider_prefixed_forms() -> None:
    assert parse_model_ref("deepseek-v4-flash", default_provider="deepseek") == (
        "deepseek",
        "deepseek-v4-flash",
    )
    assert parse_model_ref(
        "deepseek/deepseek-v4-pro",
        default_provider="deepseek",
    ) == ("deepseek", "deepseek-v4-pro")
    assert parse_model_ref(
        "org/custom-model",
        default_provider="deepseek",
    ) == ("deepseek", "org/custom-model")


def test_config_init_routes_credentials_to_deepseek_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELOON_CORE_PROVIDER", raising=False)
    path = tmp_path / "config.json"

    main(
        [
            "config",
            "init",
            "--config",
            str(path),
            "--provider",
            "deepseek",
            "--api-key",
            "deepseek-config-key",
            "--model",
            "deepseek-v4-flash",
        ]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "active" not in payload["providers"]
    assert payload["agents"]["defaults"]["provider"] == "deepseek"
    assert payload["providers"]["deepseek"]["api_key"] == "deepseek-config-key"
    assert "base_url" not in payload["providers"]["deepseek"]
    assert payload["agents"]["defaults"]["model"] == "deepseek-v4-flash"


def test_config_set_model_accepts_provider_model_ref(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    main(
        [
            "config",
            "init",
            "--config",
            str(path),
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
        ]
    )
    main(
        [
            "config",
            "set",
            "--config",
            str(path),
            "model",
            "deepseek/deepseek-v4-pro",
        ]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["agents"]["defaults"]["provider"] == "deepseek"
    assert payload["agents"]["defaults"]["model"] == "deepseek-v4-pro"
