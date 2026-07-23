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
    assert config.agents.defaults.provider == "anthropic"
    dumped_defaults = config.model_dump(mode="json")["agents"]["defaults"]
    assert "base_profile_id" not in dumped_defaults
    assert "profile_id" not in dumped_defaults
    assert "max_handoffs" not in dumped_defaults

    cleaned_path = tmp_path / "cleaned.json"
    save_config(config, cleaned_path)
    cleaned_defaults = json.loads(cleaned_path.read_text(encoding="utf-8"))["agents"][
        "defaults"
    ]
    assert "base_profile_id" not in cleaned_defaults
    assert "profile_id" not in cleaned_defaults
    assert "max_handoffs" not in cleaned_defaults


def test_model_routing_config_supports_master_and_worker_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "routing": {
                        "master": "fast-model",
                        "workers": {
                            "explorer": "search-model",
                            "reviewer": "volcengine/strong-model",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.agents.routing.master == "fast-model"
    assert config.agents.routing.workers == {
        "explorer": "search-model",
        "reviewer": "volcengine/strong-model",
    }


def test_request_budgets_support_master_and_worker_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "budgets": {
                        "master": 12,
                        "workers": {"explorer": 8, "reviewer": 30},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.agents.budgets.master == 12
    assert config.agents.budgets.workers == {"explorer": 8, "reviewer": 30}


def test_config_set_writes_role_specific_model_and_budget_routes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"

    main(["config", "set", "--config", str(path), "master-model", "fast-model"])
    main(["config", "set", "--config", str(path), "reviewer-max-iterations", "31"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["agents"]["routing"]["master"] == "fast-model"
    assert payload["agents"]["budgets"]["workers"]["reviewer"] == 31


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
    cleaned_runtime = json.loads(cleaned_path.read_text(encoding="utf-8"))["agents"][
        "defaults"
    ]["runtime"]
    assert "minimal_context_recent_turns" not in cleaned_runtime
    assert "minimal_context_tool_result_chars" not in cleaned_runtime
    assert "tool_error_guard_threshold" not in cleaned_runtime
    assert "budget_auto_continues" not in cleaned_runtime


def test_legacy_compatibility_is_limited_to_persisted_config_loading() -> None:
    with pytest.raises(ValidationError, match="profile_id"):
        Config.model_validate(
            {"agents": {"defaults": {"profile_id": "removed-profile"}}}
        )


def test_non_object_config_uses_normal_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_volcengine_environment_selects_default_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELOON_CORE_PROVIDER", "volcengine")
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    monkeypatch.setenv("ARK_MODEL", "ark-code-latest")

    config = load_config(tmp_path / "missing.json")

    assert "active" not in config.model_dump(mode="json")["providers"]
    assert config.agents.defaults.provider == "volcengine"
    assert config.providers.volcengine.api_key == "ark-test-key"
    assert (
        config.providers.volcengine.base_url
        == "https://ark.cn-beijing.volces.com/api/plan/v3"
    )
    assert config.agents.defaults.model == "ark-code-latest"
    assert config.agents.defaults.model_ref() == "volcengine/ark-code-latest"


def test_legacy_providers_active_migrates_to_defaults_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "providers": {"active": "volcengine"},
                "agents": {"defaults": {"model": "ark-code-latest"}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.agents.defaults.provider == "volcengine"
    assert config.agents.defaults.model == "ark-code-latest"
    cleaned_path = tmp_path / "cleaned.json"
    save_config(config, cleaned_path)
    cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
    assert "active" not in cleaned["providers"]
    assert cleaned["agents"]["defaults"]["provider"] == "volcengine"


def test_parse_model_ref_supports_bare_and_provider_prefixed_forms() -> None:
    assert parse_model_ref("deepseek-v4-pro", default_provider="anthropic") == (
        "anthropic",
        "deepseek-v4-pro",
    )
    assert parse_model_ref(
        "volcengine/ark-code-latest",
        default_provider="anthropic",
    ) == ("volcengine", "ark-code-latest")
    assert parse_model_ref(
        "org/custom-model",
        default_provider="anthropic",
    ) == ("anthropic", "org/custom-model")


def test_config_init_routes_credentials_to_selected_provider(
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
            "volcengine",
            "--api-key",
            "ark-config-key",
            "--model",
            "ark-code-latest",
        ]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "active" not in payload["providers"]
    assert payload["agents"]["defaults"]["provider"] == "volcengine"
    assert payload["providers"]["volcengine"]["api_key"] == "ark-config-key"
    assert payload["providers"]["anthropic"]["api_key"] == "no-key"
    assert payload["agents"]["defaults"]["model"] == "ark-code-latest"


def test_config_set_model_accepts_provider_model_ref(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    main(
        [
            "config",
            "init",
            "--config",
            str(path),
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
        ]
    )
    main(
        [
            "config",
            "set",
            "--config",
            str(path),
            "model",
            "volcengine/ark-code-latest",
        ]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["agents"]["defaults"]["provider"] == "volcengine"
    assert payload["agents"]["defaults"]["model"] == "ark-code-latest"
