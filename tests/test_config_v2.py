from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aeloon_core.__main__ import main
from aeloon_core.config import Config, load_config, save_config


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


def test_volcengine_environment_selects_agent_plan_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELOON_CORE_PROVIDER", "volcengine")
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    monkeypatch.setenv("ARK_MODEL", "ark-code-latest")

    config = load_config(tmp_path / "missing.json")

    assert config.providers.active == "volcengine"
    assert config.providers.volcengine.api_key == "ark-test-key"
    assert (
        config.providers.volcengine.base_url
        == "https://ark.cn-beijing.volces.com/api/plan/v3"
    )
    assert config.agents.defaults.model == "ark-code-latest"


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
    assert payload["providers"]["active"] == "volcengine"
    assert payload["providers"]["volcengine"]["api_key"] == "ark-config-key"
    assert payload["providers"]["anthropic"]["api_key"] == "no-key"
    assert payload["agents"]["defaults"]["model"] == "ark-code-latest"
