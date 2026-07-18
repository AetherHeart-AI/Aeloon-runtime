from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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
