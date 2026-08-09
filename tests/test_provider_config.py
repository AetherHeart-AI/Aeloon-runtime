from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from aeloon_core.config import (
    Config,
    ConfigMigrationError,
    CustomProviderConfig,
    DeepSeekProviderConfig,
    load_config,
    public_config,
    save_config,
)


def test_provider_config_requires_reserved_ids_and_exclusive_drivers() -> None:
    with pytest.raises(ValidationError, match="Reserved Provider is required: deepseek"):
        Config(providers={})

    with pytest.raises(ValidationError, match="reserved for id deepseek"):
        Config(
            providers={
                **Config().providers,
                "renamed-deepseek": DeepSeekProviderConfig(),
            }
        )

    raw = Config().model_dump(mode="json")
    raw["providers"]["deepseek"] = {
        "driver": "ollama",
        "name": "Not DeepSeek",
    }
    with pytest.raises(ValidationError, match="must use driver deepseek"):
        Config.model_validate(raw)


def test_legacy_provider_config_and_no_key_sentinel_are_rejected(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"deepseek": {"api_key": "old"}}), encoding="utf-8")

    with pytest.raises(ConfigMigrationError, match="deepseek.*MIGRATION.md"):
        load_config(path)
    with pytest.raises(ValidationError, match="no-key"):
        DeepSeekProviderConfig(api_key="no-key")


def test_current_custom_driver_names_are_normalized_and_rewritten(tmp_path) -> None:
    path = tmp_path / "config.json"
    raw = Config().model_dump(mode="json")
    raw["providers"].update(
        {
            "desktop": {"driver": "ollama"},
            "studio": {
                "driver": "openai-compatible",
                "name": "Studio",
                "endpoint": "https://studio.example/v1",
                "api_key": "studio-secret",
                "models": [{"id": "vision", "supports_image": True}],
            },
        }
    )
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config(path)

    assert isinstance(config.providers["desktop"], CustomProviderConfig)
    assert config.providers["desktop"].endpoint == "http://127.0.0.1:11434/v1"
    assert config.providers["desktop"].backend == "ollama"
    assert isinstance(config.providers["studio"], CustomProviderConfig)
    assert config.providers["studio"].backend == "openai"
    assert config.providers["studio"].api_key == "studio-secret"
    assert config.providers["studio"].models[0].supports_image is True

    save_config(config, path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["providers"]["desktop"]["driver"] == "custom"
    assert persisted["providers"]["studio"]["driver"] == "custom"
    assert persisted["providers"]["desktop"]["backend"] == "ollama"


def test_custom_provider_without_backend_is_inferred_for_existing_configs() -> None:
    raw = Config().model_dump(mode="json")
    raw["providers"]["ollama"] = {
        "driver": "custom",
        "name": "Ollama",
        "endpoint": "http://127.0.0.1:11434/v1",
    }
    raw["providers"]["studio"] = {
        "driver": "custom",
        "name": "Studio",
        "endpoint": "https://studio.example/v1",
    }

    config = Config.model_validate(raw)

    assert config.providers["ollama"].backend == "ollama"
    assert config.providers["studio"].backend == "openai"


def test_public_config_redacts_every_provider_secret_and_sensitive_header() -> None:
    deepseek = DeepSeekProviderConfig(
        api_key="deepseek-secret",
        headers={"Proxy-Authorization": "proxy-secret", "X-Public": "visible"},
    )
    studio = CustomProviderConfig(
        name="Studio",
        endpoint="https://studio.example/v1",
        api_key="studio-secret",
        headers={"Cookie": "cookie-secret", "X-Trace": "trace-id"},
    )
    config = Config(
        providers={
            **Config().providers,
            "deepseek": deepseek,
            "studio": studio,
        }
    )

    public = public_config(config)

    assert public["providers"]["deepseek"]["api_key"] == "***"
    assert public["providers"]["studio"]["api_key"] == "***"
    assert public["providers"]["deepseek"]["headers"] == {
        "Proxy-Authorization": "***",
        "X-Public": "visible",
    }
    assert public["providers"]["studio"]["headers"] == {
        "Cookie": "***",
        "X-Trace": "trace-id",
    }
