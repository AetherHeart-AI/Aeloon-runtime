from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from aeloon_core.config import (
    Config,
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
        "driver": "custom",
        "backend": "ollama",
        "name": "Not DeepSeek",
        "endpoint": "http://127.0.0.1:11434/v1",
    }
    with pytest.raises(ValidationError, match="must use driver deepseek"):
        Config.model_validate(raw)


def test_removed_provider_config_shapes_are_rejected(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"deepseek": {"api_key": "old"}}), encoding="utf-8")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(path)


def test_current_custom_provider_shape_round_trips(tmp_path) -> None:
    path = tmp_path / "config.json"
    raw = Config().model_dump(mode="json")
    raw["providers"].update(
        {
            "desktop": {
                "driver": "custom",
                "backend": "ollama",
                "name": "Ollama",
                "endpoint": "http://127.0.0.1:11434/v1",
            },
            "studio": {
                "driver": "custom",
                "backend": "openai",
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


def test_custom_provider_requires_backend() -> None:
    raw = Config().model_dump(mode="json")
    raw["providers"]["ollama"] = {
        "driver": "custom",
        "name": "Ollama",
        "endpoint": "http://127.0.0.1:11434/v1",
    }
    with pytest.raises(ValidationError, match="backend"):
        Config.model_validate(raw)


def test_removed_provider_model_max_tokens_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.json"
    raw = Config().model_dump(mode="json")
    raw["providers"]["studio"] = {
        "driver": "custom",
        "backend": "openai",
        "name": "Studio",
        "endpoint": "https://studio.example/v1",
        "models": [
            {
                "id": "removed-field",
                "context_window": 8_192,
                "max_tokens": 8_192,
            }
        ],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="max_tokens"):
        load_config(path)


def test_public_config_redacts_every_provider_secret_and_sensitive_header() -> None:
    deepseek = DeepSeekProviderConfig(
        api_key="deepseek-secret",
        headers={"Proxy-Authorization": "proxy-secret", "X-Public": "visible"},
    )
    studio = CustomProviderConfig(
        backend="openai",
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
