from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from aeloon_core.config import Config, LocalModelConfig, LocalProviderConfig, save_config
from aeloon_core.harness import DeepSeekProvider, ProviderContext, StreamOptions, UserMessage
from aeloon_core.harness.providers import collect_assistant
from aeloon_core.providers import (
    UnifiedProviderRegistry,
    normalize_model_id,
    qualify_model_id,
    resolve_model_id,
    split_model_id,
)
from aeloon_core.service import CoreService


class OfflineCloudAccount:
    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "authenticated": False,
            "user": None,
            "base_url": "https://cloud.example",
        }


def test_model_ids_use_one_provider_prefix_and_upgrade_legacy_deepseek() -> None:
    assert qualify_model_id("ollama", "llama/3.3") == "ollama/llama/3.3"
    assert qualify_model_id("ollama", "ollama/llama/3.3") == "ollama/llama/3.3"
    assert split_model_id("ollama/llama/3.3") == ("ollama", "llama/3.3")
    assert normalize_model_id("deepseek-v4-pro") == "deepseek/deepseek-v4-pro"
    with pytest.raises(ValueError, match="provider/model"):
        normalize_model_id("unqualified-model")


def test_unqualified_model_id_uses_first_matching_provider() -> None:
    available = ["studio/coder", "backup/coder", "studio/org/model"]

    assert resolve_model_id("coder", available) == "studio/coder"
    assert resolve_model_id("org/model", available) == "studio/org/model"
    assert resolve_model_id("backup/coder", available) == "backup/coder"


@pytest.mark.asyncio
async def test_registry_resolves_unqualified_model_to_first_provider() -> None:
    config = Config(
        local_providers={
            "studio": LocalProviderConfig(
                name="Studio",
                base_url="http://127.0.0.1:8000/v1",
                models=[
                    LocalModelConfig(id="coder"),
                    LocalModelConfig(id="deepseek-v4-flash"),
                ],
            ),
            "backup": LocalProviderConfig(
                name="Backup",
                base_url="http://127.0.0.1:9000/v1",
                models=[
                    LocalModelConfig(id="coder"),
                    LocalModelConfig(id="deepseek-v4-flash"),
                ],
            ),
        }
    ).normalized()
    registry = UnifiedProviderRegistry(
        config,
        OfflineCloudAccount(),  # type: ignore[arg-type]
    )

    assert (await registry.model("coder")).id == "studio/coder"
    assert (
        await registry.model("deepseek-v4-flash")
    ).id == "studio/deepseek-v4-flash"
    assert (await registry.model("backup/coder")).id == "backup/coder"


@pytest.mark.asyncio
async def test_local_provider_routes_qualified_model_to_unprefixed_api_model() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"local"},'
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = Config(
        local_providers={
            "ollama": LocalProviderConfig(
                name="Ollama",
                base_url="http://127.0.0.1:11434/v1",
                models=[LocalModelConfig(id="llama/3.3")],
            )
        }
    ).normalized()
    registry = UnifiedProviderRegistry(
        config,
        OfflineCloudAccount(),  # type: ignore[arg-type]
        local_provider_factory=lambda **kwargs: DeepSeekProvider(client=client, **kwargs),
    )

    model = await registry.model("ollama/llama/3.3")
    message = await collect_assistant(
        registry.provider(model),
        model,
        ProviderContext("", (UserMessage("hello"),), (), "session"),
        StreamOptions(max_retries=0),
    )
    await client.aclose()

    assert message.text == "local"
    assert model.id == "ollama/llama/3.3"
    assert json.loads(requests[0].content)["model"] == "llama/3.3"
    assert "authorization" not in requests[0].headers


@pytest.mark.asyncio
async def test_bridge_adds_and_removes_local_provider_without_exposing_secret(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    service = CoreService(config_path=config_path)

    added = await service.dispatch(
        "provider.local.add",
        {
            "provider_id": "studio",
            "name": "Studio API",
            "base_url": "http://127.0.0.1:9000/v1/",
            "api_key": "local-secret",
            "models": ["coder", {"id": "vision", "supports_image": True}],
        },
    )
    catalog = await service.dispatch("catalog.get")
    settings = await service.dispatch("settings.get")
    listed = await service.dispatch("provider.list")

    assert added["provider"]["id"] == "studio"
    assert added["provider"]["model_ids"] == ["studio/coder", "studio/vision"]
    assert "local-secret" not in json.dumps(added)
    assert {model["id"] for model in catalog["models"]} >= {
        "deepseek/deepseek-v4-flash",
        "studio/coder",
        "studio/vision",
    }
    assert {provider["id"] for provider in listed["providers"]} >= {
        "deepseek",
        "studio",
        "aeloon-cloud",
    }
    assert settings["local_providers"]["studio"]["credential_configured"] is True
    assert "local-secret" not in json.dumps(settings)
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["local_providers"]["studio"]["api_key"] == "local-secret"

    removed = await service.dispatch("provider.local.remove", {"provider_id": "studio"})
    assert removed["removed"] is True
    assert all(
        model["provider_id"] != "studio"
        for model in (await service.dispatch("catalog.get"))["models"]
    )
    await service.close()


@pytest.mark.asyncio
async def test_bridge_discovers_models_when_local_provider_omits_model_list(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    service = CoreService(config_path=config_path)
    calls: list[dict[str, Any]] = []

    async def discover(base_url: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"base_url": base_url, **kwargs})
        return [{"id": "discovered", "name": "Discovered Model", "context_window": 64_000}]

    service._discover_local_models = discover  # type: ignore[method-assign]
    added = await service.dispatch(
        "provider.local.add",
        {
            "provider_id": "desktop",
            "base_url": "http://127.0.0.1:8080/v1/",
            "api_key": "no-key",
        },
    )

    assert added["provider"]["model_ids"] == ["desktop/discovered"]
    assert calls == [
        {
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "no-key",
            "proxy": None,
            "extra_headers": {},
        }
    ]
    await service.close()
