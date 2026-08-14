from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from aeloon_core.config import (
    Config,
    CustomProviderConfig,
    ProviderModelConfig,
    save_config,
)
from aeloon_core.core import InferenceContext, Model, StreamOptions, UserMessage
from aeloon_core.core.inference_runtime import collect_assistant
from aeloon_core.rpc import AeloonRpcAdapter
from aeloon_core.runtime import (
    ProviderManager,
    RuntimeService,
    normalize_model_id,
    qualify_model_id,
    resolve_model_id,
    split_model_id,
)
from aeloon_core.runtime.providers import BaseProvider, CustomProvider, model_from_config


class TrackingProvider(BaseProvider):
    def __init__(
        self,
        provider_id: str,
        *,
        models: tuple[Model, ...] = (),
        authenticated: bool | None = None,
        fail_models: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            name=provider_id,
            endpoint="https://provider.example/v1",
            enabled=enabled,
        )
        self._models = {model.id: model for model in models}
        self.authenticated = authenticated
        self.fail_models = fail_models
        self.close_calls = 0

    async def models(self):
        if self.fail_models:
            raise RuntimeError("temporary discovery failure")
        return dict(self._models)

    def stream(self, _model, _context, _options):
        async def empty():
            if False:
                yield None

        return empty()

    def status(self):
        return {**super().status(), "authenticated": self.authenticated}

    async def close(self):
        self.close_calls += 1


def _provider_config(
    provider_id: str,
    *,
    enabled: bool = True,
) -> CustomProviderConfig:
    return CustomProviderConfig(
        backend="openai",
        name=provider_id,
        endpoint=f"https://{provider_id}.example/v1",
        enabled=enabled,
        models=[ProviderModelConfig(id="model")],
    )


def _provider_only_config(**providers) -> Config:
    defaults = Config().providers
    return Config(
        providers={
            "deepseek": defaults["deepseek"].model_copy(update={"enabled": False}),
            "aeloon-cloud": defaults["aeloon-cloud"].model_copy(update={"enabled": False}),
            **providers,
        }
    )


def test_model_ids_use_one_provider_prefix_without_legacy_upgrade() -> None:
    available = ["deepseek/deepseek-v4-pro", "studio/coder"]
    assert qualify_model_id("ollama", "llama/3.3") == "ollama/llama/3.3"
    assert qualify_model_id("ollama", "ollama/llama/3.3") == "ollama/llama/3.3"
    assert split_model_id("ollama/llama/3.3") == ("ollama", "llama/3.3")
    assert normalize_model_id("coder", available) == "studio/coder"
    with pytest.raises(KeyError, match="Unknown model"):
        normalize_model_id("unqualified-model", available)


def test_unqualified_model_id_uses_first_matching_provider() -> None:
    available = ["studio/coder", "backup/coder", "studio/org/model"]
    assert resolve_model_id("coder", available) == "studio/coder"
    assert resolve_model_id("org/model", available) == "studio/org/model"
    assert resolve_model_id("backup/coder", available) == "backup/coder"


@pytest.mark.asyncio
async def test_manager_resolves_unqualified_model_in_configuration_order() -> None:
    studio = CustomProviderConfig(
        backend="openai",
        name="Studio",
        endpoint="http://127.0.0.1:8000/v1",
        models=[ProviderModelConfig(id="coder")],
    )
    backup = CustomProviderConfig(
        backend="openai",
        name="Backup",
        endpoint="http://127.0.0.1:9000/v1",
        models=[ProviderModelConfig(id="coder")],
    )
    config = Config(
        providers={
            **Config().providers,
            "studio": studio,
            "backup": backup,
        }
    ).normalized()
    manager = ProviderManager(config)
    try:
        assert (await manager.model("coder")).id == "studio/coder"
        assert (await manager.model("backup/coder")).id == "backup/coder"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_is_lazy_isolates_failures_and_closes_idempotently() -> None:
    created: dict[str, TrackingProvider] = {}
    calls: list[str] = []

    def factory(provider_id: str, _configured: Any, _account: Any) -> BaseProvider:
        calls.append(provider_id)
        provider = TrackingProvider(
            provider_id,
            models=(Model(f"{provider_id}/model", "model", provider_id),),
            fail_models=provider_id == "broken",
        )
        created[provider_id] = provider
        return provider

    manager = ProviderManager(
        _provider_only_config(
            broken=_provider_config("broken"),
            studio=_provider_config("studio"),
        ),
        driver_factories={"custom": factory},
    )
    assert calls == []

    models = await manager.models()

    assert list(models) == ["studio/model"]
    assert calls == ["broken", "studio"]
    selected = await manager.model("studio/model")
    assert manager.inference(selected) is created["studio"]
    assert calls == ["broken", "studio"]

    await manager.close()
    await manager.close()
    assert created["broken"].close_calls == 1
    assert created["studio"].close_calls == 1


@pytest.mark.asyncio
async def test_manager_lazily_owns_and_closes_snapshot_account_gateway() -> None:
    accounts = []

    class Account:
        close_calls = 0

        def status(self):
            return {"authenticated": False, "user": None}

        async def models(self):
            return []

        async def close(self):
            self.close_calls += 1

    def account_factory():
        value = Account()
        accounts.append(value)
        return value

    defaults = Config().providers
    config = Config(
        providers={
            "deepseek": defaults["deepseek"].model_copy(update={"enabled": False}),
            "aeloon-cloud": defaults["aeloon-cloud"],
        }
    )
    manager = ProviderManager(
        config,
        account_factory=account_factory,
        close_account=True,
    )
    assert accounts == []

    await manager.providers()

    assert len(accounts) == 1
    await manager.close()
    await manager.close()
    assert accounts[0].close_calls == 1


@pytest.mark.asyncio
async def test_manager_rejects_disabled_and_unauthenticated_providers() -> None:
    providers = {
        "disabled": TrackingProvider(
            "disabled",
            models=(Model("disabled/model", "model", "disabled"),),
        ),
        "locked": TrackingProvider(
            "locked",
            models=(Model("locked/model", "model", "locked"),),
            authenticated=False,
        ),
    }

    def factory(provider_id: str, configured: Any, _account: Any) -> BaseProvider:
        providers[provider_id].enabled = configured.enabled
        return providers[provider_id]

    manager = ProviderManager(
        _provider_only_config(
            disabled=_provider_config("disabled", enabled=False),
            locked=_provider_config("locked"),
        ),
        driver_factories={"custom": factory},
    )
    try:
        with pytest.raises(RuntimeError, match="disabled"):
            await manager.model("disabled/model")
        with pytest.raises(PermissionError, match="Authenticate"):
            await manager.model("locked/model")
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_custom_provider_routes_qualified_model_to_unprefixed_api_model() -> None:
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
    model_config = ProviderModelConfig(id="llama/3.3")
    config = CustomProviderConfig(
        backend="ollama",
        name="Custom",
        endpoint="http://127.0.0.1:11434/v1",
        models=[model_config],
    )
    model = ProviderManager(Config(providers={**Config().providers, "ollama": config}))
    selected = await model.model("ollama/llama/3.3")
    provider = CustomProvider(
        provider_id="ollama",
        name="Ollama",
        endpoint="http://127.0.0.1:11434/v1",
        models=(selected,),
        client=client,
    )
    message = await collect_assistant(
        provider,
        selected,
        InferenceContext("", (UserMessage("hello"),), (), "session"),
        StreamOptions(max_retries=0),
    )
    await model.close()
    await client.aclose()

    assert message.text == "local"
    assert json.loads(requests[0].content)["model"] == "llama/3.3"
    assert "authorization" not in requests[0].headers


@pytest.mark.asyncio
async def test_rpc_adds_and_removes_provider_without_exposing_secret(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "coder"},
                        {"id": "vision", "supportsImage": True},
                    ]
                },
            )
        assert request.headers["authorization"] == "Bearer local-secret"
        assert json.loads(request.content)["model"] == "coder"
        return httpx.Response(400, json={"error": {"message": "images are unsupported"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def factory(config: Config) -> ProviderManager:
        def custom(provider_id: str, configured: Any, _account: Any):
            return CustomProvider(
                provider_id=provider_id,
                name=configured.name,
                endpoint=configured.endpoint,
                models=tuple(
                    model_from_config(provider_id, model) for model in configured.models
                ),
                api_key=configured.api_key,
                proxy=configured.proxy,
                headers=configured.headers,
                client=client,
            )

        return ProviderManager(config, driver_factories={"custom": custom})

    runtime = RuntimeService(config_path=config_path, provider_manager_factory=factory)
    service = AeloonRpcAdapter(runtime)

    added = await service.dispatch(
        "provider.add",
        {
            "provider_id": "studio",
            "driver": "custom",
            "backend": "openai",
            "name": "Studio API",
            "endpoint": "http://127.0.0.1:9000/v1/",
            "api_key": "local-secret",
            "models": ["coder", "vision"],
            "max_output_tokens": 4_096,
        },
    )
    catalog = await service.dispatch("catalog.get")
    settings = await service.dispatch("settings.get")

    assert added["provider"]["driver"] == "custom"
    assert added["provider"]["model_ids"] == ["studio/coder", "studio/vision"]
    assert "local-secret" not in json.dumps(added)
    catalog_models = {model["id"]: model for model in catalog["models"]}
    assert catalog_models["studio/coder"]["supports_image"] is False
    assert catalog_models["studio/vision"]["supports_image"] is True
    assert settings["providers"]["studio"]["credential_configured"] is True
    assert "local-secret" not in json.dumps(settings)
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["providers"]["studio"]["api_key"] == "local-secret"
    assert persisted["providers"]["studio"]["driver"] == "custom"
    assert {
        model["max_output_tokens"]
        for model in persisted["providers"]["studio"]["models"]
    } == {4_096}
    assert [request.url.path for request in requests] == [
        "/v1/models",
        "/v1/chat/completions",
    ]

    removed = await service.dispatch("provider.remove", {"provider_id": "studio"})
    assert removed["removed"] is True
    await service.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_rpc_discovers_models_and_persists_resolved_v1_endpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/models":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "discovered",
                        "name": "Discovered Model",
                        "context_window": 64_000,
                        "supports_image": False,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def factory(config: Config) -> ProviderManager:
        def custom(provider_id: str, configured: Any, _account: Any):
            return CustomProvider(
                provider_id=provider_id,
                name=configured.name,
                endpoint=configured.endpoint,
                models=tuple(
                    model_from_config(provider_id, model) for model in configured.models
                ),
                client=client,
            )

        return ProviderManager(config, driver_factories={"custom": custom})

    runtime = RuntimeService(
        config_path=config_path,
        provider_manager_factory=factory,
    )
    service = AeloonRpcAdapter(runtime)
    added = await service.dispatch(
        "provider.add",
        {"provider_id": "desktop", "endpoint": "http://127.0.0.1:11434"},
    )

    assert added["provider"]["model_ids"] == ["desktop/discovered"]
    assert added["provider"]["endpoint"] == "http://127.0.0.1:11434/v1"
    assert [request.url.path for request in requests] == ["/models", "/v1/models"]
    await service.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_rpc_refreshes_provider_models_without_using_configured_cache(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config = Config(
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        providers={
            **Config().providers,
            "studio": CustomProviderConfig(
                backend="openai",
                name="Studio",
                endpoint="http://127.0.0.1:9000/v1",
                models=[
                    ProviderModelConfig(id="fresh", max_output_tokens=1_234),
                    ProviderModelConfig(id="cached"),
                ],
            ),
        },
    )
    save_config(config, config_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "fresh",
                        "name": "Fresh Model",
                        "supports_image": False,
                        "context_window": 96_000,
                    },
                    {
                        "id": "new",
                        "name": "New Model",
                        "supports_image": False,
                        "context_window": 4_096,
                    },
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def factory(snapshot: Config) -> ProviderManager:
        def custom(provider_id: str, configured: Any, _account: Any):
            return CustomProvider(
                provider_id=provider_id,
                name=configured.name,
                endpoint=configured.endpoint,
                models=tuple(
                    model_from_config(provider_id, model) for model in configured.models
                ),
                client=client,
            )

        return ProviderManager(snapshot, driver_factories={"custom": custom})

    runtime = RuntimeService(config_path=config_path, provider_manager_factory=factory)
    service = AeloonRpcAdapter(runtime)
    refreshed = await service.dispatch(
        "provider.refresh",
        {"provider_id": "studio", "force": True, "revision": 1},
    )

    assert refreshed["provider"]["model_ids"] == ["studio/fresh", "studio/new"]
    assert refreshed["revision"] == 2
    assert [request.url.path for request in requests] == ["/v1/models"]
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    persisted_models = {
        model["id"]: model for model in persisted["providers"]["studio"]["models"]
    }
    assert list(persisted_models) == ["fresh", "new"]
    assert persisted_models["fresh"]["max_output_tokens"] == 1_234
    assert persisted_models["new"]["max_output_tokens"] == 1_024
    catalog = await service.dispatch("catalog.get")
    assert any(model["id"] == "studio/fresh" for model in catalog["models"])
    assert any(model["id"] == "studio/new" for model in catalog["models"])
    assert all(model["id"] != "studio/cached" for model in catalog["models"])
    await service.close()
    await client.aclose()
