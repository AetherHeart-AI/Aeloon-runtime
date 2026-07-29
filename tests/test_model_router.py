"""Tests for process-scoped model routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from aeloon_core.config import AgentRoutingConfig, AgentsConfig, Config
from aeloon_core.harness.model import ModelRouter


@pytest.mark.asyncio
async def test_deepseek_routes_all_roles_to_default_model() -> None:
    config = Config()
    router = ModelRouter(config)

    master = router.resolve_master()
    explorer = router.resolve_worker("explorer")
    researcher = router.resolve_worker("researcher")
    builder = router.resolve_worker("builder")
    reviewer = router.resolve_worker("reviewer")
    custom = router.resolve_worker("custom")

    assert master.model_name == config.agents.defaults.model
    assert master.provider == "deepseek"
    assert explorer.model is master.model
    assert researcher.model is master.model
    assert builder.model is master.model
    assert reviewer.model is master.model
    assert custom.model is master.model
    await router.close()


@pytest.mark.asyncio
async def test_explicit_routes_override_provider_defaults() -> None:
    config = Config(
        agents=AgentsConfig(
            routing=AgentRoutingConfig(
                master="master-model",
                workers={"explorer": "search-model", "builder": "build-model"},
            )
        )
    )
    router = ModelRouter(config)

    assert router.resolve_master().model_name == "master-model"
    assert router.resolve_worker("explorer").model_name == "search-model"
    assert router.resolve_worker("builder").model_name == "build-model"
    await router.close()


@pytest.mark.asyncio
async def test_routing_profile_can_override_provider_and_model() -> None:
    config = Config(
        agents=AgentsConfig(
            routing=AgentRoutingConfig(
                master="deepseek/deepseek-v4-flash",
                workers={"builder": "deepseek/deepseek-v4-pro"},
            )
        )
    )
    router = ModelRouter(config)

    master = router.resolve_master()
    builder = router.resolve_worker("builder")
    assert master.provider == "deepseek"
    assert master.model_name == "deepseek-v4-flash"
    assert master.route == "override"
    assert builder.provider == "deepseek"
    assert builder.model_name == "deepseek-v4-pro"
    await router.close()


def test_injected_model_preserves_the_legacy_all_role_override() -> None:
    model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart("done")])
    )
    router = ModelRouter(Config(), injected_model=model, injected_settings={"temperature": 0})

    assert router.resolve_master().model is model
    assert router.resolve_worker("explorer").model is model
    assert router.resolve_worker("reviewer").route == "injected"
    assert router.resolve_worker("builder").settings["temperature"] == 0


def test_python_role_model_tier_is_used_when_no_route_override() -> None:
    config = Config()
    router = ModelRouter(config)

    assert (
        router.resolve_worker("custom", preferred_tier="fast").model_name
        == config.agents.defaults.model
    )
    assert (
        router.resolve_worker("explorer", preferred_tier="strong").model_name
        == config.agents.defaults.model
    )


@pytest.mark.asyncio
async def test_router_closes_each_reused_bundle_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    def build_bundle(**kwargs):
        model_name = kwargs["model_name"]
        model = FunctionModel(
            lambda _messages, _info: ModelResponse(parts=[TextPart("done")]),
            model_name=model_name,
        )

        async def close() -> None:
            closed.append(model_name)

        return SimpleNamespace(
            model=model,
            settings={},
            prompt_cache=None,
            close=close,
        )

    monkeypatch.setattr(
        "aeloon_core.harness.model.router.build_deepseek_model",
        build_bundle,
    )
    router = ModelRouter(Config())

    router.resolve_master()
    router.resolve_worker("explorer")
    router.resolve_worker("builder")
    router.resolve_worker("reviewer")
    await router.close()

    assert sorted(closed) == sorted(
        [Config().agents.defaults.model]
    )
