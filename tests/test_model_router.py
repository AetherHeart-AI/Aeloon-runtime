"""Tests for process-scoped model routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from aeloon_core.config import AgentRoutingConfig, AgentsConfig, Config
from aeloon_core.harness.model import ModelRouter


@pytest.mark.asyncio
async def test_deepseek_routes_master_and_expert_stages_to_default_model() -> None:
    config = Config()
    router = ModelRouter(config)

    master = router.resolve_master()
    research = router.resolve_expert("builtin:research", stage_id="reduce")
    coding = router.resolve_expert("builtin:coding", stage_id="build")
    custom = router.resolve_expert("workspace:custom")

    assert master.model_name == config.agents.defaults.model
    assert master.provider == "deepseek"
    assert research.model is master.model
    assert coding.model is master.model
    assert custom.model is master.model
    await router.close()


@pytest.mark.asyncio
async def test_explicit_routes_override_provider_defaults() -> None:
    config = Config(
        agents=AgentsConfig(
            routing=AgentRoutingConfig(
                master="master-model",
                experts={
                    "builtin:research": "search-model",
                    "builtin:coding/build": "build-model",
                },
            )
        )
    )
    router = ModelRouter(config)

    assert router.resolve_master().model_name == "master-model"
    assert router.resolve_expert("builtin:research", stage_id="docs").model_name == ("search-model")
    assert router.resolve_expert("builtin:coding", stage_id="build").model_name == ("build-model")
    await router.close()


@pytest.mark.asyncio
async def test_routing_profile_can_override_provider_and_model() -> None:
    config = Config(
        agents=AgentsConfig(
            routing=AgentRoutingConfig(
                master="deepseek/deepseek-v4-flash",
                experts={"builtin:coding": "deepseek/deepseek-v4-pro"},
            )
        )
    )
    router = ModelRouter(config)

    master = router.resolve_master()
    builder = router.resolve_expert("builtin:coding", stage_id="review")
    assert master.provider == "deepseek"
    assert master.model_name == "deepseek-v4-flash"
    assert master.route == "override"
    assert builder.provider == "deepseek"
    assert builder.model_name == "deepseek-v4-pro"
    await router.close()


def test_injected_model_applies_to_master_and_all_expert_stages() -> None:
    model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")]))
    router = ModelRouter(Config(), injected_model=model, injected_settings={"temperature": 0})

    assert router.resolve_master().model is model
    assert router.resolve_expert("builtin:research").model is model
    assert router.resolve_expert("builtin:coding", stage_id="review").route == "injected"
    assert router.resolve_expert("workspace:custom").settings["temperature"] == 0


def test_expert_model_tier_uses_default_when_no_route_override() -> None:
    config = Config()
    router = ModelRouter(config)

    fast = router.resolve_expert("workspace:custom", preferred_tier="fast")
    strong = router.resolve_expert("builtin:research", preferred_tier="strong")

    assert fast.model_name == config.agents.defaults.model
    assert fast.route == "fast"
    assert strong.model_name == config.agents.defaults.model
    assert strong.route == "strong"


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
    router.resolve_expert("builtin:research", stage_id="plan")
    router.resolve_expert("builtin:coding", stage_id="build")
    router.resolve_expert("builtin:coding", stage_id="review")
    await router.close()

    assert sorted(closed) == sorted([Config().agents.defaults.model])
