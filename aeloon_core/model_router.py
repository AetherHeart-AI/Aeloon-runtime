"""Process-scoped model routing for Master and Worker responsibilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from aeloon_core.config import Config
from aeloon_core.pydantic_model import (
    PromptCacheState,
    PydanticModelBundle,
    build_anthropic_model,
    build_volcengine_model,
)

FAST_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OFFICIAL_ANTHROPIC_URLS = {
    "https://api.anthropic.com",
    "https://api.anthropic.com/v1",
}
_FAST_WORKER_TYPES = frozenset({"explorer", "researcher"})


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """One resolved model and its process-owned provider state."""

    model_name: str
    route: Literal["fast", "strong", "override", "injected"]
    model: Model
    settings: ModelSettings
    prompt_cache: PromptCacheState | None
    bundle: PydanticModelBundle | None = None


class ModelRouter:
    """Resolve roles to models and reuse one bundle per provider/model pair."""

    def __init__(
        self,
        config: Config,
        *,
        injected_model: Model | None = None,
        injected_settings: ModelSettings | None = None,
    ) -> None:
        self.config = config
        self._bundles: dict[tuple[str, str], PydanticModelBundle] = {}
        self._injected_model = injected_model
        self._injected_settings = dict(injected_settings or {})

    def set_injected_model(
        self,
        model: Model,
        *,
        settings: ModelSettings | None = None,
    ) -> None:
        """Override every route, preserving the legacy embedding/test surface."""

        self._injected_model = model
        if settings is not None:
            self._injected_settings = dict(settings)

    def resolve_master(self) -> ModelBinding:
        override = self.config.agents.routing.master
        return self._resolve(
            model_name=override or self._default_fast_model(),
            route="override" if override else self._default_fast_route(),
        )

    def resolve_worker(self, worker_type_id: str) -> ModelBinding:
        override = self.config.agents.routing.workers.get(worker_type_id)
        if override is not None:
            return self._resolve(model_name=override, route="override")
        if worker_type_id in _FAST_WORKER_TYPES:
            return self._resolve(
                model_name=self._default_fast_model(),
                route=self._default_fast_route(),
            )
        return self._resolve(
            model_name=self.config.agents.defaults.model,
            route="strong",
        )

    def resolved_model_name(
        self,
        *,
        role: Literal["master", "worker"],
        worker_type_id: str | None = None,
    ) -> str:
        binding = (
            self.resolve_master()
            if role == "master"
            else self.resolve_worker(worker_type_id or "")
        )
        return binding.model_name

    async def close(self) -> None:
        bundles = tuple(self._bundles.values())
        self._bundles.clear()
        if bundles:
            await asyncio.gather(*(bundle.close() for bundle in bundles))

    def _resolve(
        self,
        *,
        model_name: str,
        route: Literal["fast", "strong", "override"],
    ) -> ModelBinding:
        if self._injected_model is not None:
            return ModelBinding(
                model_name=str(
                    getattr(self._injected_model, "model_name", None) or "<injected>"
                ),
                route="injected",
                model=self._injected_model,
                settings=dict(self._injected_settings),
                prompt_cache=None,
            )
        provider_name = self.config.providers.active
        key = (provider_name, model_name)
        bundle = self._bundles.get(key)
        if bundle is None:
            defaults = self.config.agents.defaults
            if provider_name == "volcengine":
                bundle = build_volcengine_model(
                    provider=self.config.providers.volcengine,
                    model_name=model_name,
                    temperature=defaults.temperature,
                    reasoning_effort=defaults.reasoning_effort,
                    timeout=defaults.chat_timeout,
                )
            else:
                bundle = build_anthropic_model(
                    provider=self.config.providers.anthropic,
                    model_name=model_name,
                    temperature=defaults.temperature,
                    reasoning_effort=defaults.reasoning_effort,
                    timeout=defaults.chat_timeout,
                )
            self._bundles[key] = bundle
        return ModelBinding(
            model_name=model_name,
            route=route,
            model=bundle.model,
            settings=dict(bundle.settings),
            prompt_cache=bundle.prompt_cache,
            bundle=bundle,
        )

    def _default_fast_model(self) -> str:
        if self._uses_official_anthropic():
            return FAST_ANTHROPIC_MODEL
        return self.config.agents.defaults.model

    def _default_fast_route(self) -> Literal["fast", "strong"]:
        return "fast" if self._uses_official_anthropic() else "strong"

    def _uses_official_anthropic(self) -> bool:
        return (
            self.config.providers.active == "anthropic"
            and self.config.providers.anthropic.base_url.rstrip("/")
            in {url.rstrip("/") for url in _OFFICIAL_ANTHROPIC_URLS}
        )


__all__ = [
    "FAST_ANTHROPIC_MODEL",
    "ModelBinding",
    "ModelRouter",
]
