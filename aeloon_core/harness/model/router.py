"""Process-scoped model routing for Master and Worker responsibilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from aeloon_core.config import (
    KNOWN_PROVIDERS,
    Config,
    ProviderName,
    format_model_ref,
    parse_model_ref,
)
from aeloon_core.harness.provider import (
    PromptCacheState,
    PydanticModelBundle,
    build_deepseek_model,
)

RouteKind = Literal["fast", "strong", "override", "fallback", "injected"]


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """One resolved model and its process-owned provider state."""

    provider: ProviderName | Literal["injected"]
    model_name: str
    route: RouteKind
    model: Model
    settings: ModelSettings
    prompt_cache: PromptCacheState | None
    bundle: PydanticModelBundle | None = None

    @property
    def model_ref(self) -> str:
        """Return the resolved selection as `provider/model` when applicable."""

        if self.provider == "injected":
            return self.model_name
        return format_model_ref(self.provider, self.model_name)


class ModelRouter:
    """Resolve roles to provider/model pairs and reuse one bundle per pair."""

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
        return self._resolve_role(
            override=self.config.agents.routing.master,
            prefer_fast=True,
        )

    def resolve_worker(
        self,
        worker_type_id: str,
        *,
        preferred_tier: Literal["fast", "strong"] | None = None,
    ) -> ModelBinding:
        override = self.config.agents.routing.workers.get(worker_type_id)
        return self._resolve_role(
            override=override,
            prefer_fast=(
                preferred_tier == "fast"
                if preferred_tier is not None
                else False
            ),
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

    def _resolve_role(
        self,
        *,
        override: str | None,
        prefer_fast: bool,
    ) -> ModelBinding:
        defaults = self.config.agents.defaults
        if override is not None:
            try:
                provider, model_name = parse_model_ref(
                    override,
                    default_provider=defaults.provider,
                )
            except ValueError:
                return self._resolve(
                    provider=defaults.provider,
                    model_name=defaults.model,
                    route="fallback",
                )
            if self._is_usable(provider, model_name):
                return self._resolve(
                    provider=provider,
                    model_name=model_name,
                    route="override",
                )
            return self._resolve(
                provider=defaults.provider,
                model_name=defaults.model,
                route="fallback",
            )

        return self._resolve(
            provider=defaults.provider,
            model_name=defaults.model,
            route="strong",
        )

    def _resolve(
        self,
        *,
        provider: ProviderName,
        model_name: str,
        route: RouteKind,
    ) -> ModelBinding:
        if self._injected_model is not None:
            return ModelBinding(
                provider="injected",
                model_name=str(
                    getattr(self._injected_model, "model_name", None) or "<injected>"
                ),
                route="injected",
                model=self._injected_model,
                settings=dict(self._injected_settings),
                prompt_cache=None,
            )
        key = (provider, model_name)
        bundle = self._bundles.get(key)
        if bundle is None:
            defaults = self.config.agents.defaults
            bundle = build_deepseek_model(
                provider=self.config.providers.deepseek,
                model_name=model_name,
                temperature=defaults.temperature,
                reasoning_effort=defaults.reasoning_effort,
                timeout=defaults.chat_timeout,
            )
            self._bundles[key] = bundle
        return ModelBinding(
            provider=provider,
            model_name=model_name,
            route=route,
            model=bundle.model,
            settings=dict(bundle.settings),
            prompt_cache=bundle.prompt_cache,
            bundle=bundle,
        )

    def _is_usable(self, provider: str, model_name: str) -> bool:
        if provider not in KNOWN_PROVIDERS:
            return False
        if not model_name.strip():
            return False
        return True


__all__ = [
    "ModelBinding",
    "ModelRouter",
]
