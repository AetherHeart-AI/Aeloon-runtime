"""Unified application-level registry for local and Aeloon Cloud providers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from aeloon_core.cloud import CloudAccountService, CloudError, CloudProvider
from aeloon_core.cloud.account import CLOUD_PROVIDER_ID
from aeloon_core.config import Config, LocalModelConfig, LocalProviderConfig
from aeloon_core.harness import DEEPSEEK_MODELS, DeepSeekProvider, Model, Provider

DEEPSEEK_PROVIDER_ID = "deepseek"
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def qualify_model_id(provider_id: str, model_id: str) -> str:
    """Return the canonical ``provider/model`` identifier."""

    provider = validate_provider_id(provider_id)
    model = model_id.strip().lstrip("/")
    if not model:
        raise ValueError("model id is required")
    prefix = f"{provider}/"
    return model if model.startswith(prefix) else f"{prefix}{model}"


def split_model_id(model_id: str) -> tuple[str, str]:
    """Split a canonical model id once so model keys may themselves contain slashes."""

    value = model_id.strip()
    provider, separator, model = value.partition("/")
    if not separator or not model:
        raise ValueError("model id must use the provider/model format")
    return validate_provider_id(provider), model


def validate_provider_id(provider_id: str) -> str:
    value = provider_id.strip()
    if not _PROVIDER_ID.fullmatch(value):
        raise ValueError(
            "provider id must start with a letter or number and contain only letters, "
            "numbers, '.', '_' or '-'"
        )
    return value


def normalize_model_id(model_id: str) -> str:
    """Upgrade legacy built-in DeepSeek ids while rejecting other bare ids."""

    value = model_id.strip()
    if "/" in value:
        provider, model = split_model_id(value)
        return qualify_model_id(provider, model)
    legacy = f"{DEEPSEEK_PROVIDER_ID}/{value}"
    if legacy in DEEPSEEK_MODELS:
        return legacy
    raise ValueError("model id must use the provider/model format")


class UnifiedProviderRegistry:
    """Resolve every provider and model through one qualified namespace."""

    def __init__(
        self,
        config: Config,
        cloud_account: CloudAccountService,
        *,
        local_provider_factory: Callable[..., Provider] = DeepSeekProvider,
        cloud_provider_factory: Callable[..., Provider] = CloudProvider,
    ) -> None:
        self.config = config
        self.cloud_account = cloud_account
        self._local_provider_factory = local_provider_factory
        self._cloud_provider_factory = cloud_provider_factory

    async def models(self) -> dict[str, Model]:
        models = self._local_models()
        status = self.cloud_account.status()
        if status["enabled"] and status["authenticated"]:
            try:
                models.update(await self.cloud_account.models())
            except CloudError:
                # A transient cloud failure must not hide configured local APIs.
                pass
        return models

    async def model(self, model_id: str) -> Model:
        try:
            canonical = normalize_model_id(model_id)
        except ValueError as exc:
            raise KeyError(str(exc)) from exc
        local = self._local_models().get(canonical)
        if local is not None:
            return local
        provider_id, _ = split_model_id(canonical)
        if provider_id != CLOUD_PROVIDER_ID:
            raise KeyError(f"Unknown model: {canonical}")
        status = self.cloud_account.status()
        if not status["enabled"]:
            raise RuntimeError("Aeloon Cloud is disabled in Core settings")
        if not status["authenticated"]:
            raise PermissionError("Sign in to Aeloon Cloud first")
        cloud_model = (await self.cloud_account.models()).get(canonical)
        if cloud_model is None:
            raise KeyError(f"Unknown model: {canonical}")
        return cloud_model

    def provider(self, model: Model) -> Provider:
        provider_id, _ = split_model_id(model.id)
        if provider_id == CLOUD_PROVIDER_ID:
            return self._cloud_provider_factory(self.cloud_account)
        provider = self._local_provider(provider_id)
        return self._local_provider_factory(
            api_key=provider.api_key,
            base_url=provider.base_url,
            proxy=provider.proxy,
            headers=provider.extra_headers,
            display_name=provider.name,
            requires_api_key=provider_id == DEEPSEEK_PROVIDER_ID,
            request_model_id=lambda item: split_model_id(item.id)[1],
        )

    async def providers(self) -> list[dict[str, Any]]:
        models = await self.models()
        by_provider: dict[str, list[str]] = {}
        for model in models.values():
            by_provider.setdefault(model.provider, []).append(model.id)
        result = []
        for provider_id, provider in self._local_providers().items():
            result.append(
                {
                    "id": provider_id,
                    "name": provider.name,
                    "kind": "local",
                    "base_url": provider.base_url,
                    "authenticated": None,
                    "credential_configured": bool(
                        provider.api_key and provider.api_key != "no-key"
                    ),
                    "model_ids": by_provider.get(provider_id, []),
                }
            )
        cloud_status = self.cloud_account.status()
        result.append(
            {
                "id": CLOUD_PROVIDER_ID,
                "name": "Aeloon Cloud",
                "kind": "cloud",
                "base_url": cloud_status["base_url"],
                "authenticated": cloud_status["authenticated"],
                "credential_configured": cloud_status["authenticated"],
                "model_ids": by_provider.get(CLOUD_PROVIDER_ID, []),
                "user": cloud_status["user"],
            }
        )
        return result

    def _local_models(self) -> dict[str, Model]:
        models = {
            model_id: replace(model, base_url=self.config.deepseek.base_url)
            for model_id, model in DEEPSEEK_MODELS.items()
        }
        for provider_id, provider in self.config.local_providers.items():
            validate_provider_id(provider_id)
            for configured_model in provider.models:
                model_id = qualify_model_id(provider_id, configured_model.id)
                models[model_id] = self._configured_model(
                    provider_id, provider, configured_model, model_id
                )
        return models

    def _local_providers(self) -> dict[str, LocalProviderConfig]:
        deepseek = LocalProviderConfig(
            name="DeepSeek",
            base_url=self.config.deepseek.base_url,
            api_key=self.config.deepseek.api_key,
            proxy=self.config.deepseek.proxy,
            extra_headers=self.config.deepseek.extra_headers,
            models=[
                LocalModelConfig(
                    id=split_model_id(model.id)[1],
                    name=model.name,
                    reasoning=model.reasoning,
                    supports_image="image" in model.input,
                    context_window=model.context_window,
                    max_tokens=model.max_tokens,
                )
                for model in DEEPSEEK_MODELS.values()
            ],
        )
        return {DEEPSEEK_PROVIDER_ID: deepseek, **self.config.local_providers}

    def _local_provider(self, provider_id: str) -> LocalProviderConfig:
        try:
            return self._local_providers()[provider_id]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {provider_id}") from exc

    @staticmethod
    def _configured_model(
        provider_id: str,
        provider: LocalProviderConfig,
        model: LocalModelConfig,
        model_id: str,
    ) -> Model:
        return Model(
            id=model_id,
            name=model.name or model.id,
            provider=provider_id,
            base_url=provider.base_url,
            reasoning=model.reasoning,
            input=("text", "image") if model.supports_image else ("text",),
            context_window=model.context_window,
            max_tokens=min(model.max_tokens, model.context_window),
            thinking_level_map={"high": "high", "max": "max"} if model.reasoning else {},
        )


__all__ = [
    "DEEPSEEK_PROVIDER_ID",
    "UnifiedProviderRegistry",
    "normalize_model_id",
    "qualify_model_id",
    "split_model_id",
    "validate_provider_id",
]
