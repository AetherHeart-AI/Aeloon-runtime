"""Provider-neutral runtime catalog and model-id resolution."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from aeloon_core.config import Config, LocalModelConfig, LocalProviderConfig
from aeloon_core.core import DEEPSEEK_MODELS, DeepSeekProvider, Model, Provider

DEEPSEEK_PROVIDER_ID = "deepseek"
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class RemoteProviderSource:
    """Injected remote provider capability; runtime knows no cloud implementation."""

    id: str
    name: str
    kind: str
    status: Callable[[], dict[str, Any]]
    models: Callable[[], Awaitable[dict[str, Model]]]
    create_provider: Callable[[], Provider]


def qualify_model_id(provider_id: str, model_id: str) -> str:
    provider = validate_provider_id(provider_id)
    model = model_id.strip().lstrip("/")
    if not model:
        raise ValueError("model id is required")
    prefix = f"{provider}/"
    return model if model.startswith(prefix) else f"{prefix}{model}"


def split_model_id(model_id: str) -> tuple[str, str]:
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
    value = model_id.strip()
    if "/" in value:
        provider, model = split_model_id(value)
        return qualify_model_id(provider, model)
    legacy = f"{DEEPSEEK_PROVIDER_ID}/{value}"
    if legacy in DEEPSEEK_MODELS:
        return legacy
    raise ValueError("model id must use the provider/model format")


def resolve_model_id(model_id: str, available_model_ids: Iterable[str]) -> str:
    requested = model_id.strip()
    if not requested:
        raise KeyError("model id is required")
    candidates = list(available_model_ids)
    if requested in candidates:
        return requested
    for candidate in candidates:
        try:
            _, provider_model_id = split_model_id(candidate)
        except ValueError:
            continue
        if provider_model_id == requested:
            return candidate
    raise KeyError(f"Unknown model: {requested}")


class ProviderCatalog:
    """Resolve local and injected remote providers through one namespace."""

    def __init__(
        self,
        config: Config,
        *,
        remote_sources: Iterable[RemoteProviderSource] = (),
        local_provider_factory: Callable[..., Provider] = DeepSeekProvider,
    ) -> None:
        self.config = config
        self._remotes = {source.id: source for source in remote_sources}
        self._local_provider_factory = local_provider_factory

    async def models(self) -> dict[str, Model]:
        models = self._local_models()
        for source in self._remotes.values():
            status = source.status()
            if not status.get("enabled") or not status.get("authenticated"):
                continue
            try:
                models.update(await source.models())
            except Exception:
                # A transient remote failure must not hide configured local APIs.
                continue
        return models

    async def model(self, model_id: str) -> Model:
        requested = model_id.strip()
        local_models = self._local_models()
        if requested in local_models:
            return local_models[requested]
        provider_id, separator, _ = requested.partition("/")
        if separator and provider_id in self._local_providers():
            raise KeyError(f"Unknown model: {requested}")
        if separator and provider_id in self._remotes:
            source = self._remotes[provider_id]
            status = source.status()
            if not status.get("enabled"):
                raise RuntimeError(f"{source.name} is disabled in settings")
            if not status.get("authenticated"):
                raise PermissionError(f"Sign in to {source.name} first")
            model = (await source.models()).get(requested)
            if model is None:
                raise KeyError(f"Unknown model: {requested}")
            return model
        models = await self.models()
        connected = [
            candidate.id
            for candidate in models.values()
            if candidate.provider != DEEPSEEK_PROVIDER_ID
            or self.config.deepseek.api_key != "no-key"
        ]
        try:
            canonical = resolve_model_id(requested, connected)
        except KeyError:
            canonical = resolve_model_id(requested, models)
        return models[canonical]

    def provider(self, model: Model) -> Provider:
        provider_id, _ = split_model_id(model.id)
        remote = self._remotes.get(provider_id)
        if remote is not None:
            return remote.create_provider()
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
        result: list[dict[str, Any]] = []
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
        for source in self._remotes.values():
            status = source.status()
            result.append(
                {
                    "id": source.id,
                    "name": source.name,
                    "kind": source.kind,
                    "base_url": status.get("base_url"),
                    "authenticated": bool(status.get("authenticated")),
                    "credential_configured": bool(status.get("authenticated")),
                    "model_ids": by_provider.get(source.id, []),
                    "user": status.get("user"),
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


CatalogFactory = Callable[[Config], ProviderCatalog]

__all__ = [
    "CatalogFactory",
    "DEEPSEEK_PROVIDER_ID",
    "ProviderCatalog",
    "RemoteProviderSource",
    "normalize_model_id",
    "qualify_model_id",
    "resolve_model_id",
    "split_model_id",
    "validate_provider_id",
]
