"""Provider construction, model resolution, and operation-scoped lifecycle."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

from aeloon_runtime.config import (
    CloudProviderConfig,
    Config,
    CustomProviderConfig,
    DeepSeekProviderConfig,
    ProviderConfig,
    ProviderModelConfig,
)
from aeloon_runtime.core import InferencePort, Model
from aeloon_runtime.runtime.ports import AccountGateway, NullAccountGateway
from aeloon_runtime.runtime.providers.base import BaseProvider
from aeloon_runtime.runtime.providers.cloud import CloudProvider
from aeloon_runtime.runtime.providers.custom import CustomProvider
from aeloon_runtime.runtime.providers.deepseek import DeepSeekProvider

_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DriverFactory: TypeAlias = Callable[
    [str, ProviderConfig, AccountGateway],
    BaseProvider,
]
AccountGatewayFactory: TypeAlias = Callable[[], AccountGateway]


def validate_provider_id(provider_id: str) -> str:
    value = provider_id.strip()
    if not _PROVIDER_ID.fullmatch(value):
        raise ValueError(
            "provider id must start with a letter or number and contain only letters, "
            "numbers, '.', '_' or '-'"
        )
    return value


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


def normalize_model_id(model_id: str, available_model_ids: Iterable[str] = ()) -> str:
    value = model_id.strip()
    if "/" in value:
        provider, model = split_model_id(value)
        return qualify_model_id(provider, model)
    return resolve_model_id(value, available_model_ids)


def model_from_config(
    provider_id: str,
    configured: ProviderModelConfig,
) -> Model:
    model_id = qualify_model_id(provider_id, configured.id)
    return Model(
        id=model_id,
        name=configured.name or configured.id,
        provider=provider_id,
        reasoning=configured.reasoning,
        input=("text", "image") if configured.supports_image else ("text",),
        context_window=configured.context_window,
        max_output_tokens=min(configured.max_output_tokens, configured.context_window),
        cost=dict(configured.cost),
    )


class ProviderManager:
    """Own Providers for one immutable configuration snapshot or short operation."""

    def __init__(
        self,
        config: Config,
        *,
        account: AccountGateway | None = None,
        account_factory: AccountGatewayFactory | None = None,
        close_account: bool = False,
        driver_factories: Mapping[str, DriverFactory] | None = None,
    ) -> None:
        self.config = config.model_copy(deep=True)
        if account is not None and account_factory is not None:
            raise ValueError("Use account or account_factory, not both")
        self._account = account
        self._account_factory = account_factory
        self._close_account = close_account
        self._factories: dict[str, DriverFactory] = {
            "deepseek": _create_deepseek,
            "custom": _create_custom,
            "cloud": _create_cloud,
            **dict(driver_factories or {}),
        }
        self._instances: dict[str, BaseProvider] = {}
        self._model_cache: dict[str, dict[str, Model]] = {}
        self._closed = False

    @property
    def account(self) -> AccountGateway:
        if self._account is None:
            self._account = (
                self._account_factory()
                if self._account_factory is not None
                else NullAccountGateway()
            )
        return self._account

    async def models(self) -> dict[str, Model]:
        """Return enabled models while isolating transient failures by Provider."""

        result: dict[str, Model] = {}
        for provider_id, configured in self.config.providers.items():
            validate_provider_id(provider_id)
            if not configured.enabled:
                continue
            try:
                result.update(await self._models_for_provider(provider_id))
            except Exception:
                continue
        return result

    async def model(self, model_id: str) -> Model:
        requested = model_id.strip()
        if not requested:
            models = await self.models()
            if not models:
                raise KeyError("No models are available")
            return next(iter(models.values()))
        if "/" in requested:
            provider_id, local_id = split_model_id(requested)
            return await self._model_for_provider(provider_id, local_id)
        for provider_id, configured in self.config.providers.items():
            if not configured.enabled:
                continue
            try:
                models = await self._models_for_provider(provider_id)
            except Exception:
                continue
            qualified = qualify_model_id(provider_id, requested)
            if qualified in models:
                provider = self._provider(provider_id)
                self._require_authenticated(provider)
                return models[qualified]
        raise KeyError(f"Unknown model: {requested}")

    def inference(self, model: Model) -> InferencePort:
        provider_id = model.provider
        if not provider_id:
            provider_id, _ = split_model_id(model.id)
        provider = self._provider(provider_id)
        self._require_enabled(provider)
        self._require_authenticated(provider)
        return provider

    async def providers(self) -> list[dict[str, Any]]:
        all_models = await self.models()
        result: list[dict[str, Any]] = []
        for provider_id, configured in self.config.providers.items():
            provider = self._provider(provider_id)
            status = provider.status()
            status.update(
                {
                    "id": provider_id,
                    "name": configured.name,
                    "driver": configured.driver,
                    **(
                        {"backend": configured.backend}
                        if isinstance(configured, CustomProviderConfig)
                        else {}
                    ),
                    "kind": "cloud" if configured.driver == "cloud" else "local",
                    "endpoint": configured.endpoint,
                    "enabled": configured.enabled,
                    "model_ids": [
                        model_id
                        for model_id, model in all_models.items()
                        if model.provider == provider_id
                    ],
                }
            )
            result.append(status)
        return result

    async def discover_models(self, provider_id: str) -> list[Model]:
        provider = self._provider(provider_id)
        self._require_enabled(provider)
        self._require_authenticated(provider)
        return await provider.discover_models()

    def provider_endpoint(self, provider_id: str) -> str:
        return self._provider(provider_id).endpoint

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for provider in tuple(self._instances.values()):
            try:
                await provider.close()
            except Exception as exc:
                first_error = first_error or exc
        self._instances.clear()
        self._model_cache.clear()
        if self._close_account and self._account is not None:
            try:
                await self._account.close()
            except Exception as exc:
                first_error = first_error or exc
            self._account = None
        if first_error is not None:
            raise first_error

    async def _model_for_provider(self, provider_id: str, local_id: str) -> Model:
        provider = self._provider(provider_id)
        self._require_enabled(provider)
        self._require_authenticated(provider)
        model_id = qualify_model_id(provider_id, local_id)
        models = await self._models_for_provider(provider_id)
        try:
            return models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {model_id}") from exc

    def _provider(self, provider_id: str) -> BaseProvider:
        if self._closed:
            raise RuntimeError("ProviderManager is closed")
        configured = self.config.providers.get(provider_id)
        if configured is None:
            raise KeyError(f"Unknown provider: {provider_id}")
        validate_provider_id(provider_id)
        instance = self._instances.get(provider_id)
        if instance is None:
            try:
                factory = self._factories[configured.driver]
            except KeyError as exc:
                raise ValueError(f"Unsupported Provider driver: {configured.driver}") from exc
            account = self.account if configured.driver == "cloud" else NullAccountGateway()
            instance = factory(provider_id, configured, account)
            self._instances[provider_id] = instance
        return instance

    async def _models_for_provider(self, provider_id: str) -> dict[str, Model]:
        models = self._model_cache.get(provider_id)
        if models is None:
            models = dict(await self._provider(provider_id).models())
            self._model_cache[provider_id] = models
        return dict(models)

    @staticmethod
    def _require_enabled(provider: BaseProvider) -> None:
        if not provider.enabled:
            raise RuntimeError(f"{provider.name} is disabled in settings")

    @staticmethod
    def _require_authenticated(provider: BaseProvider) -> None:
        if provider.status().get("authenticated") is False:
            raise PermissionError(f"Authenticate with {provider.name} first")


ProviderManagerFactory: TypeAlias = Callable[[Config], ProviderManager]


def provider_manager_factory(
    *,
    account: AccountGateway | None = None,
    account_factory: AccountGatewayFactory | None = None,
    close_account: bool = False,
    driver_factories: Mapping[str, DriverFactory] | None = None,
) -> ProviderManagerFactory:
    def create(config: Config) -> ProviderManager:
        return ProviderManager(
            config,
            account=account,
            account_factory=account_factory,
            close_account=close_account,
            driver_factories=driver_factories,
        )

    return create


def _configured_models(
    provider_id: str,
    configured: CustomProviderConfig | DeepSeekProviderConfig,
) -> tuple[Model, ...]:
    return tuple(model_from_config(provider_id, item) for item in configured.models)


def _create_deepseek(
    provider_id: str,
    configured: ProviderConfig,
    account: AccountGateway,
) -> BaseProvider:
    del account
    assert isinstance(configured, DeepSeekProviderConfig)
    return DeepSeekProvider(
        name=configured.name,
        endpoint=configured.endpoint,
        api_key=configured.api_key,
        proxy=configured.proxy,
        headers=configured.headers,
        models=_configured_models(provider_id, configured) or None,
        enabled=configured.enabled,
    )


def _create_custom(
    provider_id: str,
    configured: ProviderConfig,
    account: AccountGateway,
) -> BaseProvider:
    del account
    assert isinstance(configured, CustomProviderConfig)
    return CustomProvider(
        provider_id=provider_id,
        name=configured.name,
        endpoint=configured.endpoint,
        backend=configured.backend,
        models=_configured_models(provider_id, configured),
        enabled=configured.enabled,
        api_key=configured.api_key,
        proxy=configured.proxy,
        headers=configured.headers,
    )


def _create_cloud(
    provider_id: str,
    configured: ProviderConfig,
    account: AccountGateway,
) -> BaseProvider:
    del provider_id
    assert isinstance(configured, CloudProviderConfig)
    return CloudProvider(
        account,
        name=configured.name,
        endpoint=configured.endpoint,
        enabled=configured.enabled,
        proxy=configured.proxy,
    )


__all__ = [
    "AccountGatewayFactory",
    "DriverFactory",
    "ProviderManager",
    "ProviderManagerFactory",
    "model_from_config",
    "normalize_model_id",
    "provider_manager_factory",
    "qualify_model_id",
    "resolve_model_id",
    "split_model_id",
    "validate_provider_id",
]
