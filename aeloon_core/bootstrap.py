"""Application composition root for runtime, cloud, and transport adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeloon_core.cloud import CloudAccountService, CloudProvider
from aeloon_core.cloud import CloudConfig as CloudServiceConfig
from aeloon_core.cloud.account import CLOUD_PROVIDER_ID
from aeloon_core.config import Config, load_config
from aeloon_core.core import Provider
from aeloon_core.runtime.agent import SessionAgentFactory
from aeloon_core.runtime.catalog import ProviderCatalog, RemoteProviderSource
from aeloon_core.runtime.ports import AccountConfig
from aeloon_core.runtime.service import RuntimeService


def _cloud_config(config: AccountConfig) -> CloudServiceConfig:
    return CloudServiceConfig(
        enabled=config.enabled,
        base_url=config.base_url,
        proxy=config.proxy,
        device_name=config.device_name,
        allow_insecure_http=config.allow_insecure_http,
    )


class CloudAccountGateway:
    """Composition adapter; neither runtime nor cloud imports the other."""

    def __init__(self, config: AccountConfig, *, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._service = CloudAccountService(_cloud_config(config), data_dir=data_dir)

    def status(self) -> dict[str, Any]:
        return self._service.status()

    async def models(self):
        return await self._service.models()

    def create_provider(self) -> Provider:
        return CloudProvider(self._service)

    async def login(self, *, username: str, password: str) -> dict[str, Any]:
        return await self._service.login(username=username, password=password)

    def logout(self) -> dict[str, Any]:
        return self._service.logout()

    async def configure(self, config: AccountConfig) -> None:
        service_config = _cloud_config(config)
        if service_config == self._service.config:
            return
        previous = self._service
        if service_config.base_url.rstrip("/") != previous.config.base_url.rstrip("/"):
            previous.logout()
        self._service = CloudAccountService(service_config, data_dir=self.data_dir)
        await previous.close()

    async def close(self) -> None:
        await self._service.close()


def cloud_provider_source(account: CloudAccountService) -> RemoteProviderSource:
    """Adapt an existing cloud account for a provider-neutral catalog."""

    return RemoteProviderSource(
        id=CLOUD_PROVIDER_ID,
        name="Aeloon Cloud",
        kind="cloud",
        status=account.status,
        models=account.models,
        create_provider=lambda: CloudProvider(account),
    )


def create_runtime_service(
    *,
    config_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    max_concurrent_operations: int = 4,
    agent_factory: SessionAgentFactory | None = None,
) -> RuntimeService:
    config = load_config(config_path)
    if data_dir is not None:
        config = config.model_copy(update={"data_dir": Path(data_dir)}).normalized()
    account = CloudAccountGateway(config.cloud, data_dir=config.data_dir)
    source = RemoteProviderSource(
        id=CLOUD_PROVIDER_ID,
        name="Aeloon Cloud",
        kind="cloud",
        status=account.status,
        models=account.models,
        create_provider=account.create_provider,
    )

    def catalog_factory(value: Config) -> ProviderCatalog:
        return ProviderCatalog(value, remote_sources=(source,))

    return RuntimeService(
        config_path=config_path,
        data_dir=data_dir,
        max_concurrent_operations=max_concurrent_operations,
        agent_factory=agent_factory,
        catalog_factory=catalog_factory,
        account_gateway=account,
    )


__all__ = ["CloudAccountGateway", "cloud_provider_source", "create_runtime_service"]
