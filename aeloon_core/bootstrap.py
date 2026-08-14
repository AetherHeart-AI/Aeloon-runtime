"""Application composition root for runtime, cloud, and transport adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeloon_core.cloud import CloudAccountService
from aeloon_core.cloud import CloudConfig as CloudServiceConfig
from aeloon_core.config import CloudProviderConfig, Config, load_config
from aeloon_core.runtime.agent import SessionAgentFactory
from aeloon_core.runtime.ports import AccountConfig
from aeloon_core.runtime.providers import ProviderManager, ProviderManagerFactory
from aeloon_core.runtime.service import RuntimeService


def _cloud_config(config: AccountConfig) -> CloudServiceConfig:
    return CloudServiceConfig(
        enabled=config.enabled,
        base_url=config.endpoint,
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

    async def models(self, *, force: bool = False) -> list[dict[str, Any]]:
        return await self._service.models(force=force)

    async def access_token(self, *, force: bool = False) -> str:
        return await self._service.access_token(force=force)

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
    cloud_config = config.providers["aeloon-cloud"]
    assert isinstance(cloud_config, CloudProviderConfig)
    account = CloudAccountGateway(cloud_config, data_dir=config.data_dir)

    def create_provider_manager(snapshot: Config) -> ProviderManager:
        snapshot_cloud = snapshot.providers["aeloon-cloud"]
        assert isinstance(snapshot_cloud, CloudProviderConfig)
        return ProviderManager(
            snapshot,
            account_factory=lambda: CloudAccountGateway(
                snapshot_cloud,
                data_dir=snapshot.data_dir,
            ),
            close_account=True,
        )

    manager_factory: ProviderManagerFactory = create_provider_manager

    return RuntimeService(
        config_path=config_path,
        data_dir=data_dir,
        max_concurrent_operations=max_concurrent_operations,
        agent_factory=agent_factory,
        provider_manager_factory=manager_factory,
        account_gateway=account,
    )


__all__ = ["CloudAccountGateway", "create_runtime_service"]
