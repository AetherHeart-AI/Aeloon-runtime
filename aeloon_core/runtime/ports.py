"""Injected application capabilities implemented outside runtime."""

from __future__ import annotations

from typing import Any, Protocol


class AccountConfig(Protocol):
    enabled: bool
    base_url: str
    proxy: str | None
    device_name: str
    allow_insecure_http: bool


class AccountGateway(Protocol):
    def status(self) -> dict[str, Any]: ...

    async def login(self, *, username: str, password: str) -> dict[str, Any]: ...

    def logout(self) -> dict[str, Any]: ...

    async def configure(self, config: AccountConfig) -> None: ...

    async def close(self) -> None: ...


class NullAccountGateway:
    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "authenticated": False,
            "base_url": None,
            "user": None,
        }

    async def login(self, *, username: str, password: str) -> dict[str, Any]:
        del username, password
        raise RuntimeError("No account provider is configured")

    def logout(self) -> dict[str, Any]:
        return self.status()

    async def configure(self, config: AccountConfig) -> None:
        del config

    async def close(self) -> None:
        return None


__all__ = ["AccountConfig", "AccountGateway", "NullAccountGateway"]
