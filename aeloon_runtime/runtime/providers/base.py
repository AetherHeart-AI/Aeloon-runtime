"""Common runtime Provider object model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from aeloon_runtime.core import AssistantStreamEvent, InferenceContext, Model, StreamOptions


class BaseProvider(ABC):
    driver = "base"
    kind = "local"

    def __init__(
        self,
        *,
        provider_id: str,
        name: str,
        endpoint: str,
        enabled: bool = True,
    ) -> None:
        self.id = provider_id
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.enabled = enabled

    @abstractmethod
    async def models(self) -> dict[str, Model]: ...

    @abstractmethod
    def stream(
        self,
        model: Model,
        context: InferenceContext,
        options: StreamOptions,
    ) -> AsyncIterator[AssistantStreamEvent]: ...

    async def discover_models(self) -> list[Model]:
        return list((await self.models()).values())

    async def close(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "driver": self.driver,
            "kind": self.kind,
            "endpoint": self.endpoint,
            "enabled": self.enabled,
            "authenticated": None,
            "credential_configured": False,
        }


__all__ = ["BaseProvider"]
