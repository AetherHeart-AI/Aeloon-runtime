"""Operation queues, concurrency, snapshots, and Provider-manager creation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aeloon_runtime.config import Config
from aeloon_runtime.core import Model
from aeloon_runtime.runtime.agent import SessionAgent
from aeloon_runtime.runtime.providers import ProviderManager, ProviderManagerFactory


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Operation:
    id: str
    session_id: str
    workspace: str
    kind: str
    input: dict[str, Any]
    created_at: str = field(default_factory=_now)
    status: str = "queued"
    cancel_requested: bool = False
    blocks: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    task: asyncio.Task[None] | None = None
    agent: SessionAgent | None = None
    model: Model | None = None


@dataclass(slots=True)
class SessionRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operations: dict[str, Operation] = field(default_factory=dict)
    active: Operation | None = None


class OperationCoordinator:
    def __init__(
        self,
        *,
        max_concurrent_operations: int,
        provider_manager_factory: ProviderManagerFactory,
    ) -> None:
        self.semaphore = asyncio.Semaphore(max(1, max_concurrent_operations))
        self.runtimes: dict[str, SessionRuntime] = {}
        self.provider_manager_factory = provider_manager_factory

    @property
    def active_count(self) -> int:
        # A queued operation is still owned by the Runtime and must prevent a
        # root-authority restart or uninstall.  Counting only ``runtime.active``
        # would allow those queued turns to be abandoned during a reconnect.
        return sum(
            1
            for runtime in self.runtimes.values()
            for operation in runtime.operations.values()
            if operation.status in {"queued", "active", "cancelling"}
        )

    def runtime(self, session_id: str) -> SessionRuntime:
        return self.runtimes.setdefault(session_id, SessionRuntime())

    @staticmethod
    def snapshot(config: Config) -> Config:
        return config.model_copy(deep=True)

    def provider_manager(self, config: Config) -> ProviderManager:
        return self.provider_manager_factory(config)


__all__ = ["Operation", "OperationCoordinator", "SessionRuntime"]
