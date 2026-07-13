"""Base-only atomic scheduling tools for one coordinator turn."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from aeloon_core.tools.base import FunctionTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.worker_control import WorkerControlService


class _NoArgs(BaseModel):
    pass


class _WorkerId(BaseModel):
    worker_id: str = Field(min_length=1)


class _RunId(BaseModel):
    run_id: str = Field(min_length=1)


class _SpawnArgs(BaseModel):
    profile_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    detached: bool = False


class _SendArgs(BaseModel):
    worker_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class _AwaitArgs(BaseModel):
    run_ids: list[str] = Field(min_length=1)
    timeout_seconds: float | None = Field(default=None, ge=0)


def build_base_scheduler_tools(
    *,
    control: WorkerControlService,
    base_session_id: str,
    base_turn_id: str,
    on_progress: Any | None = None,
) -> ToolRegistry:
    """Return a per-turn registry; Base never receives domain host tools."""

    registry = ToolRegistry()

    async def discover_profiles() -> str:
        return _json(control.discover_profiles())

    async def list_workers() -> str:
        return _json(control.list_workers(base_session_id))

    async def inspect_worker(worker_id: str) -> str:
        return _json(control.inspect_worker(worker_id))

    async def spawn_worker(
        profile_id: str,
        task: str,
        idempotency_key: str,
        detached: bool = False,
    ) -> str:
        return _json(
            await control.spawn_worker(
                base_session_id=base_session_id,
                base_turn_id=base_turn_id,
                profile_id=profile_id,
                task=task,
                idempotency_key=idempotency_key,
                detached=detached,
                progress=on_progress,
            )
        )

    async def send_worker(worker_id: str, task: str, idempotency_key: str) -> str:
        return _json(
            await control.send_worker(
                base_session_id=base_session_id,
                worker_id=worker_id,
                task=task,
                idempotency_key=idempotency_key,
                base_turn_id=base_turn_id,
                progress=on_progress,
            )
        )

    async def await_workers(run_ids: list[str], timeout_seconds: float | None = None) -> str:
        return _json(await control.await_workers(run_ids, timeout=timeout_seconds))

    async def resume_worker(run_id: str) -> str:
        return _json(await control.resume_worker(run_id))

    async def cancel_worker(run_id: str) -> str:
        return _json(await control.cancel_worker(run_id))

    async def archive_worker(worker_id: str) -> str:
        return _json(control.archive_worker(worker_id))

    for name, description, model, handler, concurrency_mode in (
        (
            "discover_profiles",
            "List active Worker Profile capabilities.",
            _NoArgs,
            discover_profiles,
            "read_only",
        ),
        (
            "list_workers",
            "List this Base session's Workers and whether each is safe to reuse.",
            _NoArgs,
            list_workers,
            "read_only",
        ),
        (
            "inspect_worker",
            "Read one Worker's bounded lifecycle and results.",
            _WorkerId,
            inspect_worker,
            "read_only",
        ),
        (
            "spawn_worker",
            "Create a clean Worker only when no compatible reusable Worker exists.",
            _SpawnArgs,
            spawn_worker,
            "mutating",
        ),
        (
            "send_worker",
            "Reuse one idle Worker's private context for a related follow-up task.",
            _SendArgs,
            send_worker,
            "mutating",
        ),
        (
            "await_workers",
            "Wait for one or more Worker runs.",
            _AwaitArgs,
            await_workers,
            "read_only",
        ),
        (
            "resume_worker",
            "Resume one recoverable Worker run.",
            _RunId,
            resume_worker,
            "mutating",
        ),
        (
            "cancel_worker",
            "Cancel one Worker run.",
            _RunId,
            cancel_worker,
            "mutating",
        ),
        (
            "archive_worker",
            "Archive one inactive Worker.",
            _WorkerId,
            archive_worker,
            "mutating",
        ),
    ):
        registry.register(
            FunctionTool(
                name=name,
                description=description,
                args_model=model,
                handler=handler,
                concurrency_mode=concurrency_mode,
            )
        )
    return registry


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
