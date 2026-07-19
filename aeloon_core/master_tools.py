"""Master-only atomic Worker lifecycle tools for one turn."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.tools.base import FunctionTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.worker_control import WorkerControlService

if TYPE_CHECKING:
    from aeloon_core.flow_control import FlowControlService


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NoArgs(_Args):
    pass


class _WorkerId(_Args):
    worker_id: str = Field(min_length=1)


class _RunId(_Args):
    run_id: str = Field(min_length=1)


class _SpawnArgs(_Args):
    worker_type_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    detached: bool = False


class _ReuseArgs(_Args):
    worker_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class _ResumeArgs(_Args):
    run_id: str = Field(min_length=1)
    response: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class _AwaitArgs(_Args):
    run_ids: list[str] = Field(min_length=1)
    timeout_seconds: float | None = Field(default=None, ge=0)


def build_master_scheduler_tools(
    *,
    control: WorkerControlService,
    base_session_id: str,
    base_turn_id: str,
    on_progress: Any | None = None,
    flow_control: FlowControlService | None = None,
    execution_guard: Callable[[Any], Any] | None = None,
) -> ToolRegistry:
    """Return a per-turn registry whose lifecycle calls are session-scoped."""

    registry = ToolRegistry(execution_guard=execution_guard)

    async def discover_worker_types() -> str:
        return _json(control.discover_worker_types())

    async def list_workers() -> str:
        return _json(control.list_workers(base_session_id))

    async def inspect_worker(worker_id: str) -> str:
        return _json(control.inspect_worker(worker_id, base_session_id=base_session_id))

    async def spawn_worker(
        worker_type_id: str,
        objective: str,
        idempotency_key: str,
        detached: bool = False,
    ) -> str:
        return _json(
            await control.spawn_worker(
                base_session_id=base_session_id,
                base_turn_id=base_turn_id,
                worker_type_id=worker_type_id,
                objective=objective,
                idempotency_key=idempotency_key,
                detached=detached,
                progress=on_progress,
            )
        )

    async def reuse_worker(
        worker_id: str,
        objective: str,
        idempotency_key: str,
    ) -> str:
        return _json(
            await control.reuse_worker(
                base_session_id=base_session_id,
                worker_id=worker_id,
                objective=objective,
                idempotency_key=idempotency_key,
                base_turn_id=base_turn_id,
                progress=on_progress,
            )
        )

    async def await_workers(
        run_ids: list[str], timeout_seconds: float | None = None
    ) -> str:
        return _json(
            await control.await_workers(
                run_ids,
                timeout=timeout_seconds,
                base_session_id=base_session_id,
            )
        )

    async def resume_worker(
        run_id: str,
        response: str,
        idempotency_key: str,
    ) -> str:
        return _json(
            await control.resume_worker(
                run_id,
                response=response,
                idempotency_key=idempotency_key,
                base_session_id=base_session_id,
                base_turn_id=base_turn_id,
                progress=on_progress,
            )
        )

    async def cancel_worker(run_id: str) -> str:
        return _json(
            await control.cancel_worker(run_id, base_session_id=base_session_id)
        )

    async def archive_worker(worker_id: str) -> str:
        return _json(
            control.archive_worker(worker_id, base_session_id=base_session_id)
        )

    specifications = (
        (
            "discover_worker_types",
            "List available soft Worker responsibilities and definition digests.",
            _NoArgs,
            discover_worker_types,
            "read_only",
        ),
        (
            "list_workers",
            "List this Master session's WorkerSessions and their latest bounded result.",
            _NoArgs,
            list_workers,
            "read_only",
        ),
        (
            "inspect_worker",
            "Inspect one owned WorkerSession without reading its private transcript.",
            _WorkerId,
            inspect_worker,
            "read_only",
        ),
        (
            "spawn_worker",
            "Create a WorkerSession and assign one outcome-oriented objective.",
            _SpawnArgs,
            spawn_worker,
            "mutating",
        ),
        (
            "reuse_worker",
            "Explicitly reuse an idle related WorkerSession for a new objective.",
            _ReuseArgs,
            reuse_worker,
            "mutating",
        ),
        (
            "await_workers",
            "Wait until selected Runs complete, fail, cancel, or request Master context.",
            _AwaitArgs,
            await_workers,
            "read_only",
        ),
        (
            "resume_worker",
            "Answer a waiting low-level Worker that is not Flow-owned. Use "
            "resume_flow_node for a Flow node.",
            _ResumeArgs,
            resume_worker,
            "mutating",
        ),
        (
            "cancel_worker",
            "Cancel one queued or running WorkerRun.",
            _RunId,
            cancel_worker,
            "mutating",
        ),
        (
            "archive_worker",
            "Soft-delete one inactive WorkerSession.",
            _WorkerId,
            archive_worker,
            "mutating",
        ),
    )
    for name, description, model, handler, concurrency_mode in specifications:
        registry.register(
            FunctionTool(
                name=name,
                description=description,
                args_model=model,
                handler=handler,
                concurrency_mode=concurrency_mode,
            )
        )
    if flow_control is not None:
        # Imported lazily so the low-level Worker scheduler remains independently usable.
        from aeloon_core.master_flow_tools import build_master_flow_tools

        flow_tools = build_master_flow_tools(
            control=flow_control,
            base_session_id=base_session_id,
            base_turn_id=base_turn_id,
            on_progress=on_progress,
        )
        for definition in flow_tools.get_definitions():
            name = str(definition["name"])
            tool = flow_tools.get(name)
            assert tool is not None
            registry.register(tool)
    return registry


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = ["build_master_scheduler_tools"]
