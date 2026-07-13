"""Prompt-independent atomic Worker control operations."""

from __future__ import annotations

from typing import Any

from aeloon_core.profile_registry import ProfileRegistry
from aeloon_core.worker_manager import WorkerSessionManager
from aeloon_core.worker_sessions import (
    BudgetGrant,
    ContextEnvelope,
    PermissionSnapshot,
    WorkerRunStatus,
    WorkerSessionStatus,
)


class WorkerControlService:
    """The single capability surface shared by Base and the terminal UI."""

    def __init__(self, *, manager: WorkerSessionManager, profiles: ProfileRegistry) -> None:
        self.manager = manager
        self.profiles = profiles

    def discover_profiles(self) -> list[dict[str, Any]]:
        return [descriptor.to_dict() for descriptor in self.profiles.discover()]

    def list_workers(self, base_session_id: str) -> list[dict[str, Any]]:
        workers: list[dict[str, Any]] = []
        for worker in self.manager.store.list_workers(base_session_id):
            runs = self.manager.store.list_runs(worker.worker_id)
            latest = runs[-1] if runs else None
            reusable = bool(
                worker.status is WorkerSessionStatus.IDLE
                and latest is not None
                and latest.status in {WorkerRunStatus.COMPLETED, WorkerRunStatus.PARTIAL}
            )
            workers.append(
                {
                    "worker_id": worker.worker_id,
                    "profile": worker.profile.model_dump(mode="json"),
                    "status": worker.status.value,
                    "created_at": worker.created_at,
                    "reusable": reusable,
                    "recommended_action": "send_worker" if reusable else None,
                    "run_count": len(runs),
                    "latest_run": self._run_summary(latest) if latest is not None else None,
                    "affinity": {
                        "workspace_paths": latest.context.permissions.workspace_paths
                        if latest is not None
                        else (),
                        "sensitivity": latest.context.permissions.sensitivity
                        if latest is not None
                        else "normal",
                    },
                }
            )
        return workers

    async def spawn_worker(
        self,
        *,
        base_session_id: str,
        profile_id: str,
        task: str,
        idempotency_key: str,
        base_turn_id: str | None = None,
        detached: bool = False,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        descriptor = next(
            (item for item in self.profiles.discover() if item.profile.profile_id == profile_id),
            None,
        )
        if descriptor is None:
            raise ValueError(f"unknown active Worker Profile: {profile_id}")
        context = ContextEnvelope(
            goal=task,
            permissions=PermissionSnapshot(tool_names=descriptor.requested_tools),
            budget=BudgetGrant(max_tokens=32_000, max_seconds=3_600, max_tool_calls=25),
        )
        worker, run, created = await self.manager.spawn_worker(
            base_session_id=base_session_id,
            base_turn_id=base_turn_id,
            profile=descriptor.profile,
            context=context,
            idempotency_key=idempotency_key,
            start=not detached,
            progress=progress if not detached else None,
        )
        return {
            "worker_id": worker.worker_id,
            "run_id": run.run_id,
            "run_sequence": run.run_sequence,
            "created_at": run.created_at,
            "created": created,
            "detached": detached,
            "profile": worker.profile.model_dump(mode="json"),
        }

    async def send_worker(
        self,
        *,
        base_session_id: str,
        worker_id: str,
        task: str,
        idempotency_key: str,
        base_turn_id: str | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        worker, runs = self.manager.inspect_worker(worker_id)
        if worker.base_session_id != base_session_id:
            raise ValueError("cannot reuse a Worker owned by another Base session")
        if worker.status is WorkerSessionStatus.ARCHIVED:
            raise ValueError("archived Workers cannot be reused")
        if not runs:
            raise ValueError("the Worker has no prior context to reuse")
        latest = runs[-1]
        is_idempotent_replay = latest.idempotency_key == idempotency_key
        if not latest.status.terminal and not is_idempotent_replay:
            raise ValueError("the Worker is busy; await its active run before reusing it")
        if (
            latest.status not in {WorkerRunStatus.COMPLETED, WorkerRunStatus.PARTIAL}
            and not is_idempotent_replay
        ):
            raise ValueError("the Worker has no safe completed checkpoint to reuse")
        context = ContextEnvelope(
            goal=task,
            # A continuation cannot silently expand the permissions or budget of the
            # Worker's previous security domain.
            permissions=latest.context.permissions,
            budget=latest.context.budget,
        )
        run, created = await self.manager.send_worker(
            worker_id=worker.worker_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
            progress=progress,
        )
        return {
            "worker_id": worker_id,
            "run_id": run.run_id,
            "run_sequence": run.run_sequence,
            "created_at": run.created_at,
            "created": created,
            "reused_worker": True,
            "profile": worker.profile.model_dump(mode="json"),
        }

    async def await_workers(
        self,
        run_ids: list[str],
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        runs = await self.manager.await_workers(run_ids, timeout=timeout)
        return [self._run_view(run) for run in runs]

    def inspect_worker(self, worker_id: str) -> dict[str, Any]:
        worker, runs = self.manager.inspect_worker(worker_id)
        return {
            "worker_id": worker.worker_id,
            "profile": worker.profile.model_dump(mode="json"),
            "status": worker.status.value,
            "created_at": worker.created_at,
            "runs": [self._run_view(run) for run in runs],
        }

    async def resume_worker(
        self,
        run_id: str,
        *,
        instruction: str | None = None,
        idempotency_key: str | None = None,
        base_session_id: str | None = None,
        base_turn_id: str | None = None,
        expected_run_sequence: int | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Schedule a queued run or create an explicit terminal recovery run."""

        source = self.manager.store.get_run(run_id)
        worker, runs = self.manager.inspect_worker(source.worker_id)
        if base_session_id is not None and worker.base_session_id != base_session_id:
            raise ValueError("cannot recover a Worker owned by another Base session")
        if worker.status is WorkerSessionStatus.ARCHIVED:
            raise ValueError("archived Workers cannot be recovered")
        if expected_run_sequence is not None and source.run_sequence != expected_run_sequence:
            raise ValueError("the selected Worker run is stale; inspect the Worker again")

        latest = runs[-1]
        moved_to_newer_run = latest.run_id != source.run_id

        if source.status is WorkerRunStatus.QUEUED and not moved_to_newer_run:
            resumed = await self.manager.resume_worker(run_id)
            return self._recovery_view(
                resumed,
                action="scheduled",
                source_status=source.status,
                created=False,
            )
        if source.status is WorkerRunStatus.RUNNING and not moved_to_newer_run:
            return self._recovery_view(
                source,
                action="already_running",
                source_status=source.status,
                created=False,
            )
        if source.status is WorkerRunStatus.WAITING_FOR_CONTEXT and not moved_to_newer_run:
            raise ValueError(
                "this Worker is waiting for context, but no safe continuation checkpoint exists"
            )
        if source.status not in {
            WorkerRunStatus.COMPLETED,
            WorkerRunStatus.PARTIAL,
            WorkerRunStatus.FAILED,
            WorkerRunStatus.CANCELLED,
        }:
            raise ValueError(f"Worker run status {source.status.value!r} is not recoverable")

        recovery_instruction = (instruction or "").strip()
        if not recovery_instruction:
            raise ValueError(
                "terminal Worker recovery requires an instruction to avoid repeating side effects"
            )
        context = ContextEnvelope(
            goal=recovery_instruction,
            permissions=source.context.permissions,
            budget=source.context.budget,
        )
        recovery_key = idempotency_key or f"resume:{run_id}:{source.run_sequence}"

        # The Store verifies that this source is still latest and inserts the
        # continuation in one transaction. Replays verify the instruction,
        # base turn, operation provenance, and source sequence before returning.
        recovered, created = await self.manager.send_worker(
            worker_id=source.worker_id,
            context=context,
            idempotency_key=recovery_key,
            base_turn_id=base_turn_id,
            source_run_id=source.run_id,
            progress=progress,
        )
        action = (
            "continued"
            if source.status in {WorkerRunStatus.COMPLETED, WorkerRunStatus.PARTIAL}
            else "restarted"
        )
        return self._recovery_view(
            recovered,
            action=action,
            source_status=source.status,
            created=created,
        )

    async def cancel_worker(self, run_id: str) -> dict[str, Any]:
        return self._run_view(await self.manager.cancel_worker(run_id))

    def archive_worker(self, worker_id: str) -> dict[str, Any]:
        worker = self.manager.archive_worker(worker_id)
        return {"worker_id": worker.worker_id, "status": worker.status.value}

    @staticmethod
    def _run_summary(run: Any) -> dict[str, Any]:
        result = run.result
        report = result.report if result is not None else None
        return {
            "run_id": run.run_id,
            "run_sequence": run.run_sequence,
            "created_at": run.created_at,
            "status": run.status.value,
            "goal_preview": run.context.goal[:240],
            "summary": report.summary[:400] if report is not None else None,
        }

    @staticmethod
    def _run_view(run: Any) -> dict[str, Any]:
        result = run.result
        return {
            "run_id": run.run_id,
            "worker_id": run.worker_id,
            "run_sequence": run.run_sequence,
            "created_at": run.created_at,
            "status": run.status.value,
            "goal": run.context.goal,
            "summary": result.report.summary if result and result.report else None,
            "tool_outcome": result.tool_outcome if result else None,
            "usage": result.usage if result else {},
        }

    @classmethod
    def _recovery_view(
        cls,
        run: Any,
        *,
        action: str,
        source_status: WorkerRunStatus,
        created: bool,
    ) -> dict[str, Any]:
        return {
            **cls._run_view(run),
            "action": action,
            "source_status": source_status.value,
            "created": created,
        }
