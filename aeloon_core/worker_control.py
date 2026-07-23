"""Prompt-independent atomic Worker control operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from aeloon_core.worker_manager import WorkerSessionManager
from aeloon_core.worker_sessions import (
    BudgetGrant,
    BudgetIncrease,
    ContextEnvelope,
    PermissionSnapshot,
    RelatedContextSection,
    RelatedWorkerContext,
    WorkerRunStatus,
    WorkerSessionStatus,
)
from aeloon_core.workers import WorkerRegistry


class WorkerControlService:
    """The single lifecycle capability surface shared by Master and operator UI."""

    def __init__(
        self,
        *,
        manager: WorkerSessionManager,
        worker_types: WorkerRegistry,
        worker_tool_names: tuple[str, ...] = (),
        skills_enabled: bool = True,
        default_budget: BudgetGrant | None = None,
    ) -> None:
        self.manager = manager
        self.worker_types = worker_types
        self.worker_tool_names = tuple(sorted(set(worker_tool_names)))
        self.skills_enabled = bool(skills_enabled)
        self.default_budget = default_budget or BudgetGrant()

    def discover_worker_types(self) -> list[dict[str, str]]:
        """Return bounded descriptors; prompts remain private to Worker execution."""

        return [snapshot.descriptor() for snapshot in self.worker_types.list()]

    def list_workers(self, base_session_id: str) -> list[dict[str, Any]]:
        workers: list[dict[str, Any]] = []
        for worker in self.manager.store.list_workers(base_session_id):
            runs = self.manager.store.list_runs(worker.worker_id)
            latest = runs[-1] if runs else None
            flow_owned = bool(
                latest is not None
                and latest.base_turn_id is not None
                and latest.base_turn_id.startswith("flow:")
            )
            reusable = bool(
                worker.status is WorkerSessionStatus.IDLE
                and latest is not None
                and latest.status in {WorkerRunStatus.COMPLETED, WorkerRunStatus.PARTIAL}
                and not flow_owned
            )
            waiting = bool(
                worker.status is WorkerSessionStatus.IDLE
                and latest is not None
                and latest.status is WorkerRunStatus.WAITING_FOR_CONTEXT
            )
            resumable = waiting and not flow_owned
            if waiting and flow_owned:
                recommended_action = "resume_flow_node"
            elif resumable:
                recommended_action = "resume_worker"
            elif reusable:
                recommended_action = "reuse_worker"
            else:
                recommended_action = None
            workers.append(
                {
                    "worker_id": worker.worker_id,
                    "snapshot": worker.snapshot.descriptor(),
                    "status": worker.status.value,
                    "created_at": worker.created_at,
                    "reusable": reusable,
                    "resumable": resumable,
                    "flow_owned": flow_owned,
                    "recommended_action": recommended_action,
                    "run_count": len(runs),
                    "latest_run": self._run_summary(latest) if latest is not None else None,
                }
            )
        return workers

    async def spawn_worker(
        self,
        *,
        base_session_id: str,
        worker_type_id: str,
        objective: str,
        idempotency_key: str,
        base_turn_id: str | None = None,
        detached: bool = False,
        progress: Any | None = None,
        start: bool | None = None,
        budget: BudgetGrant | None = None,
        related_contexts: Sequence[RelatedWorkerContext] = (),
    ) -> dict[str, Any]:
        snapshot = self.worker_types.get(worker_type_id)
        context = ContextEnvelope(
            objective=objective,
            permissions=PermissionSnapshot(
                tool_names=self.worker_tool_names,
                skills_enabled=self.skills_enabled,
            ),
            budget=budget or self.default_budget,
            related_contexts=tuple(related_contexts),
        )
        worker, run, created = await self.manager.spawn_worker(
            base_session_id=base_session_id,
            base_turn_id=base_turn_id,
            snapshot=snapshot,
            context=context,
            idempotency_key=idempotency_key,
            start=not detached if start is None else start,
            progress=progress if not detached else None,
        )
        return {
            "worker_id": worker.worker_id,
            "run_id": run.run_id,
            "run_sequence": run.run_sequence,
            "source_run_id": run.source_run_id,
            "created_at": run.created_at,
            "status": run.status.value,
            "created": created,
            "detached": detached,
            "worker_session_action": "new",
            "budget": run.context.budget.model_dump(mode="json"),
            "snapshot": worker.snapshot.descriptor(),
        }

    async def reuse_worker(
        self,
        *,
        base_session_id: str,
        worker_id: str,
        objective: str,
        idempotency_key: str,
        base_turn_id: str | None = None,
        progress: Any | None = None,
        budget_increase: BudgetIncrease | None = None,
    ) -> dict[str, Any]:
        worker, runs = self._owned_worker(worker_id, base_session_id)
        if worker.status is WorkerSessionStatus.ARCHIVED:
            raise ValueError("archived Workers cannot be reused")
        if not runs:
            raise ValueError("the Worker has no prior Run to reuse")
        latest = runs[-1]
        if _is_flow_owned(latest):
            raise ValueError(
                "this Worker belongs to a Flow; revise or retry the Flow node instead"
            )
        if latest.status is WorkerRunStatus.PARTIAL and budget_increase is None:
            raise ValueError(
                "reusing a partial WorkerRun requires a Master-authored budget_increase"
            )
        budget = (
            budget_increase.apply(latest.context.budget)
            if budget_increase is not None
            else self.default_budget
        )
        context = ContextEnvelope(
            objective=objective,
            permissions=latest.context.permissions,
            # Each Run receives the current host grant. In particular, a v2
            # continuation must not inherit the old context-window-as-budget cap.
            budget=budget,
            related_contexts=latest.context.related_contexts,
        )
        run, created = await self.manager.reuse_worker(
            worker_id=worker_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
            progress=progress,
        )
        return {
            "worker_id": worker_id,
            "run_id": run.run_id,
            "run_sequence": run.run_sequence,
            "source_run_id": run.source_run_id,
            "created_at": run.created_at,
            "created": created,
            "reused_worker": True,
            "budget": run.context.budget.model_dump(mode="json"),
            "snapshot": worker.snapshot.descriptor(),
        }

    async def resume_worker(
        self,
        run_id: str,
        *,
        response: str,
        idempotency_key: str,
        base_session_id: str | None = None,
        base_turn_id: str | None = None,
        progress: Any | None = None,
        budget_increase: BudgetIncrease | None = None,
    ) -> dict[str, Any]:
        """Continue the exact latest waiting Run without reopening it."""

        source = self.manager.store.get_run(run_id)
        worker = self.manager.store.get_worker(source.worker_id)
        if base_session_id is not None and worker.base_session_id != base_session_id:
            raise ValueError("cannot resume a Worker owned by another Master session")
        if worker.status is WorkerSessionStatus.ARCHIVED:
            raise ValueError("archived Workers cannot be resumed")
        if _is_flow_owned(source):
            raise ValueError(
                "this WorkerRun belongs to a Flow; use resume_flow_node so the "
                "continuation is durably attached"
            )
        context = ContextEnvelope(
            objective=response,
            permissions=source.context.permissions,
            budget=(
                budget_increase.apply(source.context.budget)
                if budget_increase is not None
                else self.default_budget
            ),
            related_contexts=source.context.related_contexts,
        )
        continuation, created = await self.manager.resume_worker(
            source_run_id=source.run_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
            progress=progress,
        )
        return {
            **self._run_view(continuation),
            "action": "resumed",
            "source_status": source.status.value,
            "created": created,
        }

    async def _reserve_flow_resume(
        self,
        run_id: str,
        *,
        flow_id: str,
        response: str,
        idempotency_key: str,
        base_session_id: str,
        progress: Any | None = None,
        budget_increase: BudgetIncrease | None = None,
    ) -> dict[str, Any]:
        """Reserve an exact Flow continuation for attach-before-start dispatch."""

        source = self._owned_run(run_id, base_session_id)
        worker = self.manager.store.get_worker(source.worker_id)
        if worker.status is WorkerSessionStatus.ARCHIVED:
            raise ValueError("archived Workers cannot be resumed")
        flow_turn_id = f"flow:{flow_id}"
        if source.base_turn_id != flow_turn_id:
            raise ValueError("the source WorkerRun is not owned by this Flow")
        context = ContextEnvelope(
            objective=response,
            permissions=source.context.permissions,
            budget=(
                budget_increase.apply(source.context.budget)
                if budget_increase is not None
                else self.default_budget
            ),
            related_contexts=source.context.related_contexts,
        )
        continuation, created = await self.manager.resume_worker(
            source_run_id=source.run_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=flow_turn_id,
            start=False,
            progress=progress,
        )
        return {
            **self._run_view(continuation),
            "action": "reserved_for_flow",
            "worker_session_action": "resume",
            "worker_session_reason": "waiting_exact_resume",
            "source_status": source.status.value,
            "created": created,
        }

    def flow_reuse_ineligibility_reason(
        self,
        source_run_id: str,
        *,
        flow_id: str,
        base_session_id: str,
        worker_type_id: str,
    ) -> str | None:
        """Return a stable reason code when exact same-node reuse is unsafe."""

        try:
            source = self.manager.store.get_run(source_run_id)
        except (KeyError, ValueError):
            return "source_run_missing"
        try:
            worker = self.manager.store.get_worker(source.worker_id)
        except (KeyError, ValueError):
            return "worker_missing"
        if worker.base_session_id != base_session_id:
            return "worker_owner_mismatch"
        if worker.snapshot.id != worker_type_id:
            return "worker_type_changed"
        if worker.status is WorkerSessionStatus.ARCHIVED:
            return "worker_archived"
        if source.base_turn_id != f"flow:{flow_id}":
            return "worker_owner_mismatch"
        runs = self.manager.store.list_runs(worker.worker_id)
        if not runs or runs[-1].run_id != source.run_id:
            return "worker_context_advanced"
        if worker.status is not WorkerSessionStatus.IDLE:
            return "worker_state_unknown"
        if source.status not in {
            WorkerRunStatus.COMPLETED,
            WorkerRunStatus.PARTIAL,
            WorkerRunStatus.FAILED,
            WorkerRunStatus.CANCELLED,
        }:
            return "worker_state_unknown"
        if source.status is WorkerRunStatus.FAILED and source.result is None:
            return "worker_state_unknown"
        if source.result is not None and source.result.tool_outcome == "unknown":
            return "worker_state_unknown"
        if (
            source.status in {WorkerRunStatus.COMPLETED, WorkerRunStatus.PARTIAL}
            and self.manager.store.load_checkpoint(source.run_id) is None
        ):
            return "checkpoint_missing"
        return None

    async def _reserve_flow_reuse(
        self,
        source_run_id: str,
        *,
        flow_id: str,
        objective: str,
        idempotency_key: str,
        base_session_id: str,
        worker_type_id: str,
        progress: Any | None = None,
        budget: BudgetGrant | None = None,
        related_contexts: Sequence[RelatedWorkerContext] = (),
    ) -> dict[str, Any]:
        """Reserve an exact same-node Flow reuse before durable attachment."""

        reason = self.flow_reuse_ineligibility_reason(
            source_run_id,
            flow_id=flow_id,
            base_session_id=base_session_id,
            worker_type_id=worker_type_id,
        )
        if reason is not None:
            raise ValueError(f"Flow WorkerSession cannot be reused: {reason}")
        source = self._owned_run(source_run_id, base_session_id)
        context = ContextEnvelope(
            objective=objective,
            permissions=source.context.permissions,
            budget=budget or self.default_budget,
            related_contexts=tuple(related_contexts),
        )
        continuation, created = await self.manager.reuse_worker(
            worker_id=source.worker_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=f"flow:{flow_id}",
            flow_source_run_id=source.run_id,
            start=False,
            progress=progress,
        )
        return {
            **self._run_view(continuation),
            "worker_session_action": "reuse",
            "worker_session_reason": "same_node_reuse",
            "created": created,
        }

    def increased_budget(
        self,
        run_id: str,
        increase: BudgetIncrease,
        *,
        base_session_id: str,
    ) -> BudgetGrant:
        """Resolve a Master-authored increase against one owned Run's exact grant."""

        source = self._owned_run(run_id, base_session_id)
        return increase.apply(source.context.budget)

    def start_worker_run(
        self,
        run_id: str,
        *,
        base_session_id: str,
        progress: Any | None = None,
    ) -> None:
        """Activate a Flow-owned reserved Run after its binding is durable."""

        self._owned_run(run_id, base_session_id)
        self.manager.start_existing(run_id, progress=progress)

    def _fence_flow_run(
        self,
        run_id: str,
        *,
        flow_id: str,
        base_session_id: str,
    ) -> dict[str, Any]:
        """Durably prevent a Flow reservation from starting, without awaiting teardown."""

        run = self._owned_run(run_id, base_session_id)
        if run.base_turn_id != f"flow:{flow_id}":
            raise ValueError("the WorkerRun is not owned by this Flow")
        fenced, changed = self.manager.store.try_cancel_run(run_id)
        return {**self._run_view(fenced), "changed": changed}

    async def await_workers(
        self,
        run_ids: list[str],
        *,
        timeout: float | None = None,
        base_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if base_session_id is not None:
            for run_id in run_ids:
                self._owned_run(run_id, base_session_id)
        runs = await self.manager.await_workers(run_ids, timeout=timeout)
        return [self._run_view(run) for run in runs]

    def inspect_worker(
        self,
        worker_id: str,
        *,
        base_session_id: str | None = None,
    ) -> dict[str, Any]:
        worker, runs = self.manager.inspect_worker(worker_id)
        if base_session_id is not None and worker.base_session_id != base_session_id:
            raise ValueError("cannot inspect a Worker owned by another Master session")
        return {
            "worker_id": worker.worker_id,
            "snapshot": worker.snapshot.descriptor(),
            "status": worker.status.value,
            "created_at": worker.created_at,
            "run_count": len(runs),
            "runs": [self._run_view(run) for run in runs[-20:]],
        }

    def find_run_by_idempotency(
        self,
        *,
        base_session_id: str,
        idempotency_key: str,
        objective: str,
        worker_type_id: str | None = None,
        worker_id: str | None = None,
        allowed_source_run_ids: set[str] | None = None,
        budget: BudgetGrant | None = None,
        related_contexts: Sequence[RelatedWorkerContext] | None = None,
    ) -> dict[str, Any] | None:
        """Recover an exact durable Run created before its caller saved a binding."""

        matches: list[Any] = []
        for worker in self.manager.store.list_workers(base_session_id):
            if worker_id is not None and worker.worker_id != worker_id:
                continue
            if worker_type_id is not None and worker.snapshot.id != worker_type_id:
                continue
            for run in self.manager.store.list_runs(worker.worker_id):
                if run.idempotency_key != idempotency_key:
                    continue
                if run.context.objective != objective:
                    raise ValueError(
                        "durable WorkerRun idempotency key has a different objective"
                    )
                if budget is not None and run.context.budget != budget:
                    raise ValueError(
                        "durable WorkerRun idempotency key has a different budget grant"
                    )
                if (
                    related_contexts is not None
                    and run.context.related_contexts != tuple(related_contexts)
                ):
                    raise ValueError(
                        "durable WorkerRun idempotency key has different related context"
                    )
                if (
                    allowed_source_run_ids is not None
                    and run.source_run_id not in allowed_source_run_ids
                ):
                    raise ValueError(
                        "durable WorkerRun idempotency key has a different source Run"
                    )
                matches.append(run)
        if len(matches) > 1:
            raise RuntimeError("Worker idempotency lookup is ambiguous")
        return self._run_view(matches[0]) if matches else None

    def related_context(
        self,
        run_id: str,
        *,
        base_session_id: str,
        source_kind: Literal["worker_run", "flow_node"],
        source_id: str,
        relation: str,
        include: tuple[RelatedContextSection, ...],
    ) -> RelatedWorkerContext:
        """Resolve one same-session settled Run into bounded untrusted reference data."""

        run = self._owned_run(run_id, base_session_id)
        if not run.status.settled:
            raise ValueError(f"related WorkerRun {run_id!r} is not settled")
        worker = self.manager.store.get_worker(run.worker_id)
        report = run.result.report if run.result is not None else None
        sections = set(include)
        return RelatedWorkerContext(
            source_kind=source_kind,
            source_id=source_id,
            relation=relation,
            run_id=run.run_id,
            worker_id=run.worker_id,
            worker_type_id=worker.snapshot.id,
            status=run.status.value,
            included_sections=include,
            objective=(run.context.objective[:2_000] if "objective" in sections else None),
            summary=(
                report.summary[:4_000]
                if report is not None and "summary" in sections
                else None
            ),
            artifacts=(
                _bounded_context_items(report.artifacts, limit=8)
                if report is not None and "artifacts" in sections
                else ()
            ),
            evidence=(
                _bounded_context_items(report.evidence, limit=8)
                if report is not None and "evidence" in sections
                else ()
            ),
            unresolved=(
                _bounded_context_items(report.unresolved, limit=4)
                if report is not None and "unresolved" in sections
                else ()
            ),
        )

    def related_contexts_for_run(
        self,
        run_id: str,
        *,
        base_session_id: str,
    ) -> tuple[RelatedWorkerContext, ...]:
        """Return the bounded associations carried by one owned Run."""

        return self._owned_run(run_id, base_session_id).context.related_contexts

    def find_continuation(
        self,
        source_run_id: str,
        *,
        base_session_id: str,
    ) -> dict[str, Any] | None:
        """Find the unique continuation created through another control surface."""

        source = self._owned_run(source_run_id, base_session_id)
        continuations = [
            run
            for run in self.manager.store.list_runs(source.worker_id)
            if run.source_run_id == source_run_id
        ]
        if len(continuations) > 1:
            raise RuntimeError("a WorkerRun has multiple direct continuations")
        return self._run_view(continuations[0]) if continuations else None

    def run_has_objective(
        self,
        run_id: str,
        objective: str,
        *,
        base_session_id: str,
    ) -> bool:
        """Compare an internal Run objective without exposing unbounded text."""

        run = self._owned_run(run_id, base_session_id)
        return run.context.objective == objective

    async def cancel_worker(
        self,
        run_id: str,
        *,
        base_session_id: str | None = None,
    ) -> dict[str, Any]:
        if base_session_id is not None:
            self._owned_run(run_id, base_session_id)
        run = await self.manager.cancel_worker(run_id)
        if run.status is WorkerRunStatus.CANCELLED:
            action = "cancelled"
        elif (
            run.status is WorkerRunStatus.RUNNING
            and run.cancel_requested_at is not None
        ):
            action = "cancellation_requested"
        else:
            action = "already_settled"
        return {
            **self._run_view(run),
            "action": action,
        }

    def archive_worker(
        self,
        worker_id: str,
        *,
        base_session_id: str | None = None,
    ) -> dict[str, Any]:
        if base_session_id is not None:
            _, runs = self._owned_worker(worker_id, base_session_id)
            if (
                runs
                and _is_flow_owned(runs[-1])
                and runs[-1].status is WorkerRunStatus.WAITING_FOR_CONTEXT
            ):
                raise ValueError(
                    "a waiting Flow Worker cannot be archived; resolve its Flow first"
                )
        worker = self.manager.archive_worker(worker_id)
        return {"worker_id": worker.worker_id, "status": worker.status.value}

    def _owned_worker(self, worker_id: str, base_session_id: str) -> tuple[Any, list[Any]]:
        worker, runs = self.manager.inspect_worker(worker_id)
        if worker.base_session_id != base_session_id:
            raise ValueError("cannot access a Worker owned by another Master session")
        return worker, runs

    def _owned_run(self, run_id: str, base_session_id: str) -> Any:
        run = self.manager.store.get_run(run_id)
        worker = self.manager.store.get_worker(run.worker_id)
        if worker.base_session_id != base_session_id:
            raise ValueError("cannot access a Worker owned by another Master session")
        return run

    @staticmethod
    def _run_summary(run: Any) -> dict[str, Any]:
        result = run.result
        report = result.report if result is not None else None
        return {
            "run_id": run.run_id,
            "run_sequence": run.run_sequence,
            "source_run_id": run.source_run_id,
            "created_at": run.created_at,
            "status": run.status.value,
            "cancel_requested": run.cancel_requested_at is not None,
            "objective_preview": run.context.objective[:240],
            "budget": run.context.budget.model_dump(mode="json"),
            "summary": report.summary[:400] if report is not None else None,
            "related_context_refs": _related_context_refs(run.context.related_contexts),
            "waiting_question": (
                run.waiting_request.question if run.waiting_request is not None else None
            ),
        }

    @staticmethod
    def _run_view(run: Any) -> dict[str, Any]:
        result = run.result
        report = result.report if result is not None else None
        return {
            "run_id": run.run_id,
            "worker_id": run.worker_id,
            "run_sequence": run.run_sequence,
            "source_run_id": run.source_run_id,
            "created_at": run.created_at,
            "status": run.status.value,
            "cancel_requested": run.cancel_requested_at is not None,
            "objective": run.context.objective[:2_000],
            "budget": run.context.budget.model_dump(mode="json"),
            "related_context_refs": _related_context_refs(run.context.related_contexts),
            "summary": report.summary[:4_000] if report is not None else None,
            "artifacts": _bounded_items(report.artifacts) if report is not None else [],
            "evidence": _bounded_items(report.evidence) if report is not None else [],
            "unresolved": _bounded_items(report.unresolved, limit=10)
            if report is not None
            else [],
            "waiting_request": (
                run.waiting_request.model_dump(mode="json")
                if run.waiting_request is not None
                else None
            ),
            "tool_outcome": result.tool_outcome if result is not None else None,
            "usage": result.usage if result is not None else {},
        }


def _bounded_items(items: tuple[str, ...], *, limit: int = 20) -> list[str]:
    return [item[:500] for item in items[:limit]]


def _bounded_context_items(items: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    return tuple(item[:500] for item in items[:limit])


def _related_context_refs(items: tuple[RelatedWorkerContext, ...]) -> list[dict[str, str]]:
    return [
        {
            "source_kind": item.source_kind,
            "source_id": item.source_id,
            "relation": item.relation,
            "run_id": item.run_id,
        }
        for item in items
    ]


def _is_flow_owned(run: Any) -> bool:
    return bool(run.base_turn_id and run.base_turn_id.startswith("flow:"))


__all__ = ["WorkerControlService"]
