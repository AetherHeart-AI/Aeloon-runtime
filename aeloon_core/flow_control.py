"""Dynamic Flow runtime layered above atomic Worker lifecycle operations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import partial
from typing import Any

from aeloon_core.flow_activation_fence import flow_activation_fence
from aeloon_core.flow_runtime import (
    _annotate_recovered_session_decision,
    _attach_run_to_flow,
    _claim_ready_frontier,
    _fail_node_for_binding_limit,
    _fail_nodes_with_missing_runs,
    _flow_summary,
    _flow_turn_id,
    _missing_run_view,
    _node_run_key,
    _node_status,
    _requested_fresh_reason,
    _resume_run_key,
    _reuse_session_reason,
    _unique,
    _worker_run_exists,
)
from aeloon_core.flow_state import (
    MAX_FLOW_RUN_BINDINGS,
    FlowAdvanceMode,
    FlowCompletion,
    FlowNode,
    FlowNodeSpec,
    FlowNodeStatus,
    FlowStatus,
    MasterFlow,
    WorkerSessionAction,
    add_flow_nodes,
    cancel_flow_state,
    finish_flow,
    flow_node_spec_payload,
    flow_run_telemetry_payload,
    pause_flow,
    reopen_flow,
    retry_flow_node,
    revise_flow_node,
    skip_flow_node,
)
from aeloon_core.flows import FlowStore
from aeloon_core.worker_control import WorkerControlService
from aeloon_core.worker_state import (
    BudgetGrant,
    BudgetIncrease,
    RelatedWorkerContext,
    WorkerRunStatus,
)


class FlowControlService:
    """Persist, evolve, and execute one ready Flow frontier at a time."""

    def __init__(
        self,
        *,
        store: FlowStore,
        workers: WorkerControlService,
    ) -> None:
        self.store = store
        self.workers = workers
        self._locks: dict[str, asyncio.Lock] = {}

    def create_flow(
        self,
        *,
        base_session_id: str,
        goal: str,
        nodes: Sequence[FlowNodeSpec],
        idempotency_key: str,
        max_nodes: int = 64,
        max_rounds: int = 12,
        advance_mode: FlowAdvanceMode = FlowAdvanceMode.CHECKPOINTED,
        auto_advance_max_frontiers: int = 4,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_worker_types(nodes)
        self._validate_external_context_refs(
            base_session_id=base_session_id,
            nodes=nodes,
        )
        flow, created = self.store.create_flow(
            base_session_id=base_session_id,
            goal=goal,
            nodes=nodes,
            idempotency_key=idempotency_key,
            max_nodes=max_nodes,
            max_rounds=max_rounds,
            advance_mode=advance_mode,
            auto_advance_max_frontiers=auto_advance_max_frontiers,
            turn_id=turn_id,
        )
        return {**flow.to_view(), "created": created}

    def list_flows(
        self,
        base_session_id: str,
        *,
        include_terminal: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            _flow_summary(flow)
            for flow in self.store.list_flows(
                base_session_id,
                include_terminal=include_terminal,
            )
        ]

    async def inspect_flow(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            flow = await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            return flow.to_view()

    async def add_nodes(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        nodes: Sequence[FlowNodeSpec],
        idempotency_key: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_worker_types(nodes)
        self._validate_external_context_refs(
            base_session_id=base_session_id,
            nodes=nodes,
        )
        async with self._lock(flow_id):
            await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="add_nodes",
                idempotency_key=idempotency_key,
                payload=[flow_node_spec_payload(node) for node in nodes],
                mutation=lambda current: add_flow_nodes(current, nodes),
                turn_id=turn_id,
            )
            return {**flow.to_view(), "changed": changed}

    async def advance_flow(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        base_turn_id: str,
        timeout_seconds: float | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Dispatch one frontier or a bounded predictable auto-advance chain."""

        async with self._lock(flow_id):
            loop = asyncio.get_running_loop()
            deadline = (
                loop.time() + timeout_seconds
                if timeout_seconds is not None
                else None
            )
            frontiers_executed: list[list[str]] = []
            all_launch_errors: list[dict[str, str]] = []
            stop_reason = "no_ready_frontier"

            while True:
                flow = await self._sync(
                    flow_id,
                    base_session_id=base_session_id,
                    turn_id=base_turn_id,
                )
                if flow.status is FlowStatus.PAUSED:
                    raise ValueError("resume the paused Flow before advancing it")
                if flow.status.terminal:
                    stop_reason = (
                        "max_rounds"
                        if flow.termination_reason == "Flow execution round limit reached"
                        else "flow_terminal"
                    )
                    break

                frontier = flow.starting_nodes()
                if not frontier:
                    flow = self.store.update_runtime(
                        flow_id,
                        base_session_id=base_session_id,
                        mutation=_claim_ready_frontier,
                        turn_id=base_turn_id,
                    )
                    if flow.status.terminal:
                        stop_reason = (
                            "max_rounds"
                            if flow.termination_reason
                            == "Flow execution round limit reached"
                            else "flow_terminal"
                        )
                        break
                    frontier = flow.starting_nodes()
                if not frontier:
                    stop_reason = "no_ready_frontier"
                    break

                frontier_ids = [node.node_id for node in frontier]
                frontiers_executed.append(frontier_ids)
                launch_errors = await self._launch_frontier(
                    flow,
                    frontier,
                    base_session_id=base_session_id,
                    turn_id=base_turn_id,
                    progress=progress,
                )
                all_launch_errors.extend(launch_errors)
                flow = await self._sync(
                    flow_id,
                    base_session_id=base_session_id,
                    turn_id=base_turn_id,
                )
                run_ids = _unique(
                    node.current_run_id
                    for node in flow.active_nodes()
                    if node.current_run_id is not None
                )
                if run_ids:
                    remaining_timeout = (
                        max(0.0, deadline - loop.time())
                        if deadline is not None
                        else None
                    )
                    await self.workers.await_workers(
                        run_ids,
                        timeout=remaining_timeout,
                        base_session_id=base_session_id,
                    )
                    flow = await self._sync(
                        flow_id,
                        base_session_id=base_session_id,
                        turn_id=base_turn_id,
                    )

                current_frontier = [flow.node(node_id) for node_id in frontier_ids]
                if launch_errors:
                    stop_reason = "launch_error"
                    break
                if any(node.status.active for node in current_frontier):
                    stop_reason = "active_or_timeout"
                    break
                if any(
                    node.status is FlowNodeStatus.WAITING_FOR_CONTEXT
                    for node in current_frontier
                ):
                    stop_reason = "waiting_for_context"
                    break
                if any(
                    node.status is not FlowNodeStatus.COMPLETED
                    for node in current_frontier
                ):
                    stop_reason = "frontier_non_success"
                    break
                if any(node.worker_type_id == "reviewer" for node in current_frontier):
                    stop_reason = "reviewer_frontier"
                    break
                if flow.advance_mode is FlowAdvanceMode.CHECKPOINTED:
                    stop_reason = "checkpointed"
                    break
                if len(frontiers_executed) >= flow.auto_advance_max_frontiers:
                    stop_reason = "auto_advance_limit"
                    break
                if deadline is not None and loop.time() >= deadline:
                    stop_reason = "active_or_timeout"
                    break

            auto_advanced_count = (
                max(0, len(frontiers_executed) - 1)
                if flow.advance_mode is FlowAdvanceMode.AUTO
                else 0
            )
            if flow.last_stop_reason != stop_reason or auto_advanced_count:
                flow = self.store.update_runtime(
                    flow_id,
                    base_session_id=base_session_id,
                    mutation=partial(
                        _record_advance_telemetry,
                        stop_reason=stop_reason,
                        auto_advanced_count=auto_advanced_count,
                    ),
                    turn_id=base_turn_id,
                )

            view = flow.to_view()
            view["frontier_node_ids"] = (
                frontiers_executed[-1] if frontiers_executed else []
            )
            view["frontiers_executed"] = frontiers_executed
            view["auto_advanced_count"] = auto_advanced_count
            view["stop_reason"] = stop_reason
            if all_launch_errors:
                view["launch_errors"] = all_launch_errors
            return view

    async def revise_node(
        self,
        flow_id: str,
        node_id: str,
        *,
        feedback: str,
        base_session_id: str,
        idempotency_key: str,
        fresh_worker: bool = False,
        fresh_reason: str | None = None,
        budget_increase: BudgetIncrease | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            current = await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            budget = self._resolve_budget_increase(
                current,
                node_id,
                budget_increase,
                base_session_id=base_session_id,
            )
            payload: dict[str, Any] = {"node_id": node_id, "feedback": feedback}
            if fresh_worker:
                payload["fresh_worker"] = True
            if fresh_reason is not None:
                payload["fresh_reason"] = fresh_reason
            if budget_increase is not None:
                payload["budget_increase"] = budget_increase.model_dump(mode="json")
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="revise_node",
                idempotency_key=idempotency_key,
                payload=payload,
                mutation=lambda current: revise_flow_node(
                    current,
                    node_id,
                    feedback,
                    fresh_worker=fresh_worker,
                    fresh_reason=fresh_reason,
                    budget=budget,
                ),
                turn_id=turn_id,
            )
            return {**flow.to_view(), "changed": changed}

    async def retry_node(
        self,
        flow_id: str,
        node_id: str,
        *,
        base_session_id: str,
        idempotency_key: str,
        fresh_worker: bool = False,
        fresh_reason: str | None = None,
        budget_increase: BudgetIncrease | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            current = await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            node = current.node(node_id)
            if (
                node.status is FlowNodeStatus.PARTIAL
                and budget_increase is None
                and not fresh_worker
                and node.current_run_id is not None
                and self.workers.flow_reuse_ineligibility_reason(
                    node.current_run_id,
                    flow_id=flow_id,
                    base_session_id=base_session_id,
                    worker_type_id=node.worker_type_id,
                )
                is None
            ):
                raise ValueError(
                    "retrying a reusable partial Flow node requires a Master-authored "
                    "budget_increase"
                )
            budget = self._resolve_budget_increase(
                current,
                node_id,
                budget_increase,
                base_session_id=base_session_id,
            )
            payload: dict[str, Any] = {"node_id": node_id}
            if fresh_worker:
                payload["fresh_worker"] = True
            if fresh_reason is not None:
                payload["fresh_reason"] = fresh_reason
            if budget_increase is not None:
                payload["budget_increase"] = budget_increase.model_dump(mode="json")
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="retry_node",
                idempotency_key=idempotency_key,
                payload=payload,
                mutation=lambda current: retry_flow_node(
                    current,
                    node_id,
                    fresh_worker=fresh_worker,
                    fresh_reason=fresh_reason,
                    budget=budget,
                ),
                turn_id=turn_id,
            )
            return {**flow.to_view(), "changed": changed}

    async def resume_node(
        self,
        flow_id: str,
        node_id: str,
        *,
        response: str,
        base_session_id: str,
        base_turn_id: str,
        idempotency_key: str,
        progress: Any | None = None,
        budget_increase: BudgetIncrease | None = None,
    ) -> dict[str, Any]:
        """Continue the exact waiting WorkerRun bound to one Flow node."""

        async with self._lock(flow_id):
            payload = {"node_id": node_id, "response": response}
            if budget_increase is not None:
                payload["budget_increase"] = budget_increase.model_dump(mode="json")
            flow, replayed = self.store.operation_state(
                flow_id,
                base_session_id=base_session_id,
                operation="resume_node",
                idempotency_key=idempotency_key,
                payload=payload,
                turn_id=base_turn_id,
            )
            if replayed:
                return flow.to_view()

            if flow.status is not FlowStatus.OPEN:
                raise ValueError("only an open Flow can resume a node")
            node = flow.node(node_id)
            if (
                node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT
                or node.current_run_id is None
            ):
                raise ValueError("resume_flow_node requires a waiting node")
            if len(node.runs) >= MAX_FLOW_RUN_BINDINGS:
                raise ValueError(
                    f"Flow node Run history limit reached ({MAX_FLOW_RUN_BINDINGS})"
                )
            source_run_id = node.current_run_id
            budget = (
                self.workers.increased_budget(
                    source_run_id,
                    budget_increase,
                    base_session_id=base_session_id,
                )
                if budget_increase is not None
                else self.workers.default_budget_for(node.worker_type_id)
            )
            worker_key = _resume_run_key(flow_id, node_id, source_run_id)
            related_contexts = self.workers.related_contexts_for_run(
                source_run_id,
                base_session_id=base_session_id,
            )
            resumed = self.workers.find_run_by_idempotency(
                base_session_id=base_session_id,
                idempotency_key=worker_key,
                objective=response,
                worker_id=node.worker_id,
                allowed_source_run_ids={source_run_id},
                budget=budget,
                related_contexts=related_contexts,
            )
            adopted_external = False
            if resumed is None:
                # Adopt a continuation created through another control surface
                # before trying to create a second one.
                external = self.workers.find_continuation(
                    source_run_id,
                    base_session_id=base_session_id,
                )
                if external is not None:
                    if not self.workers.run_has_objective(
                        str(external["run_id"]),
                        response,
                        base_session_id=base_session_id,
                    ):
                        self._attach_and_activate_run(
                            flow_id,
                            base_session_id=base_session_id,
                            node_id=node_id,
                            generation=node.generation,
                            attempt=node.attempt,
                            run_view=external,
                            turn_id=base_turn_id,
                            progress=progress,
                        )
                        await self._sync(
                            flow_id,
                            base_session_id=base_session_id,
                            turn_id=base_turn_id,
                        )
                        raise ValueError(
                            "the Flow node was already resumed elsewhere with a "
                            "different response; inspect the adopted continuation"
                        )
                    resumed = external
                    adopted_external = True
                else:
                    resumed = await self.workers._reserve_flow_resume(
                        source_run_id,
                        flow_id=flow_id,
                        response=response,
                        idempotency_key=worker_key,
                        base_session_id=base_session_id,
                        progress=progress,
                        budget_increase=budget_increase,
                    )

            def attach(current: MasterFlow) -> None:
                _attach_run_to_flow(
                    current,
                    node_id=node_id,
                    generation=node.generation,
                    attempt=node.attempt,
                    run_view=resumed,
                )

            try:
                with self._activation_fence(flow_id):
                    committed, _ = self.store.mutate(
                        flow_id,
                        base_session_id=base_session_id,
                        operation="resume_node",
                        idempotency_key=idempotency_key,
                        payload=payload,
                        mutation=attach,
                        turn_id=base_turn_id,
                    )
                    self._activate_current_bindings(
                        committed,
                        run_ids=[str(resumed["run_id"])],
                        base_session_id=base_session_id,
                        progress=progress,
                    )
            except Exception:
                if self._resume_is_obsolete(
                    flow_id,
                    base_session_id=base_session_id,
                    node_id=node_id,
                    generation=node.generation,
                    attempt=node.attempt,
                    source_run_id=source_run_id,
                    resumed_run_id=str(resumed["run_id"]),
                ):
                    await self.workers.cancel_worker(
                        str(resumed["run_id"]),
                        base_session_id=base_session_id,
                    )
                raise
            flow = await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=base_turn_id,
            )
            view = flow.to_view()
            if adopted_external:
                view["adopted_external_continuation"] = True
            return view

    async def skip_node(
        self,
        flow_id: str,
        node_id: str,
        *,
        reason: str,
        base_session_id: str,
        idempotency_key: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="skip_node",
                idempotency_key=idempotency_key,
                payload={"node_id": node_id, "reason": reason},
                mutation=lambda current: skip_flow_node(current, node_id, reason),
                turn_id=turn_id,
            )
            return {**flow.to_view(), "changed": changed}

    async def pause(
        self,
        flow_id: str,
        *,
        reason: str,
        base_session_id: str,
        idempotency_key: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="pause",
                idempotency_key=idempotency_key,
                payload={"reason": reason},
                mutation=lambda current: pause_flow(current, reason),
                turn_id=turn_id,
            )
            return {**flow.to_view(), "changed": changed}

    async def resume(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        idempotency_key: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="resume",
                idempotency_key=idempotency_key,
                payload={},
                mutation=reopen_flow,
                turn_id=turn_id,
            )
            return {**flow.to_view(), "changed": changed}

    async def complete(
        self,
        flow_id: str,
        *,
        outcome: FlowCompletion,
        summary: str,
        base_session_id: str,
        idempotency_key: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist one explicit Flow terminal decision."""

        async with self._lock(flow_id):
            current = await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            waiting_run_ids: list[str] = []
            for node in current.nodes:
                if node.current_run_id is None:
                    continue
                try:
                    run = self.workers.manager.store.get_run(node.current_run_id)
                except KeyError:
                    continue
                if run.status is WorkerRunStatus.WAITING_FOR_CONTEXT:
                    waiting_run_ids.append(run.run_id)
            flow, changed = self.store.mutate(
                flow_id,
                base_session_id=base_session_id,
                operation="complete",
                idempotency_key=idempotency_key,
                payload={"outcome": outcome.value, "summary": summary},
                mutation=lambda current: finish_flow(current, outcome, summary),
                turn_id=turn_id,
            )
            if outcome is not FlowCompletion.COMPLETED and waiting_run_ids:
                await asyncio.gather(
                    *(
                        self.workers.cancel_worker(
                            run_id,
                            base_session_id=base_session_id,
                        )
                        for run_id in waiting_run_ids
                    )
                )
            return {**flow.to_view(), "changed": changed}

    async def finish_turn(
        self,
        final_content: str,
        *,
        base_session_id: str,
        turn_id: str | None = None,
    ) -> str:
        """Return terminal text only when no open Flow can be abandoned."""

        await self.reconcile_legacy_runs(base_session_id)
        self.store.seal_session_if_quiescent(
            base_session_id,
            turn_id=turn_id,
        )
        return final_content

    async def cancel(
        self,
        flow_id: str,
        *,
        reason: str,
        base_session_id: str,
        idempotency_key: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock(flow_id):
            # Cancellation must never run the normal synchronization path before
            # its intent is durable.  A Flow can crash after attaching a reserved
            # WorkerRun but before activating it; normal sync would activate that
            # inert reservation and let a detached runner begin work solely because
            # cancellation was requested.
            flow = self.store.get_flow(flow_id)
            if flow.base_session_id != base_session_id:
                raise PermissionError("Flow belongs to a different Master session")
            recovered_run_ids = self._recover_cancellable_run_ids(
                flow,
                base_session_id=base_session_id,
            )
            with self._activation_fence(flow_id):
                flow = self.store.get_flow(flow_id, base_session_id=base_session_id)
                run_ids = _unique(
                    [
                        *recovered_run_ids,
                        *flow.cancellation_run_ids,
                        *(
                            node.current_run_id
                            for node in flow.nodes
                            if node.current_run_id is not None
                        ),
                    ]
                )
                flow, changed = self.store.mutate(
                    flow_id,
                    base_session_id=base_session_id,
                    operation="cancel",
                    idempotency_key=idempotency_key,
                    payload={"reason": reason},
                    mutation=lambda current: cancel_flow_state(
                        current,
                        reason,
                        run_ids=run_ids,
                    ),
                    turn_id=turn_id,
                )
                # This is a synchronous durable fence only. Executor teardown is
                # awaited below, after releasing the cross-controller lock.
                for run_id in run_ids:
                    if not _worker_run_exists(self.workers, run_id):
                        # Missing durable Worker state is reconciled below as an
                        # explicit unknown outcome; cancellation still commits.
                        continue
                    self.workers._fence_flow_run(
                        run_id,
                        flow_id=flow_id,
                        base_session_id=base_session_id,
                    )
            existing_run_ids = [
                run_id
                for run_id in run_ids
                if _worker_run_exists(self.workers, run_id)
            ]
            if existing_run_ids:
                await asyncio.gather(
                    *(
                        self.workers.cancel_worker(
                            run_id,
                            base_session_id=base_session_id,
                        )
                        for run_id in existing_run_ids
                    )
                )
            flow = await self._sync(
                flow_id,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
            if flow.status.terminal:
                # Catch an inert reservation created from a stale launch after
                # cancellation's pre-commit scan. It can never be recovered by
                # a terminal Flow and must not pin its WorkerSession forever.
                await self.reconcile_legacy_runs(base_session_id)
            return {**flow.to_view(), "changed": changed}

    def open_flow_ids(self, base_session_id: str) -> list[str]:
        return [
            flow.flow_id
            for flow in self.store.list_flows(base_session_id, include_terminal=False)
            if flow.status in {FlowStatus.OPEN, FlowStatus.CANCELLING}
        ]

    async def reconcile_legacy_runs(
        self,
        base_session_id: str,
        *,
        wait: bool = False,
    ) -> list[str]:
        """Fence historical Flow Runs that are live without a safe binding."""

        flows = {
            flow.flow_id: flow
            for flow in self.store.list_flows(
                base_session_id,
                include_terminal=True,
            )
        }
        safe_run_ids: dict[str, set[str]] = {}
        for flow_id, flow in flows.items():
            if flow.status not in {
                FlowStatus.OPEN,
                FlowStatus.PAUSED,
                FlowStatus.CANCELLING,
            }:
                continue
            safe_run_ids[flow_id] = {
                *flow.cancellation_run_ids,
                *(
                    node.current_run_id
                    for node in flow.nodes
                    if node.current_run_id is not None
                ),
            }

        unsafe_run_ids: list[str] = []
        for worker in self.workers.manager.store.list_workers(base_session_id):
            worker_runs = self.workers.manager.store.list_runs(worker.worker_id)
            for run in worker_runs:
                if run.status not in {
                    WorkerRunStatus.QUEUED,
                    WorkerRunStatus.RUNNING,
                    WorkerRunStatus.WAITING_FOR_CONTEXT,
                }:
                    continue
                if not run.base_turn_id or not run.base_turn_id.startswith("flow:"):
                    continue
                flow_id = run.base_turn_id.removeprefix("flow:")
                flow = flows.get(flow_id)
                if run.status is WorkerRunStatus.WAITING_FOR_CONTEXT:
                    current_binding = bool(
                        flow is not None
                        and any(
                            node.current_run_id == run.run_id for node in flow.nodes
                        )
                    )
                    historical_binding = bool(
                        flow is not None
                        and any(
                            binding.run_id == run.run_id
                            for node in flow.nodes
                            for binding in node.runs
                        )
                    )
                    if current_binding and flow is not None and flow.status in {
                        FlowStatus.OPEN,
                        FlowStatus.PAUSED,
                    }:
                        continue
                    continued_historical_binding = historical_binding and (
                        any(
                            binding.source_run_id == run.run_id
                            for node in flow.nodes
                            for binding in node.runs
                        )
                        if flow is not None
                        else False
                    )
                    continued_historical_run = any(
                        candidate.source_run_id == run.run_id
                        for candidate in worker_runs
                    )
                    if not current_binding and (
                        continued_historical_binding or continued_historical_run
                    ):
                        # A genuinely resumed source remains WAITING forever as
                        # immutable audit history; only its continuation changes
                        # state. A source merely abandoned by skip has no direct
                        # continuation and must be fenced below.
                        continue
                    unsafe_run_ids.append(run.run_id)
                    continue
                if run.status is WorkerRunStatus.QUEUED and run.activated_at is None:
                    # Only the exact current epoch of an open Flow may recover an
                    # inert reserve-before-attach Run. Terminal, paused, stale,
                    # and superseded reservations are safe to fence.
                    recoverable = bool(
                        flow is not None
                        and flow.status is FlowStatus.OPEN
                        and any(
                            (
                                node.status is FlowNodeStatus.STARTING
                                and worker.snapshot.id == node.worker_type_id
                                and run.idempotency_key == _node_run_key(flow_id, node)
                                and run.context.objective == node.execution_objective()
                                and (
                                    run.source_run_id is None
                                    or run.source_run_id == node.reuse_source_run_id
                                )
                            )
                            or (
                                node.status is FlowNodeStatus.WAITING_FOR_CONTEXT
                                and node.current_run_id == run.source_run_id
                                and node.worker_id == run.worker_id
                            )
                            for node in flow.nodes
                        )
                    )
                    if recoverable:
                        continue
                if run.run_id not in safe_run_ids.get(flow_id, set()):
                    unsafe_run_ids.append(run.run_id)

        if not unsafe_run_ids:
            return []
        await asyncio.gather(
            *(
                self.workers.cancel_worker(
                    run_id,
                    base_session_id=base_session_id,
                )
                for run_id in unsafe_run_ids
            )
        )
        views = await self.workers.await_workers(
            unsafe_run_ids,
            timeout=None if wait else 0,
            base_session_id=base_session_id,
        )
        unsettled = [
            str(view["run_id"])
            for view in views
            if str(view["status"]) in {"queued", "running"}
        ]
        if unsettled:
            raise ValueError(
                "cannot finish while legacy unbound Flow Runs are still tearing down: "
                f"{unsettled}"
            )
        status_by_run_id = {
            str(view["run_id"]): str(view["status"])
            for view in views
        }
        for flow in flows.values():
            if not any(
                binding.run_id in status_by_run_id
                for node in flow.nodes
                for binding in node.runs
            ):
                continue

            def project_binding_statuses(current: MasterFlow) -> None:
                for node in current.nodes:
                    for binding in node.runs:
                        status = status_by_run_id.get(binding.run_id)
                        if status is not None:
                            binding.status = status

            self.store.update_runtime(
                flow.flow_id,
                base_session_id=base_session_id,
                mutation=project_binding_statuses,
            )
        return unsafe_run_ids

    async def _launch_frontier(
        self,
        flow: MasterFlow,
        frontier: Sequence[FlowNode],
        *,
        base_session_id: str,
        turn_id: str | None,
        progress: Any | None,
    ) -> list[dict[str, str]]:
        if not frontier:
            return []

        async def launch(node: FlowNode) -> tuple[FlowNode, dict[str, Any]]:
            idempotency_key = _node_run_key(flow.flow_id, node)
            source_run_id = node.reuse_source_run_id
            budget = node.pending_budget or self.workers.default_budget_for(
                node.worker_type_id
            )
            related_contexts = self._resolve_related_contexts(
                flow,
                node,
                base_session_id=base_session_id,
            )
            allowed_sources = {None}
            if source_run_id is not None:
                allowed_sources.add(source_run_id)
            result = self.workers.find_run_by_idempotency(
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                objective=node.execution_objective(),
                worker_type_id=node.worker_type_id,
                allowed_source_run_ids=allowed_sources,
                budget=budget,
                related_contexts=related_contexts,
            )
            if result is not None:
                _annotate_recovered_session_decision(
                    node,
                    result,
                    fallback_reason=(
                        self.workers.flow_reuse_ineligibility_reason(
                            source_run_id,
                            flow_id=flow.flow_id,
                            base_session_id=base_session_id,
                            worker_type_id=node.worker_type_id,
                        )
                        if source_run_id is not None
                        else None
                    ),
                )
                return node, result

            fresh_reason = _requested_fresh_reason(node)
            if source_run_id is not None and fresh_reason is None:
                unavailable = self.workers.flow_reuse_ineligibility_reason(
                    source_run_id,
                    flow_id=flow.flow_id,
                    base_session_id=base_session_id,
                    worker_type_id=node.worker_type_id,
                )
                if unavailable is None:
                    try:
                        result = await self.workers._reserve_flow_reuse(
                            source_run_id,
                            flow_id=flow.flow_id,
                            objective=node.execution_objective(),
                            idempotency_key=idempotency_key,
                            base_session_id=base_session_id,
                            worker_type_id=node.worker_type_id,
                            progress=progress,
                            budget=budget,
                            related_contexts=related_contexts,
                        )
                    except (KeyError, ValueError):
                        # A second controller may have resolved this epoch while
                        # the exact-source transaction was acquiring its lock.
                        result = self.workers.find_run_by_idempotency(
                            base_session_id=base_session_id,
                            idempotency_key=idempotency_key,
                            objective=node.execution_objective(),
                            worker_type_id=node.worker_type_id,
                            allowed_source_run_ids=allowed_sources,
                            budget=budget,
                            related_contexts=related_contexts,
                        )
                        if result is not None:
                            unavailable = self.workers.flow_reuse_ineligibility_reason(
                                source_run_id,
                                flow_id=flow.flow_id,
                                base_session_id=base_session_id,
                                worker_type_id=node.worker_type_id,
                            )
                            _annotate_recovered_session_decision(
                                node,
                                result,
                                fallback_reason=unavailable,
                            )
                            return node, result
                        unavailable = self.workers.flow_reuse_ineligibility_reason(
                            source_run_id,
                            flow_id=flow.flow_id,
                            base_session_id=base_session_id,
                            worker_type_id=node.worker_type_id,
                        )
                        if unavailable is None:
                            raise
                    else:
                        result["worker_session_action"] = WorkerSessionAction.REUSE.value
                        result["worker_session_reason"] = _reuse_session_reason(node)
                        return node, result
                fresh_reason = unavailable
            if result is None:
                # Recheck after resolving reuse eligibility. Another controller
                # may have reserved this stable epoch between the first lookup
                # and an apparent worker_context_advanced decision.
                result = self.workers.find_run_by_idempotency(
                    base_session_id=base_session_id,
                    idempotency_key=idempotency_key,
                    objective=node.execution_objective(),
                    worker_type_id=node.worker_type_id,
                    allowed_source_run_ids=allowed_sources,
                    budget=budget,
                    related_contexts=related_contexts,
                )
            if result is not None:
                _annotate_recovered_session_decision(
                    node,
                    result,
                    fallback_reason=fresh_reason,
                )
                return node, result
            if result is None:
                result = await self.workers.spawn_worker(
                    base_session_id=base_session_id,
                    base_turn_id=_flow_turn_id(flow.flow_id),
                    worker_type_id=node.worker_type_id,
                    objective=node.execution_objective(),
                    idempotency_key=idempotency_key,
                    progress=progress,
                    start=False,
                    budget=budget,
                    related_contexts=related_contexts,
                )
                result["worker_session_action"] = WorkerSessionAction.NEW.value
                result["worker_session_reason"] = fresh_reason or "first_run"
            return node, result

        results = await asyncio.gather(
            *(launch(node) for node in frontier),
            return_exceptions=True,
        )
        errors: list[dict[str, str]] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(
                    {
                        "error": f"{type(result).__name__}: {result}",
                        "recovery": "call advance_flow again; the stable node key is reusable",
                    }
                )
                continue
            node, run_view = result
            try:
                self._attach_and_activate_run(
                    flow.flow_id,
                    base_session_id=base_session_id,
                    node_id=node.node_id,
                    generation=node.generation,
                    attempt=node.attempt,
                    run_view=run_view,
                    turn_id=turn_id,
                    progress=progress,
                )
            except Exception:
                # A transient Flow-store failure leaves a recoverable WorkerRun:
                # the stable key lets the next advance attach that exact run.
                # Cancel only when a concurrent Flow mutation made this launch
                # definitively obsolete (for example cancellation or revision).
                if self._launch_is_obsolete(
                    flow.flow_id,
                    base_session_id=base_session_id,
                    node_id=node.node_id,
                    generation=node.generation,
                    attempt=node.attempt,
                    run_id=str(run_view["run_id"]),
                ):
                    await self.workers.cancel_worker(
                        str(run_view["run_id"]),
                        base_session_id=base_session_id,
                    )
                raise
        return errors

    async def _sync(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        turn_id: str | None,
    ) -> MasterFlow:
        flow = self.store.get_flow(flow_id, base_session_id=base_session_id)
        if flow.status is FlowStatus.CANCELLING:
            return await self._sync_cancelling(
                flow,
                base_session_id=base_session_id,
                turn_id=turn_id,
            )
        if flow.status is not FlowStatus.OPEN:
            return flow
        missing_run_ids = self._missing_current_run_ids(flow)
        if missing_run_ids:
            flow = self.store.update_runtime(
                flow_id,
                base_session_id=base_session_id,
                mutation=partial(
                    _fail_nodes_with_missing_runs,
                    missing_run_ids=frozenset(missing_run_ids),
                ),
                turn_id=turn_id,
            )
        while flow.status is FlowStatus.OPEN:
            adopted = False
            for node in flow.nodes:
                if (
                    node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT
                    or node.current_run_id is None
                ):
                    continue
                continuation = self.workers.find_continuation(
                    node.current_run_id,
                    base_session_id=base_session_id,
                )
                if continuation is None:
                    continue
                if len(node.runs) >= MAX_FLOW_RUN_BINDINGS:
                    await self.workers.cancel_worker(
                        str(continuation["run_id"]),
                        base_session_id=base_session_id,
                    )
                    flow = self.store.update_runtime(
                        flow_id,
                        base_session_id=base_session_id,
                        mutation=partial(
                            _fail_node_for_binding_limit,
                            node_id=node.node_id,
                            source_run_id=node.current_run_id,
                        ),
                        turn_id=turn_id,
                    )
                    adopted = True
                    break
                flow = self._attach_and_activate_run(
                    flow_id,
                    base_session_id=base_session_id,
                    node_id=node.node_id,
                    generation=node.generation,
                    attempt=node.attempt,
                    run_view=continuation,
                    turn_id=turn_id,
                    progress=None,
                )
                adopted = True
                break
            if not adopted:
                break
        run_ids = _unique(
            node.current_run_id
            for node in flow.nodes
            if node.current_run_id is not None
            and node.status
            in {
                FlowNodeStatus.STARTING,
                FlowNodeStatus.RUNNING,
                FlowNodeStatus.WAITING_FOR_CONTEXT,
            }
        )
        if not run_ids:
            return flow
        active_run_ids = _unique(
            node.current_run_id
            for node in flow.nodes
            if node.current_run_id is not None and node.status.active
        )
        self._activate_bound_runs(
            flow_id,
            base_session_id=base_session_id,
            run_ids=active_run_ids,
        )
        views = await self.workers.await_workers(
            run_ids,
            timeout=0,
            base_session_id=base_session_id,
        )
        by_run_id = {str(view["run_id"]): view for view in views}

        def synchronize(current: MasterFlow) -> None:
            if current.status is not FlowStatus.OPEN:
                return
            for node in current.nodes:
                if node.current_run_id is None or node.status not in {
                    FlowNodeStatus.STARTING,
                    FlowNodeStatus.RUNNING,
                    FlowNodeStatus.WAITING_FOR_CONTEXT,
                }:
                    continue
                view = by_run_id.get(node.current_run_id)
                if view is None:
                    continue
                binding = next(
                    (
                        candidate
                        for candidate in reversed(node.runs)
                        if candidate.run_id == node.current_run_id
                    ),
                    None,
                )
                if (
                    binding is None
                    or binding.generation != node.generation
                    or binding.attempt != node.attempt
                ):
                    continue
                node.status = _node_status(str(view["status"]))
                node.result = dict(view)
                binding.status = str(view["status"])
                binding.telemetry = flow_run_telemetry_payload(view)

        return self.store.update_runtime(
            flow_id,
            base_session_id=base_session_id,
            mutation=synchronize,
            turn_id=turn_id,
        )

    async def _sync_cancelling(
        self,
        flow: MasterFlow,
        *,
        base_session_id: str,
        turn_id: str | None,
    ) -> MasterFlow:
        run_ids = _unique(
            [
                *flow.cancellation_run_ids,
                *(
                    node.current_run_id
                    for node in flow.active_nodes()
                    if node.current_run_id is not None
                ),
            ]
        )
        existing_run_ids: list[str] = []
        missing_run_ids: list[str] = []
        for run_id in run_ids:
            try:
                self.workers.manager.store.get_run(run_id)
            except (KeyError, ValueError):
                missing_run_ids.append(run_id)
            else:
                existing_run_ids.append(run_id)
        if existing_run_ids:
            await asyncio.gather(
                *(
                    self.workers.cancel_worker(
                        run_id,
                        base_session_id=base_session_id,
                    )
                    for run_id in existing_run_ids
                )
            )
        views = (
            await self.workers.await_workers(
                existing_run_ids,
                timeout=0,
                base_session_id=base_session_id,
            )
            if existing_run_ids
            else []
        )
        views.extend(_missing_run_view(run_id) for run_id in missing_run_ids)
        by_run_id = {str(view["run_id"]): view for view in views}

        def reconcile(current: MasterFlow) -> None:
            if current.status is not FlowStatus.CANCELLING:
                return
            for node in current.nodes:
                if node.current_run_id is None:
                    if node.status.active:
                        node.status = FlowNodeStatus.CANCELLED
                    continue
                view = by_run_id.get(node.current_run_id)
                if view is None:
                    continue
                node.status = _node_status(str(view["status"]))
                node.result = dict(view)
                for binding in reversed(node.runs):
                    if binding.run_id == node.current_run_id:
                        binding.status = str(view["status"])
                        binding.telemetry = flow_run_telemetry_payload(view)
                        break
            unseen_tracked = set(current.cancellation_run_ids) - set(by_run_id)
            tracked_unsettled = bool(unseen_tracked) or any(
                str(view["status"]) in {"queued", "running"} for view in views
            )
            if not current.active_nodes() and not tracked_unsettled:
                unknown_runs = [
                    view
                    for view in views
                    if str(view["status"]) == WorkerRunStatus.FAILED.value
                    and view.get("tool_outcome") == "unknown"
                ]
                if unknown_runs:
                    count = len(unknown_runs)
                    summary = (
                        "Cancellation outcome is unknown because "
                        f"{count} tracked WorkerRun{'s' if count != 1 else ''} failed "
                        "with an unknown tool outcome. "
                        "External or descendant side effects may still be running or may "
                        "already have completed; inspect side effects before retrying."
                    )
                    current.status = FlowStatus.BLOCKED
                    current.completion_summary = summary
                    current.termination_reason = summary
                else:
                    current.status = FlowStatus.CANCELLED

        return self.store.update_runtime(
            flow.flow_id,
            base_session_id=base_session_id,
            mutation=reconcile,
            turn_id=turn_id,
        )

    def _missing_current_run_ids(self, flow: MasterFlow) -> list[str]:
        missing: list[str] = []
        for node in flow.nodes:
            if node.current_run_id is None or node.status not in {
                FlowNodeStatus.STARTING,
                FlowNodeStatus.RUNNING,
                FlowNodeStatus.WAITING_FOR_CONTEXT,
            }:
                continue
            try:
                self.workers.manager.store.get_run(node.current_run_id)
            except (KeyError, ValueError):
                missing.append(node.current_run_id)
        return _unique(missing)

    def _attach_run(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        node_id: str,
        generation: int,
        attempt: int,
        run_view: dict[str, Any],
        turn_id: str | None = None,
    ) -> MasterFlow:
        def attach(flow: MasterFlow) -> None:
            _attach_run_to_flow(
                flow,
                node_id=node_id,
                generation=generation,
                attempt=attempt,
                run_view=run_view,
            )

        return self.store.update_runtime(
            flow_id,
            base_session_id=base_session_id,
            mutation=attach,
            turn_id=turn_id,
        )

    def _attach_and_activate_run(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        node_id: str,
        generation: int,
        attempt: int,
        run_view: dict[str, Any],
        turn_id: str | None,
        progress: Any | None,
    ) -> MasterFlow:
        """Linearize binding and activation against durable cancellation intent."""

        with self._activation_fence(flow_id):
            flow = self._attach_run(
                flow_id,
                base_session_id=base_session_id,
                node_id=node_id,
                generation=generation,
                attempt=attempt,
                run_view=run_view,
                turn_id=turn_id,
            )
            self._activate_current_bindings(
                flow,
                run_ids=[str(run_view["run_id"])],
                base_session_id=base_session_id,
                progress=progress,
            )
            return flow

    def _activate_bound_runs(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        run_ids: Sequence[str | None],
        progress: Any | None = None,
    ) -> list[str]:
        """Activate only bindings that remain current after acquiring the fence."""

        with self._activation_fence(flow_id):
            flow = self.store.get_flow(flow_id, base_session_id=base_session_id)
            return self._activate_current_bindings(
                flow,
                run_ids=run_ids,
                base_session_id=base_session_id,
                progress=progress,
            )

    def _activate_current_bindings(
        self,
        flow: MasterFlow,
        *,
        run_ids: Sequence[str | None],
        base_session_id: str,
        progress: Any | None,
    ) -> list[str]:
        if flow.status is not FlowStatus.OPEN:
            return []
        requested = {run_id for run_id in run_ids if run_id is not None}
        current = _unique(
            node.current_run_id
            for node in flow.nodes
            if node.current_run_id in requested and node.status.active
        )
        for run_id in current:
            self.workers.start_worker_run(
                run_id,
                base_session_id=base_session_id,
                progress=progress,
            )
        return current

    def _activation_fence(self, flow_id: str) -> Any:
        """Serialize the short cross-database activation/cancellation boundary."""

        return flow_activation_fence(self.store.path.parent, flow_id)

    def _recover_cancellable_run_ids(
        self,
        flow: MasterFlow,
        *,
        base_session_id: str,
    ) -> list[str]:
        run_ids = [
            node.current_run_id
            for node in flow.nodes
            if node.current_run_id is not None
        ]
        for node in flow.nodes:
            if node.current_run_id is not None or node.attempt < 1:
                continue
            recovered = self.workers.find_run_by_idempotency(
                base_session_id=base_session_id,
                idempotency_key=_node_run_key(flow.flow_id, node),
                objective=node.execution_objective(),
                worker_type_id=node.worker_type_id,
            )
            if recovered is not None:
                run_ids.append(str(recovered["run_id"]))
        return _unique(run_ids)

    def _validate_worker_types(self, nodes: Sequence[FlowNodeSpec]) -> None:
        for node in nodes:
            self.workers.worker_types.get(node.worker_type_id)

    def _validate_external_context_refs(
        self,
        *,
        base_session_id: str,
        nodes: Sequence[FlowNodeSpec],
    ) -> None:
        """Fail before persisting when an external Run association is inaccessible."""

        for node in nodes:
            for reference in node.context_refs:
                if reference.kind != "worker_run":
                    continue
                self.workers.related_context(
                    reference.id,
                    base_session_id=base_session_id,
                    source_kind=reference.kind,
                    source_id=reference.id,
                    relation=reference.relation,
                    include=reference.include,
                )

    def _resolve_related_contexts(
        self,
        flow: MasterFlow,
        node: FlowNode,
        *,
        base_session_id: str,
    ) -> tuple[RelatedWorkerContext, ...]:
        """Resolve explicit associations without turning dependencies into context pipes."""

        related: list[RelatedWorkerContext] = []
        for reference in node.context_refs:
            if reference.kind == "worker_run":
                run_id = reference.id
            else:
                source_node = flow.node(reference.id)
                if source_node.current_run_id is None:
                    raise ValueError(
                        f"related Flow node {reference.id!r} has no current WorkerRun"
                    )
                run_id = source_node.current_run_id
            related.append(
                self.workers.related_context(
                    run_id,
                    base_session_id=base_session_id,
                    source_kind=reference.kind,
                    source_id=reference.id,
                    relation=reference.relation,
                    include=reference.include,
                )
            )
        return tuple(related)

    def _launch_is_obsolete(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        node_id: str,
        generation: int,
        attempt: int,
        run_id: str,
    ) -> bool:
        """Return true only when durable Flow state rejects this launch epoch."""

        try:
            current = self.store.get_flow(
                flow_id,
                base_session_id=base_session_id,
            )
            node = current.node(node_id)
        except Exception:
            # Failure to inspect is not evidence that side-effecting work became
            # obsolete. Leave the run recoverable under its stable key.
            return False
        if (
            current.status is FlowStatus.OPEN
            and node.generation == generation
            and node.attempt == attempt
            and node.current_run_id == run_id
            and node.status.active
        ):
            return False
        return (
            current.status is not FlowStatus.OPEN
            or node.generation != generation
            or node.attempt != attempt
            or node.status is not FlowNodeStatus.STARTING
        )

    def _resume_is_obsolete(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        node_id: str,
        generation: int,
        attempt: int,
        source_run_id: str,
        resumed_run_id: str,
    ) -> bool:
        """Return true only when durable state rejects a continuation epoch."""

        try:
            current = self.store.get_flow(
                flow_id,
                base_session_id=base_session_id,
            )
            node = current.node(node_id)
        except Exception:
            return False
        if node.current_run_id == resumed_run_id:
            return False
        return (
            current.status is not FlowStatus.OPEN
            or node.generation != generation
            or node.attempt != attempt
            or node.status is not FlowNodeStatus.WAITING_FOR_CONTEXT
            or node.current_run_id != source_run_id
        )

    def _resolve_budget_increase(
        self,
        flow: MasterFlow,
        node_id: str,
        increase: BudgetIncrease | None,
        *,
        base_session_id: str,
    ) -> BudgetGrant | None:
        if increase is None:
            return None
        node = flow.node(node_id)
        source_run_id = node.current_run_id
        if source_run_id is None and node.runs:
            source_run_id = node.runs[-1].run_id
        if source_run_id is None:
            raise ValueError("cannot increase budget before a Flow node has run")
        return self.workers.increased_budget(
            source_run_id,
            increase,
            base_session_id=base_session_id,
        )

    def _lock(self, flow_id: str) -> asyncio.Lock:
        return self._locks.setdefault(flow_id, asyncio.Lock())


def _record_advance_telemetry(
    flow: MasterFlow,
    *,
    stop_reason: str,
    auto_advanced_count: int,
) -> None:
    flow.last_stop_reason = stop_reason[:128]
    flow.auto_advanced_frontiers += max(0, auto_advanced_count)


__all__ = ["FlowControlService"]
