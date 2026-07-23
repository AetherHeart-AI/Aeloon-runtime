"""Master-only tools for authoring and advancing dynamic Flows."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeloon_core.flow_control import FlowControlService
from aeloon_core.flows import FlowCompletion, FlowNodeSpec
from aeloon_core.tools.base import FunctionTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.worker_sessions import BudgetIncrease


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _NoArgs(_Args):
    pass


class _FlowId(_Args):
    flow_id: str = Field(min_length=1, max_length=64)


class _FlowMutation(_FlowId):
    idempotency_key: str = Field(min_length=1, max_length=256)


class _CreateFlowArgs(_Args):
    goal: str = Field(min_length=1, max_length=16_000)
    nodes: list[FlowNodeSpec] = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=256)
    max_nodes: int = Field(default=64, ge=1, le=256)
    max_rounds: int = Field(default=12, ge=1, le=64)


class _ListFlowsArgs(_Args):
    include_terminal: bool = False


class _AddNodesArgs(_FlowMutation):
    nodes: list[FlowNodeSpec] = Field(min_length=1, max_length=64)


class _AdvanceArgs(_FlowId):
    timeout_seconds: float | None = Field(default=None, ge=0)


class _NodeMutation(_FlowMutation):
    node_id: str = Field(min_length=1, max_length=64)


class _RerunNodeArgs(_NodeMutation):
    fresh_worker: bool = False
    fresh_reason: str | None = Field(default=None, min_length=1, max_length=1_000)
    budget_increase: BudgetIncrease | None = None

    @model_validator(mode="after")
    def _fresh_reason_matches_flag(self) -> _RerunNodeArgs:
        if self.fresh_worker and self.fresh_reason is None:
            raise ValueError("fresh_worker=true requires fresh_reason")
        if not self.fresh_worker and self.fresh_reason is not None:
            raise ValueError("fresh_reason requires fresh_worker=true")
        return self


class _ReviseNodeArgs(_RerunNodeArgs):
    feedback: str = Field(min_length=1, max_length=8_000)


class _RetryNodeArgs(_RerunNodeArgs):
    pass


class _ResumeNodeArgs(_NodeMutation):
    response: str = Field(min_length=1, max_length=32_000)
    budget_increase: BudgetIncrease | None = None


class _SkipNodeArgs(_NodeMutation):
    reason: str = Field(min_length=1, max_length=4_000)


class _PauseFlowArgs(_FlowMutation):
    reason: str = Field(min_length=1, max_length=4_000)


class _CompleteFlowArgs(_FlowMutation):
    outcome: FlowCompletion
    summary: str = Field(min_length=1, max_length=16_000)


class FinishTurnArgs(_Args):
    final_content: str = Field(min_length=1, max_length=64_000)


class _CancelFlowArgs(_FlowMutation):
    reason: str = Field(min_length=1, max_length=4_000)


def build_master_flow_tools(
    *,
    control: FlowControlService,
    base_session_id: str,
    base_turn_id: str,
    on_progress: Any | None = None,
) -> ToolRegistry:
    """Return session-scoped Flow authoring and frontier execution tools."""

    registry = ToolRegistry()

    async def create_flow(
        goal: str,
        nodes: list[dict[str, Any]],
        idempotency_key: str,
        max_nodes: int = 64,
        max_rounds: int = 12,
    ) -> str:
        return _json(
            control.create_flow(
                base_session_id=base_session_id,
                goal=goal,
                nodes=_node_specs(nodes),
                idempotency_key=idempotency_key,
                max_nodes=max_nodes,
                max_rounds=max_rounds,
                turn_id=base_turn_id,
            )
        )

    async def list_flows(include_terminal: bool = False) -> str:
        return _json(
            control.list_flows(
                base_session_id,
                include_terminal=include_terminal,
            )
        )

    async def inspect_flow(flow_id: str) -> str:
        return _json(
            await control.inspect_flow(
                flow_id,
                base_session_id=base_session_id,
                turn_id=base_turn_id,
            )
        )

    async def add_flow_nodes(
        flow_id: str,
        nodes: list[dict[str, Any]],
        idempotency_key: str,
    ) -> str:
        return _json(
            await control.add_nodes(
                flow_id,
                base_session_id=base_session_id,
                nodes=_node_specs(nodes),
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def advance_flow(
        flow_id: str,
        timeout_seconds: float | None = None,
    ) -> str:
        return _json(
            await control.advance_flow(
                flow_id,
                base_session_id=base_session_id,
                base_turn_id=base_turn_id,
                timeout_seconds=timeout_seconds,
                progress=on_progress,
            )
        )

    async def revise_flow_node(
        flow_id: str,
        node_id: str,
        feedback: str,
        idempotency_key: str,
        fresh_worker: bool = False,
        fresh_reason: str | None = None,
        budget_increase: dict[str, Any] | None = None,
    ) -> str:
        return _json(
            await control.revise_node(
                flow_id,
                node_id,
                feedback=feedback,
                fresh_worker=fresh_worker,
                fresh_reason=fresh_reason,
                budget_increase=(
                    BudgetIncrease.model_validate(budget_increase)
                    if budget_increase is not None
                    else None
                ),
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def retry_flow_node(
        flow_id: str,
        node_id: str,
        idempotency_key: str,
        fresh_worker: bool = False,
        fresh_reason: str | None = None,
        budget_increase: dict[str, Any] | None = None,
    ) -> str:
        return _json(
            await control.retry_node(
                flow_id,
                node_id,
                fresh_worker=fresh_worker,
                fresh_reason=fresh_reason,
                budget_increase=(
                    BudgetIncrease.model_validate(budget_increase)
                    if budget_increase is not None
                    else None
                ),
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def resume_flow_node(
        flow_id: str,
        node_id: str,
        response: str,
        idempotency_key: str,
        budget_increase: dict[str, Any] | None = None,
    ) -> str:
        return _json(
            await control.resume_node(
                flow_id,
                node_id,
                response=response,
                base_session_id=base_session_id,
                base_turn_id=base_turn_id,
                idempotency_key=idempotency_key,
                progress=on_progress,
                budget_increase=(
                    BudgetIncrease.model_validate(budget_increase)
                    if budget_increase is not None
                    else None
                ),
            )
        )

    async def skip_flow_node(
        flow_id: str,
        node_id: str,
        reason: str,
        idempotency_key: str,
    ) -> str:
        return _json(
            await control.skip_node(
                flow_id,
                node_id,
                reason=reason,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def pause_flow(
        flow_id: str,
        reason: str,
        idempotency_key: str,
    ) -> str:
        return _json(
            await control.pause(
                flow_id,
                reason=reason,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def resume_flow(flow_id: str, idempotency_key: str) -> str:
        return _json(
            await control.resume(
                flow_id,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def complete_flow(
        flow_id: str,
        idempotency_key: str,
        outcome: FlowCompletion,
        summary: str,
    ) -> str:
        return _json(
            await control.complete(
                flow_id,
                outcome=outcome,
                summary=summary,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    async def cancel_flow(
        flow_id: str,
        idempotency_key: str,
        reason: str,
    ) -> str:
        return _json(
            await control.cancel(
                flow_id,
                reason=reason,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                turn_id=base_turn_id,
            )
        )

    specifications = (
        (
            "create_flow",
            "Create a durable dynamic DAG for a multi-stage outcome. Nodes describe "
            "semantic objectives, dependencies, soft Worker responsibilities, and an "
            "optional worker_session_policy of auto or fresh. context_refs explicitly "
            "associate bounded untrusted context from an ancestor flow_node or a settled "
            "same-session worker_run.",
            _CreateFlowArgs,
            create_flow,
            "mutating",
        ),
        (
            "list_flows",
            "List this Master session's open or paused Flows.",
            _ListFlowsArgs,
            list_flows,
            "read_only",
        ),
        (
            "inspect_flow",
            "Synchronize and inspect one Flow, including bounded node results.",
            _FlowId,
            inspect_flow,
            "read_only",
        ),
        (
            "add_flow_nodes",
            "Dynamically append validated nodes to an open Flow after observing results. "
            "Set worker_session_policy=fresh for a node that requires a clean, "
            "independent WorkerSession on every non-resume execution. Use context_refs "
            "when that fresh Worker needs bounded evidence from related prior work.",
            _AddNodesArgs,
            add_flow_nodes,
            "mutating",
        ),
        (
            "advance_flow",
            "Execute exactly one ready frontier: launch all independent nodes in "
            "parallel, optionally wait, synchronize results, then return to Master.",
            _AdvanceArgs,
            advance_flow,
            "mutating",
        ),
        (
            "revise_flow_node",
            "Create a new generation with review feedback and mark only affected "
            "descendants stale for targeted re-execution. The same healthy "
            "WorkerSession is reused by default; set fresh_worker with fresh_reason "
            "when its context must be discarded. budget_increase raises target limits "
            "for the next Run.",
            _ReviseNodeArgs,
            revise_flow_node,
            "mutating",
        ),
        (
            "retry_flow_node",
            "Retry a partial, failed, or cancelled node without changing its semantics. "
            "The same healthy WorkerSession is reused by default; set fresh_worker "
            "with fresh_reason for a clean retry. Partial nodes require a strictly "
            "larger Master-authored budget_increase.",
            _RetryNodeArgs,
            retry_flow_node,
            "mutating",
        ),
        (
            "resume_flow_node",
            "Answer the exact waiting WorkerRun bound to a Flow node. "
            "budget_increase may raise the continuation's target limits.",
            _ResumeNodeArgs,
            resume_flow_node,
            "mutating",
        ),
        (
            "skip_flow_node",
            "Explicitly waive one non-active, non-partial node so successful joins can "
            "proceed. A partial node must be retried with budget_increase, revised, or "
            "reported through a partial Flow outcome.",
            _SkipNodeArgs,
            skip_flow_node,
            "mutating",
        ),
        (
            "pause_flow",
            "Pause a quiescent Flow before asking the user for information.",
            _PauseFlowArgs,
            pause_flow,
            "mutating",
        ),
        (
            "resume_flow",
            "Reopen a paused Flow after the user provides information.",
            _FlowMutation,
            resume_flow,
            "mutating",
        ),
        (
            "complete_flow",
            "Persist an explicit Flow outcome. Completed requires every node completed "
            "or skipped. This does not end the Master turn.",
            _CompleteFlowArgs,
            complete_flow,
            "mutating",
        ),
        (
            "cancel_flow",
            "Request WorkerRun cancellation. The Flow remains cancelling and blocks "
            "finish_turn until each healthy owner confirms teardown or a crashed "
            "owner loses its durable, tool-fenced lease.",
            _CancelFlowArgs,
            cancel_flow,
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
    return registry


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _node_specs(values: list[dict[str, Any]]) -> list[FlowNodeSpec]:
    return [FlowNodeSpec.model_validate(value) for value in values]


__all__ = ["FinishTurnArgs", "build_master_flow_tools"]
