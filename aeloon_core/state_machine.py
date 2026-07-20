"""Unified Agentic State Machine runner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aeloon_core.agents import (
    AgentRuntime,
    BaseAgent,
    GuardAgent,
    ModelAgent,
    RouterAgent,
    ToolAgent,
)
from aeloon_core.context_view import ContextViewPipeline
from aeloon_core.loop_guard import (
    GuardEvent,
    GuardEvidence,
    GuardReviewer,
    local_failure_message,
)
from aeloon_core.runtime_support import default_add_tool_result
from aeloon_core.state import (
    AgentNode,
    LightweightState,
    RunStatus,
    StateMetadata,
)
from aeloon_core.transitions import TransitionRecord, TransitionRecorder

if TYPE_CHECKING:
    from aeloon_core.agents import CompletionGate
    from aeloon_core.providers.base import LLMProvider
    from aeloon_core.tools.registry import ToolRegistry


async def run_agent_loop(
    *,
    provider: LLMProvider,
    model: str,
    tools: ToolRegistry,
    messages: list[dict[str, Any]],
    max_iterations: int = 25,
    transition_trace_enabled: bool = True,
    tool_error_guard_threshold: int = 3,
    budget_auto_continues: int = 2,
    stuck_detection_enabled: bool = True,
    stuck_detection_threshold: int = 4,
    max_tokens: int | None = None,
    max_tool_calls: int | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    on_transition: Callable[[TransitionRecord], None] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    context_pipeline: ContextViewPipeline | None = None,
    require_terminal: bool = False,
    completion_gate: CompletionGate | None = None,
) -> LightweightState:
    """Run the generic Router -> Model/Tool/Guard state machine."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tool_error_guard_threshold < 1:
        raise ValueError("tool_error_guard_threshold must be at least 1")
    if budget_auto_continues < 0:
        raise ValueError("budget_auto_continues must be non-negative")
    if not 3 <= stuck_detection_threshold <= 20:
        raise ValueError("stuck_detection_threshold must be between 3 and 20")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if max_tool_calls is not None and max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")

    metadata = StateMetadata(
        session_id=session_id,
        turn_id=turn_id,
    )
    state = LightweightState.from_messages(
        messages,
        active_tools=_active_tool_names(tools.get_definitions()),
        permissions={"model": True, "tools": True},
        max_iterations=max_iterations,
        metadata=metadata,
    )
    resolved_context_pipeline = context_pipeline or ContextViewPipeline(
        provider=provider,
        model=model,
    )
    recorder = TransitionRecorder(
        session_id=session_id,
        turn_id=turn_id,
        persist=on_transition if transition_trace_enabled else None,
    )
    runtime = AgentRuntime(
        provider=provider,
        model=model,
        tools=tools,
        guard=GuardReviewer(provider=provider, model=model),
        base_iteration_budget=max_iterations,
        context_pipeline=resolved_context_pipeline,
        recorder=recorder,
        require_terminal=require_terminal,
        tool_error_guard_threshold=tool_error_guard_threshold,
        budget_auto_continues=budget_auto_continues,
        stuck_detection_enabled=stuck_detection_enabled,
        stuck_detection_threshold=stuck_detection_threshold,
        max_tokens=max_tokens,
        max_tool_calls=max_tool_calls,
        trace_enabled=transition_trace_enabled,
        on_progress=on_progress,
        completion_gate=completion_gate,
    )
    agents: dict[AgentNode, BaseAgent] = {
        AgentNode.ROUTER: RouterAgent(runtime),
        AgentNode.MODEL: ModelAgent(runtime),
        AgentNode.TOOL: ToolAgent(runtime),
        AgentNode.GUARD: GuardAgent(runtime),
    }

    await runtime.emit_hook("on_turn_start")
    current_digest = state.digest() if transition_trace_enabled else ""
    while state.metadata.phase != AgentNode.DONE:
        node = state.metadata.phase
        agent = agents.get(node)
        if agent is None:
            await runtime.finish(
                state,
                content=f"The state machine reached an unknown node: {node}.",
                status=RunStatus.FAILED,
                reason="unknown state-machine node",
                add_message=True,
            )
            break

        if transition_trace_enabled:
            before_digest = current_digest
            started_at = perf_counter()
            runtime.begin_step(before_digest=before_digest, started_at=started_at)
        else:
            before_digest = ""
            started_at = 0.0
            runtime.begin_step()
        try:
            state = await agent.run(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if node == AgentNode.GUARD or state.metadata.status == RunStatus.FINALIZING:
                evidence = state.metadata.finalization_evidence or GuardEvidence(
                    event=GuardEvent.RUNTIME_ERROR,
                    cause=f"{type(exc).__name__}: {exc}",
                    iteration=state.metadata.iteration,
                    iteration_limit=state.metadata.iteration_limit,
                    node=node.value,
                )
                await _finish_with_local_fallback(
                    runtime,
                    state,
                    content=local_failure_message(evidence),
                    reason="guard or finalization failed",
                )
            else:
                failures, outcomes, side_effects = _pair_interrupted_tool_calls(state, exc)
                state = await runtime.queue_guard(
                    state,
                    event=GuardEvent.RUNTIME_ERROR,
                    cause=f"{type(exc).__name__}: {exc}",
                    reason_code="runtime_exception",
                    failures=failures,
                    recent_outcomes=outcomes,
                    successful_side_effects=side_effects,
                )
        if transition_trace_enabled:
            node_kind = agent.node_kind_for(state)
            component = agent.component_for(state)
            transition = recorder.record(
                iteration=state.metadata.iteration,
                node=node,
                node_kind=node_kind,
                component=component,
                before_digest=runtime.step_before_digest or before_digest,
                after_digest=state.digest(),
                decision=runtime.last_decision,
                token_usage=runtime.last_usage,
                wall_time_ms=(perf_counter() - (runtime.segment_started_at or started_at)) * 1_000,
            )
            state.transitions.append(transition)
            current_digest = transition.after_digest

    if state.metadata.final_content is None:
        await runtime.finish(
            state,
            content="The state machine stopped without a visible final response.",
            status=RunStatus.FAILED,
            reason="missing final response",
            add_message=True,
        )
    return state


def _active_tool_names(tool_defs: list[dict[str, Any]]) -> list[str]:
    return [
        str(tool["name"])
        for tool in tool_defs
        if isinstance(tool.get("name"), str)
    ]


def _pair_interrupted_tool_calls(
    state: LightweightState,
    exc: Exception,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    tuple[dict[str, Any], ...],
]:
    """Close any pending protocol calls without replaying uncertain side effects."""

    answered = {
        str(block.get("tool_use_id"))
        for message in state.messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id")
    }
    declared = {
        str(block.get("id"))
        for message in state.messages
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("id")
    }
    for call in state.pending_tool_calls:
        if call.id in answered or call.id not in declared:
            continue
        state.messages = default_add_tool_result(
            state.messages,
            call.id,
            call.name,
            "Error: The call outcome is uncertain because execution was interrupted; "
            "do not replay it automatically.",
        )
    state.pending_response = None
    state.pending_tool_calls = []
    failures: list[dict[str, Any]] = []
    outcomes: list[str] = []
    side_effects: list[dict[str, Any]] = []
    for node in state.pending_tool_nodes:
        result = node.result or node.error or "unknown tool outcome"
        outcomes.append(result)
        item = {
            "tool_name": node.tool_name,
            "arguments": node.arguments,
            "result": result,
        }
        if str(result).lstrip().lower().startswith("error"):
            failures.append({**item, "kind": "tool_result"})
        elif node.mode != "read_only":
            side_effects.append(item)
    state.pending_tool_nodes = []
    failures.append(
        {
            "kind": "runtime_exception",
            "result": f"{type(exc).__name__}: {exc}",
        }
    )
    return tuple(failures), tuple(outcomes), tuple(side_effects)


async def _finish_with_local_fallback(
    runtime: AgentRuntime,
    state: LightweightState,
    *,
    content: str,
    reason: str,
) -> None:
    """Make the terminal fallback independent of adapters and model output."""

    try:
        await runtime.finish(
            state,
            content=content,
            status=RunStatus.FAILED,
            reason=reason,
            add_message=True,
        )
        return
    except Exception:
        state.messages.append({"role": "assistant", "content": content})
        state.metadata.finish(
            status=RunStatus.FAILED,
            final_content=content,
            reason=reason,
        )
        if not state.final_emitted:
            state.final_emitted = True
            await runtime.emit_hook("on_final", content, messages=state.messages)
