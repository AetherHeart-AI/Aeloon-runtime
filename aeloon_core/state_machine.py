"""Unified Agentic State Machine runner."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aeloon_core.agents import (
    AgentRuntime,
    BaseAgent,
    MasterAgent,
    TemporaryGuardAgent,
    ToolAgent,
    WorkerAgent,
)
from aeloon_core.loop_guard import SimpleRuleEngine
from aeloon_core.minimal_context import (
    ContextProcessor,
    IdentityContextProcessor,
    MinimalContextProcessor,
)
from aeloon_core.model_input import PrepareModelInput
from aeloon_core.state import AgentNode, LightweightState, RunStatus, StateMetadata
from aeloon_core.temporary_guard import TemporaryGuard
from aeloon_core.transitions import TransitionRecord, TransitionRecorder

if TYPE_CHECKING:
    from aeloon_core.providers.base import LLMProvider, ToolCallRequest
    from aeloon_core.tools.registry import ToolRegistry


async def run_agent_loop(
    *,
    provider: LLMProvider,
    model: str,
    tools: ToolRegistry,
    messages: list[dict[str, Any]],
    max_iterations: int = 25,
    max_auto_continue_iterations: int = 25,
    max_finalization_iterations: int = 2,
    rule_engine_enabled: bool = True,
    temporary_guard_enabled: bool = True,
    minimal_context_enabled: bool = True,
    transition_trace_enabled: bool = True,
    guard_decision_mode: str = "full",
    minimal_context_recent_turns: int = 2,
    minimal_context_tool_result_chars: int = 1_200,
    session_id: str | None = None,
    turn_id: str | None = None,
    experiment_labels: dict[str, str] | None = None,
    on_transition: Callable[[TransitionRecord], None] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    prepare_model_input: PrepareModelInput | None = None,
    context_processor: ContextProcessor | None = None,
    add_assistant_message: Callable[..., list[dict[str, Any]]] | None = None,
    add_tool_result: Callable[[list[dict[str, Any]], str, str, str], list[dict[str, Any]]]
    | None = None,
    strip_think: Callable[[str | None], str | None] | None = None,
    tool_hint: Callable[[list[ToolCallRequest]], str] | None = None,
) -> LightweightState:
    """Run the explicit Master -> Worker/Tool/Guard state machine."""

    metadata = StateMetadata(
        session_id=session_id,
        turn_id=turn_id,
        experiment_labels=dict(experiment_labels or {}),
    )
    state = LightweightState.from_messages(
        messages,
        active_tools=_active_tool_names(tools.get_definitions()),
        permissions={"model": True, "tools": True},
        max_iterations=max_iterations,
        max_auto_continue_iterations=max_auto_continue_iterations,
        max_finalization_iterations=max_finalization_iterations,
        metadata=metadata,
    )
    rule_engine = SimpleRuleEngine(
        max_iterations=max_iterations,
        max_auto_continue_iterations=max_auto_continue_iterations,
        max_finalization_iterations=max_finalization_iterations,
        state=state.guard_state,
    )
    if context_processor is None:
        context_processor = (
            MinimalContextProcessor(
                preserve_recent_turns=minimal_context_recent_turns,
                max_tool_result_chars=minimal_context_tool_result_chars,
            )
            if minimal_context_enabled
            else IdentityContextProcessor()
        )
    recorder = TransitionRecorder(
        session_id=session_id,
        turn_id=turn_id,
        persist=on_transition if transition_trace_enabled else None,
    )
    temporary_guard = (
        TemporaryGuard(
            provider=provider,
            model=model,
            action_space=guard_decision_mode,
        )
        if temporary_guard_enabled and rule_engine_enabled
        else None
    )
    runtime_kwargs: dict[str, Any] = {}
    if add_assistant_message is not None:
        runtime_kwargs["add_assistant_message"] = add_assistant_message
    if add_tool_result is not None:
        runtime_kwargs["add_tool_result"] = add_tool_result
    if strip_think is not None:
        runtime_kwargs["strip_think"] = strip_think
    if tool_hint is not None:
        runtime_kwargs["tool_hint"] = tool_hint
    runtime = AgentRuntime(
        provider=provider,
        model=model,
        tools=tools,
        rule_engine=rule_engine,
        context_processor=context_processor,
        recorder=recorder,
        temporary_guard=temporary_guard,
        rule_engine_enabled=rule_engine_enabled,
        trace_enabled=transition_trace_enabled,
        on_progress=on_progress,
        prepare_model_input=prepare_model_input,
        **runtime_kwargs,
    )
    agents: dict[AgentNode, BaseAgent] = {
        AgentNode.MASTER: MasterAgent(runtime),
        AgentNode.WORKER: WorkerAgent(runtime),
        AgentNode.TOOL: ToolAgent(runtime),
        AgentNode.TEMPORARY_GUARD: TemporaryGuardAgent(runtime),
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
        state = await agent.run(state)
        if transition_trace_enabled:
            transition = recorder.record(
                iteration=state.metadata.iteration,
                node=node,
                node_kind=agent.node_kind,
                before_digest=runtime.step_before_digest or before_digest,
                after_digest=state.digest(),
                decision=runtime.last_decision,
                token_usage=runtime.last_usage,
                wall_time_ms=(
                    perf_counter() - (runtime.segment_started_at or started_at)
                )
                * 1_000,
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


async def run_uasm_kernel(**kwargs: Any) -> tuple[str | None, list[str], list[dict[str, Any]]]:
    """Compatibility wrapper returning the legacy kernel result tuple."""

    state = await run_agent_loop(**kwargs)
    return state.metadata.final_content, state.tools_used, state.messages


def _active_tool_names(tool_defs: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tool_defs:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names
