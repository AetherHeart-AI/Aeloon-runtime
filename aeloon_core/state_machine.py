"""Unified Agentic State Machine runner."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aeloon_core.agents import (
    AgentRuntime,
    BaseAgent,
    MasterAgent,
    WorkerAgent,
)
from aeloon_core.loop_guard import SimpleRuleEngine
from aeloon_core.minimal_context import MinimalContextProcessor
from aeloon_core.model_input import PrepareModelInput
from aeloon_core.profile_agents import ControlAgent, ProfileDomainAgent
from aeloon_core.profile_runtime import LLMProfileMaster
from aeloon_core.state import (
    AgentNode,
    LightweightState,
    ProfileRef,
    RunStatus,
    StateMetadata,
)
from aeloon_core.temporary_guard import TemporaryGuard
from aeloon_core.tool_agents import TemporaryGuardAgent, ToolAgent
from aeloon_core.transitions import TransitionRecord, TransitionRecorder

if TYPE_CHECKING:
    from aeloon_core.profiles import RuntimeProfileSpec
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
    transition_trace_enabled: bool = True,
    minimal_context_recent_turns: int = 2,
    minimal_context_tool_result_chars: int = 1_200,
    session_id: str | None = None,
    turn_id: str | None = None,
    on_transition: Callable[[TransitionRecord], None] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    prepare_model_input: PrepareModelInput | None = None,
    add_assistant_message: Callable[..., list[dict[str, Any]]] | None = None,
    add_tool_result: Callable[[list[dict[str, Any]], str, str, str], list[dict[str, Any]]]
    | None = None,
    strip_think: Callable[[str | None], str | None] | None = None,
    tool_hint: Callable[[list[ToolCallRequest]], str] | None = None,
    profile: RuntimeProfileSpec | None = None,
    max_handoffs: int = 8,
) -> LightweightState:
    """Run the explicit Master -> Worker/Tool/Guard state machine."""

    metadata = StateMetadata(
        session_id=session_id,
        turn_id=turn_id,
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
    if profile is not None:
        artifact_id = profile.artifact_id or (
            f"inline:{profile.profile_id}:revision-{profile.revision}"
        )
        state.profile_ref = ProfileRef(
            profile_id=profile.profile_id,
            revision=profile.revision,
            artifact_id=artifact_id,
            generation=profile.generation,
        )
        state.active_tools = []
    rule_engine = SimpleRuleEngine(
        max_iterations=max_iterations,
        max_auto_continue_iterations=max_auto_continue_iterations,
        max_finalization_iterations=max_finalization_iterations,
        state=state.guard_state,
    )
    context_processor = MinimalContextProcessor(
        preserve_recent_turns=minimal_context_recent_turns,
        max_tool_result_chars=minimal_context_tool_result_chars,
    )
    recorder = TransitionRecorder(
        session_id=session_id,
        turn_id=turn_id,
        persist=on_transition if transition_trace_enabled else None,
    )
    temporary_guard = TemporaryGuard(
        provider=provider,
        model=model,
        action_space="full",
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
        profile=profile,
        profile_master=LLMProfileMaster(provider=provider, model=model)
        if profile is not None
        else None,
        max_handoffs=max_handoffs,
        trace_enabled=transition_trace_enabled,
        on_progress=on_progress,
        prepare_model_input=prepare_model_input,
        **runtime_kwargs,
    )
    agents: dict[AgentNode, BaseAgent] = {
        AgentNode.MASTER: MasterAgent(runtime),
        AgentNode.WORKER: WorkerAgent(runtime),
        AgentNode.CONTROL: ControlAgent(runtime),
        AgentNode.TOOL: ToolAgent(runtime),
        AgentNode.TEMPORARY_GUARD: TemporaryGuardAgent(runtime),
    }
    profile_agents = (
        {agent.id: ProfileDomainAgent(runtime, agent.id) for agent in profile.agents}
        if profile is not None
        else {}
    )

    await runtime.emit_hook("on_turn_start")
    if state.profile_ref is not None:
        await runtime.emit_hook("on_profile_pinned", state.profile_ref.to_dict())
    current_digest = state.digest() if transition_trace_enabled else ""
    while state.metadata.phase != AgentNode.DONE:
        node = state.metadata.phase
        if (
            node == AgentNode.WORKER
            and profile is not None
            and state.metadata.status != RunStatus.FINALIZING
        ):
            agent = profile_agents.get(state.active_agent_id or "")
        else:
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
            node_kind = agent.node_kind_for(state)
            component = agent.component_for(state)
            transition = recorder.record(
                iteration=state.metadata.iteration,
                node=node,
                node_kind=node_kind,
                component=component,
                profile=(state.profile_ref.to_dict() if state.profile_ref is not None else None),
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


def _active_tool_names(tool_defs: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tool_defs:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names
