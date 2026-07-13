"""Explicit UASM agent nodes and their shared runtime services."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon_core.context_compaction import estimate_request_tokens
from aeloon_core.loop_guard import (
    AgentLoopGuard,
    LoopGuardAction,
    LoopGuardDecision,
    rejected_arguments_summary,
)
from aeloon_core.minimal_context import ContextProcessor
from aeloon_core.model_input import PrepareModelInput, unpack_prepared_model_input
from aeloon_core.profile_runtime import (
    COMPLETE_TOOL_NAME,
    CONTROL_TOOL_DEFINITIONS,
    DELEGATE_TOOL_NAME,
    HANDOFF_TOOL_NAME,
    MAX_DELEGATION_ROUNDS,
    LLMProfileMaster,
    fallback_agent_id,
    role_context_messages,
)
from aeloon_core.runtime_support import (
    ThinkTagDeltaFilter,
    default_add_assistant_message,
    default_add_tool_result,
    default_strip_think,
    default_tool_hint,
    execute_tool_batch,
    provider_supports_streaming,
    shrink_answered_tool_args_for_provider,
)
from aeloon_core.state import AgentNode, LightweightState, RunStatus
from aeloon_core.temporary_guard import GuardEvidence, TemporaryGuard
from aeloon_core.tools.registry import ScopedToolRegistry
from aeloon_core.transitions import NodeKind, TransitionRecorder, normalize_usage

if TYPE_CHECKING:
    from aeloon_core.profiles import RuntimeAgentSpec, RuntimeProfileSpec
    from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
    from aeloon_core.task_graph import TaskNode
    from aeloon_core.tools.registry import ToolRegistry

@dataclass
class AgentRuntime:
    """Dependencies and side-effect boundaries shared by all UASM nodes."""

    provider: LLMProvider
    model: str
    tools: ToolRegistry
    rule_engine: AgentLoopGuard
    context_processor: ContextProcessor
    recorder: TransitionRecorder
    temporary_guard: TemporaryGuard | None = None
    profile: RuntimeProfileSpec | None = None
    profile_master: LLMProfileMaster | None = None
    max_handoffs: int = 8
    trace_enabled: bool = True
    on_progress: Callable[..., Awaitable[None]] | None = None
    prepare_model_input: PrepareModelInput | None = None
    add_assistant_message: Callable[..., list[dict[str, Any]]] = (
        default_add_assistant_message
    )
    add_tool_result: Callable[[list[dict[str, Any]], str, str, str], list[dict[str, Any]]] = (
        default_add_tool_result
    )
    strip_think: Callable[[str | None], str | None] = default_strip_think
    tool_hint: Callable[[list[ToolCallRequest]], str] = default_tool_hint
    last_decision: Any = None
    last_usage: dict[str, int] = field(default_factory=dict)
    step_before_digest: str | None = None
    segment_started_at: float | None = None

    def profile_handoff_limit(self) -> int:
        if self.profile is None:
            return 0
        return max(0, min(int(self.profile.max_handoffs), max(0, int(self.max_handoffs))))

    def profile_delegation_enabled(self) -> bool:
        return bool(
            self.profile is not None
            and self.profile.control_protocol_version == 2
            and self.provider.supports_concurrent_calls
        )

    def profile_control_tool_names(self) -> frozenset[str]:
        names = {HANDOFF_TOOL_NAME, COMPLETE_TOOL_NAME}
        if self.profile is not None and self.profile.control_protocol_version == 2:
            names.add(DELEGATE_TOOL_NAME)
        return frozenset(names)

    def active_profile_agent(self, state: LightweightState) -> RuntimeAgentSpec:
        if self.profile is None or state.active_agent_id is None:
            raise RuntimeError("an active profile role is required")
        return self.profile.agent(state.active_agent_id)

    def begin_step(
        self,
        *,
        before_digest: str | None = None,
        started_at: float | None = None,
    ) -> None:
        self.last_decision = None
        self.last_usage = {}
        self.step_before_digest = before_digest
        self.segment_started_at = started_at

    def describe_step(
        self,
        decision: Any,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        self.last_decision = decision
        self.last_usage = normalize_usage(usage)

    async def emit_progress(self, text: str, *, tool_hint: bool = False) -> None:
        if self.on_progress is not None:
            await self.on_progress(text, tool_hint=tool_hint)

    async def emit_hook(self, name: str, *args: Any, **kwargs: Any) -> None:
        if self.on_progress is None:
            return
        hook = getattr(self.on_progress, name, None)
        if hook is None:
            return
        try:
            parameters = inspect.signature(hook).parameters.values()
        except (TypeError, ValueError):
            filtered_kwargs = kwargs
        else:
            accepts_arbitrary_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            accepted_names = {parameter.name for parameter in parameters}
            filtered_kwargs = (
                kwargs
                if accepts_arbitrary_kwargs
                else {key: value for key, value in kwargs.items() if key in accepted_names}
            )
        result = hook(*args, **filtered_kwargs)
        if inspect.isawaitable(result):
            await result

    async def emit_guard_decision(
        self,
        decision: LoopGuardDecision,
        *,
        event: str,
        source: str = "rule_engine",
        fallback_used: bool = False,
        budget_grant: int | None = None,
    ) -> None:
        """Emit the sanitized control outcome without forwarding guard evidence."""

        await self.emit_hook(
            "on_loop_guard_decision",
            decision,
            event=event,
            source=source,
            fallback_used=fallback_used,
            budget_grant=budget_grant,
        )

    async def do_llm_call(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
    ) -> LLMResponse:
        provider_messages = shrink_answered_tool_args_for_provider(messages)
        delta_hook = (
            getattr(self.on_progress, "on_llm_delta", None) if self.on_progress else None
        )
        reasoning_delta_hook = (
            getattr(self.on_progress, "on_llm_reasoning_delta", None)
            if self.on_progress
            else None
        )
        if (
            delta_hook is not None or reasoning_delta_hook is not None
        ) and provider_supports_streaming(self.provider):
            think_filter = ThinkTagDeltaFilter()

            async def _on_delta(delta: str) -> None:
                if delta_hook is None:
                    return
                visible = think_filter.feed(delta)
                if not visible:
                    return
                result = delta_hook(visible)
                if inspect.isawaitable(result):
                    await result

            async def _on_reasoning_delta(delta: str) -> None:
                if reasoning_delta_hook is None or not delta:
                    return
                result = reasoning_delta_hook(delta)
                if inspect.isawaitable(result):
                    await result

            response = await self.provider.chat_stream_with_retry(
                messages=provider_messages,
                tools=tool_defs,
                model=self.model,
                on_delta=_on_delta if delta_hook is not None else None,
                on_reasoning_delta=(
                    _on_reasoning_delta if reasoning_delta_hook is not None else None
                ),
            )
            tail = think_filter.flush() if delta_hook is not None else ""
            if tail and delta_hook is not None:
                result = delta_hook(tail)
                if inspect.isawaitable(result):
                    await result
            return response

        return await self.provider.chat_with_retry(
            messages=provider_messages,
            tools=tool_defs,
            model=self.model,
        )

    async def finish(
        self,
        state: LightweightState,
        *,
        content: str | None,
        status: RunStatus,
        reason: str,
        add_message: bool,
    ) -> LightweightState:
        final_content = (content or reason or "The agent loop stopped.").strip()
        if add_message:
            state.messages = self.add_assistant_message(state.messages, final_content)
        state.metadata.finish(
            status=status,
            final_content=final_content,
            reason=reason,
        )
        if not state.final_emitted:
            state.final_emitted = True
            await self.emit_hook("on_final", final_content, messages=state.messages)
        return state

    async def profile_protocol_error(
        self,
        state: LightweightState,
        *,
        reason: str,
        visible_content: str | None = None,
        correction: str | None = None,
    ) -> LightweightState:
        """Correct one profile protocol violation, then terminate on repetition."""

        state.control_protocol_retries += 1
        if state.control_protocol_retries >= 2:
            content = (
                (visible_content or "").strip()
                or f"The profile role stopped after a repeated protocol violation: {reason}."
            )
            self.describe_step(
                {"action": "terminate", "source": "control_protocol", "reason": reason},
                usage=self.last_usage,
            )
            return await self.finish(
                state,
                content=content,
                status=RunStatus.TERMINATED_BY_RULE,
                reason=f"profile control protocol violation: {reason}",
                add_message=False,
            )

        if correction is None:
            delegation_guidance = (
                " To run independent read-only branches, call delegate_tasks as the only "
                "tool call."
                if self.profile_delegation_enabled()
                else ""
            )
            message = (
                "PROFILE CONTROL PROTOCOL: Your previous response was rejected. External "
                "work must use only your visible external tools. To finish, call "
                "complete_task as the only tool call. To transfer work, call handoff_agent "
                "as the only tool call."
                f"{delegation_guidance} Do not mix a control call with any other call."
            )
        else:
            message = correction
        state.pending_profile_correction = message
        state.metadata.phase = AgentNode.MASTER
        self.describe_step(
            {"action": "correct", "source": "control_protocol", "reason": reason},
            usage=self.last_usage,
        )
        await self.emit_progress("Correcting the active profile role's control response...")
        return state

    async def switch_to_finalization(
        self,
        state: LightweightState,
        prompt_message: dict[str, str] | None,
        *,
        reason: str,
    ) -> bool:
        if (
            state.metadata.status == RunStatus.FINALIZING
            or self.rule_engine.finalization_budget <= 0
        ):
            return False
        state.metadata.status = RunStatus.FINALIZING
        state.metadata.finalization_prompt = (
            prompt_message or self.rule_engine.finalization_prompt_message()
        )
        logger.info(
            "Entering UASM finalization because {}; base={}, auto_continue={}, finalization={}",
            reason,
            self.rule_engine.max_iterations,
            self.rule_engine.max_auto_continue_iterations,
            self.rule_engine.max_finalization_iterations,
        )
        await self.emit_progress(
            f"{reason}; asking for a text-only wrap-up with tools disabled."
        )
        state.metadata.phase = AgentNode.MASTER
        return True

    async def grant_more_or_finalize(self, state: LightweightState) -> LightweightState:
        if state.metadata.iteration < self.rule_engine.iteration_limit:
            state.metadata.phase = AgentNode.MASTER
            return state
        decision = self.rule_engine.handle_iteration_budget_reached()
        return await self.apply_budget_decision(state, decision)

    async def apply_budget_decision(
        self,
        state: LightweightState,
        decision: LoopGuardDecision,
    ) -> LightweightState:
        await self.emit_guard_decision(decision, event="iteration_budget")
        self.describe_step(_decision_summary(decision), usage=self.last_usage)
        if decision.action == LoopGuardAction.EXTEND_BUDGET:
            if decision.progress_message:
                await self.emit_progress(decision.progress_message)
            state.metadata.phase = AgentNode.MASTER
            return state
        if decision.action == LoopGuardAction.FINALIZE:
            switched = await self.switch_to_finalization(
                state,
                decision.prompt_message,
                reason=decision.reason or "iteration budgets exhausted",
            )
            if switched:
                return state
        return await self.finish(
            state,
            content=decision.final_content,
            status=RunStatus.TERMINATED_BY_RULE,
            reason=decision.reason or "iteration budgets exhausted",
            add_message=False,
        )

    async def return_to_model(
        self,
        state: LightweightState,
        decision: LoopGuardDecision,
    ) -> LightweightState:
        if decision.prompt_message:
            state.messages.append(dict(decision.prompt_message))
        if decision.progress_message:
            await self.emit_progress(decision.progress_message)
        self.describe_step(_decision_summary(decision), usage=self.last_usage)
        return await self.grant_more_or_finalize(state)

    def record_context_transition(
        self,
        state: LightweightState,
        *,
        before_digest: str,
        decision: Mapping[str, Any],
        usage: Mapping[str, Any],
        started_at: float,
    ) -> None:
        if not self.trace_enabled:
            return
        transition = self.recorder.record(
            iteration=state.metadata.iteration,
            node="minimal_context",
            node_kind=NodeKind.CONTEXT_PROCESSING,
            component="minimal_context",
            profile=(state.profile_ref.to_dict() if state.profile_ref is not None else None),
            before_digest=self.step_before_digest or before_digest,
            after_digest=state.digest(),
            decision=decision,
            token_usage=usage,
            wall_time_ms=(perf_counter() - (self.segment_started_at or started_at)) * 1_000,
        )
        state.transitions.append(transition)
        self.step_before_digest = transition.after_digest
        self.segment_started_at = perf_counter()


class BaseAgent(ABC):
    """Uniform State -> Agent Node -> State contract."""

    node: AgentNode
    node_kind: NodeKind = NodeKind.DOMAIN

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def node_kind_for(self, state: LightweightState) -> NodeKind:
        del state
        return self.node_kind

    def component_for(self, state: LightweightState) -> str:
        del state
        return self.node.value

    @abstractmethod
    async def run(self, state: LightweightState) -> LightweightState:
        """Execute one explicit transition."""


class MasterAgent(BaseAgent):
    """Deterministically route the next node without an LLM call."""

    node = AgentNode.MASTER

    def node_kind_for(self, state: LightweightState) -> NodeKind:
        return NodeKind.HARNESS if state.profile_ref is not None else self.node_kind

    def component_for(self, state: LightweightState) -> str:
        return "profile_master" if state.profile_ref is not None else self.node.value

    async def run(self, state: LightweightState) -> LightweightState:
        if state.metadata.is_terminal:
            next_node = AgentNode.DONE
        elif state.pending_guard_evidence is not None:
            next_node = AgentNode.TEMPORARY_GUARD
        elif state.metadata.status == RunStatus.FINALIZING:
            next_node = AgentNode.WORKER
        elif state.pending_control_call is not None:
            next_node = AgentNode.CONTROL
        elif state.pending_tool_calls:
            next_node = AgentNode.TOOL
        elif state.profile_ref is not None and state.resume_agent_id is not None:
            state.active_agent_id = state.resume_agent_id
            state.resume_agent_id = None
            next_node = AgentNode.WORKER
        elif (
            state.profile_ref is not None
            and self.runtime.profile is not None
            and (state.active_agent_id is None or state.pending_handoff is not None)
        ):
            selector = self.runtime.profile_master
            if selector is None:
                raise RuntimeError("profile runtime requires a profile master")
            result = await selector.select(
                profile=self.runtime.profile,
                state=state,
                handoff=state.pending_handoff,
            )
            selected_agent_id = result.agent_id
            source = result.source
            fallback_used = result.fallback_used
            diagnostic = result.diagnostic
            declared_agent_ids = {agent.id for agent in self.runtime.profile.agents}
            if selected_agent_id not in declared_agent_ids:
                selected_agent_id = fallback_agent_id(
                    self.runtime.profile,
                    state.pending_handoff,
                )
                source = "fallback"
                fallback_used = True
                diagnostic = "profile master strategy returned an undeclared role"
            state.active_agent_id = selected_agent_id
            usage = state.token_ledger.record(
                NodeKind.HARNESS,
                result.usage,
                component="profile_master",
            )
            self.runtime.describe_step(
                {
                    "route": f"domain:{selected_agent_id}",
                    "source": source,
                    "fallback_used": fallback_used,
                    "diagnostic": diagnostic,
                },
                usage=usage,
            )
            await self.runtime.emit_hook(
                "on_profile_route",
                selected_agent_id,
                source=source,
                fallback_used=fallback_used,
            )
            if result.usage:
                await self.runtime.emit_hook(
                    "on_usage",
                    result.usage,
                    node_kind=NodeKind.HARNESS.value,
                    component="profile_master",
                )
            state.metadata.phase = AgentNode.WORKER
            return state
        else:
            next_node = AgentNode.WORKER
        state.metadata.phase = next_node
        self.runtime.describe_step({"route": next_node.value})
        return state


class WorkerAgent(BaseAgent):
    """Construct model input and perform one domain-model call."""

    node = AgentNode.WORKER

    def component_for(self, state: LightweightState) -> str:
        if state.profile_ref is not None and state.active_agent_id is not None:
            return f"domain:{state.active_agent_id}"
        return self.node.value

    async def run(self, state: LightweightState) -> LightweightState:
        guard = self.runtime.rule_engine
        if state.metadata.status == RunStatus.FINALIZING:
            if state.metadata.finalization_iteration >= guard.finalization_budget:
                decision = LoopGuardDecision(
                    LoopGuardAction.FINAL_RESPONSE,
                    reason="finalization budget exhausted",
                )
                await self.runtime.emit_guard_decision(
                    decision,
                    event="output_exhausted",
                )
                return await self.runtime.finish(
                    state,
                    content=guard.finalization_exhausted_message(),
                    status=RunStatus.TERMINATED_BY_RULE,
                    reason="finalization budget exhausted",
                    add_message=True,
                )
            state.metadata.finalization_iteration += 1
            tool_defs: list[dict[str, Any]] = []
            await self.runtime.emit_hook(
                "on_agent_activity",
                phase="finalizing",
                role_id=state.active_agent_id,
            )
            await self.runtime.emit_progress(
                "Wrapping up..."
                if state.metadata.finalization_iteration == 1
                else f"Wrapping up (attempt {state.metadata.finalization_iteration})..."
            )
        else:
            if state.metadata.iteration >= guard.iteration_limit:
                return await self.runtime.grant_more_or_finalize(state)
            state.metadata.iteration += 1
            if self.runtime.profile is not None:
                agent = self.runtime.active_profile_agent(state)
                control_names = self.runtime.profile_control_tool_names()
                scoped = ScopedToolRegistry(
                    self.runtime.tools,
                    (name for name in agent.tools if name not in control_names),
                )
                tool_defs = scoped.get_definitions()
                state.active_tools = _active_tool_names(tool_defs)
                delegation_enabled = self.runtime.profile_delegation_enabled()
                control_definitions = [
                    dict(item)
                    for item in CONTROL_TOOL_DEFINITIONS
                    if delegation_enabled
                    or item.get("function", {}).get("name") != DELEGATE_TOOL_NAME
                ]
                tool_defs = [*tool_defs, *control_definitions]
            else:
                tool_defs = self.runtime.tools.get_definitions()
            await self.runtime.emit_hook(
                "on_agent_activity",
                phase="analyzing" if state.metadata.iteration == 1 else "planning",
                role_id=state.active_agent_id,
            )
            await self.runtime.emit_progress(
                "Thinking..."
                if state.metadata.iteration == 1
                else f"Thinking (step {state.metadata.iteration})..."
            )

        if state.metadata.status == RunStatus.FINALIZING:
            additional_messages = [
                state.metadata.finalization_prompt or guard.finalization_prompt_message()
            ]
        elif self.runtime.profile is not None:
            active_agent = self.runtime.active_profile_agent(state)
            additional_messages = role_context_messages(
                self.runtime.profile,
                active_agent,
                effective_tools=list(state.active_tools),
                handoff=state.pending_handoff,
                handoff_count=state.handoff_count,
                handoff_limit=self.runtime.profile_handoff_limit(),
                delegation_count=state.delegation_count,
                delegation_limit=MAX_DELEGATION_ROUNDS,
                delegation_enabled=(
                    self.runtime.profile_delegation_enabled()
                ),
            )
            if state.pending_profile_correction is not None:
                additional_messages.append(
                    {
                        "role": "system",
                        "content": state.pending_profile_correction,
                    }
                )
                state.pending_profile_correction = None
            state.pending_handoff = None
        else:
            additional_messages = []
        context_before = (
            self.runtime.step_before_digest or state.digest()
            if self.runtime.trace_enabled
            else ""
        )
        context_started = perf_counter()
        context_usage: dict[str, int] = {}
        prepared_tokens: int | None = None
        if self.runtime.prepare_model_input is not None:
            prepared = await self.runtime.prepare_model_input(
                state.messages,
                tool_defs,
                additional_messages,
            )
            state.messages, context_usage, prepared_tokens = unpack_prepared_model_input(
                prepared
            )

        original_tokens = (
            prepared_tokens
            if prepared_tokens is not None
            else estimate_request_tokens(
                [*state.messages, *additional_messages],
                tools=tool_defs,
                model=self.runtime.model,
            )
        )
        context = self.runtime.context_processor.process(
            state=state,
            messages=state.messages,
            tools=tool_defs,
            additional_messages=additional_messages,
        )
        state.minimal_context = context.messages
        compact_tokens = estimate_request_tokens(
            context.messages,
            tools=context.tools,
            model=self.runtime.model,
        )
        context_metrics = {
            **context_usage,
            "estimated_input_tokens_before": original_tokens,
            "estimated_input_tokens_after": compact_tokens,
            "estimated_input_tokens_saved": max(0, original_tokens - compact_tokens),
        }
        normalized_context_usage = state.token_ledger.record(
            NodeKind.CONTEXT_PROCESSING,
            context_metrics,
            component="minimal_context",
        )
        self.runtime.record_context_transition(
            state,
            before_digest=context_before,
            decision={
                "messages_before": len(state.messages),
                "messages_after": len(context.messages),
                "lazy_references": list(context.lazy_references),
            },
            usage=normalized_context_usage,
            started_at=context_started,
        )

        response = await self.runtime.do_llm_call(context.messages, context.tools)
        component = self.component_for(state)
        domain_usage = state.token_ledger.record(
            NodeKind.DOMAIN,
            response.usage,
            component=component,
        )
        self.runtime.describe_step(
            {
                "finish_reason": response.finish_reason,
                "tool_calls": len(response.tool_calls),
            },
            usage=domain_usage,
        )
        await self.runtime.emit_hook(
            "on_llm_response",
            response,
            component=component,
        )

        if response.tool_calls:
            if state.metadata.status == RunStatus.FINALIZING:
                return await self._handle_finalization_tool_calls(state, response)
            if self.runtime.profile is not None:
                control_names = self.runtime.profile_control_tool_names()
                control_calls = [
                    tool_call
                    for tool_call in response.tool_calls
                    if tool_call.name in control_names
                ]
                if control_calls and len(response.tool_calls) != 1:
                    return await self._reject_mixed_control_batch(state, response)
                if control_calls:
                    state.pending_response = response
                    state.pending_control_call = control_calls[0]
                    state.pending_tool_calls = []
                    state.metadata.phase = AgentNode.MASTER
                    return state
                state.resume_agent_id = state.active_agent_id
            state.pending_response = response
            state.pending_tool_calls = list(response.tool_calls)
            state.metadata.phase = AgentNode.MASTER
            return state
        return await self._handle_text_response(state, response)

    async def _reject_mixed_control_batch(
        self,
        state: LightweightState,
        response: LLMResponse,
    ) -> LightweightState:
        """Pair every rejected call locally without executing any external handler."""

        call_dicts = [tool_call.to_openai_tool_call() for tool_call in response.tool_calls]
        state.messages = self.runtime.add_assistant_message(
            state.messages,
            response.content,
            tool_calls=call_dicts,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        for tool_call in response.tool_calls:
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call.id,
                tool_call.name,
                "Error: Control calls must be the response's only tool call; the entire "
                "batch was rejected without external execution.",
            )
        state.pending_response = None
        state.pending_control_call = None
        state.pending_tool_calls = []
        state.resume_agent_id = None
        return await self.runtime.profile_protocol_error(
            state,
            reason="mixed control and external tool calls",
        )

    async def _handle_finalization_tool_calls(
        self,
        state: LightweightState,
        response: LLMResponse,
    ) -> LightweightState:
        logger.warning(
            "Ignoring {} tool call(s) attempted during UASM finalization",
            len(response.tool_calls),
        )
        clean = self.runtime.strip_think(response.content)
        if clean is not None:
            await self.runtime.emit_guard_decision(
                LoopGuardDecision(
                    LoopGuardAction.FINALIZE,
                    reason=(
                        "tool calls attempted during finalization were ignored because "
                        "visible text was available"
                    ),
                ),
                event="finalization_violation",
            )
            state.messages = self.runtime.add_assistant_message(
                state.messages,
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            return await self.runtime.finish(
                state,
                content=clean,
                status=RunStatus.COMPLETED,
                reason="finalization completed with visible text",
                add_message=False,
            )
        decision = self.runtime.rule_engine.handle_finalization_tool_call_violation()
        await self.runtime.emit_guard_decision(
            decision,
            event="finalization_violation",
        )
        self.runtime.describe_step(_decision_summary(decision), usage=self.runtime.last_usage)
        return await self.runtime.finish(
            state,
            content=decision.final_content,
            status=RunStatus.TERMINATED_BY_RULE,
            reason=decision.reason,
            add_message=True,
        )

    async def _handle_text_response(
        self,
        state: LightweightState,
        response: LLMResponse,
    ) -> LightweightState:
        clean = self.runtime.strip_think(response.content)
        logger.debug(
            "UASM LLM response - content={!r}, reasoning={!r}, finish={}",
            (response.content or "")[:200],
            (response.reasoning_content or "")[:200],
            response.finish_reason,
        )
        if response.finish_reason == "error":
            final_content = clean or "Sorry, I encountered an error calling the AI model."
            return await self.runtime.finish(
                state,
                content=final_content,
                status=RunStatus.FAILED,
                reason="provider error",
                add_message=False,
            )

        if clean is None:
            decision = self.runtime.rule_engine.handle_empty_or_exhausted_response(
                finish_reason=response.finish_reason,
                finalizing=state.metadata.status == RunStatus.FINALIZING,
                finalization_iteration=state.metadata.finalization_iteration,
            )
            await self.runtime.emit_guard_decision(
                decision,
                event=_empty_response_event(response.finish_reason),
            )
            self.runtime.describe_step(
                _decision_summary(decision), usage=self.runtime.last_usage
            )
            if decision.action == LoopGuardAction.CONTINUE:
                state.metadata.phase = AgentNode.MASTER
                return state
            if decision.action == LoopGuardAction.FINALIZE:
                if decision.progress_message:
                    await self.runtime.emit_progress(decision.progress_message)
                switched = await self.runtime.switch_to_finalization(
                    state,
                    decision.prompt_message,
                    reason=decision.reason or "output budget exhausted without visible answer",
                )
                if switched:
                    return state
            return await self.runtime.finish(
                state,
                content=decision.final_content,
                status=RunStatus.TERMINATED_BY_RULE,
                reason=decision.reason or "empty response",
                add_message=True,
            )

        state.messages = self.runtime.add_assistant_message(
            state.messages,
            clean,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        if (
            self.runtime.profile is not None
            and state.metadata.status != RunStatus.FINALIZING
        ):
            return await self.runtime.profile_protocol_error(
                state,
                reason="bare text completion",
                visible_content=clean,
                correction=(
                    "PROFILE CONTROL PROTOCOL: Bare text cannot complete this profile "
                    "turn. Call complete_task(final_content=...) as the only tool call."
                ),
            )
        return await self.runtime.finish(
            state,
            content=clean,
            status=RunStatus.COMPLETED,
            reason="model completed",
            add_message=False,
        )


class ToolAgent(BaseAgent):
    """Apply deterministic rules and execute one tool-call batch."""

    node = AgentNode.TOOL

    def component_for(self, state: LightweightState) -> str:
        del state
        return "tool"

    async def run(self, state: LightweightState) -> LightweightState:
        response = state.pending_response
        if response is None or not state.pending_tool_calls:
            state.pending_response = None
            state.pending_tool_calls = []
            state.metadata.phase = AgentNode.MASTER
            self.runtime.describe_step({"route": "master", "reason": "no pending tools"})
            return state

        thought = self.runtime.strip_think(response.content)
        if thought:
            await self.runtime.emit_progress(thought)

        malformed = self.runtime.rule_engine.handle_malformed_tool_calls(
            state.pending_tool_calls,
            apply_rules=True,
        )
        malformed_ids = {tool_call.id for tool_call in malformed.malformed_calls}
        tool_call_dicts: list[dict[str, Any]] = []
        for tool_call in state.pending_tool_calls:
            call_dict = tool_call.to_openai_tool_call()
            if tool_call.id in malformed_ids:
                call_dict["function"]["arguments"] = rejected_arguments_summary(tool_call)
            tool_call_dicts.append(call_dict)
        state.messages = self.runtime.add_assistant_message(
            state.messages,
            response.content,
            tool_calls=tool_call_dicts,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        for tool_result in malformed.tool_results:
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_result.call_id,
                tool_result.tool_name,
                tool_result.content,
            )

        if malformed.decision.action == LoopGuardAction.STOP_OFF_TRACK:
            return await self._escalate_or_stop(
                state,
                malformed.decision,
                event="malformed_tool_calls",
                failures=_tool_call_failures(
                    malformed.malformed_calls,
                    malformed.tool_results,
                ),
            )
        if malformed.malformed_calls:
            visible_decision = malformed.decision
            if visible_decision.action == LoopGuardAction.CONTINUE:
                visible_decision = LoopGuardDecision(
                    LoopGuardAction.CONTINUE,
                    reason=("malformed tool calls were rejected while valid calls continue"),
                )
            await self.runtime.emit_guard_decision(
                visible_decision,
                event="malformed_tool_calls",
            )
        if malformed.decision.action == LoopGuardAction.RETURN_TO_MODEL:
            self._clear_pending(state)
            return await self.runtime.return_to_model(state, malformed.decision)

        duplicate = self.runtime.rule_engine.handle_duplicate_tool_calls(
            state.messages,
            malformed.executable_calls,
            apply_rules=True,
        )
        for tool_call in duplicate.duplicate_calls:
            logger.warning(
                "Skipping duplicate UASM tool_call '{}' with identical arguments",
                tool_call.name,
            )
        for tool_result in duplicate.tool_results:
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_result.call_id,
                tool_result.tool_name,
                tool_result.content,
            )
        if duplicate.decision.action == LoopGuardAction.STOP_OFF_TRACK:
            return await self._escalate_or_stop(
                state,
                duplicate.decision,
                event="duplicate_tool_calls",
                failures=_tool_call_failures(
                    duplicate.duplicate_calls,
                    duplicate.tool_results,
                ),
            )
        if duplicate.duplicate_calls:
            visible_decision = duplicate.decision
            if visible_decision.action == LoopGuardAction.CONTINUE:
                visible_decision = LoopGuardDecision(
                    LoopGuardAction.CONTINUE,
                    reason=("duplicate tool calls were skipped while new calls continue"),
                )
            await self.runtime.emit_guard_decision(
                visible_decision,
                event="duplicate_tool_calls",
            )
        if duplicate.decision.action == LoopGuardAction.RETURN_TO_MODEL:
            self._clear_pending(state)
            return await self.runtime.return_to_model(state, duplicate.decision)

        tool_calls = duplicate.executable_calls
        hint = self.runtime.strip_think(self.runtime.tool_hint(tool_calls))
        if hint:
            await self.runtime.emit_progress(hint, tool_hint=True)
        await self.runtime.emit_hook("on_tool_calls", tool_calls)
        for tool_call in tool_calls:
            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
            logger.info("UASM tool call: {}({})", tool_call.name, args_str[:200])

        execution_tools = (
            ScopedToolRegistry(self.runtime.tools, state.active_tools)
            if self.runtime.profile is not None
            else self.runtime.tools
        )
        async def report_tool_result(node: TaskNode) -> None:
            await self.runtime.emit_hook("on_tool_result", node)

        executed_nodes = await execute_tool_batch(
            tool_calls=tool_calls,
            tools=execution_tools,
            on_node_complete=report_tool_result,
        )
        state.pending_tool_nodes = executed_nodes
        for node in sorted(executed_nodes, key=lambda item: item.index):
            if self.runtime.profile is None or node.tool_name in state.active_tools:
                state.tools_used.append(node.tool_name)
            state.messages = self.runtime.add_tool_result(
                state.messages,
                node.call_id,
                node.tool_name,
                node.result or f"Error executing {node.tool_name}: unknown failure",
            )
        self._clear_pending(state)

        decision = self.runtime.rule_engine.handle_tool_results(executed_nodes)
        self.runtime.describe_step(_decision_summary(decision))
        if decision.action == LoopGuardAction.STOP_OFF_TRACK:
            return await self._escalate_or_stop(
                state,
                decision,
                event=_tool_result_guard_event(decision),
                failures=_node_failures(executed_nodes),
            )
        if decision.action == LoopGuardAction.RETURN_TO_MODEL:
            await self.runtime.emit_guard_decision(
                decision,
                event=_tool_result_guard_event(decision),
            )
            return await self.runtime.return_to_model(state, decision)
        return await self.runtime.grant_more_or_finalize(state)

    async def _escalate_or_stop(
        self,
        state: LightweightState,
        decision: LoopGuardDecision,
        *,
        event: str,
        failures: tuple[Mapping[str, Any], ...],
    ) -> LightweightState:
        self._clear_pending(state)
        if self.runtime.temporary_guard is None:
            self.runtime.describe_step(_decision_summary(decision))
            await self.runtime.emit_guard_decision(decision, event=event)
            return await self.runtime.finish(
                state,
                content=decision.final_content,
                status=RunStatus.TERMINATED_BY_RULE,
                reason=decision.reason,
                add_message=True,
            )
        state.pending_guard_evidence = GuardEvidence(
            event=event,
            reason=decision.reason,
            iteration=state.metadata.iteration,
            phase=state.metadata.status.value,
            state_digest=state.digest(),
            budgets={
                "iteration_limit": self.runtime.rule_engine.iteration_limit,
                "auto_continue_remaining": (
                    self.runtime.rule_engine.auto_continue_remaining
                ),
                "finalization_budget": self.runtime.rule_engine.finalization_budget,
            },
            counters={
                "unproductive_tool_rounds": (
                    self.runtime.rule_engine.unproductive_tool_rounds
                ),
                "exec_timeout_rounds": self.runtime.rule_engine.exec_timeout_rounds,
                "empty_stop_retries": self.runtime.rule_engine.empty_stop_retries,
            },
            context={
                "message_count": len(state.minimal_context or []),
                "lazy_reference_count": len(state.lazy_values),
            },
            failures=failures,
        )
        state.pending_guard_fallback = decision
        state.metadata.phase = AgentNode.MASTER
        self.runtime.describe_step({"action": "escalate", "event": event})
        return state

    @staticmethod
    def _clear_pending(state: LightweightState) -> None:
        state.pending_response = None
        state.pending_tool_calls = []


class TemporaryGuardAgent(BaseAgent):
    """Invoke the stateless LLM guard only for an eligible ambiguous event."""

    node = AgentNode.TEMPORARY_GUARD
    node_kind = NodeKind.HARNESS

    async def run(self, state: LightweightState) -> LightweightState:
        guard = self.runtime.temporary_guard
        evidence = state.pending_guard_evidence
        fallback = state.pending_guard_fallback
        state.pending_guard_evidence = None
        state.pending_guard_fallback = None
        if guard is None or not isinstance(evidence, GuardEvidence) or not isinstance(
            fallback, LoopGuardDecision
        ):
            state.metadata.phase = AgentNode.MASTER
            self.runtime.describe_step({"route": "master", "reason": "no guard evidence"})
            return state

        resolution = await guard.decide(evidence, fallback)
        usage = state.token_ledger.record(
            NodeKind.HARNESS,
            resolution.usage,
            component="temporary_guard",
        )
        decision = resolution.decision
        self.runtime.describe_step(
            {
                **_decision_summary(decision),
                "source": resolution.source,
                "fallback_used": resolution.fallback_used,
            },
            usage=usage,
        )

        terminal_status = (
            RunStatus.TERMINATED_BY_RULE
            if resolution.fallback_used
            else RunStatus.TERMINATED_BY_GUARD
        )
        effective_budget_grant = decision.budget_grant
        if decision.action == LoopGuardAction.EXTEND_BUDGET:
            requested = max(1, decision.budget_grant)
            effective_budget_grant = min(
                requested,
                state.guard_state.auto_continue_remaining,
            )
        await self.runtime.emit_hook("on_guard_decision", resolution)
        if decision.action == LoopGuardAction.EXTEND_BUDGET:
            state.guard_state.iteration_limit += effective_budget_grant
            state.guard_state.auto_continue_remaining -= effective_budget_grant
        await self.runtime.emit_guard_decision(
            decision,
            event=evidence.event,
            source=resolution.source,
            fallback_used=resolution.fallback_used,
            budget_grant=effective_budget_grant,
        )
        if decision.action == LoopGuardAction.CONTINUE:
            state.metadata.phase = AgentNode.MASTER
            return state
        if decision.action == LoopGuardAction.RETURN_TO_MODEL:
            return await self.runtime.return_to_model(state, decision)
        if decision.action == LoopGuardAction.EXTEND_BUDGET:
            if effective_budget_grant > 0 and decision.progress_message:
                await self.runtime.emit_progress(decision.progress_message)
            state.metadata.phase = AgentNode.MASTER
            return state
        if decision.action == LoopGuardAction.FINALIZE:
            switched = await self.runtime.switch_to_finalization(
                state,
                decision.prompt_message,
                reason=decision.reason or "temporary guard requested finalization",
            )
            if switched:
                return state
        return await self.runtime.finish(
            state,
            content=decision.final_content,
            status=terminal_status,
            reason=decision.reason or "temporary guard stopped the loop",
            add_message=True,
        )


def _decision_summary(decision: LoopGuardDecision) -> dict[str, Any]:
    return {
        "action": decision.action.value,
        "reason": decision.reason,
        "budget_grant": decision.budget_grant,
    }


def _empty_response_event(finish_reason: str) -> str:
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return "output_exhausted"
    return "empty_response"


def _tool_result_guard_event(decision: LoopGuardDecision) -> str:
    return "tool_timeout" if "timed out" in decision.reason.lower() else "tool_result_failed"


def _tool_call_failures(
    tool_calls: list[ToolCallRequest],
    tool_results: list[Any],
) -> tuple[Mapping[str, Any], ...]:
    result_by_id = {result.call_id: result.content for result in tool_results}
    return tuple(
        {
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
            "result": result_by_id.get(tool_call.id, "rejected tool call"),
        }
        for tool_call in tool_calls
    )


def _node_failures(nodes: list[TaskNode]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "tool_name": node.tool_name,
            "arguments": node.arguments,
            "result": node.result or node.error or "unknown tool failure",
        }
        for node in nodes
    )


def agent_node_kind(node: AgentNode) -> NodeKind:
    if node in {AgentNode.CONTROL, AgentNode.TEMPORARY_GUARD}:
        return NodeKind.HARNESS
    return NodeKind.DOMAIN


def _active_tool_names(tool_defs: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for definition in tool_defs:
        function = definition.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names
