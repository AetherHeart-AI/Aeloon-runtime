"""Explicit UASM agent nodes and their shared runtime services."""

from __future__ import annotations

import inspect
import json
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon_core.context_compaction import estimate_request_tokens
from aeloon_core.loop_guard import (
    GuardAction,
    GuardEvent,
    GuardEvidence,
    GuardRequest,
    GuardResolution,
    GuardReviewer,
    GuardSource,
    classify_malformed_tool_calls,
    finalization_prompt_message,
    guard_progress_message,
    local_failure_message,
    recovery_prompt_message,
    rejected_arguments_summary,
    suppress_successful_side_effect_duplicates,
    tool_result_failed,
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
    guard: GuardReviewer
    base_iteration_budget: int
    context_processor: ContextProcessor
    recorder: TransitionRecorder
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
            try:
                await self.on_progress(text, tool_hint=tool_hint)
            except Exception as exc:
                logger.warning("Ignoring progress callback failure: {}", exc)

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
        try:
            result = hook(*args, **filtered_kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Ignoring telemetry hook {} failure: {}", name, exc)

    async def emit_guard_resolution(self, resolution: GuardResolution) -> None:
        """Emit the same bounded record used by traces and accounting."""

        await self.emit_hook("on_guard_resolution", resolution)

    async def do_llm_call(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        *,
        allow_streaming: bool = True,
    ) -> LLMResponse:
        provider_messages = shrink_answered_tool_args_for_provider(messages)
        delta_hook = (
            getattr(self.on_progress, "on_llm_delta", None)
            if self.on_progress and allow_streaming
            else None
        )
        reasoning_delta_hook = (
            getattr(self.on_progress, "on_llm_reasoning_delta", None)
            if self.on_progress and allow_streaming
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
            try:
                await self.emit_hook("on_final", final_content, messages=state.messages)
            except Exception as exc:
                logger.warning("Ignoring final telemetry callback failure: {}", exc)
        return state

    async def profile_protocol_error(
        self,
        state: LightweightState,
        *,
        reason: str,
        visible_content: str | None = None,
        correction: str | None = None,
    ) -> LightweightState:
        """Normalize every profile protocol violation into a Guard review."""

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
        outcomes = ((visible_content or "").strip(),) if visible_content else ()
        return await self.queue_guard(
            state,
            event=GuardEvent.RUNTIME_ERROR,
            cause=f"profile control protocol error: {reason}",
            recent_outcomes=outcomes,
        )

    async def queue_guard(
        self,
        state: LightweightState,
        *,
        event: GuardEvent,
        cause: str,
        failures: tuple[Mapping[str, Any], ...] = (),
        recent_outcomes: tuple[Any, ...] = (),
        successful_side_effects: tuple[Mapping[str, Any], ...] = (),
        allowed_actions: tuple[GuardAction, ...] | None = None,
    ) -> LightweightState:
        if allowed_actions is None:
            allowed_actions = (
                (GuardAction.CONTINUE, GuardAction.FINALIZE)
                if event == GuardEvent.BUDGET_EXHAUSTED
                else (GuardAction.RETRY, GuardAction.FINALIZE)
            )
        evidence = GuardEvidence(
            event=event,
            cause=cause,
            goal=_last_user_goal(state),
            iteration=state.metadata.iteration,
            iteration_limit=state.metadata.iteration_limit,
            phase=state.metadata.status.value,
            node=state.metadata.phase.value,
            state_digest=_safe_state_digest(state),
            failures=failures,
            recent_outcomes=recent_outcomes or _recent_outcomes(state),
            successful_side_effects=successful_side_effects,
            context={
                "message_count": len(state.messages),
                "minimal_context_count": len(state.minimal_context or []),
                "lazy_reference_count": len(state.lazy_values),
                "active_agent_id": state.active_agent_id or "",
            },
        )
        state.pending_guard_request = GuardRequest(
            evidence=evidence,
            allowed_actions=allowed_actions,
            fallback_action=(
                GuardAction.RETRY
                if event == GuardEvent.TOOL_ERROR
                and GuardAction.RETRY in allowed_actions
                and state.metadata.iteration < state.metadata.iteration_limit
                else GuardAction.FINALIZE
            ),
        )
        state.metadata.phase = AgentNode.MASTER
        self.describe_step({"action": "review", "event": event.value, "cause": cause})
        await self.emit_progress(guard_progress_message(event))
        return state

    async def switch_to_finalization(
        self,
        state: LightweightState,
        *,
        evidence: GuardEvidence,
        source: GuardSource,
    ) -> LightweightState:
        state.metadata.status = RunStatus.FINALIZING
        state.metadata.finalization_prompt = finalization_prompt_message(evidence)
        state.metadata.finalization_source = source
        state.metadata.finalization_evidence = evidence
        state.metadata.phase = AgentNode.MASTER
        return state

    async def grant_more_or_finalize(self, state: LightweightState) -> LightweightState:
        if state.metadata.iteration < state.metadata.iteration_limit:
            state.metadata.phase = AgentNode.MASTER
            return state
        return await self.queue_guard(
            state,
            event=GuardEvent.BUDGET_EXHAUSTED,
            cause="agent loop iteration budget reached",
        )

    async def return_to_model(
        self,
        state: LightweightState,
        evidence: GuardEvidence,
    ) -> LightweightState:
        state.messages.append(recovery_prompt_message(evidence))
        if state.metadata.iteration >= state.metadata.iteration_limit:
            state.metadata.iteration_limit += 1
        state.metadata.phase = AgentNode.MASTER
        return state

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
        elif state.pending_guard_request is not None:
            next_node = AgentNode.GUARD
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
        if state.metadata.status == RunStatus.FINALIZING:
            tool_defs: list[dict[str, Any]] = []
            await self.runtime.emit_hook(
                "on_agent_activity",
                phase="finalizing",
                role_id=state.active_agent_id,
            )
            await self.runtime.emit_progress("Wrapping up...")
        else:
            if state.metadata.iteration >= state.metadata.iteration_limit:
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
            additional_messages = [state.metadata.finalization_prompt or {
                "role": "user",
                "content": "Respond with one concise, honest text-only wrap-up.",
            }]
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

        response = await self.runtime.do_llm_call(
            context.messages,
            context.tools,
            allow_streaming=state.metadata.status != RunStatus.FINALIZING,
        )
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
        evidence = state.metadata.finalization_evidence or _fallback_evidence(
            state,
            cause="model attempted tools during text-only finalization",
        )
        return await self.runtime.finish(
            state,
            content=local_failure_message(evidence),
            status=RunStatus.FAILED,
            reason="tool call attempted during finalization",
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
        if state.metadata.status == RunStatus.FINALIZING:
            evidence = state.metadata.finalization_evidence or _fallback_evidence(
                state,
                cause="finalization failed",
            )
            if (
                response.finish_reason != "stop"
                or clean is None
                or _is_bare_dsml_tool_envelope(clean)
            ):
                return await self.runtime.finish(
                    state,
                    content=local_failure_message(evidence),
                    status=RunStatus.FAILED,
                    reason="text-only finalization failed",
                    add_message=True,
                )
            state.messages = self.runtime.add_assistant_message(
                state.messages,
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            status = (
                RunStatus.TERMINATED_BY_GUARD
                if state.metadata.finalization_source == "guard"
                else RunStatus.FAILED
            )
            return await self.runtime.finish(
                state,
                content=clean,
                status=status,
                reason="guard requested an honest wrap-up",
                add_message=False,
            )

        if response.finish_reason == "error":
            return await self.runtime.queue_guard(
                state,
                event=GuardEvent.RUNTIME_ERROR,
                cause=clean or "provider returned finish_reason=error",
            )

        if clean is None:
            exhausted = response.finish_reason in {
                "length",
                "max_tokens",
                "max_output_tokens",
            }
            return await self.runtime.queue_guard(
                state,
                event=GuardEvent.RUNTIME_ERROR,
                cause=(
                    "model exhausted its output budget without visible text"
                    if exhausted
                    else "model returned an empty response"
                ),
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
    """Validate and execute one batch, escalating any failed call to Guard."""

    node = AgentNode.TOOL

    def component_for(self, state: LightweightState) -> str:
        del state
        return "tool"

    async def run(self, state: LightweightState) -> LightweightState:
        response = state.pending_response
        if response is None or not state.pending_tool_calls:
            self._clear_pending(state)
            state.metadata.phase = AgentNode.MASTER
            self.runtime.describe_step({"route": "master", "reason": "no pending tools"})
            return state

        state.pending_tool_nodes = []
        thought = self.runtime.strip_think(response.content)
        if thought:
            await self.runtime.emit_progress(thought)

        malformed = classify_malformed_tool_calls(state.pending_tool_calls)
        malformed_ids = {call.id for call in malformed.rejected_calls}
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
        _append_tool_patches(state, self.runtime, malformed.tool_results)

        execution_tools = (
            ScopedToolRegistry(self.runtime.tools, state.active_tools)
            if self.runtime.profile is not None
            else self.runtime.tools
        )
        tool_modes = {
            call.name: (
                tool.concurrency_mode
                if (tool := execution_tools.get(call.name)) is not None
                else "exclusive"
            )
            for call in malformed.executable_calls
        }
        duplicates = suppress_successful_side_effect_duplicates(
            state.messages,
            malformed.executable_calls,
            tool_modes=tool_modes,
        )
        _append_tool_patches(state, self.runtime, duplicates.tool_results)

        tool_calls = list(duplicates.executable_calls)
        executed_nodes: list[TaskNode] = []
        if tool_calls:
            hint = self.runtime.strip_think(self.runtime.tool_hint(tool_calls))
            if hint:
                await self.runtime.emit_progress(hint, tool_hint=True)
            await self.runtime.emit_hook("on_tool_calls", tool_calls)
            for tool_call in tool_calls:
                args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                logger.info("UASM tool call: {}({})", tool_call.name, args_str[:200])

            state.pending_tool_nodes = []

            async def report_tool_result(node: TaskNode) -> None:
                state.pending_tool_nodes.append(node)
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

        failures = (*malformed.failures, *duplicates.failures, *_node_failures(executed_nodes))
        side_effects = _successful_side_effects(executed_nodes)
        outcomes = tuple(
            node.result or node.error or "unknown tool outcome" for node in executed_nodes
        )
        self._clear_pending(state)
        self.runtime.describe_step(
            {
                "executed": len(executed_nodes),
                "failed": len(failures),
                "rejected": len(malformed.rejected_calls) + len(duplicates.rejected_calls),
            }
        )
        if failures:
            return await self.runtime.queue_guard(
                state,
                event=GuardEvent.TOOL_ERROR,
                cause="one or more calls in the latest tool round failed",
                failures=tuple(failures),
                recent_outcomes=outcomes,
                successful_side_effects=side_effects,
            )
        return await self.runtime.grant_more_or_finalize(state)

    @staticmethod
    def _clear_pending(state: LightweightState) -> None:
        state.pending_response = None
        state.pending_tool_calls = []


class GuardAgent(BaseAgent):
    """Run the stateless reviewer on the exceptional branch only."""

    node = AgentNode.GUARD
    node_kind = NodeKind.HARNESS

    async def run(self, state: LightweightState) -> LightweightState:
        request = state.pending_guard_request
        state.pending_guard_request = None
        if request is None:
            state.metadata.phase = AgentNode.MASTER
            self.runtime.describe_step({"route": "master", "reason": "no guard request"})
            return state

        resolution = await self.runtime.guard.decide(request)
        usage = state.token_ledger.record(
            NodeKind.HARNESS,
            resolution.usage,
            component="guard",
        )
        self.runtime.describe_step(resolution.to_record(), usage=usage)
        await self.runtime.emit_guard_resolution(resolution)

        if resolution.action == GuardAction.RETRY:
            return await self.runtime.return_to_model(state, request.evidence)
        if resolution.action == GuardAction.CONTINUE:
            state.metadata.iteration_limit += self.runtime.base_iteration_budget
            state.metadata.phase = AgentNode.MASTER
            return state
        return await self.runtime.switch_to_finalization(
            state,
            evidence=request.evidence,
            source=resolution.source,
        )


def _append_tool_patches(
    state: LightweightState,
    runtime: AgentRuntime,
    patches: Sequence[Any],
) -> None:
    for patch in patches:
        state.messages = runtime.add_tool_result(
            state.messages,
            patch.call_id,
            patch.tool_name,
            patch.content,
        )


def _node_failures(nodes: Sequence[TaskNode]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "tool_name": node.tool_name,
            "arguments": node.arguments,
            "result": node.result or node.error or "unknown tool failure",
            "kind": "tool_result",
        }
        for node in nodes
        if tool_result_failed(node.result or node.error)
    )


def _successful_side_effects(
    nodes: Sequence[TaskNode],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "tool_name": node.tool_name,
            "arguments": node.arguments,
            "result": node.result or "completed",
        }
        for node in nodes
        if node.mode != "read_only"
        and not tool_result_failed(node.result or node.error)
    )


def _last_user_goal(state: LightweightState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _recent_outcomes(state: LightweightState) -> tuple[str, ...]:
    outcomes: list[str] = []
    for message in reversed(state.messages):
        if message.get("role") not in {"assistant", "tool"}:
            continue
        content = str(message.get("content") or "").strip()
        if content:
            outcomes.append(content)
        if len(outcomes) >= 5:
            break
    return tuple(reversed(outcomes))


def _safe_state_digest(state: LightweightState) -> str:
    try:
        return state.digest()
    except Exception:
        return "unavailable"


def _fallback_evidence(state: LightweightState, *, cause: str) -> GuardEvidence:
    return GuardEvidence(
        event=GuardEvent.RUNTIME_ERROR,
        cause=cause,
        goal=_last_user_goal(state),
        iteration=state.metadata.iteration,
        iteration_limit=state.metadata.iteration_limit,
        phase=state.metadata.status.value,
        node=state.metadata.phase.value,
        state_digest=_safe_state_digest(state),
        recent_outcomes=_recent_outcomes(state),
    )


def _is_bare_dsml_tool_envelope(content: str) -> bool:
    """Reject a bare DeepSeek tool envelope from the text-only finalizer."""

    text = unicodedata.normalize("NFKC", content).strip()
    prefix = "<||DSML||tool_calls>"
    return text.startswith(prefix)


def agent_node_kind(node: AgentNode) -> NodeKind:
    if node in {AgentNode.CONTROL, AgentNode.GUARD}:
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
