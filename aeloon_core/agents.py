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
from aeloon_core.task_graph import TaskNode
from aeloon_core.transitions import NodeKind, TransitionRecorder, normalize_usage

if TYPE_CHECKING:
    from aeloon_core.providers.base import LLMProvider, LLMResponse
    from aeloon_core.tools.registry import ToolRegistry

CompletionGate = Callable[
    [LightweightState, str],
    Awaitable[str | None] | str | None,
]


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
    # A strict loop may only finish through a successful terminal tool.
    require_terminal: bool = False
    # Soft-feed Error* tool results this many consecutive rounds before Guard.
    tool_error_guard_threshold: int = 3
    # Automatic budget extensions before a Guard budget review is required.
    budget_auto_continues: int = 2
    # Optional hard WorkerRun grants. Master turns leave these unset.
    max_tokens: int | None = None
    max_tool_calls: int | None = None
    trace_enabled: bool = True
    on_progress: Callable[..., Awaitable[None]] | None = None
    prepare_model_input: PrepareModelInput | None = None
    # Optional semantic completion policy above the generic UASM. Returning a
    # reason rejects bare text and gives the model another planning turn.
    completion_gate: CompletionGate | None = None
    last_decision: Any = None
    last_usage: dict[str, int] = field(default_factory=dict)
    step_before_digest: str | None = None
    segment_started_at: float | None = None

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
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
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
        max_tokens: int | None = None,
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

            stream = (
                self.provider.chat_stream
                if max_tokens is not None
                else self.provider.chat_stream_with_retry
            )
            stream_kwargs: dict[str, Any] = {}
            if max_tokens is not None:
                stream_kwargs = {
                    "temperature": self.provider.generation.temperature,
                    "reasoning_effort": self.provider.generation.reasoning_effort,
                }
            response = await stream(
                messages=provider_messages,
                tools=tool_defs,
                model=self.model,
                max_tokens=max_tokens,
                on_delta=_on_delta if delta_hook is not None else None,
                on_reasoning_delta=(
                    _on_reasoning_delta if reasoning_delta_hook is not None else None
                ),
                **stream_kwargs,
            )
            tail = think_filter.flush() if delta_hook is not None else ""
            if tail and delta_hook is not None:
                result = delta_hook(tail)
                if inspect.isawaitable(result):
                    await result
            return response

        if max_tokens is not None:
            return await self.provider.chat(
                messages=provider_messages,
                tools=tool_defs,
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.provider.generation.temperature,
                reasoning_effort=self.provider.generation.reasoning_effort,
            )
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
            state.messages = default_add_assistant_message(state.messages, final_content)
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

    def hard_budget_exhaustion(
        self,
        state: LightweightState,
        *,
        pending_tool_calls: int = 0,
        check_tokens: bool = True,
    ) -> str | None:
        """Return a host-owned hard-budget reason without invoking another model."""

        if (
            check_tokens
            and self.max_tokens is not None
            and state.token_ledger.total_tokens >= self.max_tokens
        ):
            return f"WorkerRun token budget exhausted ({self.max_tokens} tokens)"
        if self.max_tool_calls is not None:
            used = len(state.tools_used)
            exhausted = (
                used >= self.max_tool_calls
                if pending_tool_calls == 0
                else used + pending_tool_calls > self.max_tool_calls
            )
            if exhausted:
                return f"WorkerRun tool-call budget exhausted ({self.max_tool_calls} calls)"
        return None

    async def finish_for_hard_budget(
        self,
        state: LightweightState,
        reason: str,
    ) -> LightweightState:
        state.pending_response = None
        state.pending_tool_calls = []
        return await self.finish(
            state,
            content=reason + ".",
            status=RunStatus.TERMINATED_BY_GUARD,
            reason=reason,
            add_message=True,
        )

    async def terminal_protocol_error(
        self,
        state: LightweightState,
        *,
        reason: str,
        visible_content: str | None = None,
    ) -> LightweightState:
        """Turn a strict-loop completion violation into a correctable Guard path."""

        terminal_names = sorted(
            name
            for name in state.active_tools
            if (tool := self.tools.get(name)) is not None
            and bool(getattr(tool, "terminal", False))
        )
        advertised = ", ".join(terminal_names) or "a terminal tool"
        state.messages.append(
            {
                "role": "system",
                "content": (
                    "TERMINAL TOOL PROTOCOL: The previous response did not finish this run. "
                    f"To finish, call exactly one of [{advertised}] as the response's only "
                    "tool call. Do not mix a terminal call with any other call."
                ),
            }
        )
        outcomes = ((visible_content or "").strip(),) if visible_content else ()
        return await self.queue_guard(
            state,
            event=GuardEvent.RUNTIME_ERROR,
            cause=f"terminal tool protocol error: {reason}",
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
                "consecutive_tool_failure_rounds": state.metadata.consecutive_tool_failure_rounds,
                "budget_auto_continues_used": state.metadata.budget_auto_continues_used,
            },
        )
        state.pending_guard_request = GuardRequest(
            evidence=evidence,
            allowed_actions=allowed_actions,
            fallback_action=_guard_fallback_action(
                event=event,
                allowed_actions=allowed_actions,
                iteration=state.metadata.iteration,
                iteration_limit=state.metadata.iteration_limit,
            ),
        )
        state.metadata.phase = AgentNode.ROUTER
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
        state.metadata.phase = AgentNode.ROUTER
        return state

    async def grant_more_or_finalize(self, state: LightweightState) -> LightweightState:
        if state.metadata.iteration < state.metadata.iteration_limit:
            state.metadata.phase = AgentNode.ROUTER
            return state
        # Prefer local budget extensions over an immediate Guard finalize.
        if state.metadata.budget_auto_continues_used < max(0, int(self.budget_auto_continues)):
            state.metadata.budget_auto_continues_used += 1
            state.metadata.iteration_limit += max(1, int(self.base_iteration_budget))
            state.metadata.phase = AgentNode.ROUTER
            self.describe_step(
                {
                    "action": "auto_continue",
                    "budget_auto_continues_used": state.metadata.budget_auto_continues_used,
                    "iteration_limit": state.metadata.iteration_limit,
                }
            )
            await self.emit_progress(
                f"Extending budget ({state.metadata.budget_auto_continues_used}/"
                f"{self.budget_auto_continues})..."
            )
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
        state.metadata.phase = AgentNode.ROUTER
        return state

    async def return_after_tool_errors(
        self,
        state: LightweightState,
        *,
        failures: tuple[Mapping[str, Any], ...],
        recent_outcomes: tuple[Any, ...] = (),
        successful_side_effects: tuple[Mapping[str, Any], ...] = (),
    ) -> LightweightState:
        """Feed Error* tool results back to the model; escalate only after N rounds."""

        state.metadata.consecutive_tool_failure_rounds += 1
        threshold = max(1, int(self.tool_error_guard_threshold))
        if state.metadata.consecutive_tool_failure_rounds < threshold:
            self.describe_step(
                {
                    "action": "soft_tool_feedback",
                    "failed": len(failures),
                    "consecutive_tool_failure_rounds": (
                        state.metadata.consecutive_tool_failure_rounds
                    ),
                    "threshold": threshold,
                }
            )
            await self.emit_progress(
                "Tool error returned to the model "
                f"({state.metadata.consecutive_tool_failure_rounds}/{threshold})..."
            )
            return await self.grant_more_or_finalize(state)
        return await self.queue_guard(
            state,
            event=GuardEvent.TOOL_ERROR,
            cause=(
                "one or more calls failed in "
                f"{state.metadata.consecutive_tool_failure_rounds} consecutive tool rounds"
            ),
            failures=failures,
            recent_outcomes=recent_outcomes,
            successful_side_effects=successful_side_effects,
        )

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
            node=AgentNode.MODEL,
            node_kind=NodeKind.CONTEXT_PROCESSING,
            component="minimal_context",
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


class RouterAgent(BaseAgent):
    """Deterministically route the next generic runtime node."""

    node = AgentNode.ROUTER

    async def run(self, state: LightweightState) -> LightweightState:
        if state.metadata.is_terminal:
            next_node = AgentNode.DONE
        elif state.pending_tool_calls:
            next_node = AgentNode.TOOL
        elif reason := self.runtime.hard_budget_exhaustion(state):
            return await self.runtime.finish_for_hard_budget(state, reason)
        elif state.pending_guard_request is not None:
            next_node = AgentNode.GUARD
        elif state.metadata.status == RunStatus.FINALIZING:
            next_node = AgentNode.MODEL
        else:
            next_node = AgentNode.MODEL
        state.metadata.phase = next_node
        self.runtime.describe_step({"route": next_node.value})
        return state


class ModelAgent(BaseAgent):
    """Construct model input and perform one provider-model call."""

    node = AgentNode.MODEL

    async def run(self, state: LightweightState) -> LightweightState:
        if state.metadata.status == RunStatus.FINALIZING:
            tool_defs: list[dict[str, Any]] = []
            await self.runtime.emit_hook(
                "on_agent_activity",
                phase="finalizing",
            )
            await self.runtime.emit_progress("Wrapping up...")
        else:
            if state.metadata.iteration >= state.metadata.iteration_limit:
                return await self.runtime.grant_more_or_finalize(state)
            state.metadata.iteration += 1
            tool_defs = self.runtime.tools.get_definitions()
            state.active_tools = _active_tool_names(tool_defs)
            await self.runtime.emit_hook(
                "on_agent_activity",
                phase="analyzing" if state.metadata.iteration == 1 else "planning",
            )
            await self.runtime.emit_progress(
                "Thinking..."
                if state.metadata.iteration == 1
                else f"Thinking (step {state.metadata.iteration})..."
            )

        if state.metadata.status == RunStatus.FINALIZING:
            additional_messages = [
                state.metadata.finalization_prompt
                or {
                    "role": "user",
                    "content": "Respond with one concise, honest text-only wrap-up.",
                }
            ]
        else:
            additional_messages = []
        context_before = (
            self.runtime.step_before_digest or state.digest() if self.runtime.trace_enabled else ""
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
            state.messages, context_usage, prepared_tokens = unpack_prepared_model_input(prepared)

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

        call_max_tokens: int | None = None
        if self.runtime.max_tokens is not None:
            remaining = self.runtime.max_tokens - state.token_ledger.total_tokens
            if compact_tokens >= remaining:
                return await self.runtime.finish_for_hard_budget(
                    state,
                    f"WorkerRun token budget cannot fit the next model input "
                    f"({self.runtime.max_tokens} tokens)",
                )
            call_max_tokens = max(1, remaining - compact_tokens)

        response = await self.runtime.do_llm_call(
            context.messages,
            context.tools,
            allow_streaming=state.metadata.status != RunStatus.FINALIZING,
            max_tokens=call_max_tokens,
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
            if any(call.arguments_error is not None for call in response.tool_calls):
                state.pending_response = response
                state.pending_tool_calls = list(response.tool_calls)
                state.metadata.phase = AgentNode.ROUTER
                return state
            state.pending_response = response
            state.pending_tool_calls = list(response.tool_calls)
            state.metadata.phase = AgentNode.ROUTER
            return state
        return await self._handle_text_response(state, response)

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
        clean = default_strip_think(response.content)
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
                response.finish_reason != "end_turn"
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
            state.messages = default_add_assistant_message(
                state.messages,
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            # A successful text-only wrap-up is partial success, whether Guard or
            # the local fallback produced the decision. Hard FAILED is reserved for
            # paths that cannot emit a usable answer (see local_failure_message).
            return await self.runtime.finish(
                state,
                content=clean,
                status=RunStatus.TERMINATED_BY_GUARD,
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

        if (
            state.metadata.status != RunStatus.FINALIZING
            and self.runtime.completion_gate is not None
        ):
            blocked = self.runtime.completion_gate(state, clean)
            if inspect.isawaitable(blocked):
                blocked = await blocked
            if blocked:
                state.messages = default_add_assistant_message(
                    state.messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                state.messages.append(
                    {
                        "role": "system",
                        "content": (
                            "COMPLETION GATE: This response did not finish the Master "
                            f"turn. {blocked} Continue with the advertised tools."
                        ),
                    }
                )
                self.runtime.describe_step(
                    {"action": "completion_rejected", "reason": str(blocked)}
                )
                return await self.runtime.grant_more_or_finalize(state)

        state.messages = default_add_assistant_message(
            state.messages,
            clean,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        if self.runtime.require_terminal:
            return await self.runtime.terminal_protocol_error(
                state,
                reason="bare text completion",
                visible_content=clean,
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
            state.metadata.phase = AgentNode.ROUTER
            self.runtime.describe_step({"route": "router", "reason": "no pending tools"})
            return state

        state.pending_tool_nodes = []
        thought = default_strip_think(response.content)
        if thought:
            await self.runtime.emit_progress(thought)

        if len(state.pending_tool_calls) > 1 and any(
            (tool := self.runtime.tools.get(call.name)) is not None
            and bool(getattr(tool, "terminal", False))
            for call in state.pending_tool_calls
        ):
            return await self._reject_terminal_batch(state, response)

        budget_reason = self.runtime.hard_budget_exhaustion(
            state,
            pending_tool_calls=len(state.pending_tool_calls),
            check_tokens=False,
        )
        if budget_reason is not None:
            return await self.runtime.finish_for_hard_budget(state, budget_reason)

        malformed = classify_malformed_tool_calls(state.pending_tool_calls)
        malformed_ids = {call.id for call in malformed.rejected_calls}
        tool_use_blocks: list[dict[str, Any]] = []
        for tool_call in state.pending_tool_calls:
            input_override = None
            if tool_call.id in malformed_ids:
                input_override = rejected_arguments_summary(tool_call)
            tool_use_blocks.append(
                tool_call.to_anthropic_tool_use(input_override=input_override)
            )
        state.messages = default_add_assistant_message(
            state.messages,
            response.content,
            tool_uses=tool_use_blocks,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        _append_tool_patches(state, malformed.tool_results)

        execution_tools = self.runtime.tools
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
        _append_tool_patches(state, duplicates.tool_results)

        tool_calls = list(duplicates.executable_calls)
        executed_nodes: list[TaskNode] = []
        if tool_calls:
            hint = default_strip_think(default_tool_hint(tool_calls))
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
                state.tools_used.append(node.tool_name)
                state.messages = default_add_tool_result(
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
            # Error strings are already in conversation history as tool results.
            # Prefer model self-correction; only escalate after consecutive rounds.
            return await self.runtime.return_after_tool_errors(
                state,
                failures=tuple(failures),
                recent_outcomes=outcomes,
                successful_side_effects=side_effects,
            )
        terminal_nodes = [
            node
            for node in executed_nodes
            if (tool := execution_tools.get(node.tool_name)) is not None
            and bool(getattr(tool, "terminal", False))
        ]
        if terminal_nodes:
            terminal = terminal_nodes[-1]
            state.metadata.extras["terminal_tool"] = {
                "name": terminal.tool_name,
                "call_id": terminal.call_id,
            }
            return await self.runtime.finish(
                state,
                content=terminal.result or f"Completed via {terminal.tool_name}.",
                status=RunStatus.COMPLETED,
                reason=f"terminal tool {terminal.tool_name}",
                add_message=False,
            )
        state.metadata.consecutive_tool_failure_rounds = 0
        return await self.runtime.grant_more_or_finalize(state)

    async def _reject_terminal_batch(
        self,
        state: LightweightState,
        response: LLMResponse,
    ) -> LightweightState:
        """Reject a mixed terminal batch before invoking any tool handler."""

        tool_uses = [call.to_anthropic_tool_use() for call in state.pending_tool_calls]
        state.messages = default_add_assistant_message(
            state.messages,
            response.content,
            tool_uses=tool_uses,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        failures: list[dict[str, Any]] = []
        for call in state.pending_tool_calls:
            result = (
                "Error: A terminal tool must be the response's only tool call; "
                "the entire batch was rejected without execution."
            )
            state.messages = default_add_tool_result(
                state.messages,
                call.id,
                call.name,
                result,
            )
            failures.append(
                {
                    "kind": "terminal_batch",
                    "tool_name": call.name,
                    "arguments": call.arguments,
                    "result": result,
                }
            )
        self._clear_pending(state)
        self.runtime.describe_step(
            {"executed": 0, "failed": len(failures), "rejected_terminal_batch": True}
        )
        return await self.runtime.return_after_tool_errors(
            state,
            failures=tuple(failures),
        )

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
            state.metadata.phase = AgentNode.ROUTER
            self.runtime.describe_step({"route": "router", "reason": "no guard request"})
            return state

        remaining_tokens = (
            None
            if self.runtime.max_tokens is None
            else max(0, self.runtime.max_tokens - state.token_ledger.total_tokens)
        )
        resolution = await self.runtime.guard.decide(
            request,
            token_budget=remaining_tokens,
        )
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
            state.metadata.phase = AgentNode.ROUTER
            return state
        return await self.runtime.switch_to_finalization(
            state,
            evidence=request.evidence,
            source=resolution.source,
        )


def _append_tool_patches(
    state: LightweightState,
    patches: Sequence[Any],
) -> None:
    for patch in patches:
        state.messages = default_add_tool_result(
            state.messages,
            patch.call_id,
            patch.tool_name,
            patch.content,
        )


def _guard_fallback_action(
    *,
    event: GuardEvent,
    allowed_actions: tuple[GuardAction, ...],
    iteration: int,
    iteration_limit: int,
) -> GuardAction:
    """Host-owned fallback when Guard itself fails.

    Budget reviews prefer CONTINUE so a flaky Guard response does not force
    wrap-up. Tool errors prefer RETRY while budget remains; otherwise finalize.
    """

    allowed = set(allowed_actions)
    if event == GuardEvent.BUDGET_EXHAUSTED and GuardAction.CONTINUE in allowed:
        return GuardAction.CONTINUE
    if (
        event == GuardEvent.TOOL_ERROR
        and GuardAction.RETRY in allowed
        and iteration < iteration_limit
    ):
        return GuardAction.RETRY
    if GuardAction.FINALIZE in allowed:
        return GuardAction.FINALIZE
    if GuardAction.CONTINUE in allowed:
        return GuardAction.CONTINUE
    if GuardAction.RETRY in allowed:
        return GuardAction.RETRY
    return next(iter(allowed_actions))


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
        if node.mode != "read_only" and not tool_result_failed(node.result or node.error)
    )


def _last_user_goal(state: LightweightState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message.get("content") or "")
    return ""


def _recent_outcomes(state: LightweightState) -> tuple[str, ...]:
    outcomes: list[str] = []
    for message in reversed(state.messages):
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            text = "\n".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
            if text:
                outcomes.append(text)
        elif role == "user" and isinstance(content, list):
            text = "\n".join(
                str(block.get("content") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ).strip()
            if text:
                outcomes.append(text)
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
    if node is AgentNode.GUARD:
        return NodeKind.HARNESS
    return NodeKind.DOMAIN


def _active_tool_names(tool_defs: list[dict[str, Any]]) -> list[str]:
    return [
        str(definition["name"])
        for definition in tool_defs
        if isinstance(definition.get("name"), str)
    ]
