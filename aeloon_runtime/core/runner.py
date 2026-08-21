"""Stateless public agent-run entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from aeloon_runtime.core.compaction import (
    ContextCompactor,
    ContextPolicy,
    ContextUpdate,
    is_context_overflow,
    should_compact,
)
from aeloon_runtime.core.context_stats import estimate_context_tokens, estimate_tokens
from aeloon_runtime.core.control import RunController
from aeloon_runtime.core.events import RunEventDispatcher
from aeloon_runtime.core.inference_runtime import InferenceRuntime, project_inference_messages
from aeloon_runtime.core.tool_runtime import ToolRuntime
from aeloon_runtime.core.types import (
    AgentMessage,
    AssistantMessage,
    InferencePort,
    Model,
    QueueMode,
    RunError,
    RunEventSink,
    RunHook,
    StopReason,
    StreamOptions,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
    message_from_dict,
    message_to_dict,
)


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Every dependency and input required for exactly one agent run."""

    run_id: str
    inference: InferencePort
    model: Model
    messages: tuple[AgentMessage, ...]
    input: tuple[UserMessage, ...]
    system_prompt: str
    tools: tuple[Tool, ...] = ()
    active_tool_names: tuple[str, ...] | None = None
    stream_options: StreamOptions = field(default_factory=StreamOptions)
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    context_id: str | None = None
    compaction_boundary_ms: int | None = None
    compaction_boundary_index: int | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    final_message: AssistantMessage
    new_messages: tuple[AgentMessage, ...]
    usage: Usage
    stop_reason: StopReason
    context_update: ContextUpdate | None = None


RunHooks = Mapping[str, Sequence[RunHook]]


async def run_agent(
    request: RunRequest,
    *,
    controller: RunController | None = None,
    emit: RunEventSink | None = None,
    hooks: RunHooks | None = None,
    compactor: ContextCompactor | None = None,
) -> RunResult:
    """Execute one run without retaining state after the await completes."""

    if not request.run_id.strip():
        raise RunError("invalid_argument", "run_id must not be empty")
    if not request.input:
        raise RunError("invalid_argument", "Run input must not be empty")
    events = RunEventDispatcher(emit)
    for event_type, handlers in (hooks or {}).items():
        for handler in handlers:
            events.on(event_type, handler)
    run_controller = controller or RunController(
        steering_mode=request.steering_mode,
        follow_up_mode=request.follow_up_mode,
    )
    engine = _RunEngine(request, run_controller, events, compactor)
    return await engine.run()


class _RunEngine:
    """Invocation-local mutable state; never escapes ``run_agent``."""

    def __init__(
        self,
        request: RunRequest,
        controller: RunController,
        events: RunEventDispatcher,
        compactor: ContextCompactor | None,
    ) -> None:
        self.request = request
        self.controller = controller
        self.events = events
        self.compactor = compactor
        self.messages = list(request.messages)
        self.compaction_boundary_ms = request.compaction_boundary_ms
        self.compaction_boundary_index = request.compaction_boundary_index
        self.new_messages: list[AgentMessage] = []
        self.context_update: ContextUpdate | None = None
        active_names = request.active_tool_names
        if active_names is None:
            active_names = tuple(tool.name for tool in request.tools)
        self.tools = ToolRuntime(request.tools, active_names, events)
        self.inference = InferenceRuntime(request.inference, events)

    async def run(self) -> RunResult:
        await self.controller._bind(self.events, self._cancel)
        try:
            initial = list(self.request.input)
            hook = await self.events.hook(
                "before_agent_start",
                {
                    "messages": [message_to_dict(message) for message in initial],
                    "systemPrompt": self.request.system_prompt,
                },
            )
            if "messages" in hook:
                initial = [
                    item if not isinstance(item, Mapping) else message_from_dict(item)
                    for item in hook["messages"]
                ]
            system_prompt = str(hook.get("systemPrompt") or self.request.system_prompt)
            oversized = self._oversized_input_error(initial)
            if oversized is not None:
                final = await self._finish_input_error(initial, oversized)
                return RunResult(
                    final_message=final,
                    new_messages=tuple(self.new_messages),
                    usage=_aggregate_usage(self.new_messages),
                    stop_reason=final.stop_reason,
                    context_update=None,
                )
            if self.compactor is not None and should_compact(
                self._estimate_inference_context((*self.messages, *initial)),
                self.request.model.context_window,
                self.request.context_policy,
            ):
                try:
                    await self._compact("threshold")
                except Exception as exc:
                    await self._emit_compaction_failure("threshold", exc)
            final = await self._run_loop(initial, system_prompt)
            successful_overflow = is_context_overflow(
                final,
                self.request.model.context_window,
            ) and final.stop_reason == "stop"
            if (
                successful_overflow
                and not self.controller.abort_requested
                and self.compactor is not None
                and self.request.context_policy.enabled
            ):
                try:
                    await self._compact("overflow", will_retry=False)
                except Exception as exc:
                    await self._emit_compaction_failure("overflow", exc)
            elif (
                final.stop_reason not in {"error", "aborted"}
                and not self.controller.abort_requested
                and self.compactor is not None
                and should_compact(
                    self._estimate_inference_context(self.messages),
                    self.request.model.context_window,
                    self.request.context_policy,
                )
            ):
                try:
                    await self._compact("threshold")
                except Exception as exc:
                    await self._emit_compaction_failure("threshold", exc)
            return RunResult(
                final_message=final,
                new_messages=tuple(self.new_messages),
                usage=_aggregate_usage(self.new_messages),
                stop_reason=final.stop_reason,
                context_update=self.context_update,
            )
        finally:
            await self.events.emit("settled", {"nextTurnCount": 0})
            self.controller._release()

    def _oversized_input_error(
        self,
        messages: Sequence[AgentMessage],
    ) -> AssistantMessage | None:
        context_window = self.request.model.context_window
        reserve_tokens = self.request.context_policy.reserve_tokens
        usable_budget = max(0, context_window - reserve_tokens)
        for message in messages:
            if not isinstance(message, UserMessage):
                continue
            estimated_tokens = estimate_tokens(message)
            if estimated_tokens <= usable_budget:
                continue
            return AssistantMessage(
                content=(),
                provider=self.request.model.provider,
                model=self.request.model.id,
                stop_reason="error",
                error_message=(
                    "User input exceeds usable context budget: "
                    f"estimated {estimated_tokens} tokens, budget {usable_budget} tokens "
                    f"(context window {context_window}, reserve {reserve_tokens}). "
                    "Historical compaction cannot reduce a single oversized input."
                ),
            )
        return None

    def _estimate_inference_context(self, messages: Sequence[AgentMessage]) -> int:
        projected, boundary_index = project_inference_messages(
            messages,
            boundary_index=self.compaction_boundary_index,
        )
        return estimate_context_tokens(
            projected,
            usage_after_ms=self.compaction_boundary_ms,
            usage_after_index=boundary_index,
        )

    async def _finish_input_error(
        self,
        initial: Sequence[AgentMessage],
        final: AssistantMessage,
    ) -> AssistantMessage:
        await self.events.emit("agent_start")
        await self.events.emit("turn_start")
        for message in initial:
            await self._append(message)
        await self._append(final, attempt=0)
        await self.events.emit(
            "turn_end",
            {"message": message_to_dict(final), "toolResults": []},
        )
        await self._agent_end()
        return final

    async def _run_loop(
        self,
        initial: list[AgentMessage],
        system_prompt: str,
    ) -> AssistantMessage:
        await self.events.emit("agent_start")
        await self.events.emit("turn_start")
        for message in initial:
            await self._append(message)
        if self.controller.abort_requested:
            final = self._aborted_message()
            await self._append(final, attempt=self.inference.last_attempt + 1)
            await self.events.emit(
                "turn_end",
                {"message": message_to_dict(final), "toolResults": []},
            )
            await self._agent_end()
            return final
        pending = await self.controller._drain_steering()
        first_turn = True
        overflow_attempts = 0
        final: AssistantMessage | None = None
        while True:
            has_more_tools = True
            while has_more_tools or pending:
                if not first_turn:
                    await self.events.emit("turn_start")
                first_turn = False
                for message in pending:
                    await self._append(message)
                pending = []
                final = await self._stream_response(system_prompt)
                if (
                    final.stop_reason in {"error", "length"}
                    and overflow_attempts < 2
                    and self.compactor is not None
                    and self.request.context_policy.enabled
                    and is_context_overflow(final, self.request.model.context_window)
                ):
                    overflow_attempts += 1
                    await self.events.emit(
                        "message_end",
                        {
                            "message": message_to_dict(final),
                            "overflowAttempt": True,
                            "attempt": self.inference.last_attempt,
                        },
                    )
                    self.new_messages.append(final)
                    await self.events.emit(
                        "turn_end",
                        {"message": message_to_dict(final), "toolResults": []},
                    )
                    try:
                        await self._compact("overflow")
                    except Exception as exc:
                        await self.events.emit(
                            "compaction_end",
                            {
                                "reason": "overflow",
                                "aborted": False,
                                "willRetry": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        if self.controller.abort_requested:
                            final = self._aborted_message()
                            await self._append(final, attempt=self.inference.last_attempt + 1)
                        await self._agent_end()
                        return final
                    continue
                if (
                    final.stop_reason in {"error", "length"}
                    and overflow_attempts >= 2
                    and is_context_overflow(final, self.request.model.context_window)
                ):
                    policy = self.request.context_policy
                    final = replace(
                        final,
                        stop_reason="error",
                        error_message=(
                            "Context still exceeds the model window after 2 compaction attempts "
                            f"(context window {self.request.model.context_window}, "
                            "effective reserve "
                            f"{policy.reserve_tokens}, effective keep {policy.keep_recent_tokens})."
                        ),
                    )
                await self._append(
                    final,
                    message_started=True,
                    attempt=self.inference.last_attempt,
                )
                if final.stop_reason in {"error", "aborted"}:
                    await self.events.emit(
                        "turn_end", {"message": message_to_dict(final), "toolResults": []}
                    )
                    await self._agent_end()
                    return final
                tool_results: list[ToolResultMessage] = []
                terminate = False
                if final.tool_calls:
                    if final.stop_reason == "length":
                        tool_results = await self.tools.fail_truncated_calls(final.tool_calls)
                    else:
                        tool_results, terminate = await self.tools.execute_calls(
                            final.tool_calls,
                            is_aborted=lambda: self.controller.abort_requested,
                        )
                    for result in tool_results:
                        await self._append(result)
                has_more_tools = bool(final.tool_calls) and not terminate
                await self.events.emit(
                    "turn_end",
                    {
                        "message": message_to_dict(final),
                        "toolResults": [message_to_dict(item) for item in tool_results],
                    },
                )
                if self.controller.abort_requested:
                    await self._agent_end()
                    return final
                pending = await self.controller._drain_steering()
            follow_up = await self.controller._drain_follow_up()
            if follow_up:
                pending = follow_up
                continue
            break
        assert final is not None
        await self._agent_end()
        return final

    async def _stream_response(self, system_prompt: str) -> AssistantMessage:
        return await self.inference.request(
            model=self.request.model,
            messages=self.messages,
            system_prompt=system_prompt,
            tools=self.tools.active_tools,
            session_id=self.request.context_id or self.request.run_id,
            stream_options=self.request.stream_options,
            on_retry=self._inference_retry,
        )

    async def _append(
        self,
        message: AgentMessage,
        *,
        message_started: bool = False,
        attempt: int | None = None,
    ) -> None:
        payload = {"message": message_to_dict(message)}
        if attempt is not None:
            payload["attempt"] = attempt
        if not message_started:
            await self.events.emit("message_start", payload)
        self.messages.append(message)
        self.new_messages.append(message)
        await self.events.emit("message_end", payload)

    async def _compact(self, reason: str, *, will_retry: bool | None = None) -> None:
        assert self.compactor is not None
        await self.events.emit("compaction_start", {"reason": reason})
        update = await self.compactor.compact(
            tuple(self.messages),
            reason=reason,  # type: ignore[arg-type]
        )
        self.messages = list(update.messages)
        self.compaction_boundary_ms = update.compaction_boundary_ms
        self.compaction_boundary_index = update.compaction_boundary_index
        self.context_update = update
        await self.events.emit(
            "context_compacted",
            {
                "compactionEntryId": update.details.get("compactionEntryId"),
                "summary": update.summary,
                "fromHook": False,
            },
        )
        await self.events.emit(
            "compaction_end",
            {
                "reason": reason,
                "aborted": False,
                "willRetry": reason == "overflow" if will_retry is None else will_retry,
            },
        )

    async def _emit_compaction_failure(self, reason: str, exc: Exception) -> None:
        await self.events.emit(
            "compaction_end",
            {
                "reason": reason,
                "aborted": False,
                "willRetry": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    async def _inference_retry(self, data: dict[str, Any]) -> None:
        event_type = "auto_retry_start" if data.get("stage") == "start" else "auto_retry_end"
        await self.events.emit(event_type, data)  # type: ignore[arg-type]

    async def _agent_end(self) -> None:
        await self.events.emit(
            "agent_end",
            {"messages": [message_to_dict(item) for item in self.new_messages]},
        )

    def _aborted_message(self) -> AssistantMessage:
        return AssistantMessage(
            content=(),
            provider=self.request.model.provider,
            model=self.request.model.id,
            stop_reason="aborted",
            error_message="Operation aborted",
        )

    def _cancel(self) -> None:
        self.inference.cancel()
        self.tools.cancel()
        cancel_compaction = getattr(self.compactor, "cancel", None)
        if callable(cancel_compaction):
            cancel_compaction()


def _aggregate_usage(messages: Sequence[AgentMessage]) -> Usage:
    usages: list[Usage] = []
    for message in messages:
        usage = getattr(message, "usage", None)
        if isinstance(usage, Usage):
            usages.append(usage)
    costs: dict[str, float] = {}
    for usage in usages:
        for key, value in usage.cost.items():
            costs[key] = costs.get(key, 0.0) + float(value)
    return Usage(
        input=sum(item.input for item in usages),
        output=sum(item.output for item in usages),
        cache_read=sum(item.cache_read for item in usages),
        cache_write=sum(item.cache_write for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        reasoning=sum(item.reasoning or 0 for item in usages) or None,
        cost=costs,
    )


__all__ = ["RunHooks", "RunRequest", "RunResult", "run_agent"]
