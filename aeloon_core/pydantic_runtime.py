"""One PydanticAI execution engine shared by Master and Worker actors."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ValidationError
from pydantic_ai import (
    Agent,
    ModelRetry,
    RunContext,
    Tool,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.capabilities import Hooks, ProcessHistory
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from aeloon_core.message_history import (
    MESSAGE_FORMAT,
    MESSAGE_SCHEMA_VERSION,
    deserialize_messages,
    serialize_messages,
)
from aeloon_core.pydantic_model import (
    PromptCacheState,
    is_prompt_caching_unsupported_error,
    prompt_caching_enabled,
    without_prompt_caching,
)
from aeloon_core.runtime_events import (
    ModelResponseView,
    ToolCallView,
    ToolExecutionRecord,
    ToolExecutionState,
)
from aeloon_core.stuck_detection import detect_repeated_tool_exchanges
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import NodeKind, TransitionRecord

AgentRole = Literal["master", "worker"]
OutputValidator = Callable[..., Awaitable[Any] | Any]
HistoryProcessor = Callable[
    [RunContext["AeloonRunDeps"], list[ModelMessage]],
    Awaitable[list[ModelMessage]] | list[ModelMessage],
]


class AgentRunStatus(StrEnum):
    COMPLETED = "completed"
    LIMIT_EXCEEDED = "limit_exceeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    """Immutable host-owned capability boundary for one model run."""

    namespace: str
    tool_names: tuple[str, ...]
    terminal_names: tuple[str, ...]
    snapshot_digest: str | None = None

    @classmethod
    def from_registry(
        cls,
        registry: ToolRegistry,
        *,
        namespace: str,
        terminal_names: Sequence[str] = (),
        snapshot_digest: str | None = None,
    ) -> CapabilityManifest:
        return cls(
            namespace=namespace,
            tool_names=tuple(
                str(definition["name"]) for definition in registry.get_definitions()
            ),
            terminal_names=tuple(terminal_names),
            snapshot_digest=snapshot_digest,
        )


@dataclass(slots=True)
class AeloonRunDeps:
    """Host-owned policy and observability state available to one Agent run."""

    role: AgentRole
    tools: ToolRegistry
    terminal_models: dict[str, type[BaseModel]]
    capability_manifest: CapabilityManifest
    progress: Any | None = None
    session_id: str | None = None
    turn_id: str | None = None
    max_tokens: int | None = None
    stuck_detection_enabled: bool = True
    stuck_detection_threshold: int = 4
    prompt_cache: PromptCacheState | None = None
    tools_used: list[str] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    tool_calls: dict[str, ToolCallView] = field(default_factory=dict)
    progress_call_ids: set[str] = field(default_factory=set)
    progress_result_ids: set[str] = field(default_factory=set)
    transition_sequence: int = 0
    transition_digest: str = ""

    @property
    def tool_modes(self) -> dict[str, str]:
        return {
            definition["name"]: self.tools.get(str(definition["name"])).concurrency_mode
            for definition in self.tools.get_definitions()
            if self.tools.get(str(definition["name"])) is not None
        }


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """One host tool result retained for deterministic terminal validation."""

    name: str
    arguments: dict[str, Any]
    result: str


@dataclass(slots=True)
class AgentRunSpec:
    """Everything that varies between one Master or Worker invocation."""

    role: AgentRole
    model: Model
    instructions: str
    prompt: str
    history: list[ModelMessage]
    tools: ToolRegistry
    output_type: Any
    terminal_models: dict[str, type[BaseModel]]
    capability_manifest: CapabilityManifest | None = None
    model_settings: ModelSettings | None = None
    request_limit: int = 25
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    progress: Any | None = None
    session_id: str | None = None
    turn_id: str | None = None
    output_validator: OutputValidator | None = None
    history_processor: HistoryProcessor | None = None
    transition_trace_enabled: bool = True
    stuck_detection_enabled: bool = True
    stuck_detection_threshold: int = 4
    prompt_cache: PromptCacheState | None = None
    on_transition: Callable[[TransitionRecord], Any] | None = None
    capabilities: Sequence[Any] = ()


@dataclass(slots=True)
class AgentRunOutcome:
    status: AgentRunStatus
    output: Any | None
    messages: list[ModelMessage]
    usage: dict[str, int]
    tools_used: list[str]
    transitions: list[TransitionRecord]
    failure: str | None = None


class AeloonToolset(FunctionToolset[AeloonRunDeps]):
    """Expose exactly one host-resolved ToolRegistry to PydanticAI."""

    def __init__(self, registry: ToolRegistry, *, max_retries: int) -> None:
        super().__init__(max_retries=max_retries, id="aeloon-tools")
        self.registry = registry
        for definition in registry.get_definitions():
            name = str(definition["name"])
            host_tool = registry.get(name)
            assert host_tool is not None
            self.add_tool(
                Tool.from_schema(
                    self._call_host_tool(name),
                    name=name,
                    description=str(definition.get("description") or ""),
                    json_schema=dict(definition["input_schema"]),
                    takes_ctx=True,
                    sequential=host_tool.concurrency_mode != "read_only",
                    args_validator=self._validate_host_tool(name),
                )
            )

    def _call_host_tool(self, name: str) -> Callable[..., Awaitable[str]]:
        async def call(ctx: RunContext[AeloonRunDeps], **kwargs: Any) -> str:
            result = await self.registry.execute(name, kwargs)
            ctx.deps.tools_used.append(name)
            ctx.deps.tool_observations.append(
                ToolObservation(
                    name=name,
                    arguments=dict(kwargs),
                    result=result,
                )
            )
            return result

        return call

    def _validate_host_tool(self, name: str) -> Callable[..., None]:
        def validate(_ctx: RunContext[AeloonRunDeps], **kwargs: Any) -> None:
            tool = self.registry.get(name)
            if tool is None:
                raise ModelRetry(f"Tool {name!r} is no longer available in this namespace.")
            try:
                tool.args_model.model_validate(kwargs)
            except ValidationError as exc:
                raise ModelRetry(_validation_retry(name, exc)) from exc

        return validate


class HarnessAgentRuntime:
    """Compose Pydantic AI with Harness capabilities and Aeloon policy hooks."""

    async def run(self, spec: AgentRunSpec) -> AgentRunOutcome:
        request_limit = max(1, int(spec.request_limit))
        manifest = spec.capability_manifest or CapabilityManifest.from_registry(
            spec.tools,
            namespace=spec.role,
            terminal_names=spec.terminal_models,
        )
        _validate_capability_manifest(spec, manifest)
        deps = AeloonRunDeps(
            role=spec.role,
            tools=spec.tools,
            terminal_models=dict(spec.terminal_models),
            capability_manifest=manifest,
            progress=spec.progress,
            session_id=spec.session_id,
            turn_id=spec.turn_id,
            max_tokens=spec.max_tokens,
            stuck_detection_enabled=spec.stuck_detection_enabled,
            stuck_detection_threshold=spec.stuck_detection_threshold,
            prompt_cache=spec.prompt_cache,
            transition_digest=_messages_digest(spec.history),
        )
        hooks = self._hooks(spec, deps)
        capabilities: list[Any] = [hooks, *spec.capabilities]
        if spec.history_processor is not None:
            capabilities.append(ProcessHistory(spec.history_processor))

        agent = Agent[AeloonRunDeps, Any](
            spec.model,
            output_type=spec.output_type,
            instructions=spec.instructions,
            deps_type=AeloonRunDeps,
            toolsets=[AeloonToolset(spec.tools, max_retries=request_limit)],
            retries=request_limit,
            end_strategy="early",
            capabilities=capabilities,
        )
        if spec.output_validator is not None:
            validator = spec.output_validator

            @agent.output_validator
            async def validate_output(
                _ctx: RunContext[AeloonRunDeps], output: Any
            ) -> Any:
                validated = _invoke_output_validator(validator, _ctx, output)
                if inspect.isawaitable(validated):
                    return await validated
                return validated

        usage = RunUsage()
        captured: list[ModelMessage]
        result: AgentRunResult[Any] | None = None
        status = AgentRunStatus.COMPLETED
        failure: str | None = None
        with capture_run_messages() as captured:
            try:
                model_settings = dict(spec.model_settings or {})
                if spec.max_output_tokens is not None:
                    configured = model_settings.get("max_tokens")
                    model_settings["max_tokens"] = (
                        min(int(configured), spec.max_output_tokens)
                        if configured is not None
                        else spec.max_output_tokens
                    )
                event_stream_handler = self._event_stream_handler
                if (
                    hasattr(spec.model, "stream_function")
                    and spec.model.stream_function is None
                ):
                    event_stream_handler = None

                result = await agent.run(
                    spec.prompt,
                    message_history=spec.history,
                    deps=deps,
                    model_settings=model_settings,
                    usage_limits=UsageLimits(
                        request_limit=request_limit,
                        tool_calls_limit=spec.max_tool_calls,
                        total_tokens_limit=spec.max_tokens,
                    ),
                    usage=usage,
                    event_stream_handler=event_stream_handler,
                )
            except UsageLimitExceeded as exc:
                status = AgentRunStatus.LIMIT_EXCEEDED
                failure = str(exc)
            except UnexpectedModelBehavior as exc:
                status = AgentRunStatus.FAILED
                failure = str(exc)

        messages = list(result.all_messages() if result is not None else captured)
        if not messages:
            messages = list(spec.history)
        output = result.output if result is not None else None
        if result is not None:
            await _emit_progress(
                spec.progress,
                "on_final",
                _output_text(output),
                messages=serialize_messages(messages),
            )
            await self._record_transition(
                deps,
                spec,
                node="run_finished",
                node_kind=NodeKind.HARNESS,
                decision={"status": status.value},
                after_messages=messages,
            )
        return AgentRunOutcome(
            status=status,
            output=output,
            messages=messages,
            usage=_usage_dict(usage),
            tools_used=list(deps.tools_used),
            transitions=list(deps.transitions),
            failure=failure,
        )

    def _hooks(self, spec: AgentRunSpec, deps: AeloonRunDeps) -> Hooks:
        hooks = Hooks[AeloonRunDeps]()

        @hooks.on.before_model_request
        async def before_model_request(
            ctx: RunContext[AeloonRunDeps], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if ctx.deps.prompt_cache is not None and ctx.deps.prompt_cache.disabled:
                request_context.model_settings = without_prompt_caching(
                    dict(request_context.model_settings or {})
                )
            if ctx.deps.max_tokens is not None:
                await _apply_hard_token_budget(ctx, request_context)
            await self._record_transition(
                deps,
                spec,
                node="model_request",
                node_kind=NodeKind.HARNESS,
                decision={"run_step": ctx.run_step},
                before_messages=request_context.messages,
                after_messages=request_context.messages,
            )
            return request_context

        @hooks.on.model_request
        async def model_request(
            _ctx: RunContext[AeloonRunDeps], *, request_context: ModelRequestContext, handler: Any
        ) -> ModelResponse:
            try:
                return await handler(request_context)
            except ModelHTTPError as exc:
                cache = deps.prompt_cache
                settings = dict(request_context.model_settings or {})
                if (
                    cache is not None
                    and not cache.disabled
                    and prompt_caching_enabled(settings)
                    and is_prompt_caching_unsupported_error(exc)
                ):
                    cache.disabled = True
                    logger.warning(
                        "Provider rejected prompt caching; retrying this model request without it"
                    )
                    return await handler(
                        replace(
                            request_context,
                            model_settings=without_prompt_caching(settings),
                        )
                    )
                raise

        @hooks.on.after_model_request
        async def after_model_request(
            ctx: RunContext[AeloonRunDeps],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            _validate_entire_tool_batch(ctx.deps, calls)
            _reject_repeated_stuck_call(ctx, calls)
            view = _model_response_view(response)
            await _emit_progress(
                ctx.deps.progress,
                "on_llm_response",
                view,
                component=ctx.deps.role,
            )
            await self._record_transition(
                deps,
                spec,
                node="model_response",
                node_kind=NodeKind.HARNESS,
                decision={
                    "finish_reason": response.finish_reason,
                    "tool_names": [call.tool_name for call in calls],
                },
                usage=_usage_dict(response.usage),
                before_messages=request_context.messages,
                after_messages=[*request_context.messages, response],
            )
            return response

        @hooks.on.before_tool_execute
        async def before_tool_execute(
            ctx: RunContext[AeloonRunDeps],
            *,
            call: ToolCallPart,
            tool_def: Any,
            args: Any,
        ) -> Any:
            del tool_def
            view = _tool_call_view(call)
            ctx.deps.tool_calls[view.id] = view
            if view.id not in ctx.deps.progress_call_ids:
                ctx.deps.progress_call_ids.add(view.id)
                await _emit_progress(ctx.deps.progress, "on_tool_calls", [view])
            await self._record_transition(
                deps,
                spec,
                node="tool_call",
                node_kind=NodeKind.DOMAIN,
                decision={"id": view.id, "name": view.name, "arguments": view.arguments},
            )
            return args

        @hooks.on.after_tool_execute
        async def after_tool_execute(
            ctx: RunContext[AeloonRunDeps],
            *,
            call: ToolCallPart,
            tool_def: Any,
            args: Any,
            result: Any,
        ) -> Any:
            del tool_def, args
            view = ctx.deps.tool_calls.get(call.tool_call_id) or _tool_call_view(call)
            result_text = result if isinstance(result, str) else json.dumps(
                result, ensure_ascii=False, default=str
            )
            if ctx.deps.tools.get(view.name) is None:
                # Harness and other capability tools execute outside
                # AeloonToolset, so account for them here.
                ctx.deps.tools_used.append(view.name)
                ctx.deps.tool_observations.append(
                    ToolObservation(
                        name=view.name,
                        arguments=dict(view.arguments),
                        result=result_text,
                    )
                )
            if view.id not in ctx.deps.progress_result_ids:
                ctx.deps.progress_result_ids.add(view.id)
                await _emit_progress(
                    ctx.deps.progress,
                    "on_tool_result",
                    _execution_record(ctx.deps, view, result_text),
                )
            await self._record_transition(
                deps,
                spec,
                node="tool_result",
                node_kind=NodeKind.DOMAIN,
                decision={"id": view.id, "name": view.name, "result": result_text},
            )
            return result

        @hooks.on.after_output_validate
        async def after_output_validate(
            ctx: RunContext[AeloonRunDeps],
            *,
            output_context: Any,
            output: Any,
        ) -> Any:
            del ctx, output_context
            await self._record_transition(
                deps,
                spec,
                node="output",
                node_kind=NodeKind.HARNESS,
                decision={"type": type(output).__name__},
            )
            return output

        return hooks

    async def _event_stream_handler(
        self,
        ctx: RunContext[AeloonRunDeps],
        events: Any,
    ) -> None:
        tool_index = 0
        async for event in events:
            if isinstance(event, PartDeltaEvent):
                delta = event.delta
                if isinstance(delta, TextPartDelta) and delta.content_delta:
                    await _emit_progress(ctx.deps.progress, "on_llm_delta", delta.content_delta)
                elif isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                    await _emit_progress(
                        ctx.deps.progress,
                        "on_llm_reasoning_delta",
                        delta.content_delta,
                    )
                continue
            if isinstance(event, FunctionToolCallEvent):
                call = _tool_call_view(event.part)
                ctx.deps.tool_calls[call.id] = call
                if call.id not in ctx.deps.progress_call_ids:
                    ctx.deps.progress_call_ids.add(call.id)
                    await _emit_progress(ctx.deps.progress, "on_tool_calls", [call])
                tool_index += 1
                continue
            if isinstance(event, FunctionToolResultEvent):
                part = event.part
                call = ctx.deps.tool_calls.get(part.tool_call_id)
                if call is None:
                    call = ToolCallView(part.tool_call_id, part.tool_name or "tool", {})
                result = _tool_result_text(part)
                if call.id not in ctx.deps.progress_result_ids:
                    ctx.deps.progress_result_ids.add(call.id)
                    record = _execution_record(
                        ctx.deps,
                        call,
                        result,
                        index=max(0, tool_index - 1),
                        failed=isinstance(part, RetryPromptPart),
                    )
                    await _emit_progress(ctx.deps.progress, "on_tool_result", record)

    async def _record_transition(
        self,
        deps: AeloonRunDeps,
        spec: AgentRunSpec,
        *,
        node: str,
        node_kind: NodeKind,
        decision: Any,
        usage: dict[str, int] | None = None,
        before_messages: Sequence[ModelMessage] | None = None,
        after_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        if not spec.transition_trace_enabled:
            return
        deps.transition_sequence += 1
        before_digest = (
            _messages_digest(before_messages)
            if before_messages is not None
            else deps.transition_digest
        )
        after_digest = (
            _messages_digest(after_messages)
            if after_messages is not None
            else before_digest
        )
        deps.transition_digest = after_digest
        record = TransitionRecord(
            sequence=deps.transition_sequence,
            iteration=deps.transition_sequence,
            node=node,
            node_kind=node_kind,
            before_digest=before_digest,
            after_digest=after_digest,
            session_id=deps.session_id,
            turn_id=deps.turn_id,
            decision=decision,
            token_usage=usage or {},
            component="pydantic_ai",
        )
        deps.transitions.append(record)
        if spec.on_transition is not None:
            emitted = spec.on_transition(record)
            if inspect.isawaitable(emitted):
                await emitted


def output_tools(*specs: tuple[type[BaseModel], str, str]) -> list[ToolOutput[Any]]:
    """Build typed terminal outputs with stable names and descriptions."""

    return [
        ToolOutput(model, name=name, description=description, sequential=True)
        for model, name, description in specs
    ]


def _validate_capability_manifest(
    spec: AgentRunSpec,
    manifest: CapabilityManifest,
) -> None:
    exposed = tuple(
        str(definition["name"]) for definition in spec.tools.get_definitions()
    )
    if exposed != manifest.tool_names:
        raise ValueError("ToolRegistry does not match the host capability manifest")
    if set(spec.terminal_models) != set(manifest.terminal_names):
        raise ValueError("typed outputs do not match the host capability manifest")
    if set(exposed) & set(manifest.terminal_names):
        raise ValueError("terminal outputs must not be registered as ordinary tools")


def _validate_entire_tool_batch(
    deps: AeloonRunDeps,
    calls: list[ToolCallPart],
) -> None:
    terminal_names = set(deps.terminal_models)
    called_terminals = [call for call in calls if call.tool_name in terminal_names]
    if called_terminals and (len(calls) != 1 or len(called_terminals) != 1):
        raise ModelRetry(
            "A terminal output must be the response's only tool call. No tool was executed."
        )
    errors: list[str] = []
    for call in calls:
        model = deps.terminal_models.get(call.tool_name)
        if model is None:
            tool = deps.tools.get(call.tool_name)
            model = tool.args_model if tool is not None else None
        if model is None:
            # Capability-contributed tools (for example Harness
            # `run_workflow`) are not part of Aeloon's host ToolRegistry.
            # Pydantic AI owns their schema validation and execution.
            continue
        try:
            arguments = call.args_as_dict(raise_if_invalid=True)
            model.model_validate(arguments)
        except (ValueError, ValidationError) as exc:
            errors.append(f"{call.tool_name}: {exc}")
    if errors:
        raise ModelRetry(
            "The entire tool batch was rejected before execution: " + "; ".join(errors)
        )


def _reject_repeated_stuck_call(
    ctx: RunContext[AeloonRunDeps],
    calls: list[ToolCallPart],
) -> None:
    deps = ctx.deps
    if not deps.stuck_detection_enabled or not calls:
        return
    detection = detect_repeated_tool_exchanges(
        _legacy_detection_messages(ctx.messages),
        tool_modes=deps.tool_modes,
        threshold=deps.stuck_detection_threshold,
    )
    if detection is None:
        return
    repeated = any(
        _action_digest(call.tool_name, call.args_as_dict()) == detection.action_digest
        for call in calls
    )
    if repeated:
        raise ModelRetry(
            f"The same successful read-only action and observation repeated "
            f"{detection.repetitions} times. Change strategy before calling another tool."
        )


async def _apply_hard_token_budget(
    ctx: RunContext[AeloonRunDeps],
    request_context: ModelRequestContext,
) -> None:
    limit = ctx.deps.max_tokens
    assert limit is not None
    try:
        counted = await request_context.model.count_tokens(
            request_context.messages,
            request_context.model_settings,
            request_context.model_request_parameters,
        )
        next_input = max(0, int(counted.input_tokens))
    except Exception as exc:
        logger.warning("Falling back to conservative local token preflight: {}", exc)
        payload = ModelMessagesTypeAdapter.dump_json(request_context.messages)
        next_input = max(1, len(payload) // 3 + 512)
    remaining = limit - ctx.usage.total_tokens - next_input
    if remaining < 1:
        raise UsageLimitExceeded(
            f"The next model request would exceed the hard token budget ({limit})."
        )
    settings = dict(request_context.model_settings or {})
    configured = settings.get("max_tokens")
    settings["max_tokens"] = min(int(configured), remaining) if configured else remaining
    request_context.model_settings = settings


def _model_response_view(response: ModelResponse) -> ModelResponseView:
    text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
    reasoning = "".join(
        part.content for part in response.parts if isinstance(part, ThinkingPart)
    )
    calls = tuple(
        _tool_call_view(part) for part in response.parts if isinstance(part, ToolCallPart)
    )
    return ModelResponseView(
        content=text or None,
        reasoning_content=reasoning or None,
        tool_calls=calls,
        usage=_usage_dict(response.usage),
        finish_reason=str(response.finish_reason) if response.finish_reason else None,
    )


def _tool_call_view(part: ToolCallPart) -> ToolCallView:
    try:
        arguments = part.args_as_dict()
    except ValueError:
        arguments = {}
    return ToolCallView(id=part.tool_call_id, name=part.tool_name, arguments=arguments)


def _tool_result_text(part: ToolReturnPart | RetryPromptPart) -> str:
    content = part.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _execution_record(
    deps: AeloonRunDeps,
    call: ToolCallView,
    result: str,
    *,
    index: int | None = None,
    failed: bool | None = None,
) -> ToolExecutionRecord:
    is_failed = result.lstrip().lower().startswith("error") if failed is None else failed
    tool = deps.tools.get(call.name)
    return ToolExecutionRecord(
        index=max(0, len(deps.tools_used) - 1) if index is None else index,
        call_id=call.id,
        tool_name=call.name,
        arguments=call.arguments,
        mode=tool.concurrency_mode if tool is not None else "exclusive",
        state=ToolExecutionState.FAILED if is_failed else ToolExecutionState.DONE,
        result=result,
        error=result if is_failed else None,
    )


def _usage_dict(usage: Any) -> dict[str, int]:
    try:
        values = asdict(usage)
    except TypeError:
        values = dict(getattr(usage, "__dict__", {}) or {})
    normalized = {
        str(key): max(0, int(value))
        for key, value in values.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int | float):
        normalized["total_tokens"] = max(0, int(total))
    return normalized


def _invoke_output_validator(
    validator: OutputValidator,
    ctx: RunContext[AeloonRunDeps],
    output: Any,
) -> Any:
    """Call new context-aware validators without breaking one-argument callers."""

    try:
        inspect.signature(validator).bind(ctx, output)
    except (TypeError, ValueError):
        return validator(output)
    return validator(ctx, output)


async def _emit_progress(progress: Any, name: str, *args: Any, **kwargs: Any) -> None:
    hook = getattr(progress, name, None)
    if hook is None:
        return
    try:
        value = hook(*args, **kwargs)
        if inspect.isawaitable(value):
            await value
    except Exception as exc:
        logger.warning("Ignoring progress observer failure in {}: {}", name, exc)


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    final_content = getattr(output, "final_content", None)
    if isinstance(final_content, str):
        return final_content
    summary = getattr(output, "summary", None)
    return summary if isinstance(summary, str) else ""


def _validation_retry(name: str, exc: ValidationError) -> str:
    errors = "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: "
        f"{error.get('msg', 'invalid value')}"
        for error in exc.errors()
    )
    return f"Invalid arguments for {name}: {errors}"


def _messages_digest(messages: Sequence[ModelMessage]) -> str:
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json(list(messages))).hexdigest()


def _action_digest(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _legacy_detection_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Project typed history into the detector's bounded compatibility shape."""

    projected: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            blocks: list[dict[str, Any]] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    blocks.append({"type": "text", "text": part.content})
                elif isinstance(part, ToolCallPart):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "input": part.args_as_dict(),
                        }
                    )
            projected.append({"role": "assistant", "content": blocks})
            continue
        if not isinstance(message, ModelRequest):
            continue
        blocks = []
        real_user: list[str] = []
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": part.tool_call_id,
                        "content": _tool_result_text(part),
                        "is_error": part.outcome != "success",
                    }
                )
            elif getattr(part, "part_kind", None) == "user-prompt":
                real_user.append(str(getattr(part, "content", "")))
        projected.append(
            {"role": "user", "content": blocks if blocks else "\n".join(real_user)}
        )
    return projected


# Compatibility for callers that imported the pre-Harness runtime name.
PydanticAgentRuntime = HarnessAgentRuntime


__all__ = [
    "AeloonRunDeps",
    "AeloonToolset",
    "AgentRunOutcome",
    "AgentRunSpec",
    "AgentRunStatus",
    "CapabilityManifest",
    "HarnessAgentRuntime",
    "MESSAGE_FORMAT",
    "MESSAGE_SCHEMA_VERSION",
    "PydanticAgentRuntime",
    "deserialize_messages",
    "output_tools",
    "serialize_messages",
]
