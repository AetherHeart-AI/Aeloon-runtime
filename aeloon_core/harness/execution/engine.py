"""One pi-core execution engine shared by Master and Expert actors."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from loguru import logger
from pydantic import BaseModel, ValidationError

from aeloon_core.conversation.history import (
    MESSAGE_FORMAT,
    MESSAGE_SCHEMA_VERSION,
    PiMessage,
    deserialize_messages,
    serialize_messages,
)
from aeloon_core.harness.execution.bridge import PiRuntimeBridge
from aeloon_core.harness.execution.events import (
    ModelResponseView,
    ToolCallView,
    ToolExecutionRecord,
    ToolExecutionState,
)
from aeloon_core.harness.execution.stuck import detect_repeated_tool_exchanges
from aeloon_core.harness.execution.transitions import NodeKind, TransitionRecord
from aeloon_core.harness.mcp.registry import connect_mcp_toolsets
from aeloon_core.harness.provider import PiModelLike, PiModelSettings
from aeloon_core.harness.tool.registry import ToolRegistry

AgentRole = Literal["master", "expert"]
OutputValidator = Callable[..., Awaitable[Any] | Any]


class AgentRunStatus(StrEnum):
    COMPLETED = "completed"
    LIMIT_EXCEEDED = "limit_exceeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TerminalOutput:
    """One model-visible terminal tool backed by a Pydantic result model."""

    model: type[BaseModel]
    name: str
    description: str

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _strip_titles(self.model.model_json_schema()),
        }


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    """Immutable host-owned capability boundary for one Pi run."""

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
        additional_tool_names: Sequence[str] = (),
    ) -> CapabilityManifest:
        return cls(
            namespace=namespace,
            tool_names=(
                *(str(definition["name"]) for definition in registry.get_definitions()),
                *additional_tool_names,
            ),
            terminal_names=tuple(terminal_names),
            snapshot_digest=snapshot_digest,
        )


@dataclass(slots=True)
class AeloonRunDeps:
    """Host-owned policy and observability state for one pi-core loop."""

    role: AgentRole
    tools: ToolRegistry
    terminal_models: dict[str, type[BaseModel]]
    capability_manifest: CapabilityManifest
    progress: Any | None = None
    session_id: str | None = None
    turn_id: str | None = None
    max_tokens: int | None = None
    max_tool_calls: int | None = None
    max_retries: int = 3
    stuck_detection_enabled: bool = True
    stuck_detection_threshold: int = 4
    tools_used: list[str] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    tool_calls: dict[str, ToolCallView] = field(default_factory=dict)
    transition_sequence: int = 0
    transition_digest: str = ""
    tool_call_count: int = 0
    validation_failures: int = 0

    @property
    def tool_modes(self) -> dict[str, str]:
        modes = {
            definition["name"]: self.tools.get(str(definition["name"])).concurrency_mode
            for definition in self.tools.get_definitions()
            if self.tools.get(str(definition["name"])) is not None
        }
        for name in self.capability_manifest.tool_names:
            modes.setdefault(name, "exclusive")
        return modes


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """One host tool result retained for deterministic terminal validation."""

    name: str
    arguments: dict[str, Any]
    result: str


@dataclass(slots=True)
class PiRunContext:
    """Minimal context supplied to compatibility output validators."""

    deps: AeloonRunDeps
    messages: list[PiMessage]


@dataclass(slots=True)
class AgentRunSpec:
    """Everything that varies between one Master or Expert invocation."""

    role: AgentRole
    model: PiModelLike
    instructions: str
    prompt: str
    history: list[PiMessage]
    tools: ToolRegistry
    output_type: Any
    terminal_models: dict[str, type[BaseModel]]
    capability_manifest: CapabilityManifest | None = None
    model_settings: PiModelSettings | None = None
    request_limit: int | None = None
    max_retries: int = 3
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    progress: Any | None = None
    session_id: str | None = None
    turn_id: str | None = None
    output_validator: OutputValidator | None = None
    on_transition: Callable[[TransitionRecord], Any] | None = None
    transition_trace_enabled: bool = True
    stuck_detection_enabled: bool = True
    stuck_detection_threshold: int = 4
    capabilities: Sequence[Any] = ()
    toolsets: Sequence[Any] = ()


@dataclass(slots=True)
class AgentRunOutcome:
    status: AgentRunStatus
    output: Any | None
    messages: list[PiMessage]
    usage: dict[str, int]
    tools_used: list[str]
    transitions: list[TransitionRecord]
    failure: str | None = None


class PiAgentRuntime:
    """Drive pi-agent-core while keeping tools and policy in Python."""

    def __init__(self, *, bridge: PiRuntimeBridge | None = None) -> None:
        self.bridge = bridge or PiRuntimeBridge()

    async def run(self, spec: AgentRunSpec) -> AgentRunOutcome:
        if spec.toolsets:
            async with connect_mcp_toolsets(tuple(spec.toolsets)) as mcp_tools:
                tools = spec.tools.copy()
                for tool in mcp_tools:
                    tools.register(tool)
                return await self.run(
                    replace(
                        spec,
                        tools=tools,
                        toolsets=(),
                        capability_manifest=None,
                    )
                )
        request_limit = None if spec.request_limit is None else max(1, int(spec.request_limit))
        max_retries = max(0, int(spec.max_retries))
        terminals = _terminal_outputs(spec)
        terminal_models = {terminal.name: terminal.model for terminal in terminals}
        registry = spec.tools.copy()
        runtime_capabilities: list[dict[str, Any]] = []
        builtin_names: list[str] = []
        workspace = Path.cwd()
        for capability in spec.capabilities:
            for tool in getattr(capability, "host_tools", lambda: ())():
                registry.register(tool)
            runtime = getattr(capability, "runtime_config", lambda: None)()
            if runtime is None:
                continue
            runtime_capabilities.append(runtime)
            if runtime.get("kind") == "filesystem":
                workspace = Path(runtime["cwd"])
                names = runtime.get("tool_names") or {}
                builtin_names.extend(
                    (
                        str(names.get("read") or "read"),
                        str(names.get("write") or "write"),
                        str(names.get("edit") or "edit"),
                    )
                )
            elif runtime.get("kind") == "shell":
                workspace = Path(runtime["cwd"])
                builtin_names.append(str(runtime.get("tool_name") or "bash"))

        manifest = spec.capability_manifest or CapabilityManifest.from_registry(
            registry,
            namespace=spec.role,
            terminal_names=terminal_models,
            additional_tool_names=builtin_names,
        )
        _validate_capability_manifest(registry, terminal_models, builtin_names, manifest)
        deps = AeloonRunDeps(
            role=spec.role,
            tools=registry,
            terminal_models=terminal_models,
            capability_manifest=manifest,
            progress=spec.progress,
            session_id=spec.session_id,
            turn_id=spec.turn_id,
            max_tokens=spec.max_tokens,
            max_tool_calls=spec.max_tool_calls,
            max_retries=max_retries,
            stuck_detection_enabled=spec.stuck_detection_enabled,
            stuck_detection_threshold=spec.stuck_detection_threshold,
            transition_digest=_messages_digest(spec.history),
        )
        tool_schemas: dict[str, dict[str, Any]] = {
            str(definition["name"]): dict(definition["input_schema"])
            for definition in registry.get_definitions()
        }
        terminal_definitions = [terminal.definition() for terminal in terminals]
        tool_schemas.update(
            {str(definition["name"]): dict(definition["input_schema"])
             for definition in terminal_definitions}
        )

        async def on_rpc(message: dict[str, Any]) -> dict[str, Any]:
            method = message.get("method")
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Pi RPC payload must be an object")
            if method == "tool_call":
                name = str(payload.get("name") or "")
                arguments = payload.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                result = await registry.execute(name, arguments)
                deps.tool_observations.append(ToolObservation(name, dict(arguments), result))
                return {
                    "result": result,
                    "is_error": result.lstrip().lower().startswith("error"),
                }
            if method == "preflight":
                return _preflight(deps, payload, tool_schemas)
            raise ValueError(f"unknown Pi RPC method: {method!r}")

        async def on_event(message: dict[str, Any]) -> None:
            await self._on_event(deps, spec, message)

        instructions = spec.instructions
        if terminals:
            names = ", ".join(terminal.name for terminal in terminals)
            instructions += (
                "\n\nSTRUCTURED COMPLETION: Plain text cannot complete this run. "
                f"Finish by calling exactly one terminal tool: {names}."
            )
        payload = {
            "model": spec.model.to_runtime(),
            "settings": dict(spec.model_settings or {}),
            "instructions": instructions,
            "prompt": spec.prompt,
            "history": serialize_messages(spec.history),
            "tools": [
                {
                    **definition,
                    "mode": registry.get(str(definition["name"])).concurrency_mode,
                }
                for definition in registry.get_definitions()
            ],
            "terminals": terminal_definitions,
            "capabilities": runtime_capabilities,
            "workspace": str(workspace),
            "request_limit": request_limit,
            "max_retries": max_retries,
            "max_tool_calls": spec.max_tool_calls,
            "max_tokens": spec.max_tokens,
            "max_output_tokens": spec.max_output_tokens,
            "session_id": spec.session_id,
            "turn_id": spec.turn_id,
        }
        try:
            raw = await self.bridge.run(payload, on_rpc=on_rpc, on_event=on_event)
        finally:
            for capability in spec.capabilities:
                close = getattr(capability, "close", None)
                if close is None:
                    continue
                try:
                    closed = close()
                    if inspect.isawaitable(closed):
                        await closed
                except Exception as exc:
                    logger.warning(
                        "Ignoring capability cleanup failure for {}: {}",
                        type(capability).__name__,
                        exc,
                    )
        status = AgentRunStatus(str(raw.get("status") or AgentRunStatus.FAILED))
        messages = deserialize_messages(raw.get("messages") or spec.history)
        output = raw.get("output")
        output_name = raw.get("output_name")
        if status is AgentRunStatus.COMPLETED and terminals:
            model = terminal_models.get(str(output_name))
            if model is None:
                status = AgentRunStatus.FAILED
                output = None
                raw["failure"] = "Pi runtime completed without a known terminal output"
            else:
                try:
                    output = model.model_validate(output)
                except ValidationError as exc:
                    status = AgentRunStatus.FAILED
                    output = None
                    raw["failure"] = f"invalid terminal output: {exc}"
        if status is AgentRunStatus.COMPLETED and spec.output_validator is not None:
            context = PiRunContext(deps=deps, messages=messages)
            validated = _invoke_output_validator(spec.output_validator, context, output)
            if inspect.isawaitable(validated):
                validated = await validated
            output = validated
        usage = _usage_dict(raw.get("usage"))
        failure = str(raw.get("failure")) if raw.get("failure") else None
        if status is AgentRunStatus.COMPLETED:
            await _emit_progress(
                spec.progress,
                "on_final",
                _output_text(output),
                messages=serialize_messages(messages),
                status=status.value,
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
            usage=usage,
            tools_used=list(deps.tools_used),
            transitions=list(deps.transitions),
            failure=failure,
        )

    async def _on_event(
        self,
        deps: AeloonRunDeps,
        spec: AgentRunSpec,
        envelope: dict[str, Any],
    ) -> None:
        event = envelope.get("event")
        if event == "text_delta":
            await _emit_progress(deps.progress, "on_llm_delta", str(envelope.get("delta") or ""))
            return
        if event == "thinking_delta":
            await _emit_progress(
                deps.progress,
                "on_llm_reasoning_delta",
                str(envelope.get("delta") or ""),
            )
            return
        if event == "model_request":
            messages = _message_list(envelope.get("messages"))
            await self._record_transition(
                deps,
                spec,
                node="model_request",
                node_kind=NodeKind.HARNESS,
                decision={
                    "request": int(envelope.get("request_number") or 0),
                    "tool_names": list(envelope.get("tool_names") or []),
                    "max_tokens": envelope.get("max_tokens"),
                },
                before_messages=messages,
                after_messages=messages,
            )
            return
        if event == "model_response":
            message = envelope.get("message")
            if not isinstance(message, dict):
                return
            view = _model_response_view(message)
            await _emit_progress(deps.progress, "on_llm_response", view, component=deps.role)
            await self._record_transition(
                deps,
                spec,
                node="model_response",
                node_kind=NodeKind.HARNESS,
                decision={
                    "finish_reason": message.get("stopReason"),
                    "tool_names": [call.name for call in view.tool_calls],
                },
                usage=_usage_dict(message.get("usage")),
            )
            return
        if event == "tool_start":
            call = ToolCallView(
                id=str(envelope.get("call_id") or ""),
                name=str(envelope.get("name") or "tool"),
                arguments=dict(envelope.get("arguments") or {}),
            )
            deps.tool_calls[call.id] = call
            await _emit_progress(deps.progress, "on_tool_calls", [call])
            await self._record_transition(
                deps,
                spec,
                node="tool_call",
                node_kind=NodeKind.DOMAIN,
                decision={"id": call.id, "name": call.name, "arguments": call.arguments},
            )
            return
        if event == "tool_end":
            call_id = str(envelope.get("call_id") or "")
            call = deps.tool_calls.get(call_id) or ToolCallView(
                call_id,
                str(envelope.get("name") or "tool"),
                dict(envelope.get("arguments") or {}),
            )
            result = str(envelope.get("result") or "")
            failed = bool(envelope.get("is_error"))
            if call.name not in deps.terminal_models:
                deps.tools_used.append(call.name)
            await _emit_progress(
                deps.progress,
                "on_tool_result",
                _execution_record(deps, call, result, failed=failed),
            )
            await self._record_transition(
                deps,
                spec,
                node="tool_result",
                node_kind=NodeKind.DOMAIN,
                decision={"id": call.id, "name": call.name, "result": result},
            )

    async def _record_transition(
        self,
        deps: AeloonRunDeps,
        spec: AgentRunSpec,
        *,
        node: str,
        node_kind: NodeKind,
        decision: Any,
        usage: dict[str, int] | None = None,
        before_messages: Sequence[Mapping[str, Any]] | None = None,
        after_messages: Sequence[Mapping[str, Any]] | None = None,
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
            component="pi-core",
        )
        deps.transitions.append(record)
        if spec.on_transition is not None:
            emitted = spec.on_transition(record)
            if inspect.isawaitable(emitted):
                await emitted


# Compatibility name retained for the rest of Aeloon's stable Python surface.
HarnessAgentRuntime = PiAgentRuntime


def output_tools(*specs: tuple[type[BaseModel], str, str]) -> list[TerminalOutput]:
    """Build typed Pi terminal outputs with stable names and descriptions."""

    return [TerminalOutput(model, name, description) for model, name, description in specs]


def _terminal_outputs(spec: AgentRunSpec) -> list[TerminalOutput]:
    if isinstance(spec.output_type, list | tuple) and all(
        isinstance(item, TerminalOutput) for item in spec.output_type
    ):
        terminals = list(spec.output_type)
    elif isinstance(spec.output_type, type) and issubclass(spec.output_type, BaseModel):
        if spec.terminal_models:
            terminals = [
                TerminalOutput(model, name, f"Return the final {name} structured result.")
                for name, model in spec.terminal_models.items()
            ]
        else:
            terminals = [
                TerminalOutput(
                    spec.output_type,
                    "final_result",
                    "Return the final structured result for this stage.",
                )
            ]
    else:
        terminals = []
    configured = set(spec.terminal_models)
    derived = {terminal.name for terminal in terminals}
    if configured and configured != derived:
        raise ValueError("terminal_models does not match configured terminal outputs")
    return terminals


def _validate_capability_manifest(
    registry: ToolRegistry,
    terminal_models: Mapping[str, type[BaseModel]],
    builtin_names: Sequence[str],
    manifest: CapabilityManifest,
) -> None:
    exposed = (
        *(str(definition["name"]) for definition in registry.get_definitions()),
        *builtin_names,
    )
    if tuple(exposed) != manifest.tool_names:
        raise ValueError("ToolRegistry does not match the host capability manifest")
    if set(terminal_models) != set(manifest.terminal_names):
        raise ValueError("typed outputs do not match the host capability manifest")
    if set(exposed) & set(manifest.terminal_names):
        raise ValueError("terminal outputs must not be registered as ordinary tools")


def _preflight(
    deps: AeloonRunDeps,
    payload: dict[str, Any],
    schemas: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_calls = payload.get("calls")
    calls = (
        [call for call in raw_calls if isinstance(call, dict)]
        if isinstance(raw_calls, list)
        else []
    )
    terminal_names = set(deps.terminal_models)
    called_terminals = [call for call in calls if call.get("name") in terminal_names]
    if called_terminals and (len(calls) != 1 or len(called_terminals) != 1):
        return _retry_rejection(
            deps,
            "A terminal output must be the response's only tool call. No tool was executed.",
        )
    if deps.max_tool_calls is not None and deps.tool_call_count + len(calls) > deps.max_tool_calls:
        reason = f"The {deps.max_tool_calls}-tool-call limit would be exceeded by this batch."
        return {
            "allowed": False,
            "reason": reason,
            "limit_reason": reason,
            "terminate": True,
        }
    usage = _usage_dict(payload.get("usage"))
    if deps.max_tokens is not None and usage.get("total_tokens", 0) > deps.max_tokens:
        reason = f"The {deps.max_tokens}-token limit was exceeded before tool execution."
        return {
            "allowed": False,
            "reason": reason,
            "limit_reason": reason,
            "terminate": True,
        }
    errors: list[str] = []
    for call in calls:
        name = str(call.get("name") or "")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            errors.append(f"{name}: arguments must be an object")
            continue
        model = deps.terminal_models.get(name)
        tool = deps.tools.get(name)
        if model is not None:
            try:
                model.model_validate(arguments)
            except ValidationError as exc:
                errors.append(f"{name}: {_validation_error(exc)}")
            continue
        if tool is not None:
            try:
                tool.args_model.model_validate(arguments)
            except ValidationError as exc:
                errors.append(f"{name}: {_validation_error(exc)}")
            continue
        schema = schemas.get(name)
        if schema is not None:
            validation = next(iter(Draft202012Validator(schema).iter_errors(arguments)), None)
            if validation is not None:
                errors.append(f"{name}: {validation.message}")
    if errors:
        return _retry_rejection(
            deps,
            "The entire tool batch was rejected before execution: " + "; ".join(errors),
        )
    if deps.stuck_detection_enabled and calls:
        messages = _message_list(payload.get("messages"))
        detection = detect_repeated_tool_exchanges(
            _legacy_detection_messages(messages),
            tool_modes=deps.tool_modes,
            threshold=deps.stuck_detection_threshold,
        )
        if detection is not None:
            repeated = any(
                _action_digest(str(call.get("name") or ""), dict(call.get("arguments") or {}))
                == detection.action_digest
                for call in calls
            )
            if repeated:
                return _retry_rejection(
                    deps,
                    "The same successful read-only action and observation repeated "
                    f"{detection.repetitions} times. Change strategy before another tool call.",
                )
    deps.tool_call_count += len(calls)
    return {"allowed": True}


def _retry_rejection(deps: AeloonRunDeps, reason: str) -> dict[str, Any]:
    deps.validation_failures += 1
    exhausted = deps.validation_failures > deps.max_retries
    response: dict[str, Any] = {
        "allowed": False,
        "reason": reason,
        "terminate": exhausted,
    }
    if exhausted:
        response["failure_reason"] = (
            f"Pi tool/output validation failed after {deps.max_retries + 1} attempts: {reason}"
        )
    return response


def _model_response_view(message: Mapping[str, Any]) -> ModelResponseView:
    content = message.get("content")
    parts = content if isinstance(content, list) else []
    text = "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, Mapping) and part.get("type") == "text"
    )
    reasoning = "".join(
        str(part.get("thinking") or "")
        for part in parts
        if isinstance(part, Mapping) and part.get("type") == "thinking"
    )
    calls = tuple(
        ToolCallView(
            id=str(part.get("id") or ""),
            name=str(part.get("name") or "tool"),
            arguments=dict(part.get("arguments") or {}),
        )
        for part in parts
        if isinstance(part, Mapping) and part.get("type") == "toolCall"
    )
    return ModelResponseView(
        content=text or None,
        reasoning_content=reasoning or None,
        tool_calls=calls,
        usage=_usage_dict(message.get("usage")),
        finish_reason=str(message.get("stopReason") or "") or None,
    )


def _execution_record(
    deps: AeloonRunDeps,
    call: ToolCallView,
    result: str,
    *,
    failed: bool,
) -> ToolExecutionRecord:
    tool = deps.tools.get(call.name)
    return ToolExecutionRecord(
        index=max(0, len(deps.tools_used) - 1),
        call_id=call.id,
        tool_name=call.name,
        arguments=call.arguments,
        mode=tool.concurrency_mode if tool is not None else "exclusive",
        state=ToolExecutionState.FAILED if failed else ToolExecutionState.DONE,
        result=result,
        error=result if failed else None,
    )


def _usage_dict(usage: Any) -> dict[str, int]:
    if not isinstance(usage, Mapping):
        return {}
    aliases = {
        "input": "input_tokens",
        "output": "output_tokens",
        "cacheRead": "cache_read_tokens",
        "cacheWrite": "cache_write_tokens",
        "reasoning": "reasoning_tokens",
        "totalTokens": "total_tokens",
        "requests": "requests",
    }
    return {
        normalized: max(0, int(usage[key]))
        for key, normalized in aliases.items()
        if isinstance(usage.get(key), int | float) and not isinstance(usage.get(key), bool)
    }


def _invoke_output_validator(
    validator: OutputValidator,
    ctx: PiRunContext,
    output: Any,
) -> Any:
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
    for field_name in ("final_content", "summary", "answer"):
        value = getattr(output, field_name, None)
        if isinstance(value, str):
            return value
    return ""


def _messages_digest(messages: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _action_digest(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _legacy_detection_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "toolCall":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": part.get("id"),
                            "name": part.get("name"),
                            "input": part.get("arguments", {}),
                        }
                    )
            projected.append({"role": "assistant", "content": blocks})
        elif role == "toolResult":
            projected.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("toolCallId"),
                            "content": content,
                            "is_error": bool(message.get("isError")),
                        }
                    ],
                }
            )
        elif role == "user":
            projected.append({"role": "user", "content": content})
    return projected


def _validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: "
        f"{error.get('msg', 'invalid value')}"
        for error in exc.errors()
    )


def _message_list(value: Any) -> list[PiMessage]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _strip_titles(value) for key, value in node.items() if key != "title"}
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


__all__ = [
    "AeloonRunDeps",
    "AgentRunOutcome",
    "AgentRunSpec",
    "AgentRunStatus",
    "CapabilityManifest",
    "HarnessAgentRuntime",
    "MESSAGE_FORMAT",
    "MESSAGE_SCHEMA_VERSION",
    "PiAgentRuntime",
    "PiRunContext",
    "TerminalOutput",
    "deserialize_messages",
    "output_tools",
    "serialize_messages",
]
