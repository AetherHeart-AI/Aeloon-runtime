from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, OutputToolCallEvent, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from aeloon_core.pydantic_model import PromptCacheState
from aeloon_core.pydantic_runtime import (
    AgentRunSpec,
    AgentRunStatus,
    PydanticAgentRuntime,
    deserialize_messages,
    output_tools,
    serialize_messages,
)
from aeloon_core.tools.base import FunctionTool, Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.workers import WorkerReport as CompleteWorkArgs


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


class _RecordingRead(Tool):
    name = "read"
    description = "Read one test path."
    args_model = _ReadArgs
    concurrency_mode = "read_only"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, path: str) -> str:
        self.calls.append(path)
        return f"contents:{path}"


@pytest.mark.asyncio
async def test_output_tool_calls_are_not_projected_as_executing_tools() -> None:
    runtime = PydanticAgentRuntime()

    async def events():
        yield OutputToolCallEvent(
            ToolCallPart("complete_work", {"summary": "done"}, "done-1")
        )

    await runtime._event_stream_handler(SimpleNamespace(deps=object()), events())


def _spec(model: FunctionModel, registry: ToolRegistry, *, request_limit: int = 4) -> AgentRunSpec:
    return AgentRunSpec(
        role="worker",
        model=model,
        instructions="Complete explicitly.",
        prompt="read and finish",
        history=[],
        tools=registry,
        output_type=output_tools(
            (CompleteWorkArgs, "complete_work", "Finish with a structured report."),
        ),
        terminal_models={"complete_work": CompleteWorkArgs},
        request_limit=request_limit,
        transition_trace_enabled=True,
    )


@pytest.mark.asyncio
async def test_runtime_executes_host_tool_and_returns_typed_output() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)

    def script(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        if not read.calls:
            return ModelResponse(parts=[ToolCallPart("read", {"path": "README.md"}, "read-1")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "complete_work",
                    {
                        "summary": "done",
                        "artifacts": [],
                        "evidence": [
                            {
                                "kind": "file",
                                "locator": "README.md:1",
                                "claim": "README was inspected",
                                "status": "observed",
                            }
                        ],
                    },
                    "done-1",
                )
            ]
        )

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry)
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert isinstance(outcome.output, CompleteWorkArgs)
    assert outcome.output.summary == "done"
    assert read.calls == ["README.md"]
    assert outcome.tools_used == ["read"]
    assert deserialize_messages(serialize_messages(outcome.messages)) == outcome.messages


@pytest.mark.asyncio
async def test_mixed_terminal_batch_is_retried_before_any_side_effect() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    attempts = 0

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("read", {"path": "secret.txt"}, "read-1"),
                    ToolCallPart("complete_work", {"summary": "unsafe"}, "done-1"),
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "safe"}, "done-2")]
        )

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry)
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert outcome.output.summary == "safe"
    assert attempts == 2
    assert read.calls == []


@pytest.mark.asyncio
async def test_malformed_batch_is_retried_as_a_unit() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    attempts = 0

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("read", {}, "invalid"),
                    ToolCallPart("read", {"path": "must-not-run"}, "valid"),
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "recovered"}, "done")]
        )

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry)
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert attempts == 2
    assert read.calls == []


@pytest.mark.asyncio
async def test_request_limit_returns_exact_partial_history() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("read", {"path": f"call-{len(read.calls)}"}, "read")]
        )

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry, request_limit=1)
    )

    assert outcome.status is AgentRunStatus.LIMIT_EXCEEDED
    assert outcome.failure is not None
    assert read.calls == ["call-0"]
    assert outcome.messages
    assert deserialize_messages(serialize_messages(outcome.messages)) == outcome.messages


@pytest.mark.asyncio
async def test_plain_text_cannot_complete_worker() -> None:
    registry = ToolRegistry()

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("I am done")])

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry, request_limit=2)
    )

    assert outcome.status is not AgentRunStatus.COMPLETED
    assert outcome.output is None


@pytest.mark.asyncio
async def test_multiple_terminal_outputs_are_retried_before_completion() -> None:
    registry = ToolRegistry()
    attempts = 0

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("complete_work", {"summary": "one"}, "done-1"),
                    ToolCallPart("complete_work", {"summary": "two"}, "done-2"),
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "safe"}, "done-3")]
        )

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry)
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert outcome.output.summary == "safe"
    assert attempts == 2


@pytest.mark.asyncio
async def test_read_only_tools_run_concurrently_and_mutation_is_a_barrier() -> None:
    class EmptyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    active = 0
    max_active = 0
    order: list[str] = []

    async def read(name: str) -> str:
        nonlocal active, max_active
        order.append(f"{name}:start")
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        order.append(f"{name}:end")
        return name

    async def read_a() -> str:
        return await read("a")

    async def read_b() -> str:
        return await read("b")

    async def mutate() -> str:
        order.append("write:start")
        await asyncio.sleep(0)
        order.append("write:end")
        return "written"

    registry = ToolRegistry()
    for name, handler, mode in (
        ("read_a", read_a, "read_only"),
        ("read_b", read_b, "read_only"),
        ("write", mutate, "mutating"),
    ):
        registry.register(
            FunctionTool(
                name=name,
                description=name,
                args_model=EmptyArgs,
                handler=handler,
                concurrency_mode=mode,
            )
        )
    round_number = 0

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal round_number
        round_number += 1
        if round_number == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("read_a", {}, "a"),
                    ToolCallPart("read_b", {}, "b"),
                ]
            )
        if round_number == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart("read_a", {}, "a2"),
                    ToolCallPart("write", {}, "w"),
                    ToolCallPart("read_b", {}, "b2"),
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "done"}, "done")]
        )

    outcome = await PydanticAgentRuntime().run(
        _spec(FunctionModel(script), registry)
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert max_active == 2
    second_round = order[4:]
    assert second_round == [
        "a:start",
        "a:end",
        "write:start",
        "write:end",
        "b:start",
        "b:end",
    ]


@pytest.mark.asyncio
async def test_response_over_token_limit_never_executes_its_tool() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("read", {"path": "must-not-run"}, "read")],
            usage=RequestUsage(input_tokens=1, output_tokens=50),
        )

    spec = _spec(FunctionModel(script), registry)
    spec.max_tokens = 20
    outcome = await PydanticAgentRuntime().run(spec)

    assert outcome.status is AgentRunStatus.LIMIT_EXCEEDED
    assert read.calls == []


@pytest.mark.asyncio
async def test_finite_token_budget_preemptively_caps_model_output() -> None:
    registry = ToolRegistry()
    observed_max_tokens: list[int | None] = []

    def script(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        observed_max_tokens.append((info.model_settings or {}).get("max_tokens"))
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "done"}, "done")],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
        )

    spec = _spec(FunctionModel(script), registry)
    spec.max_tokens = 5_000
    outcome = await PydanticAgentRuntime().run(spec)

    assert outcome.status is AgentRunStatus.COMPLETED
    assert observed_max_tokens[0] is not None
    assert 0 < observed_max_tokens[0] < 5_000


@pytest.mark.asyncio
async def test_master_granted_output_limit_reaches_the_next_model_request() -> None:
    registry = ToolRegistry()
    observed_max_tokens: list[int | None] = []

    def script(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        observed_max_tokens.append((info.model_settings or {}).get("max_tokens"))
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "done"}, "done")]
        )

    spec = _spec(FunctionModel(script), registry)
    spec.max_output_tokens = 16_384
    outcome = await PydanticAgentRuntime().run(spec)

    assert outcome.status is AgentRunStatus.COMPLETED
    assert observed_max_tokens == [16_384]


@pytest.mark.asyncio
async def test_tool_call_limit_rejects_whole_batch_without_side_effects() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)

    def script(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart("read", {"path": "one"}, "one"),
                ToolCallPart("read", {"path": "two"}, "two"),
            ]
        )

    spec = _spec(FunctionModel(script), registry)
    spec.max_tool_calls = 1
    outcome = await PydanticAgentRuntime().run(spec)

    assert outcome.status is AgentRunStatus.LIMIT_EXCEEDED
    assert read.calls == []


@pytest.mark.asyncio
async def test_prompt_cache_rejection_retries_only_the_current_model_request() -> None:
    registry = ToolRegistry()
    cache = PromptCacheState()
    seen_settings: list[dict[str, Any]] = []

    def script(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        settings = dict(info.model_settings or {})
        seen_settings.append(settings)
        if len(seen_settings) == 1:
            raise ModelHTTPError(
                status_code=400,
                model_name="gateway",
                body={"error": "cache_control is an unknown field"},
            )
        return ModelResponse(
            parts=[ToolCallPart("complete_work", {"summary": "done"}, "done")]
        )

    spec = _spec(FunctionModel(script), registry)
    spec.model_settings = {"anthropic_cache_messages": True}
    spec.prompt_cache = cache
    outcome = await PydanticAgentRuntime().run(spec)

    assert outcome.status is AgentRunStatus.COMPLETED
    assert len(seen_settings) == 2
    assert seen_settings[0]["anthropic_cache_messages"] is True
    assert "anthropic_cache_messages" not in seen_settings[1]
    assert cache.disabled is True
    assert outcome.tools_used == []
    assert {record.node for record in outcome.transitions} >= {
        "model_request",
        "model_response",
        "output",
        "run_finished",
    }
