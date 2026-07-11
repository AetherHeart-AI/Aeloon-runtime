from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.state import RunStatus
from aeloon_core.state_machine import run_agent_loop
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import NodeKind


class ValueArgs(BaseModel):
    value: str


class CommandArgs(BaseModel):
    command: str


class EchoTool(Tool):
    name = "echo"
    description = "Echo a value."
    args_model = ValueArgs

    async def execute(self, value: str) -> str:
        return f"echo:{value}"


class FailingTool(Tool):
    name = "fail"
    description = "Return an error."
    args_model = ValueArgs

    async def execute(self, value: str) -> str:
        return f"Error: failed for {value}"


class TimeoutExecTool(Tool):
    name = "exec"
    description = "Return a timeout."
    args_model = CommandArgs

    async def execute(self, command: str) -> str:
        del command
        return "Error: Command timed out after 3 seconds"


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del reasoning_effort, tool_choice
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools or []),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": copy.deepcopy(response_format),
            }
        )
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)


class ProgressRecorder:
    def __init__(self) -> None:
        self.final_calls: list[str] = []

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        del text, tool_hint

    async def on_turn_start(self) -> None:
        return None

    async def on_final(self, content: str, **kwargs: Any) -> None:
        del kwargs
        self.final_calls.append(content)


def registry(*tools: Tool) -> ToolRegistry:
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


@pytest.mark.asyncio
async def test_state_machine_routes_explicit_nodes_and_attributes_usage() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            LLMResponse(
                content="done",
                usage={"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
            ),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        session_id="session-1",
        turn_id="turn-1",
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "done"
    assert state.tools_used == ["echo"]
    assert state.messages[-1] == {"role": "assistant", "content": "done"}
    assert state.token_ledger.total_tokens == 20
    assert state.token_ledger.for_kind(NodeKind.DOMAIN)["total_tokens"] == 20
    agent_nodes = [
        record.node for record in state.transitions if record.node != "minimal_context"
    ]
    assert agent_nodes == ["master", "worker", "master", "tool", "master", "worker"]
    assert [record.sequence for record in state.transitions] == list(
        range(1, len(state.transitions) + 1)
    )
    assert all(
        previous.after_digest == current.before_digest
        for previous, current in zip(state.transitions, state.transitions[1:], strict=False)
    )
    assert all(record.session_id == "session-1" for record in state.transitions)


@pytest.mark.asyncio
async def test_rule_engine_auto_continues_after_base_iteration_limit() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done after echo", finish_reason="stop"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=1,
        max_auto_continue_iterations=1,
        max_finalization_iterations=1,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "done after echo"
    assert state.tools_used == ["echo"]
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"]
    assert "MAXIMUM ITERATIONS REACHED" not in provider.calls[1]["messages"][-1]["content"]
    assert state.guard_state.iteration_limit == 2
    assert state.guard_state.auto_continue_remaining == 0


@pytest.mark.asyncio
async def test_runtime_recovers_after_failed_tool_result() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="fail-1", name="fail", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="recovered"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "recover"}],
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "recovered"
    assert state.tools_used == ["fail"]
    assert len(provider.calls) == 2
    assert "TOOL ERROR RECOVERY" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_runtime_recovers_after_exec_timeout() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="exec-1",
                        name="exec",
                        arguments={"command": "slow command"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="recovered"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(TimeoutExecTool()),
        messages=[{"role": "user", "content": "recover"}],
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "recovered"
    assert state.tools_used == ["exec"]
    assert state.guard_state.exec_timeout_rounds == 1
    assert "TOOL ERROR RECOVERY" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_runtime_stops_after_repeated_malformed_calls() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="bad-1", name="echo", arguments=["bad"])],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="bad-2", name="echo", arguments=["bad"])],
                finish_reason="tool_calls",
            ),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "malformed"}],
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "malformed tool arguments" in (state.metadata.final_content or "")
    assert state.tools_used == []
    assert state.guard_state.unproductive_tool_rounds == 2
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_repeated_duplicate_escalates_to_temporary_guard_then_recovers() -> None:
    duplicate_call = lambda call_id: LLMResponse(  # noqa: E731
        content=None,
        tool_calls=[
            ToolCallRequest(id=call_id, name="echo", arguments={"value": "one"})
        ],
        finish_reason="tool_calls",
        usage={"total_tokens": 5},
    )
    provider = ScriptedProvider(
        [
            duplicate_call("call-1"),
            duplicate_call("call-2"),
            duplicate_call("call-3"),
            LLMResponse(
                content='{"action":"continue"}',
                usage={"total_tokens": 3},
            ),
            LLMResponse(content="recovered", usage={"total_tokens": 4}),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=10,
        max_auto_continue_iterations=0,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "recovered"
    assert state.tools_used == ["echo"]
    assert provider.calls[3]["tools"] == []
    assert provider.calls[3]["response_format"] == {"type": "json_object"}
    assert state.token_ledger.for_kind(NodeKind.HARNESS)["total_tokens"] == 3
    assert any(record.node == "temporary_guard" for record in state.transitions)


@pytest.mark.asyncio
async def test_invalid_temporary_guard_output_falls_back_to_rule_termination() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-2", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-3", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="not json", usage={"total_tokens": 2}),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=10,
        max_auto_continue_iterations=0,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "repeated tool calls" in (state.metadata.final_content or "")
    assert state.token_ledger.for_kind(NodeKind.HARNESS)["total_tokens"] == 2


@pytest.mark.asyncio
async def test_preparation_persists_canonical_messages_and_context_usage() -> None:
    provider = ScriptedProvider([LLMResponse(content="done", usage={"total_tokens": 4})])

    async def prepare_model_input(
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        additional_messages: list[dict[str, Any]],
    ) -> Any:
        del tool_defs, additional_messages
        return SimpleNamespace(
            messages=[*messages, {"role": "system", "content": "prepared"}],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "answer"}],
        prepare_model_input=prepare_model_input,
    )

    assert state.messages[-2] == {"role": "system", "content": "prepared"}
    assert provider.calls[0]["messages"][-1] == {
        "role": "system",
        "content": "prepared",
    }
    context_usage = state.token_ledger.for_kind(NodeKind.CONTEXT_PROCESSING)
    assert context_usage["total_tokens"] == 5
    assert context_usage["estimated_input_tokens_before"] > 0


@pytest.mark.asyncio
async def test_preparation_runs_before_every_sampling_call() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )
    preparations: list[list[dict[str, Any]]] = []

    async def prepare_model_input(
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        additional_messages: list[dict[str, Any]],
    ) -> Any:
        del tool_defs, additional_messages
        preparations.append(copy.deepcopy(messages))
        prepared_messages = messages
        if messages[-1].get("role") == "tool":
            prepared_messages = [
                *messages,
                {"role": "system", "content": "prepared after tool output"},
            ]
        return SimpleNamespace(messages=prepared_messages, usage={})

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        prepare_model_input=prepare_model_input,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert len(preparations) == 2
    assert preparations[1][-1]["role"] == "tool"
    assert provider.calls[1]["messages"][-1] == {
        "role": "system",
        "content": "prepared after tool output",
    }
    assert state.messages[-2] == {
        "role": "system",
        "content": "prepared after tool output",
    }


@pytest.mark.asyncio
async def test_exhausted_auto_continue_budget_enters_finalization() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-2", name="echo", arguments={"value": "two"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="wrapped up", finish_reason="stop"),
        ]
    )
    prepared_additional_messages: list[list[dict[str, Any]]] = []

    async def prepare_model_input(
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        additional_messages: list[dict[str, Any]],
    ) -> Any:
        del tool_defs
        prepared_additional_messages.append(copy.deepcopy(additional_messages))
        return SimpleNamespace(messages=messages, usage={})

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo twice"}],
        max_iterations=1,
        max_auto_continue_iterations=1,
        max_finalization_iterations=1,
        prepare_model_input=prepare_model_input,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "wrapped up"
    assert state.tools_used == ["echo", "echo"]
    assert len(provider.calls) == 3
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"]
    assert provider.calls[2]["tools"] == []
    assert "MAXIMUM ITERATIONS REACHED" in provider.calls[2]["messages"][-1]["content"]
    assert prepared_additional_messages[:2] == [[], []]
    assert "MAXIMUM ITERATIONS REACHED" in prepared_additional_messages[2][0]["content"]


@pytest.mark.asyncio
async def test_uasm_budget_exhaustion_enters_text_only_finalization() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="wrapped up"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo then wrap"}],
        max_iterations=1,
        max_auto_continue_iterations=0,
        max_finalization_iterations=1,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "wrapped up"
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"] == []
    assert "MAXIMUM ITERATIONS REACHED" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_uasm_finalization_tool_violation_terminates_without_execution() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-2", name="echo", arguments={"value": "two"})
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo then wrap"}],
        max_iterations=1,
        max_auto_continue_iterations=0,
        max_finalization_iterations=1,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "after tools were disabled" in (state.metadata.final_content or "")
    assert state.tools_used == ["echo"]


@pytest.mark.asyncio
async def test_output_budget_exhaustion_recovers_with_visible_text() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                finish_reason="length",
                reasoning_content="hidden reasoning used the whole output budget",
            ),
            LLMResponse(content="visible answer", finish_reason="stop"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "answer visibly"}],
        max_iterations=5,
        max_auto_continue_iterations=0,
        max_finalization_iterations=1,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "visible answer"
    assert state.tools_used == []
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"] == []
    assert provider.calls[1]["messages"][-1]["role"] == "user"
    assert "VISIBLE ANSWER REQUIRED" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_repeated_output_budget_exhaustion_returns_visible_failure() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, finish_reason="length", reasoning_content="hidden"),
            LLMResponse(content=None, finish_reason="length", reasoning_content="hidden"),
            LLMResponse(content=None, finish_reason="length", reasoning_content="hidden"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "answer visibly"}],
        max_iterations=5,
        max_auto_continue_iterations=0,
        max_finalization_iterations=2,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "repeatedly exhausted its output budget" in (state.metadata.final_content or "")
    assert "No final artifact was produced" in (state.metadata.final_content or "")
    assert state.messages[-1] == {
        "role": "assistant",
        "content": state.metadata.final_content,
    }
    assert len(provider.calls) == 3
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"] == []
    assert provider.calls[2]["tools"] == []


@pytest.mark.asyncio
async def test_output_budget_exhaustion_without_finalization_retries_once() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, finish_reason="length"),
            LLMResponse(content=None, finish_reason="length"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "answer visibly"}],
        max_iterations=5,
        max_auto_continue_iterations=0,
        max_finalization_iterations=0,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "empty response" in (state.metadata.final_content or "")
    assert state.metadata.finalization_iteration == 0
    assert state.guard_state.empty_stop_retries == 1
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"]


@pytest.mark.asyncio
async def test_uasm_provider_error_is_failed_and_emits_final_once_without_trace() -> None:
    progress = ProgressRecorder()

    state = await run_agent_loop(
        provider=ScriptedProvider(
            [LLMResponse(content="Error calling LLM", finish_reason="error")]
        ),
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "answer"}],
        transition_trace_enabled=False,
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.FAILED
    assert state.metadata.final_content == "Error calling LLM"
    assert progress.final_calls == ["Error calling LLM"]
    assert state.transitions == []


@pytest.mark.asyncio
async def test_trace_persistence_failure_does_not_interrupt_tool_side_effect_path() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )
    attempts = 0

    def fail_trace(_record) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("trace disk full")

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        on_transition=fail_trace,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "done"
    assert state.tools_used == ["echo"]
    assert attempts == 1


@pytest.mark.asyncio
async def test_temporary_guard_budget_extension_consumes_finite_auto_budget() -> None:
    def duplicate(call_id: str) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id=call_id, name="echo", arguments={"value": "same"})
            ],
            finish_reason="tool_calls",
        )

    provider = ScriptedProvider(
        [
            duplicate("call-1"),
            duplicate("call-2"),
            duplicate("call-3"),
            LLMResponse(content='{"action":"extend_budget"}'),
            duplicate("call-4"),
            LLMResponse(content='{"action":"extend_budget"}'),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "repeat forever"}],
        max_iterations=3,
        max_auto_continue_iterations=1,
        max_finalization_iterations=0,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "maximum number of tool call iterations" in (state.metadata.final_content or "")
    assert state.guard_state.iteration_limit == 4
    assert state.guard_state.auto_continue_remaining == 0
    assert len(provider.calls) == 6
