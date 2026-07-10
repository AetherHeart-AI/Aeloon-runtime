from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from aeloon_core.kernel import run_agent_kernel
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.state import RunStatus
from aeloon_core.state_machine import run_agent_loop, run_uasm_kernel
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
        minimal_context_enabled=False,
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
async def test_a0_stops_immediately_after_tool_error_without_recovery_rules() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="fail", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            )
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "fail"}],
        rule_engine_enabled=False,
        temporary_guard_enabled=False,
        minimal_context_enabled=False,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "recovery rules are disabled" in (state.metadata.final_content or "")
    assert state.messages[-1]["role"] == "assistant"
    assert len(provider.calls) == 1


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
        guard_decision_mode="binary",
        minimal_context_enabled=False,
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
        minimal_context_enabled=False,
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
        minimal_context_enabled=False,
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
async def test_uasm_tuple_wrapper_matches_public_result_shape() -> None:
    result = await run_uasm_kernel(
        provider=ScriptedProvider([LLMResponse(content="done")]),
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "answer"}],
        minimal_context_enabled=False,
    )

    assert result[0] == "done"
    assert result[1] == []
    assert result[2][-1] == {"role": "assistant", "content": "done"}


@pytest.mark.asyncio
async def test_rule_only_uasm_matches_legacy_kernel_for_normal_tool_flow() -> None:
    def responses() -> list[LLMResponse]:
        return [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]

    legacy = await run_agent_kernel(
        provider=ScriptedProvider(responses()),
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
    )
    uasm = await run_uasm_kernel(
        provider=ScriptedProvider(responses()),
        model="test-model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        temporary_guard_enabled=False,
        minimal_context_enabled=False,
        transition_trace_enabled=False,
    )

    assert uasm == legacy


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
        temporary_guard_enabled=False,
        minimal_context_enabled=False,
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
        temporary_guard_enabled=False,
        minimal_context_enabled=False,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "after tools were disabled" in (state.metadata.final_content or "")
    assert state.tools_used == ["echo"]


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
        minimal_context_enabled=False,
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
        minimal_context_enabled=False,
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
        minimal_context_enabled=False,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_RULE
    assert "maximum number of tool call iterations" in (state.metadata.final_content or "")
    assert state.guard_state.iteration_limit == 4
    assert state.guard_state.auto_continue_remaining == 0
    assert len(provider.calls) == 6


@pytest.mark.asyncio
async def test_rule_only_uasm_matches_legacy_across_recovery_and_terminal_paths() -> None:
    def tool_call(
        call_id: str,
        name: str,
        arguments: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
            finish_reason="tool_calls",
        )

    cases = [
        (
            lambda: [
                tool_call("call-1", "echo", {"value": "one"}),
                tool_call("call-2", "echo", {"value": "two"}),
                LLMResponse(content="wrapped"),
            ],
            lambda: registry(EchoTool()),
            {
                "max_iterations": 1,
                "max_auto_continue_iterations": 1,
                "max_finalization_iterations": 1,
            },
        ),
        (
            lambda: [
                tool_call(f"call-{index}", "echo", {"value": "same"})
                for index in range(1, 4)
            ],
            lambda: registry(EchoTool()),
            {
                "max_iterations": 1,
                "max_auto_continue_iterations": 5,
                "max_finalization_iterations": 1,
            },
        ),
        (
            lambda: [
                tool_call("bad-1", "echo", ["invalid"]),
                tool_call("bad-2", "echo", ["invalid"]),
            ],
            lambda: registry(EchoTool()),
            {},
        ),
        (
            lambda: [
                tool_call("exec-1", "exec", {"command": "first"}),
                tool_call("exec-2", "exec", {"command": "second"}),
                LLMResponse(content="recovered"),
            ],
            lambda: registry(TimeoutExecTool()),
            {},
        ),
        (
            lambda: [
                tool_call("fail-1", "fail", {"value": "one"}),
                LLMResponse(content="recovered"),
            ],
            lambda: registry(FailingTool()),
            {},
        ),
        (
            lambda: [LLMResponse(content="provider failed", finish_reason="error")],
            lambda: registry(EchoTool()),
            {},
        ),
    ]

    for responses, tools, limits in cases:
        legacy = await run_agent_kernel(
            provider=ScriptedProvider(responses()),
            model="test-model",
            tools=tools(),
            messages=[{"role": "user", "content": "run case"}],
            **limits,
        )
        uasm = await run_uasm_kernel(
            provider=ScriptedProvider(responses()),
            model="test-model",
            tools=tools(),
            messages=[{"role": "user", "content": "run case"}],
            temporary_guard_enabled=False,
            minimal_context_enabled=False,
            transition_trace_enabled=False,
            **limits,
        )

        assert uasm == legacy
