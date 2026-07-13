from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from aeloon_core.loop_guard import GuardAction, GuardEvent
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.state import RunStatus
from aeloon_core.state_machine import run_agent_loop
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import NodeKind


class ValueArgs(BaseModel):
    value: str


class EchoTool(Tool):
    name = "echo"
    description = "Echo a value."
    args_model = ValueArgs

    async def execute(self, value: str) -> str:
        return f"echo:{value}"


class FailingTool(Tool):
    name = "fail"
    description = "Fail."
    args_model = ValueArgs

    async def execute(self, value: str) -> str:
        return f"Error: failed for {value}"


class PollingTool(Tool):
    name = "poll"
    description = "Poll one run."
    args_model = ValueArgs
    concurrency_mode = "read_only"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, value: str) -> str:
        self.calls += 1
        return "running" if self.calls == 1 else "completed"


class FailOnceTool(Tool):
    name = "fail_once"
    description = "Fail once, then succeed."
    args_model = ValueArgs
    concurrency_mode = "mutating"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, value: str) -> str:
        self.calls += 1
        return "Error: temporary write failure" if self.calls == 1 else f"wrote:{value}"


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools or []), **kwargs}
        )
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)


class Progress:
    def __init__(self) -> None:
        self.resolutions = []
        self.final_calls: list[str] = []
        self.messages: list[str] = []
        self.deltas: list[str] = []

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        del tool_hint
        self.messages.append(text)

    async def on_guard_resolution(self, resolution) -> None:
        self.resolutions.append(resolution)

    async def on_llm_delta(self, delta: str) -> None:
        self.deltas.append(delta)

    async def on_final(self, content: str, **kwargs) -> None:
        del kwargs
        self.final_calls.append(content)


def registry(*tools: Tool) -> ToolRegistry:
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


def call(call_id: str, name: str, value: str = "one") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments={"value": value})],
        finish_reason="tool_calls",
    )


@pytest.mark.asyncio
async def test_normal_path_never_invokes_guard() -> None:
    progress = Progress()
    provider = ScriptedProvider([call("echo-1", "echo"), LLMResponse(content="done")])

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo once"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "done"
    assert progress.resolutions == []
    assert all(record.node != "guard" for record in state.transitions)
    assert progress.final_calls == ["done"]


@pytest.mark.asyncio
async def test_identical_read_only_poll_runs_again_without_guard() -> None:
    progress = Progress()
    poll = PollingTool()
    provider = ScriptedProvider(
        [
            call("poll-1", "poll"),
            call("poll-2", "poll"),
            LLMResponse(content="done"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(poll),
        messages=[{"role": "user", "content": "wait for it"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "done"
    assert poll.calls == 2
    assert progress.resolutions == []


@pytest.mark.asyncio
async def test_budget_continue_adds_exactly_the_original_budget() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("echo-1", "echo"),
            LLMResponse(content='{"action":"continue"}', usage={"total_tokens": 3}),
            LLMResponse(content="done after review"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "echo"}],
        max_iterations=1,
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.iteration_limit == 2
    assert progress.resolutions[0].event == GuardEvent.BUDGET_EXHAUSTED
    assert progress.resolutions[0].action == GuardAction.CONTINUE
    assert provider.calls[1]["tools"] == []
    assert any("已达步数上限" in message for message in progress.messages)


@pytest.mark.asyncio
async def test_tool_failure_is_reviewed_then_retried() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content='{"action":"retry"}', usage={"total_tokens": 2}),
            LLMResponse(content="recovered"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "recover"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "recovered"
    assert progress.resolutions[0].action == GuardAction.RETRY
    assert "AGENT LOOP RECOVERY" in provider.calls[2]["messages"][-1]["content"]
    assert state.token_ledger.for_kind(NodeKind.HARNESS)["total_tokens"] == 2


@pytest.mark.asyncio
async def test_guard_finalize_uses_one_text_only_finalizer() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content='{"action":"finalize"}'),
            LLMResponse(content="partial work; tool failed"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_GUARD
    assert state.metadata.final_content == "partial work; tool failed"
    assert provider.calls[2]["tools"] == []
    assert progress.final_calls == ["partial work; tool failed"]


@pytest.mark.asyncio
async def test_invalid_guard_output_falls_back_to_one_recovery_attempt() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content="invalid"),
            LLMResponse(content="recovered after fallback"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "recovered after fallback"
    assert progress.resolutions[0].source == "fallback"
    assert progress.resolutions[0].action == GuardAction.RETRY


@pytest.mark.asyncio
async def test_failed_mutation_can_recover_after_invalid_guard_output() -> None:
    progress = Progress()
    tool = FailOnceTool()
    provider = ScriptedProvider(
        [
            call("write-1", "fail_once"),
            LLMResponse(content="invalid"),
            call("write-2", "fail_once"),
            LLMResponse(content="finished after retry"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(tool),
        messages=[{"role": "user", "content": "write it"}],
        max_iterations=3,
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "finished after retry"
    assert tool.calls == 2
    assert progress.resolutions[0].source == "fallback"
    assert progress.resolutions[0].action == GuardAction.RETRY


@pytest.mark.asyncio
async def test_invalid_tool_guard_at_iteration_limit_falls_back_to_finalization() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content="invalid"),
            LLMResponse(content="tool failure wrap-up"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
        max_iterations=1,
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.FAILED
    assert state.metadata.final_content == "tool failure wrap-up"
    assert progress.resolutions[0].source == "fallback"
    assert progress.resolutions[0].action == GuardAction.FINALIZE


@pytest.mark.asyncio
async def test_invalid_budget_guard_output_still_falls_back_to_finalization() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("echo-1", "echo"),
            LLMResponse(content="invalid"),
            LLMResponse(content="budget wrap-up"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(EchoTool()),
        messages=[{"role": "user", "content": "work"}],
        max_iterations=1,
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.FAILED
    assert state.metadata.final_content == "budget wrap-up"
    assert progress.resolutions[0].source == "fallback"
    assert progress.resolutions[0].action == GuardAction.FINALIZE


@pytest.mark.asyncio
async def test_invalid_runtime_guard_output_falls_back_to_finalization() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            LLMResponse(content="provider failed", finish_reason="error"),
            LLMResponse(content="invalid"),
            LLMResponse(content="runtime wrap-up"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(),
        messages=[{"role": "user", "content": "work"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.FAILED
    assert state.metadata.final_content == "runtime wrap-up"
    assert progress.resolutions[0].source == "fallback"
    assert progress.resolutions[0].action == GuardAction.FINALIZE


@pytest.mark.asyncio
async def test_failed_finalizer_uses_hardcoded_honest_message_once() -> None:
    progress = Progress()
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content='{"action":"finalize"}'),
            call("bad-final", "fail"),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.FAILED
    assert "could not safely complete" in (state.metadata.final_content or "")
    assert progress.final_calls == [state.metadata.final_content]


@pytest.mark.asyncio
async def test_serialized_dsml_tool_call_is_rejected_during_finalization() -> None:
    class StreamingProvider(ScriptedProvider):
        async def chat_stream(
            self,
            messages,
            tools=None,
            on_delta=None,
            on_reasoning_delta=None,
            **kwargs,
        ):
            del on_reasoning_delta
            self.calls.append(
                {
                    "kind": "stream",
                    "messages": copy.deepcopy(messages),
                    "tools": copy.deepcopy(tools or []),
                    **kwargs,
                }
            )
            response = self.responses.pop(0)
            if response.content and on_delta is not None:
                await on_delta(response.content)
            return response

    dsml = (
        '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="write">\n'
        "</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>"
    )
    progress = Progress()
    provider = StreamingProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content='{"action":"finalize"}'),
            LLMResponse(content=dsml),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.FAILED
    assert "could not safely complete" in (state.metadata.final_content or "")
    assert dsml not in progress.deltas
    assert all(dsml != message.get("content") for message in state.messages)
    assert provider.calls[-1].get("kind") != "stream"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final_response",
    [
        LLMResponse(content="partial wrap-up", finish_reason="length"),
        LLMResponse(content="<｜｜DSML｜｜tool_calls>"),
    ],
)
async def test_truncated_finalizer_output_uses_hardcoded_message(
    final_response: LLMResponse,
) -> None:
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content='{"action":"finalize"}'),
            final_response,
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
    )

    assert state.metadata.status == RunStatus.FAILED
    assert "could not safely complete" in (state.metadata.final_content or "")


@pytest.mark.asyncio
async def test_tool_protocol_code_on_normal_path_remains_plain_text() -> None:
    example = '<tool_calls><invoke name="write"></invoke></tool_calls>'
    progress = Progress()
    provider = ScriptedProvider([LLMResponse(content=example)])

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(),
        messages=[{"role": "user", "content": "work"}],
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == example


@pytest.mark.asyncio
async def test_fenced_dsml_example_is_allowed_during_finalization() -> None:
    dsml_example = (
        "```xml\n"
        '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="write">'
        "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>\n"
        "```"
    )
    provider = ScriptedProvider(
        [
            call("fail-1", "fail"),
            LLMResponse(content='{"action":"finalize"}'),
            LLMResponse(content=dsml_example),
        ]
    )

    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(FailingTool()),
        messages=[{"role": "user", "content": "work"}],
    )

    assert state.metadata.status == RunStatus.TERMINATED_BY_GUARD
    assert state.metadata.final_content == dsml_example


@pytest.mark.asyncio
async def test_transition_trace_is_correlated_and_contiguous() -> None:
    provider = ScriptedProvider([LLMResponse(content="done")])
    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(),
        messages=[{"role": "user", "content": "answer"}],
        session_id="session-1",
        turn_id="turn-1",
    )

    assert all(record.session_id == "session-1" for record in state.transitions)
    assert all(record.turn_id == "turn-1" for record in state.transitions)
    assert all(
        left.after_digest == right.before_digest
        for left, right in zip(state.transitions, state.transitions[1:], strict=False)
    )


@pytest.mark.asyncio
async def test_invalid_zero_iteration_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        await run_agent_loop(
            provider=ScriptedProvider([]),
            model="model",
            tools=registry(),
            messages=[],
            max_iterations=0,
        )


@pytest.mark.asyncio
async def test_ordinary_runtime_exception_is_reviewed_by_guard() -> None:
    attempts = 0

    async def prepare(messages, minimal_context, additional):
        nonlocal attempts
        del minimal_context
        attempts += 1
        if attempts == 1:
            raise RuntimeError("compaction failed")
        return SimpleNamespace(messages=[*messages, *additional], usage={})

    progress = Progress()
    provider = ScriptedProvider(
        [LLMResponse(content='{"action":"retry"}'), LLMResponse(content="recovered")]
    )
    state = await run_agent_loop(
        provider=provider,
        model="model",
        tools=registry(),
        messages=[{"role": "user", "content": "answer"}],
        prepare_model_input=prepare,
        on_progress=progress,
    )

    assert state.metadata.status == RunStatus.COMPLETED
    assert progress.resolutions[0].event == GuardEvent.RUNTIME_ERROR
    assert progress.resolutions[0].action == GuardAction.RETRY
