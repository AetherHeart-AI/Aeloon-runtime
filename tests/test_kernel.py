from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from aeloon_core.kernel import run_agent_kernel
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry


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
    description = "Execute a command."
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
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del model, max_tokens, temperature, reasoning_effort, tool_choice, response_format
        self.calls.append({"messages": messages, "tools": tools or []})
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)


def registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def registry_with_echo_and_fail() -> ToolRegistry:
    registry = registry_with_echo()
    registry.register(FailingTool())
    return registry


def registry_with_timeout_exec() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(TimeoutExecTool())
    return registry


@pytest.mark.asyncio
async def test_kernel_auto_continues_after_tool_iteration_limit() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done after echo", finish_reason="stop"),
        ]
    )

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=1,
        max_auto_continue_iterations=1,
        max_finalization_iterations=1,
    )

    assert final_content == "done after echo"
    assert tools_used == ["echo"]
    assert messages[-1] == {"role": "assistant", "content": "done after echo"}
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"]
    assert "MAXIMUM ITERATIONS REACHED" not in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_kernel_finalizes_after_auto_continue_budget_is_exhausted() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-2", name="echo", arguments={"value": "two"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="wrapped up", finish_reason="stop"),
        ]
    )

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "echo twice"}],
        max_iterations=1,
        max_auto_continue_iterations=1,
        max_finalization_iterations=1,
    )

    assert final_content == "wrapped up"
    assert tools_used == ["echo", "echo"]
    assert messages[-1] == {"role": "assistant", "content": "wrapped up"}
    assert len(provider.calls) == 3
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"]
    assert provider.calls[2]["tools"] == []
    assert "MAXIMUM ITERATIONS REACHED" in provider.calls[2]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_kernel_recovers_when_output_budget_exhausts_without_visible_text() -> None:
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

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "answer visibly"}],
        max_iterations=5,
        max_auto_continue_iterations=0,
        max_finalization_iterations=1,
    )

    assert final_content == "visible answer"
    assert tools_used == []
    assert messages[-1] == {"role": "assistant", "content": "visible answer"}
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"] == []
    assert provider.calls[1]["messages"][-1]["role"] == "user"
    assert "VISIBLE ANSWER REQUIRED" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_kernel_reports_when_finalization_budget_exhausts_without_visible_text() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                finish_reason="length",
                reasoning_content="hidden reasoning used the whole output budget",
            ),
            LLMResponse(
                content=None,
                finish_reason="length",
                reasoning_content="still hidden",
            ),
            LLMResponse(
                content=None,
                finish_reason="length",
                reasoning_content="still hidden again",
            ),
        ]
    )

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "answer visibly"}],
        max_iterations=5,
        max_auto_continue_iterations=0,
        max_finalization_iterations=2,
    )

    assert "repeatedly exhausted its output budget" in (final_content or "")
    assert "No final artifact was produced" in (final_content or "")
    assert tools_used == []
    assert messages[-1] == {"role": "assistant", "content": final_content}
    assert len(provider.calls) == 3
    assert provider.calls[0]["tools"]
    assert provider.calls[1]["tools"] == []
    assert provider.calls[2]["tools"] == []


@pytest.mark.asyncio
async def test_kernel_preserves_old_failure_when_finalization_budget_disabled() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
        ]
    )

    final_content, tools_used, _messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=1,
        max_auto_continue_iterations=0,
        max_finalization_iterations=0,
    )

    assert "maximum number of tool call iterations (1)" in (final_content or "")
    assert "automatic continuation budget (0)" in (final_content or "")
    assert tools_used == ["echo"]
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_kernel_allows_model_to_recover_after_duplicate_tool_call() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-2", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done with existing echo result", finish_reason="stop"),
        ]
    )

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )

    assert final_content == "done with existing echo result"
    assert tools_used == ["echo"]
    assert messages[-1] == {"role": "assistant", "content": "done with existing echo result"}
    assert len(provider.calls) == 3
    assert messages[-2]["role"] == "tool"
    assert messages[-2]["tool_call_id"] == "call-2"
    assert "Skipped duplicate call to echo" in messages[-2]["content"]


@pytest.mark.asyncio
async def test_kernel_stops_after_consecutive_duplicate_tool_rounds() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-2", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-3", name="echo", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
        ]
    )

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo(),
        messages=[{"role": "user", "content": "echo once"}],
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )

    assert "off track" in (final_content or "")
    assert "repeated tool calls" in (final_content or "")
    assert tools_used == ["echo"]
    assert messages[-1]["role"] == "assistant"
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_kernel_stops_after_consecutive_failed_tool_rounds() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="fail", arguments={"value": "one"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-2", name="fail", arguments={"value": "two"})],
                finish_reason="tool_calls",
            ),
        ]
    )

    final_content, tools_used, _messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_echo_and_fail(),
        messages=[{"role": "user", "content": "fail twice"}],
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )

    assert "off track" in (final_content or "")
    assert "failed or returned errors" in (final_content or "")
    assert tools_used == ["fail", "fail"]
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_kernel_allows_recovery_after_consecutive_exec_timeouts() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call-1",
                        name="exec",
                        arguments={"command": "python3 -m http.server 8765"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call-2",
                        name="exec",
                        arguments={"command": "python3 -m http.server 8765 & echo pid=$!"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="I will use a shorter verification command.", finish_reason="stop"),
        ]
    )

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=registry_with_timeout_exec(),
        messages=[{"role": "user", "content": "start a server"}],
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )

    assert final_content == "I will use a shorter verification command."
    assert tools_used == ["exec", "exec"]
    assert messages[-1] == {
        "role": "assistant",
        "content": "I will use a shorter verification command.",
    }
    assert len(provider.calls) == 3
