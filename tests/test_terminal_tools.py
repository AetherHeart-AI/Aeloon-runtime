"""Protocol tests for explicit terminal tools in the generic UASM."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from aeloon_core.agents import AgentRuntime, ModelAgent, RouterAgent, ToolAgent
from aeloon_core.loop_guard import GuardReviewer
from aeloon_core.minimal_context import MinimalContextProcessor
from aeloon_core.providers.base import LLMResponse, ToolCallRequest
from aeloon_core.state import AgentNode, LightweightState, RunStatus, StateMetadata
from aeloon_core.tools.base import FunctionTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import TransitionRecorder


class _EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def test_generic_uasm_exposes_only_shared_runtime_nodes() -> None:
    assert {node.value for node in AgentNode} == {
        "router",
        "model",
        "tool",
        "guard",
        "done",
    }


def _runtime(
    tools: ToolRegistry,
    *,
    require_terminal: bool = False,
) -> AgentRuntime:
    provider = MagicMock()
    provider.supports_concurrent_calls = False
    return AgentRuntime(
        provider=provider,
        model="test-model",
        tools=tools,
        guard=GuardReviewer(provider=provider, model="test-model"),
        base_iteration_budget=5,
        context_processor=MinimalContextProcessor(),
        recorder=TransitionRecorder(persist=None),
        require_terminal=require_terminal,
        on_progress=AsyncMock(),
    )


def _tool_state(*calls: ToolCallRequest) -> LightweightState:
    state = LightweightState.from_messages(
        [{"role": "user", "content": "finish the work"}],
        max_iterations=10,
        metadata=StateMetadata(phase=AgentNode.TOOL),
    )
    state.pending_tool_calls = list(calls)
    state.pending_response = LLMResponse(
        content=None,
        tool_calls=list(calls),
        finish_reason="tool_use",
    )
    return state


@pytest.mark.asyncio
async def test_mixed_terminal_batch_has_zero_tool_side_effects() -> None:
    calls = {"complete": 0, "write": 0}

    async def complete(**kwargs: Any) -> str:
        del kwargs
        calls["complete"] += 1
        return "complete"

    async def write(**kwargs: Any) -> str:
        del kwargs
        calls["write"] += 1
        return "written"

    tools = ToolRegistry()
    tools.register(
        FunctionTool(
            name="complete_work",
            description="Complete the run.",
            args_model=_EmptyArgs,
            handler=complete,
            terminal=True,
        )
    )
    tools.register(
        FunctionTool(
            name="write",
            description="Mutate a file.",
            args_model=_EmptyArgs,
            handler=write,
            concurrency_mode="mutating",
        )
    )
    state = _tool_state(
        ToolCallRequest(id="terminal", name="complete_work", arguments={}),
        ToolCallRequest(id="write", name="write", arguments={}),
    )

    state = await ToolAgent(_runtime(tools)).run(state)

    assert calls == {"complete": 0, "write": 0}
    assert state.metadata.status is RunStatus.RUNNING
    assert state.metadata.phase is AgentNode.ROUTER
    results = [
        block
        for message in state.messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(results) == 2
    assert all("rejected without execution" in str(result["content"]) for result in results)


@pytest.mark.asyncio
async def test_multiple_terminal_calls_are_rejected_before_execution() -> None:
    executions = 0

    async def complete(**kwargs: Any) -> str:
        nonlocal executions
        del kwargs
        executions += 1
        return "complete"

    tools = ToolRegistry()
    tools.register(
        FunctionTool(
            name="complete_work",
            description="Complete the run.",
            args_model=_EmptyArgs,
            handler=complete,
            terminal=True,
        )
    )
    state = _tool_state(
        ToolCallRequest(id="one", name="complete_work", arguments={}),
        ToolCallRequest(id="two", name="complete_work", arguments={}),
    )

    await ToolAgent(_runtime(tools)).run(state)

    assert executions == 0


@pytest.mark.asyncio
async def test_successful_sole_terminal_tool_finishes_loop() -> None:
    executions = 0

    async def complete(**kwargs: Any) -> str:
        nonlocal executions
        del kwargs
        executions += 1
        return "accepted report"

    tools = ToolRegistry()
    terminal = FunctionTool(
        name="complete_work",
        description="Complete the run.",
        args_model=_EmptyArgs,
        handler=complete,
        terminal=True,
    )
    tools.register(terminal)
    state = _tool_state(ToolCallRequest(id="done", name="complete_work", arguments={}))

    state = await ToolAgent(_runtime(tools)).run(state)

    assert terminal.terminal is True
    assert executions == 1
    assert state.metadata.status is RunStatus.COMPLETED
    assert state.metadata.phase is AgentNode.DONE
    assert state.metadata.final_content == "accepted report"
    assert state.metadata.extras["terminal_tool"] == {
        "name": "complete_work",
        "call_id": "done",
    }


@pytest.mark.asyncio
async def test_new_user_assignment_resets_side_effect_deduplication() -> None:
    executions = 0

    async def complete(**kwargs: Any) -> str:
        nonlocal executions
        del kwargs
        executions += 1
        return "accepted again"

    tools = ToolRegistry()
    tools.register(
        FunctionTool(
            name="complete_work",
            description="Complete the run.",
            args_model=_EmptyArgs,
            handler=complete,
            terminal=True,
        )
    )
    current = ToolCallRequest(id="current", name="complete_work", arguments={})
    state = _tool_state(current)
    state.messages = [
        {"role": "user", "content": "first WorkerRun"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "prior",
                    "name": "complete_work",
                    "input": {},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "prior",
                    "content": "accepted",
                }
            ],
        },
        {"role": "user", "content": "second WorkerRun"},
    ]

    state = await ToolAgent(_runtime(tools)).run(state)

    assert executions == 1
    assert state.metadata.status is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_generated_terminal_call_executes_at_exact_token_limit() -> None:
    executions = 0

    async def complete(**kwargs: Any) -> str:
        nonlocal executions
        del kwargs
        executions += 1
        return "accepted at the limit"

    tools = ToolRegistry()
    tools.register(
        FunctionTool(
            name="complete_work",
            description="Complete the run.",
            args_model=_EmptyArgs,
            handler=complete,
            terminal=True,
        )
    )
    runtime = _runtime(tools, require_terminal=True)
    runtime.max_tokens = 10
    call = ToolCallRequest(id="done", name="complete_work", arguments={})
    state = _tool_state(call)
    state.metadata.phase = AgentNode.ROUTER
    state.token_ledger.record("domain", {"total_tokens": 10})

    state = await RouterAgent(runtime).run(state)
    assert state.metadata.phase is AgentNode.TOOL
    state = await ToolAgent(runtime).run(state)

    assert executions == 1
    assert state.metadata.status is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_strict_loop_routes_bare_text_through_guard() -> None:
    tools = ToolRegistry()

    async def complete(**kwargs: Any) -> str:
        del kwargs
        return "complete"

    tools.register(
        FunctionTool(
            name="complete_work",
            description="Complete the run.",
            args_model=_EmptyArgs,
            handler=complete,
            terminal=True,
        )
    )
    runtime = _runtime(tools, require_terminal=True)
    state = LightweightState.from_messages(
        [{"role": "user", "content": "finish the work"}],
        active_tools=["complete_work"],
        max_iterations=10,
        metadata=StateMetadata(phase=AgentNode.MODEL),
    )

    state = await ModelAgent(runtime)._handle_text_response(
        state,
        LLMResponse(content="I am done.", finish_reason="end_turn"),
    )

    assert state.metadata.status is RunStatus.RUNNING
    assert state.metadata.phase is AgentNode.ROUTER
    assert state.pending_guard_request is not None
    assert "terminal tool protocol error" in state.pending_guard_request.evidence.cause
    assert any(
        message.get("role") == "system"
        and "TERMINAL TOOL PROTOCOL" in str(message.get("content"))
        for message in state.messages
    )
