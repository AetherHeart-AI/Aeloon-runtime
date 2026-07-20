from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from aeloon_core.agents import AgentRuntime, GuardAgent, ToolAgent
from aeloon_core.context_view import ContextViewPipeline
from aeloon_core.loop_guard import GuardAction, GuardEvent, GuardReviewer
from aeloon_core.providers.base import LLMResponse, ToolCallRequest
from aeloon_core.state import AgentNode, LightweightState
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import TransitionRecorder


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class _StableReadTool(Tool):
    name = "read"
    description = "Return stable contents."
    concurrency_mode = "read_only"
    args_model = _ReadArgs

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return "unchanged contents"


def _runtime(*, tools: ToolRegistry | None = None) -> tuple[AgentRuntime, MagicMock]:
    provider = MagicMock()
    provider.supports_concurrent_calls = False
    registry = tools or ToolRegistry()
    runtime = AgentRuntime(
        provider=provider,
        model="test-model",
        tools=registry,
        guard=GuardReviewer(provider=provider, model="test-model"),
        base_iteration_budget=5,
        context_pipeline=ContextViewPipeline(provider=provider, model="test-model"),
        recorder=TransitionRecorder(persist=None),
        on_progress=AsyncMock(),
    )
    return runtime, provider


@pytest.mark.asyncio
async def test_transient_guard_retries_once_without_model_then_caps_recovery() -> None:
    runtime, provider = _runtime()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content='{"action":"retry"}', finish_reason="end_turn")
    )
    state = LightweightState.from_messages(
        [{"role": "user", "content": "finish the task"}],
        max_iterations=20,
    )

    async def queue() -> None:
        await runtime.queue_guard(
            state,
            event=GuardEvent.RUNTIME_ERROR,
            cause="HTTP 503 temporarily unavailable",
            reason_code="provider_error",
        )

    await queue()
    state = await GuardAgent(runtime).run(state)
    provider.chat_with_retry.assert_not_awaited()
    assert runtime.last_decision["source"] == "policy"
    assert runtime.last_decision["action"] == GuardAction.RETRY.value

    await queue()
    state = await GuardAgent(runtime).run(state)
    provider.chat_with_retry.assert_awaited_once()
    assert runtime.last_decision["source"] == "guard"

    await queue()
    state = await GuardAgent(runtime).run(state)
    assert provider.chat_with_retry.await_count == 1
    assert state.metadata.status.value == "finalizing"
    assert state.metadata.finalization_source == "policy"


@pytest.mark.parametrize(
    "cause",
    [
        "Error code: 429 - {'type': 'rate_limit_error'}",
        "Error code: 529 - {'type': 'overloaded_error'}",
    ],
)
@pytest.mark.asyncio
async def test_provider_native_transient_errors_retry_without_model(cause: str) -> None:
    runtime, provider = _runtime()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content='{"action":"finalize"}', finish_reason="end_turn")
    )
    state = LightweightState.from_messages(
        [{"role": "user", "content": "finish the task"}],
        max_iterations=20,
    )
    await runtime.queue_guard(
        state,
        event=GuardEvent.RUNTIME_ERROR,
        cause=cause,
        reason_code="provider_error",
    )

    state = await GuardAgent(runtime).run(state)

    provider.chat_with_retry.assert_not_awaited()
    assert runtime.last_decision["source"] == "policy"
    assert runtime.last_decision["action"] == GuardAction.RETRY.value
    assert state.metadata.status.value == "running"


@pytest.mark.asyncio
async def test_heterogeneous_tool_failures_require_model_judgment() -> None:
    runtime, provider = _runtime()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content='{"action":"finalize"}', finish_reason="end_turn")
    )
    state = LightweightState.from_messages(
        [{"role": "user", "content": "finish the task"}],
        max_iterations=20,
    )
    await runtime.queue_guard(
        state,
        event=GuardEvent.TOOL_ERROR,
        cause="multiple tools failed",
        reason_code="repeated_tool_error",
        failures=(
            {"tool_name": "read", "result": "HTTP 503 temporarily unavailable"},
            {"tool_name": "read", "result": "permission denied"},
        ),
    )

    state = await GuardAgent(runtime).run(state)

    provider.chat_with_retry.assert_awaited_once()
    assert runtime.last_decision["source"] == "guard"
    assert state.metadata.finalization_source == "guard"


@pytest.mark.asyncio
async def test_budget_guard_can_continue_only_once_even_when_model_falls_back() -> None:
    runtime, provider = _runtime()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="not-json", finish_reason="end_turn")
    )
    state = LightweightState.from_messages(
        [{"role": "user", "content": "finish the task"}],
        max_iterations=5,
    )

    await runtime.queue_guard(
        state,
        event=GuardEvent.BUDGET_EXHAUSTED,
        cause="iteration budget",
        reason_code="iteration_budget",
    )
    state = await GuardAgent(runtime).run(state)
    assert state.metadata.budget_guard_continues == 1
    assert state.metadata.status.value == "running"

    await runtime.queue_guard(
        state,
        event=GuardEvent.BUDGET_EXHAUSTED,
        cause="iteration budget",
        reason_code="iteration_budget",
    )
    state = await GuardAgent(runtime).run(state)
    assert provider.chat_with_retry.await_count == 1
    assert state.metadata.status.value == "finalizing"
    assert state.metadata.finalization_source == "policy"


@pytest.mark.asyncio
async def test_tool_agent_queues_stuck_after_four_cross_step_read_exchanges() -> None:
    tools = ToolRegistry()
    tools.register(_StableReadTool())
    runtime, _provider = _runtime(tools=tools)
    state = LightweightState.from_messages(
        [{"role": "user", "content": "inspect the same file"}],
        max_iterations=10,
    )
    agent = ToolAgent(runtime)

    for index in range(4):
        call = ToolCallRequest(
            id=f"read-{index}",
            name="read",
            arguments={"path": "README.md"},
        )
        state.pending_response = LLMResponse(
            content=f"wording {index}",
            tool_calls=[call],
            finish_reason="tool_use",
        )
        state.pending_tool_calls = [call]
        state.metadata.phase = AgentNode.TOOL
        state = await agent.run(state)
        if index < 3:
            assert state.pending_guard_request is None

    request = state.pending_guard_request
    assert request is not None
    assert request.evidence.event == GuardEvent.STUCK
    assert request.evidence.reason_code == "repeated_action_observation"
    assert request.evidence.stuck["repetitions"] == 4
    assert request.fallback_action == GuardAction.FINALIZE
