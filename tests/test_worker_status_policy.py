"""Policy tests for relaxed Worker tool-error and budget handling."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, ConfigDict

from aeloon_core.agents import (
    AgentRuntime,
    GuardAgent,
    RouterAgent,
    ToolAgent,
    _guard_fallback_action,
)
from aeloon_core.loop_guard import GuardAction, GuardEvent, GuardReviewer
from aeloon_core.minimal_context import MinimalContextProcessor
from aeloon_core.providers.base import LLMResponse, ToolCallRequest
from aeloon_core.state import AgentNode, LightweightState, RunStatus, StateMetadata
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import TransitionRecorder
from aeloon_core.worker_sessions import WorkerRunStatus


class _EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FailThenSucceedTool(Tool):
    name = "probe"
    description = "Probe tool for soft-error policy tests."
    concurrency_mode = "exclusive"
    args_model = _EmptyArgs

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        self.calls += 1
        if self.calls <= 2:
            return "Error [PROBE_FAILED]: intentional failure for soft feedback."
        return "ok"


class GuardFallbackPolicyTests(unittest.TestCase):
    def test_budget_fallback_prefers_continue(self) -> None:
        action = _guard_fallback_action(
            event=GuardEvent.BUDGET_EXHAUSTED,
            allowed_actions=(GuardAction.CONTINUE, GuardAction.FINALIZE),
            iteration=25,
            iteration_limit=25,
        )
        self.assertEqual(action, GuardAction.CONTINUE)

    def test_tool_error_fallback_prefers_retry_within_budget(self) -> None:
        action = _guard_fallback_action(
            event=GuardEvent.TOOL_ERROR,
            allowed_actions=(GuardAction.RETRY, GuardAction.FINALIZE),
            iteration=3,
            iteration_limit=10,
        )
        self.assertEqual(action, GuardAction.RETRY)

    def test_tool_error_fallback_finalizes_when_budget_spent(self) -> None:
        action = _guard_fallback_action(
            event=GuardEvent.TOOL_ERROR,
            allowed_actions=(GuardAction.RETRY, GuardAction.FINALIZE),
            iteration=10,
            iteration_limit=10,
        )
        self.assertEqual(action, GuardAction.FINALIZE)


class WorkerRunStatusMappingTests(unittest.TestCase):
    def test_waiting_is_settled_but_not_terminal(self) -> None:
        self.assertTrue(WorkerRunStatus.WAITING_FOR_CONTEXT.settled)
        self.assertFalse(WorkerRunStatus.WAITING_FOR_CONTEXT.terminal)

    def test_completed_is_terminal_and_settled(self) -> None:
        self.assertTrue(WorkerRunStatus.COMPLETED.terminal)
        self.assertTrue(WorkerRunStatus.COMPLETED.settled)

    def test_running_is_not_settled(self) -> None:
        self.assertFalse(WorkerRunStatus.RUNNING.settled)


class SoftToolFeedbackTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self, *, threshold: int = 3, auto_continues: int = 2) -> AgentRuntime:
        tools = ToolRegistry()
        tools.register(_FailThenSucceedTool())
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
            tool_error_guard_threshold=threshold,
            budget_auto_continues=auto_continues,
            on_progress=AsyncMock(),
        )

    async def test_tool_errors_soft_feed_until_threshold(self) -> None:
        runtime = self._runtime(threshold=3)
        agent = ToolAgent(runtime)
        state = LightweightState.from_messages(
            [{"role": "user", "content": "do work"}],
            max_iterations=10,
            metadata=StateMetadata(phase=AgentNode.TOOL),
        )
        call = ToolCallRequest(id="c1", name="probe", arguments={})
        state.pending_tool_calls = [call]
        state.pending_response = LLMResponse(
            content="",
            tool_calls=[call],
            finish_reason="tool_use",
        )

        # Round 1 — soft feedback, no Guard.
        state = await agent.run(state)
        self.assertIsNone(state.pending_guard_request)
        self.assertEqual(state.metadata.consecutive_tool_failure_rounds, 1)
        self.assertEqual(state.metadata.phase, AgentNode.ROUTER)
        self.assertTrue(
            any(
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and str(block.get("content", "")).startswith("Error")
                for m in state.messages
                if m.get("role") == "user" and isinstance(m.get("content"), list)
                for block in m["content"]
            )
        )

        # Round 2 — still soft.
        state.metadata.phase = AgentNode.TOOL
        call2 = ToolCallRequest(id="c2", name="probe", arguments={})
        state.pending_tool_calls = [call2]
        state.pending_response = LLMResponse(
            content="",
            tool_calls=[call2],
            finish_reason="tool_use",
        )
        state = await agent.run(state)
        self.assertIsNone(state.pending_guard_request)
        self.assertEqual(state.metadata.consecutive_tool_failure_rounds, 2)

        # Round 3 — escalate to Guard.
        state.metadata.phase = AgentNode.TOOL
        call3 = ToolCallRequest(id="c3", name="probe", arguments={})
        state.pending_tool_calls = [call3]
        state.pending_response = LLMResponse(
            content="",
            tool_calls=[call3],
            finish_reason="tool_use",
        )
        # Force another failure for the third round.
        tool = runtime.tools.get("probe")
        assert isinstance(tool, _FailThenSucceedTool)
        tool.calls = 0
        state = await agent.run(state)
        self.assertIsNotNone(state.pending_guard_request)
        assert state.pending_guard_request is not None
        self.assertEqual(state.pending_guard_request.evidence.event, GuardEvent.TOOL_ERROR)
        self.assertEqual(state.metadata.consecutive_tool_failure_rounds, 3)

    async def test_clean_tool_round_resets_failure_counter(self) -> None:
        runtime = self._runtime(threshold=3)
        agent = ToolAgent(runtime)
        state = LightweightState.from_messages(
            [{"role": "user", "content": "do work"}],
            max_iterations=10,
            metadata=StateMetadata(phase=AgentNode.TOOL),
        )
        tool = runtime.tools.get("probe")
        assert isinstance(tool, _FailThenSucceedTool)

        # One failure.
        tool.calls = 0
        call = ToolCallRequest(id="c1", name="probe", arguments={})
        state.pending_tool_calls = [call]
        state.pending_response = LLMResponse(
            content="",
            tool_calls=[call],
            finish_reason="tool_use",
        )
        state = await agent.run(state)
        self.assertEqual(state.metadata.consecutive_tool_failure_rounds, 1)

        # Success resets.
        tool.calls = 10
        state.metadata.phase = AgentNode.TOOL
        call2 = ToolCallRequest(id="c2", name="probe", arguments={})
        state.pending_tool_calls = [call2]
        state.pending_response = LLMResponse(
            content="",
            tool_calls=[call2],
            finish_reason="tool_use",
        )
        state = await agent.run(state)
        self.assertEqual(state.metadata.consecutive_tool_failure_rounds, 0)
        self.assertIsNone(state.pending_guard_request)

    async def test_budget_auto_continues_before_guard(self) -> None:
        runtime = self._runtime(auto_continues=2)
        # base_iteration_budget is 5 in the test runtime fixture.
        state = LightweightState.from_messages(
            [{"role": "user", "content": "do work"}],
            max_iterations=2,
        )
        state.metadata.iteration = 2
        state.metadata.iteration_limit = 2

        state = await runtime.grant_more_or_finalize(state)
        self.assertIsNone(state.pending_guard_request)
        self.assertEqual(state.metadata.budget_auto_continues_used, 1)
        self.assertEqual(state.metadata.iteration_limit, 2 + runtime.base_iteration_budget)
        self.assertEqual(state.metadata.phase, AgentNode.ROUTER)

        state.metadata.iteration = state.metadata.iteration_limit
        state = await runtime.grant_more_or_finalize(state)
        self.assertEqual(state.metadata.budget_auto_continues_used, 2)
        self.assertEqual(state.metadata.iteration_limit, 2 + 2 * runtime.base_iteration_budget)
        self.assertIsNone(state.pending_guard_request)

        state.metadata.iteration = state.metadata.iteration_limit
        state = await runtime.grant_more_or_finalize(state)
        self.assertIsNotNone(state.pending_guard_request)
        assert state.pending_guard_request is not None
        self.assertEqual(state.pending_guard_request.evidence.event, GuardEvent.BUDGET_EXHAUSTED)
        self.assertEqual(state.pending_guard_request.fallback_action, GuardAction.CONTINUE)

    async def test_hard_token_budget_stops_without_another_model_call(self) -> None:
        runtime = self._runtime()
        runtime.max_tokens = 10
        state = LightweightState.from_messages(
            [{"role": "user", "content": "do work"}],
            max_iterations=10,
        )
        state.token_ledger.record("domain", {"total_tokens": 10})

        state = await RouterAgent(runtime).run(state)

        self.assertEqual(state.metadata.status, RunStatus.TERMINATED_BY_GUARD)
        self.assertEqual(state.metadata.phase, AgentNode.DONE)
        self.assertIn("token budget exhausted", state.metadata.final_content or "")

    async def test_hard_tool_budget_rejects_whole_batch_without_side_effects(self) -> None:
        runtime = self._runtime()
        runtime.max_tool_calls = 1
        state = LightweightState.from_messages(
            [{"role": "user", "content": "do work"}],
            max_iterations=10,
            metadata=StateMetadata(phase=AgentNode.TOOL),
        )
        calls = [
            ToolCallRequest(id="one", name="probe", arguments={}),
            ToolCallRequest(id="two", name="probe", arguments={}),
        ]
        state.pending_tool_calls = calls
        state.pending_response = LLMResponse(
            content=None,
            tool_calls=calls,
            finish_reason="tool_use",
        )

        state = await ToolAgent(runtime).run(state)

        tool = runtime.tools.get("probe")
        assert isinstance(tool, _FailThenSucceedTool)
        self.assertEqual(tool.calls, 0)
        self.assertEqual(state.metadata.status, RunStatus.TERMINATED_BY_GUARD)
        self.assertIn("tool-call budget exhausted", state.metadata.final_content or "")

    async def test_guard_falls_back_locally_when_remaining_tokens_cannot_fit(self) -> None:
        runtime = self._runtime()
        runtime.max_tokens = 100
        state = LightweightState.from_messages(
            [{"role": "user", "content": "do work"}],
            max_iterations=10,
        )
        state.token_ledger.record("domain", {"total_tokens": 99})
        state = await runtime.queue_guard(
            state,
            event=GuardEvent.RUNTIME_ERROR,
            cause="needs review",
        )

        state = await GuardAgent(runtime).run(state)

        runtime.provider.chat.assert_not_called()
        self.assertEqual(state.metadata.phase, AgentNode.ROUTER)

    async def test_queue_guard_budget_fallback_is_continue(self) -> None:
        runtime = self._runtime()
        state = LightweightState.from_messages(
            [{"role": "user", "content": "goal"}],
            max_iterations=5,
        )
        state = await runtime.queue_guard(
            state,
            event=GuardEvent.BUDGET_EXHAUSTED,
            cause="budget",
        )
        assert state.pending_guard_request is not None
        self.assertEqual(state.pending_guard_request.fallback_action, GuardAction.CONTINUE)


if __name__ == "__main__":
    unittest.main()
