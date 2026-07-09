from __future__ import annotations

from types import SimpleNamespace

from aeloon_core.loop_guard import AgentLoopGuard, LoopGuardAction
from aeloon_core.providers.base import ToolCallRequest


def _answered_tool_call() -> list[dict]:
    tool_call = ToolCallRequest(id="call-1", name="echo", arguments={"value": "one"})
    return [
        {"role": "assistant", "content": None, "tool_calls": [tool_call.to_openai_tool_call()]},
        {"role": "tool", "tool_call_id": "call-1", "name": "echo", "content": "echo:one"},
    ]


def test_duplicate_only_round_returns_to_model_then_stops() -> None:
    guard = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )
    messages = _answered_tool_call()

    first = guard.handle_duplicate_tool_calls(
        messages,
        [ToolCallRequest(id="call-2", name="echo", arguments={"value": "one"})],
    )

    assert first.decision.action == LoopGuardAction.RETURN_TO_MODEL
    assert first.executable_calls == []
    assert first.duplicate_calls[0].id == "call-2"
    assert first.tool_results[0].call_id == "call-2"
    assert "Skipped duplicate call to echo" in first.tool_results[0].content

    second = guard.handle_duplicate_tool_calls(
        messages,
        [ToolCallRequest(id="call-3", name="echo", arguments={"value": "one"})],
    )

    assert second.decision.action == LoopGuardAction.STOP_OFF_TRACK
    assert "repeated tool calls" in second.decision.reason
    assert "off track" in (second.decision.final_content or "")


def test_malformed_only_round_returns_tool_error_and_counts_unproductive() -> None:
    guard = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )

    result = guard.handle_malformed_tool_calls(
        [ToolCallRequest(id="bad-1", name="echo", arguments=["not-an-object"])]
    )

    assert result.decision.action == LoopGuardAction.RETURN_TO_MODEL
    assert result.executable_calls == []
    assert result.malformed_calls[0].id == "bad-1"
    assert result.tool_results[0].call_id == "bad-1"
    assert "must be a JSON object" in result.tool_results[0].content
    assert guard.unproductive_tool_rounds == 1


def test_productive_round_resets_unproductive_counter() -> None:
    guard = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )

    first = guard.handle_unproductive_tool_round("first miss")
    guard.record_productive_tool_round()
    second = guard.handle_unproductive_tool_round("second miss")

    assert first.action == LoopGuardAction.RETURN_TO_MODEL
    assert second.action == LoopGuardAction.RETURN_TO_MODEL
    assert guard.unproductive_tool_rounds == 1


def test_consecutive_failed_tool_rounds_stop_off_track() -> None:
    guard = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )
    failed_node = SimpleNamespace(result="Error: failed")

    first = guard.handle_tool_results([failed_node])
    second = guard.handle_tool_results([failed_node])

    assert first.action == LoopGuardAction.RETURN_TO_MODEL
    assert second.action == LoopGuardAction.STOP_OFF_TRACK
    assert "failed or returned errors" in second.reason


def test_exec_timeout_rounds_get_extra_recovery_before_stopping() -> None:
    guard = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=5,
        max_finalization_iterations=1,
    )
    timed_out_node = SimpleNamespace(
        tool_name="exec",
        result="Error: Command timed out after 3 seconds",
    )

    first = guard.handle_tool_results([timed_out_node])
    second = guard.handle_tool_results([timed_out_node])
    third = guard.handle_tool_results([timed_out_node])

    assert first.action == LoopGuardAction.RETURN_TO_MODEL
    assert second.action == LoopGuardAction.RETURN_TO_MODEL
    assert third.action == LoopGuardAction.STOP_OFF_TRACK
    assert "repeatedly timed out" in third.reason


def test_iteration_budget_decisions_auto_continue_then_finalize_then_exhaust() -> None:
    guard = AgentLoopGuard(
        max_iterations=2,
        max_auto_continue_iterations=3,
        max_finalization_iterations=1,
    )

    first = guard.handle_iteration_budget_reached()
    second = guard.handle_iteration_budget_reached()
    third = guard.handle_iteration_budget_reached()

    assert first.action == LoopGuardAction.EXTEND_BUDGET
    assert first.budget_grant == 2
    assert second.action == LoopGuardAction.EXTEND_BUDGET
    assert second.budget_grant == 1
    assert third.action == LoopGuardAction.FINALIZE
    assert third.prompt_message is not None
    assert "MAXIMUM ITERATIONS REACHED" in third.prompt_message["content"]

    exhausted = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=0,
        max_finalization_iterations=0,
    ).handle_iteration_budget_reached()

    assert exhausted.action == LoopGuardAction.FINAL_RESPONSE
    assert "maximum number of tool call iterations" in (exhausted.final_content or "")


def test_empty_response_retries_once_then_returns_final_error() -> None:
    guard = AgentLoopGuard(
        max_iterations=1,
        max_auto_continue_iterations=0,
        max_finalization_iterations=1,
    )

    first = guard.handle_empty_or_exhausted_response(
        finish_reason="stop",
        finalizing=False,
        finalization_iteration=0,
    )
    second = guard.handle_empty_or_exhausted_response(
        finish_reason="stop",
        finalizing=False,
        finalization_iteration=0,
    )

    assert first.action == LoopGuardAction.CONTINUE
    assert second.action == LoopGuardAction.FINAL_RESPONSE
    assert "empty response" in (second.final_content or "")


def test_output_budget_exhaustion_enters_and_limits_finalization() -> None:
    guard = AgentLoopGuard(
        max_iterations=5,
        max_auto_continue_iterations=0,
        max_finalization_iterations=2,
    )

    first = guard.handle_empty_or_exhausted_response(
        finish_reason="length",
        finalizing=False,
        finalization_iteration=0,
    )
    retry = guard.handle_empty_or_exhausted_response(
        finish_reason="length",
        finalizing=True,
        finalization_iteration=1,
    )
    exhausted = guard.handle_empty_or_exhausted_response(
        finish_reason="length",
        finalizing=True,
        finalization_iteration=2,
    )

    assert first.action == LoopGuardAction.FINALIZE
    assert first.prompt_message is not None
    assert "VISIBLE ANSWER REQUIRED" in first.prompt_message["content"]
    assert retry.action == LoopGuardAction.CONTINUE
    assert exhausted.action == LoopGuardAction.FINAL_RESPONSE
    assert "repeatedly exhausted its output budget" in (exhausted.final_content or "")
