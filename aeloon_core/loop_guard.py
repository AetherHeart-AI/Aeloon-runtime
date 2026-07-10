"""Deterministic guard decisions for the agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger

from aeloon_core.providers.base import ToolCallRequest
from aeloon_core.utils.tool_history import (
    collect_tool_call_fingerprints,
    duplicate_tool_result,
    tool_call_fingerprint,
)


class LoopGuardAction(StrEnum):
    """Actions the kernel can take after a guard decision."""

    CONTINUE = "continue"
    RETURN_TO_MODEL = "return_to_model"
    EXTEND_BUDGET = "extend_budget"
    FINALIZE = "finalize"
    FINAL_RESPONSE = "final_response"
    STOP_OFF_TRACK = "stop_off_track"


@dataclass(frozen=True)
class ToolResultPatch:
    """A tool result the kernel should append to the message history."""

    call_id: str
    tool_name: str
    content: str


@dataclass(frozen=True)
class LoopGuardDecision:
    """A guard decision with optional payload for the kernel."""

    action: LoopGuardAction
    reason: str = ""
    final_content: str | None = None
    prompt_message: dict[str, str] | None = None
    progress_message: str | None = None
    budget_grant: int = 0


@dataclass(frozen=True)
class ToolCallGuardResult:
    """Classified tool calls plus the resulting guard decision."""

    executable_calls: list[ToolCallRequest] = field(default_factory=list)
    malformed_calls: list[ToolCallRequest] = field(default_factory=list)
    duplicate_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[ToolResultPatch] = field(default_factory=list)
    decision: LoopGuardDecision = field(
        default_factory=lambda: LoopGuardDecision(LoopGuardAction.CONTINUE)
    )


_MAX_UNPRODUCTIVE_TOOL_ROUNDS = 2
_MAX_EXEC_TIMEOUT_ROUNDS = 3
_MAX_EMPTY_STOP_RETRIES = 1
_TOOL_RECOVERY_MAX_FAILURES = 5
_TOOL_RECOVERY_RESULT_MAX_CHARS = 1_200
_TOOL_RECOVERY_ARGUMENTS_MAX_CHARS = 600
_MAX_ITERATIONS_FINALIZATION_PROMPT = """CRITICAL - MAXIMUM ITERATIONS REACHED

The normal tool-call iteration budget for this task has been reached.
Tools are disabled for this finalization pass.
Respond with text only.

STRICT REQUIREMENTS:
1. Do NOT make any tool calls.
2. Provide a concise text response summarizing work done so far.
3. Clearly state any remaining work that could not be completed.
4. Recommend the next best action.

Any attempt to use tools is a critical violation. Respond with text ONLY."""
_VISIBLE_ANSWER_FINALIZATION_PROMPT = """VISIBLE ANSWER REQUIRED

The previous model response exhausted its output token budget without producing visible answer text.
Tools are disabled for this recovery pass. Respond with concise visible text only.

STRICT REQUIREMENTS:
1. Do NOT make any tool calls.
2. Do NOT continue hidden reasoning.
3. Provide the best answer possible from the context already available.
4. If the task is incomplete, clearly state what remains and what should happen next.

Respond with text ONLY."""
_OUTPUT_EXHAUSTED_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}
_TOOL_ERROR_RECOVERY_PROMPT = """TOOL ERROR RECOVERY

The latest tool round failed. The tool results below may be recoverable error
signals, not a reason to abandon the user's task.

Reason: {reason}
Failed tool call(s):
{failures}

Continue the user's task. Before calling tools again, choose a corrected
approach based on the exact error text above. Do not repeat a failed call
unchanged. If a write failed, use edit for existing files when possible; for an
intentional full replacement, pass overwrite=true, and for large writes pass an
end_marker and make content end with that marker, or split the change into
smaller edits."""


def rejected_arguments_summary(tool_call: ToolCallRequest) -> str:
    """Serialize rejected non-object arguments without echoing large payloads."""

    return json.dumps(
        {
            "_rejected_malformed_arguments": True,
            "original_type": type(tool_call.arguments).__name__,
        },
        ensure_ascii=False,
    )


def tool_result_failed(result: str | None) -> bool:
    """Whether a tool result should count as an unproductive outcome."""

    text = (result or "").lstrip().lower()
    return text.startswith("error") or text.startswith("skipped duplicate call")


def _exec_command_timed_out(node: Any) -> bool:
    if getattr(node, "tool_name", None) != "exec":
        return False
    text = (getattr(node, "result", None) or "").lstrip().lower()
    return text.startswith("error: command timed out after")


def _partition_duplicate_tool_calls(
    messages: list[dict],
    tool_calls: list[ToolCallRequest],
) -> tuple[list[ToolCallRequest], list[ToolCallRequest]]:
    seen = collect_tool_call_fingerprints(messages)
    executable: list[ToolCallRequest] = []
    duplicates: list[ToolCallRequest] = []
    batch_seen: set[str] = set()
    for tool_call in tool_calls:
        fingerprint = tool_call_fingerprint(tool_call.name, tool_call.arguments)
        if fingerprint in seen or fingerprint in batch_seen:
            duplicates.append(tool_call)
            continue
        batch_seen.add(fingerprint)
        executable.append(tool_call)
    return executable, duplicates


def _format_malformed_arguments_error(tool_call: ToolCallRequest) -> str:
    try:
        raw = json.dumps(tool_call.arguments, ensure_ascii=False, default=str)
    except Exception:
        raw = repr(tool_call.arguments)
    if len(raw) > 500:
        raw = raw[:500] + "..."
    return (
        f"Error: arguments for tool '{tool_call.name}' must be a JSON object, "
        f"but received {type(tool_call.arguments).__name__}: {raw}. Retry with a "
        "single JSON object whose keys match the tool schema."
    )


def _off_track_message(reason: str) -> str:
    return (
        "I stopped the agent loop because it appears to be off track: "
        f"{reason}. I did not continue automatically to avoid spending more "
        "iterations on a loop. Please review the last tool result or provide "
        "narrower instructions."
    )


def _finalization_exhausted_message(finalization_budget: int) -> str:
    return (
        "I stopped because the model repeatedly exhausted its output budget "
        "without producing a visible final answer. No final artifact was produced. "
        f"The text-only recovery budget was {finalization_budget} attempt(s). "
        "Increase max_tokens or ask for a smaller first step, then retry."
    )


def _preview(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _json_preview(value: Any, *, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    return _preview(text, limit=limit)


def _indent(text: str) -> str:
    return "\n   ".join(text.splitlines()) if text else "(empty)"


def _tool_error_recovery_prompt_message(
    reason: str,
    executed_nodes: list[Any],
) -> dict[str, str]:
    failures: list[str] = []
    for index, node in enumerate(executed_nodes[:_TOOL_RECOVERY_MAX_FAILURES], start=1):
        tool_name = str(getattr(node, "tool_name", "tool"))
        arguments = _json_preview(
            getattr(node, "arguments", {}),
            limit=_TOOL_RECOVERY_ARGUMENTS_MAX_CHARS,
        )
        result = _preview(
            getattr(node, "result", None) or getattr(node, "error", None),
            limit=_TOOL_RECOVERY_RESULT_MAX_CHARS,
        )
        failures.append(
            f"{index}. {tool_name} arguments: {arguments}\n"
            f"   result: {_indent(result)}"
        )
    if len(executed_nodes) > _TOOL_RECOVERY_MAX_FAILURES:
        failures.append(
            f"... {len(executed_nodes) - _TOOL_RECOVERY_MAX_FAILURES} more failed "
            "tool call(s) omitted."
        )
    if not failures:
        failures.append("1. No structured tool result was available.")
    return {
        "role": "system",
        "content": _TOOL_ERROR_RECOVERY_PROMPT.format(
            reason=reason,
            failures="\n".join(failures),
        ),
    }


class AgentLoopGuard:
    """Stateful, deterministic supervisor for recoverable loop failures."""

    def __init__(
        self,
        *,
        max_iterations: int,
        max_auto_continue_iterations: int,
        max_finalization_iterations: int,
        max_unproductive_tool_rounds: int = _MAX_UNPRODUCTIVE_TOOL_ROUNDS,
        max_exec_timeout_rounds: int = _MAX_EXEC_TIMEOUT_ROUNDS,
        max_empty_stop_retries: int = _MAX_EMPTY_STOP_RETRIES,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_auto_continue_iterations = max_auto_continue_iterations
        self.max_finalization_iterations = max_finalization_iterations
        self.base_budget = max(0, max_iterations)
        self.auto_continue_remaining = max(0, max_auto_continue_iterations)
        self.finalization_budget = max(0, max_finalization_iterations)
        self.iteration_limit = self.base_budget
        self.max_unproductive_tool_rounds = max(0, max_unproductive_tool_rounds)
        self.max_exec_timeout_rounds = max(0, max_exec_timeout_rounds)
        self.max_empty_stop_retries = max(0, max_empty_stop_retries)
        self.unproductive_tool_rounds = 0
        self.exec_timeout_rounds = 0
        self.empty_stop_retries = 0

    def finalization_prompt_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                f"{_MAX_ITERATIONS_FINALIZATION_PROMPT}\n\n"
                f"Configured normal iteration budget: {self.max_iterations}."
            ),
        }

    def visible_answer_prompt_message(self) -> dict[str, str]:
        return {"role": "user", "content": _VISIBLE_ANSWER_FINALIZATION_PROMPT}

    def final_message_for_exhausted_loop(self) -> str:
        if self.finalization_budget > 0:
            return (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                f"plus the automatic continuation budget ({self.max_auto_continue_iterations}) "
                f"and could not produce a final text response within the finalization budget "
                f"({self.max_finalization_iterations}). Try breaking the task into smaller steps."
            )
        return (
            f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
            f"plus the automatic continuation budget ({self.max_auto_continue_iterations}) "
            "without completing the task. Try breaking the task into smaller steps."
        )

    def finalization_exhausted_message(self) -> str:
        return _finalization_exhausted_message(self.max_finalization_iterations)

    def off_track_decision(self, reason: str) -> LoopGuardDecision:
        return LoopGuardDecision(
            LoopGuardAction.STOP_OFF_TRACK,
            reason=reason,
            final_content=_off_track_message(reason),
        )

    def record_productive_tool_round(self) -> None:
        self.unproductive_tool_rounds = 0
        self.exec_timeout_rounds = 0

    def handle_unproductive_tool_round(
        self,
        reason: str,
        *,
        immediate: bool = False,
    ) -> LoopGuardDecision:
        self.unproductive_tool_rounds += 1
        if immediate or self.unproductive_tool_rounds >= self.max_unproductive_tool_rounds:
            return self.off_track_decision(reason)
        logger.info(
            "Unproductive tool round ({}/{}): {}",
            self.unproductive_tool_rounds,
            self.max_unproductive_tool_rounds,
            reason,
        )
        return LoopGuardDecision(LoopGuardAction.RETURN_TO_MODEL, reason=reason)

    def handle_exec_timeout_round(
        self,
        executed_nodes: list[Any] | None = None,
    ) -> LoopGuardDecision:
        reason = (
            "the latest exec command timed out; if it was starting a long-running server, "
            "retry with a truly detached command or verify the server with a short command"
        )
        self.exec_timeout_rounds += 1
        if self.exec_timeout_rounds >= self.max_exec_timeout_rounds:
            return self.off_track_decision("exec commands repeatedly timed out")
        logger.info(
            "Recoverable exec timeout round ({}/{}): {}",
            self.exec_timeout_rounds,
            self.max_exec_timeout_rounds,
            reason,
        )
        return LoopGuardDecision(
            LoopGuardAction.RETURN_TO_MODEL,
            reason=reason,
            prompt_message=_tool_error_recovery_prompt_message(
                reason,
                executed_nodes or [],
            ),
            progress_message="Tool error recovery prompt added; continuing.",
        )

    def handle_recoverable_tool_error_round(
        self,
        reason: str,
        executed_nodes: list[Any],
    ) -> LoopGuardDecision:
        self.unproductive_tool_rounds += 1
        logger.info(
            "Recoverable tool error round ({}): {}",
            self.unproductive_tool_rounds,
            reason,
        )
        return LoopGuardDecision(
            LoopGuardAction.RETURN_TO_MODEL,
            reason=reason,
            prompt_message=_tool_error_recovery_prompt_message(reason, executed_nodes),
            progress_message="Tool error recovery prompt added; continuing.",
        )

    def handle_malformed_tool_calls(
        self,
        tool_calls: list[ToolCallRequest],
    ) -> ToolCallGuardResult:
        valid: list[ToolCallRequest] = []
        malformed: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            if isinstance(tool_call.arguments, dict):
                valid.append(tool_call)
            else:
                malformed.append(tool_call)

        tool_results = [
            ToolResultPatch(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                content=_format_malformed_arguments_error(tool_call),
            )
            for tool_call in malformed
        ]
        decision = LoopGuardDecision(LoopGuardAction.CONTINUE)
        if malformed and not valid:
            decision = self.handle_unproductive_tool_round(
                "the model only supplied malformed tool arguments"
            )
        return ToolCallGuardResult(
            executable_calls=valid,
            malformed_calls=malformed,
            tool_results=tool_results,
            decision=decision,
        )

    def handle_duplicate_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list[ToolCallRequest],
    ) -> ToolCallGuardResult:
        executable, duplicates = _partition_duplicate_tool_calls(messages, tool_calls)
        tool_results = [
            ToolResultPatch(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                content=duplicate_tool_result(tool_call.name),
            )
            for tool_call in duplicates
        ]
        decision = LoopGuardDecision(LoopGuardAction.CONTINUE)
        if duplicates and not executable:
            decision = self.handle_unproductive_tool_round(
                "the model repeated tool calls that already ran with identical arguments"
            )
        return ToolCallGuardResult(
            executable_calls=executable,
            duplicate_calls=duplicates,
            tool_results=tool_results,
            decision=decision,
        )

    def handle_tool_results(self, executed_nodes: list[Any]) -> LoopGuardDecision:
        if executed_nodes and all(tool_result_failed(node.result) for node in executed_nodes):
            if all(_exec_command_timed_out(node) for node in executed_nodes):
                return self.handle_exec_timeout_round(executed_nodes)
            return self.handle_recoverable_tool_error_round(
                "all tool calls in the latest round failed or returned errors",
                executed_nodes,
            )
        self.record_productive_tool_round()
        return LoopGuardDecision(LoopGuardAction.CONTINUE)

    def handle_iteration_budget_reached(self) -> LoopGuardDecision:
        if self.auto_continue_remaining > 0:
            grant_size = self.base_budget if self.base_budget > 0 else 1
            grant = min(grant_size, self.auto_continue_remaining)
            self.iteration_limit += grant
            self.auto_continue_remaining -= grant
            logger.info(
                "Iteration budget reached; automatically continuing with {} more iteration(s), "
                "{} auto-continue iteration(s) remaining",
                grant,
                self.auto_continue_remaining,
            )
            return LoopGuardDecision(
                LoopGuardAction.EXTEND_BUDGET,
                reason="iteration budget reached",
                progress_message=(
                    f"Iteration budget reached; automatically continuing with {grant} more step(s)."
                ),
                budget_grant=grant,
            )
        if self.finalization_budget > 0:
            return LoopGuardDecision(
                LoopGuardAction.FINALIZE,
                reason="iteration budgets exhausted",
                prompt_message=self.finalization_prompt_message(),
            )
        return LoopGuardDecision(
            LoopGuardAction.FINAL_RESPONSE,
            reason="iteration budgets exhausted",
            final_content=self.final_message_for_exhausted_loop(),
        )

    def handle_finalization_tool_call_violation(self) -> LoopGuardDecision:
        return self.off_track_decision(
            "the model attempted to call tools after tools were disabled for finalization"
        )

    def handle_empty_or_exhausted_response(
        self,
        *,
        finish_reason: str,
        finalizing: bool,
        finalization_iteration: int,
    ) -> LoopGuardDecision:
        if finalizing and finish_reason in _OUTPUT_EXHAUSTED_FINISH_REASONS:
            logger.warning(
                "Finalization output budget exhausted (attempt {}/{})",
                finalization_iteration,
                self.finalization_budget,
            )
            if finalization_iteration >= self.finalization_budget:
                return LoopGuardDecision(
                    LoopGuardAction.FINAL_RESPONSE,
                    reason="finalization output budget exhausted",
                    final_content=self.finalization_exhausted_message(),
                )
            return LoopGuardDecision(
                LoopGuardAction.CONTINUE,
                reason="finalization output budget exhausted",
            )

        if (
            finish_reason in _OUTPUT_EXHAUSTED_FINISH_REASONS
            and not finalizing
            and self.finalization_budget > 0
        ):
            logger.warning(
                "LLM exhausted output budget without visible answer; entering finalization"
            )
            return LoopGuardDecision(
                LoopGuardAction.FINALIZE,
                reason="output budget exhausted without visible answer",
                prompt_message=self.visible_answer_prompt_message(),
                progress_message=(
                    "The model used its output budget without a visible answer; "
                    "asking for a concise text-only answer."
                ),
            )

        logger.warning(
            "LLM returned empty stop response (attempt {}/{})",
            self.empty_stop_retries + 1,
            self.max_empty_stop_retries + 1,
        )
        if self.empty_stop_retries < self.max_empty_stop_retries:
            self.empty_stop_retries += 1
            return LoopGuardDecision(LoopGuardAction.CONTINUE, reason="empty response")
        return LoopGuardDecision(
            LoopGuardAction.FINAL_RESPONSE,
            reason="empty response",
            final_content="Sorry, the AI model returned an empty response. Please try again.",
        )
