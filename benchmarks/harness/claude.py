"""Claude Code CLI harness."""

from __future__ import annotations

import shutil
from typing import Any

from benchmarks.harness.base import (
    Harness,
    HarnessInvocation,
    HarnessRequest,
    ProcessOutcome,
    first_string,
    json_payloads,
    process_failure,
)


class ClaudeHarness(Harness):
    name = "claude"

    def resolve_executable(self) -> str:
        executable = shutil.which("claude")
        if executable is None:
            raise RuntimeError("Harness 'claude' requires the 'claude' CLI on PATH.")
        return executable

    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        return HarnessInvocation(
            command=[
                self.executable,
                "--print",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--dangerously-skip-permissions",
                request.prompt,
            ],
            cwd=request.workspace,
            prompt_argument=True,
        )

    def interpret(self, outcome: ProcessOutcome) -> dict[str, Any]:
        failure = process_failure(outcome)
        if failure is not None:
            return {
                "status": "timeout" if outcome.timed_out else "process_error",
                "final_content": None,
                "payload_error": failure,
            }
        payloads, error = json_payloads(outcome.stdout)
        if error is not None:
            return {
                "status": "invalid_output",
                "final_content": None,
                "payload_error": error,
            }
        payload = payloads[-1]
        success = payload.get("type") == "result" and not bool(payload.get("is_error"))
        subtype = payload.get("subtype")
        if isinstance(subtype, str) and subtype not in {"success", "completed"}:
            success = False
        return {
            "status": "completed" if success else "agent_error",
            "session_id": first_string(payload, "session_id", "sessionId"),
            "duration_ms": payload.get("duration_ms"),
            "final_content": (
                payload.get("result") if isinstance(payload.get("result"), str) else None
            ),
            "usage": (payload.get("usage") if isinstance(payload.get("usage"), dict) else {}),
            "models": (
                payload.get("modelUsage") if isinstance(payload.get("modelUsage"), dict) else {}
            ),
            "cost_usd": (
                payload.get("total_cost_usd")
                if isinstance(payload.get("total_cost_usd"), int | float)
                else None
            ),
            "payload_error": (
                None if success else str(subtype or "Claude returned an error result")
            ),
        }
