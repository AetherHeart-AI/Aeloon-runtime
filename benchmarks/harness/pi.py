"""Pi coding-agent harness."""

from __future__ import annotations

import shutil
from typing import Any

from benchmarks.harness.base import (
    Harness,
    HarnessInvocation,
    HarnessRequest,
    ProcessOutcome,
    content_text,
    first_string,
    json_payloads,
    process_failure,
)


class PiHarness(Harness):
    name = "pi"

    def resolve_executable(self) -> str:
        executable = shutil.which("pi")
        if executable is None:
            raise RuntimeError("Harness 'pi' requires the 'pi' CLI on PATH.")
        return executable

    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        return HarnessInvocation(
            command=[
                self.executable,
                "--print",
                "--mode",
                "json",
                "--no-session",
                "--approve",
                request.prompt,
            ],
            cwd=request.workspace,
            prompt_argument=True,
        )

    def interpret(self, outcome: ProcessOutcome) -> dict[str, Any]:
        failure = process_failure(outcome)
        if failure is not None:
            return _failed(outcome, failure)
        payloads, error = json_payloads(outcome.stdout)
        if error is not None:
            return _invalid(error)

        assistant: dict[str, Any] | None = None
        session_id: str | None = None
        for payload in payloads:
            session_id = session_id or first_string(payload, "id", "session_id", "sessionId")
            if payload.get("type") == "message_end":
                message = payload.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    assistant = message
            if payload.get("type") == "agent_end" and assistant is None:
                messages = payload.get("messages")
                if isinstance(messages, list):
                    assistant = next(
                        (
                            message
                            for message in reversed(messages)
                            if isinstance(message, dict) and message.get("role") == "assistant"
                        ),
                        None,
                    )
        if assistant is None:
            return _invalid("Pi JSON stream contained no final assistant message")

        stop_reason = assistant.get("stopReason") or assistant.get("stop_reason")
        usage = dict(assistant["usage"]) if isinstance(assistant.get("usage"), dict) else {}
        if isinstance(usage.get("input"), int | float):
            usage["input_tokens"] = usage["input"]
        if isinstance(usage.get("output"), int | float):
            usage["output_tokens"] = usage["output"]
        return {
            "status": ("agent_error" if stop_reason in {"error", "aborted"} else "completed"),
            "session_id": session_id,
            "final_content": content_text(assistant.get("content")),
            "usage": usage,
            "cost_usd": (
                usage.get("cost", {}).get("total") if isinstance(usage.get("cost"), dict) else None
            ),
            "models": {
                key: assistant[key]
                for key in ("provider", "model")
                if isinstance(assistant.get(key), str)
            },
            "payload_error": (
                str(assistant.get("errorMessage") or stop_reason)
                if stop_reason in {"error", "aborted"}
                else None
            ),
        }


def _failed(outcome: ProcessOutcome, error: str) -> dict[str, Any]:
    return {
        "status": "timeout" if outcome.timed_out else "process_error",
        "final_content": None,
        "payload_error": error,
    }


def _invalid(error: str) -> dict[str, Any]:
    return {
        "status": "invalid_output",
        "final_content": None,
        "payload_error": error,
    }
