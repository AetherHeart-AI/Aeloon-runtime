"""OpenClaw CLI harness."""

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


class OpenClawHarness(Harness):
    name = "openclaw"

    def resolve_executable(self) -> str:
        executable = shutil.which("openclaw")
        if executable is None:
            raise RuntimeError("Harness 'openclaw' requires the 'openclaw' CLI on PATH.")
        return executable

    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        return HarnessInvocation(
            command=[
                self.executable,
                "agent",
                "exec",
                "--message-file",
                "-",
                "--cwd",
                str(request.workspace),
                "--model",
                self.model,
                "--timeout",
                "0",
                "--json",
            ],
            cwd=request.workspace,
            input_text=request.prompt,
        )

    def interpret(self, outcome: ProcessOutcome) -> dict[str, Any]:
        if outcome.timed_out:
            return {
                "status": "timeout",
                "final_content": None,
                "payload_error": "agent process timed out",
            }
        payloads, error = json_payloads(outcome.stdout)
        if error is not None:
            failure = process_failure(outcome)
            return {
                "status": "process_error" if failure is not None else "invalid_output",
                "final_content": None,
                "payload_error": failure or error,
            }

        payload = payloads[-1]
        final_content = first_string(payload, "final")
        success = payload.get("ok") is True and payload.get("status") == "ok"
        failure = process_failure(outcome)
        if success and failure is not None:
            return {
                "status": "process_error",
                "final_content": None,
                "payload_error": failure,
            }
        if success and not final_content:
            return {
                "status": "invalid_output",
                "final_content": None,
                "payload_error": "OpenClaw JSON envelope contained no final assistant message",
            }

        tool_summary = payload.get("toolSummary")
        tools_used = (
            tool_summary.get("tools", [])
            if isinstance(tool_summary, dict)
            and isinstance(tool_summary.get("tools", []), list)
            else []
        )
        models = {
            key: payload[key]
            for key in ("model", "provider")
            if isinstance(payload.get(key), str)
        }
        payload_error = None if success else _payload_error(payload)
        return {
            "status": (
                "completed"
                if success
                else ("timeout" if payload.get("status") == "timeout" else "agent_error")
            ),
            "session_id": first_string(payload, "sessionId", "session_id"),
            "duration_ms": _number(payload, "durationMs", "duration_ms"),
            "final_content": final_content,
            "tools_used": tools_used,
            "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
            "models": models,
            "cost_usd": _number(payload, "costUsd", "cost_usd"),
            "payload_error": payload_error,
        }


def _payload_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = first_string(error, "message")
        kind = first_string(error, "kind")
        if message and kind:
            return f"{kind}: {message}"
        if message or kind:
            return str(message or kind)
    return str(payload.get("status") or "OpenClaw returned an error result")


def _number(payload: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    return None
