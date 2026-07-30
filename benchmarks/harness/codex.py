"""Codex CLI harness."""

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


class CodexHarness(Harness):
    name = "codex"

    def resolve_executable(self) -> str:
        executable = shutil.which("codex")
        if executable is None:
            raise RuntimeError("Harness 'codex' requires the 'codex' CLI on PATH.")
        return executable

    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        return HarnessInvocation(
            command=[
                self.executable,
                "exec",
                "--sandbox",
                "workspace-write",
                "--ephemeral",
                "--json",
                "--model",
                self.model,
                "-",
            ],
            cwd=request.workspace,
            input_text=request.prompt,
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

        final_content: str | None = None
        usage: dict[str, Any] = {}
        thread_id: str | None = None
        completed = False
        failure = None
        for payload in payloads:
            event_type = payload.get("type")
            if event_type == "thread.started":
                thread_id = first_string(payload, "thread_id", "threadId", "id")
            elif event_type == "item.completed":
                item = payload.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    final_content = first_string(item, "text", "content")
            elif event_type == "turn.completed":
                completed = True
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
            elif event_type in {"turn.failed", "error"}:
                failure = first_string(payload, "message", "error") or str(payload)
        return {
            "status": (
                "agent_error" if failure else ("completed" if completed else "invalid_output")
            ),
            "session_id": thread_id,
            "final_content": final_content,
            "usage": usage,
            "payload_error": (
                failure or (None if completed else "Codex JSON stream did not complete")
            ),
        }
