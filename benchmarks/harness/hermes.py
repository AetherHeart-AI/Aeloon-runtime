"""Hermes Agent CLI harness."""

from __future__ import annotations

import shutil
from typing import Any

from benchmarks.harness.base import (
    Harness,
    HarnessInvocation,
    HarnessRequest,
    ProcessOutcome,
    process_failure,
)


class HermesHarness(Harness):
    name = "hermes"

    def resolve_executable(self) -> str:
        executable = shutil.which("hermes")
        if executable is None:
            raise RuntimeError("Harness 'hermes' requires the 'hermes' CLI on PATH.")
        return executable

    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        return HarnessInvocation(
            command=[
                self.executable,
                "--yolo",
                "--model",
                self.model,
                "-z",
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
        final_content = outcome.stdout.strip()
        if not final_content:
            return {
                "status": "invalid_output",
                "final_content": None,
                "payload_error": "Hermes text output contained no final assistant message",
            }
        return {
            "status": "completed",
            "final_content": final_content,
            "payload_error": None,
        }
