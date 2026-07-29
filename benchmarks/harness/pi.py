"""Pi coding-agent harness."""

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
                "text",
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
        final_content = outcome.stdout.strip()
        if not final_content:
            return _invalid("Pi text output contained no final assistant message")
        return {
            "status": "completed",
            "final_content": final_content,
            "payload_error": None,
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
