"""Aeloon Core harness."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from benchmarks.harness.base import (
    Harness,
    HarnessInvocation,
    HarnessRequest,
    ProcessOutcome,
    process_failure,
)


class AeloonHarness(Harness):
    name = "aeloon"

    def resolve_executable(self) -> str:
        return sys.executable

    def resolve_version(self) -> str | None:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        revision = completed.stdout.strip() if completed.returncode == 0 else None
        return f"aeloon-core@{revision}" if revision else "aeloon-core"

    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        command = [
            self.executable,
            "-m",
            "aeloon_core",
            "run",
            "--workspace",
            str(request.workspace),
            "--data-dir",
            str(request.session_dir),
            "--stdin",
            "--output",
            "json",
            "--model",
            self.model,
        ]
        if request.config_path is not None:
            command.extend(["--config", str(request.config_path.expanduser().resolve())])
        return HarnessInvocation(
            command=command,
            cwd=request.project_root,
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
        try:
            payload = json.loads(outcome.stdout)
        except json.JSONDecodeError as exc:
            return {
                "status": "invalid_output",
                "final_content": None,
                "payload_error": f"agent stdout was not JSON: {exc}",
            }
        if not isinstance(payload, dict):
            return {
                "status": "invalid_output",
                "final_content": None,
                "payload_error": "agent JSON payload was not an object",
            }
        return {
            "status": (
                "completed"
                if payload.get("status") in {"completed", "success"}
                else str(payload.get("status") or "agent_error")
            ),
            "session_id": _string(payload, "session_id"),
            "duration_ms": payload.get("duration_ms"),
            "final_content": _string(payload, "final_content"),
            "tools_used": _list(payload, "tools_used"),
            "usage": _dict(payload, "usage"),
            "model": _string(payload, "model"),
            "payload_error": None,
        }


def _string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []
