"""Common contract and subprocess handling for coding harnesses."""

from __future__ import annotations

import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_CAPTURE_CHARS = 20_000


@dataclass(frozen=True)
class ProcessOutcome:
    """Bounded result of one child process."""

    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


@dataclass(frozen=True)
class HarnessInvocation:
    """One non-interactive harness invocation."""

    command: list[str]
    cwd: Path
    input_text: str | None = None
    prompt_argument: bool = False

    @property
    def display_command(self) -> list[str]:
        if not self.prompt_argument:
            return self.command
        return [*self.command[:-1], "<prompt>"]


@dataclass(frozen=True)
class HarnessRequest:
    """Benchmark-neutral input to one coding harness call."""

    prompt: str
    workspace: Path
    session_dir: Path
    project_root: Path
    timeout: float = 900.0
    config_path: Path | None = None


@dataclass(frozen=True)
class HarnessResult:
    """Normalized result shared by benchmark adapters."""

    harness: str
    version: str | None
    invocation: HarnessInvocation
    process: ProcessOutcome
    status: str
    final_content: str | None
    session_id: str | None = None
    turn_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    tools_used: list[Any] = field(default_factory=list)
    transitions: list[Any] = field(default_factory=list)
    cost_usd: int | float | None = None
    payload_error: str | None = None
    duration_ms: int | float | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the stable on-disk representation."""

        return {
            "harness": self.harness,
            "version": self.version,
            "status": self.status,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "duration_ms": self.duration_ms,
            "final_content": self.final_content,
            "tools_used": self.tools_used,
            "usage": self.usage,
            "transitions": self.transitions,
            "models": self.models,
            "cost_usd": self.cost_usd,
            "payload_error": self.payload_error,
            "command": self.invocation.display_command,
            "wall_time_ms": self.process.duration_ms,
            "returncode": self.process.returncode,
            "timed_out": self.process.timed_out,
            "stdout": bounded(self.process.stdout) if self.payload_error else None,
            "stderr": bounded(self.process.stderr) if self.process.stderr else None,
        }


class Harness(ABC):
    """Base for a non-interactive coding-agent CLI."""

    name: str

    def __init__(self, *, project_root: Path, model: str) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.model = model.strip()
        if not self.model:
            raise ValueError("model name must not be empty")
        self.executable = self.resolve_executable()
        self.version = self.resolve_version()

    @abstractmethod
    def resolve_executable(self) -> str:
        """Resolve the CLI executable or fail with an actionable error."""

    @abstractmethod
    def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
        """Build a non-interactive process invocation."""

    @abstractmethod
    def interpret(self, outcome: ProcessOutcome) -> dict[str, Any]:
        """Normalize harness-specific stdout into the shared result fields."""

    def resolve_version(self) -> str | None:
        outcome = run_process(
            [self.executable, "--version"],
            cwd=self.project_root,
            timeout=10.0,
        )
        if outcome.returncode != 0 or outcome.timed_out:
            return None
        value = outcome.stdout.strip() or outcome.stderr.strip()
        return bounded(value, limit=500) or None

    def run(self, request: HarnessRequest) -> HarnessResult:
        invocation = self.build_invocation(request)
        outcome = run_process(
            invocation.command,
            cwd=invocation.cwd,
            timeout=request.timeout,
            input_text=invocation.input_text,
        )
        fields = self.interpret(outcome)
        captured_outcome = ProcessOutcome(
            returncode=outcome.returncode,
            stdout=bounded(outcome.stdout),
            stderr=bounded(outcome.stderr),
            duration_ms=outcome.duration_ms,
            timed_out=outcome.timed_out,
        )
        return HarnessResult(
            harness=self.name,
            version=self.version,
            invocation=invocation,
            process=captured_outcome,
            status=str(fields.pop("status")),
            final_content=fields.pop("final_content", None),
            **fields,
        )


def run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
) -> ProcessOutcome:
    """Run a process without a shell and preserve timeout diagnostics."""

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return ProcessOutcome(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessOutcome(
            returncode=None,
            stdout=process_text(exc.stdout),
            stderr=process_text(exc.stderr),
            duration_ms=round((time.monotonic() - started) * 1000),
            timed_out=True,
        )


def process_failure(outcome: ProcessOutcome) -> str | None:
    if outcome.timed_out:
        return "agent process timed out"
    if outcome.returncode != 0:
        return f"agent process exited with code {outcome.returncode}"
    return None


def json_payloads(stdout: str) -> tuple[list[dict[str, Any]], str | None]:
    """Accept either one JSON object or a JSONL event stream."""

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                return [], f"agent JSONL was invalid at line {line_number}: {exc}"
            if not isinstance(value, dict):
                return [], f"agent JSONL line {line_number} was not an object"
            payloads.append(value)
        if not payloads:
            return [], "agent stdout contained no JSON objects"
        return payloads, None
    if not isinstance(payload, dict):
        return [], "agent JSON payload was not an object"
    return [payload], None


def first_string(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    return None


def content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return "\n".join(parts) or None


def bounded(value: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"[... {omitted} earlier characters omitted ...]\n{value[-limit:]}"


def process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value
