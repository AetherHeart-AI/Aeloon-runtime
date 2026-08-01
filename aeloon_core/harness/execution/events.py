"""Engine-neutral events emitted by Harness execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCallView:
    """A bounded, provider-independent view of one requested tool call."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecutionState(StrEnum):
    """Lifecycle state for one pi-core-managed tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ToolExecutionRecord:
    """Completed tool execution projected without exposing engine internals."""

    index: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    mode: str
    state: ToolExecutionState = ToolExecutionState.PENDING
    result: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponseView:
    """Small compatibility view consumed by existing progress observers."""

    content: str | None
    reasoning_content: str | None
    tool_calls: tuple[ToolCallView, ...]
    usage: dict[str, int]
    finish_reason: str | None


def tool_result_failed(result: str | None) -> bool:
    """Return whether an Aeloon tool result represents recoverable failure."""

    normalized = str(result or "").lstrip().lower()
    return normalized.startswith("error") or normalized.startswith(
        "skipped duplicate call"
    )


__all__ = [
    "ModelResponseView",
    "ToolCallView",
    "ToolExecutionRecord",
    "ToolExecutionState",
    "tool_result_failed",
]
