"""Typed Worker outputs interpreted only by the Aeloon control plane."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from aeloon_core.worker_sessions import WaitingRequest, WorkerReport, WorkerRunStatus

TerminalItem = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
]


class _TerminalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompleteWorkArgs(_TerminalArgs):
    summary: str = Field(min_length=1, max_length=8_000)
    artifacts: list[TerminalItem] = Field(default_factory=list, max_length=32)
    evidence: list[TerminalItem] = Field(default_factory=list, max_length=32)


class RequestMasterArgs(_TerminalArgs):
    summary: str = Field(min_length=1, max_length=8_000)
    question: str = Field(min_length=1, max_length=1_000)


def worker_terminal_result(
    output: CompleteWorkArgs | RequestMasterArgs,
) -> tuple[WorkerRunStatus, WorkerReport, WaitingRequest | None]:
    """Map a validated PydanticAI output into durable Aeloon records."""

    if isinstance(output, CompleteWorkArgs):
        return (
            WorkerRunStatus.COMPLETED,
            WorkerReport(
                summary=output.summary,
                artifacts=tuple(output.artifacts),
                evidence=tuple(output.evidence),
            ),
            None,
        )
    request = WaitingRequest(summary=output.summary, question=output.question)
    return (
        WorkerRunStatus.WAITING_FOR_CONTEXT,
        WorkerReport(summary=output.summary, unresolved=(output.question,)),
        request,
    )


__all__ = [
    "CompleteWorkArgs",
    "RequestMasterArgs",
    "worker_terminal_result",
]
