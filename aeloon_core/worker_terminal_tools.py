"""Worker-only explicit completion and Master-context terminal tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
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


@dataclass(frozen=True)
class WorkerTerminalSignal:
    status: WorkerRunStatus
    report: WorkerReport
    waiting_request: WaitingRequest | None = None


class CompleteWorkTool(Tool):
    name = "complete_work"
    description = (
        "Finish the WorkerRun after verifying the objective. Must be the only tool call "
        "in the model response."
    )
    args_model = CompleteWorkArgs
    concurrency_mode = "mutating"
    terminal = True

    def __init__(self, controller: WorkerTerminalController) -> None:
        self.controller = controller

    async def execute(
        self,
        summary: str,
        artifacts: list[str] | None = None,
        evidence: list[str] | None = None,
    ) -> str:
        self.controller.signal = WorkerTerminalSignal(
            status=WorkerRunStatus.COMPLETED,
            report=WorkerReport(
                summary=summary,
                artifacts=tuple(artifacts or ()),
                evidence=tuple(evidence or ()),
            ),
        )
        return "WorkerRun completed with a structured report."


class RequestMasterTool(Tool):
    name = "request_master"
    description = (
        "Pause because one specific answer from Master is required. Must be the only tool "
        "call in the model response."
    )
    args_model = RequestMasterArgs
    concurrency_mode = "mutating"
    terminal = True

    def __init__(self, controller: WorkerTerminalController) -> None:
        self.controller = controller

    async def execute(self, summary: str, question: str) -> str:
        request = WaitingRequest(summary=summary, question=question)
        self.controller.signal = WorkerTerminalSignal(
            status=WorkerRunStatus.WAITING_FOR_CONTEXT,
            report=WorkerReport(summary=summary, unresolved=(question,)),
            waiting_request=request,
        )
        return "WorkerRun paused with a structured request for Master."


class WorkerTerminalController:
    """Per-Run signal holder; never share this instance across WorkerRuns."""

    def __init__(self) -> None:
        self.signal: WorkerTerminalSignal | None = None
        self.complete_work = CompleteWorkTool(self)
        self.request_master = RequestMasterTool(self)

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(self.complete_work)
        registry.register(self.request_master)


__all__ = [
    "CompleteWorkArgs",
    "RequestMasterArgs",
    "WorkerTerminalController",
    "WorkerTerminalSignal",
]
