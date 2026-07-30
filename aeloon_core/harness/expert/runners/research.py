"""Built-in bounded research ExpertSkill runner."""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeloon_core.harness.capabilities import CapabilityUnavailable
from aeloon_core.harness.execution import accumulate_usage
from aeloon_core.harness.expert.base import (
    ExpertEvidence,
    ExpertResult,
    ExpertRunContext,
    ExpertRunRequest,
    StageOutcome,
    StageStatus,
)


class ResearchAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    question: str = Field(min_length=1, max_length=4_000)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str = Field(min_length=1, max_length=4_000)
    assignments: tuple[ResearchAssignment, ...] = Field(min_length=2, max_length=4)
    verification_focus: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def assignment_ids_are_unique(self) -> ResearchPlan:
        ids = [assignment.id for assignment in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("research assignment ids must be unique")
        return self


class ResearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=12_000)
    evidence: tuple[ExpertEvidence, ...] = Field(default=(), max_length=48)
    unresolved: tuple[str, ...] = Field(default=(), max_length=24)


class ResearchSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    final_content: str = Field(min_length=1, max_length=48_000)
    evidence: tuple[ExpertEvidence, ...] = Field(default=(), max_length=96)
    unresolved: tuple[str, ...] = Field(default=(), max_length=48)


class ResearchExpertRunner:
    """plan -> parallel explorers -> primary-source docs -> reduce."""

    async def run(
        self,
        request: ExpertRunRequest,
        context: ExpertRunContext,
    ) -> ExpertResult:
        usage: dict[str, int] = {}
        statuses: list[StageStatus] = []
        plan_outcome = await context.stages.run(
            stage_id="plan",
            task=request.task,
            instructions=(
                "Create a research plan with two to four genuinely independent "
                "assignments. Define what the final verification pass must establish."
            ),
            output_type=ResearchPlan,
            capabilities=(),
            model_tier="fast",
        )
        _record(plan_outcome, usage, statuses)
        if not isinstance(plan_outcome.output, ResearchPlan):
            return _blocked(
                "The research planner did not produce a valid plan.",
                usage,
                statuses,
                plan_outcome.failure,
            )

        semaphore = asyncio.Semaphore(context.config.experts.max_concurrency)

        async def explore(assignment: ResearchAssignment) -> StageOutcome:
            async with semaphore:
                return await context.stages.run(
                    stage_id=f"explore-{assignment.id}",
                    task=assignment.question,
                    instructions=(
                        "Explore this assignment independently. Use web_search and "
                        "get_page. Return direct source URLs and distinguish observed "
                        "facts from inference. Prefer authoritative sources."
                    ),
                    output_type=ResearchReport,
                    capabilities=("web_search",),
                    model_tier="fast",
                )

        try:
            raw_explorers = await asyncio.gather(
                *(explore(item) for item in plan_outcome.output.assignments),
                return_exceptions=True,
            )
        except CapabilityUnavailable as exc:
            return _blocked(str(exc), usage, statuses, str(exc))

        explorer_reports: list[ResearchReport] = []
        explorer_failures: list[str] = []
        for assignment, result in zip(
            plan_outcome.output.assignments,
            raw_explorers,
            strict=True,
        ):
            if isinstance(result, CapabilityUnavailable):
                return _blocked(str(result), usage, statuses, str(result))
            if isinstance(result, BaseException):
                explorer_failures.append(
                    f"explorer {assignment.id} failed: {type(result).__name__}: {result}"
                )
                statuses.append(
                    StageStatus(
                        stage_id=f"explore-{assignment.id}",
                        status="failed",
                        summary=str(result),
                    )
                )
                continue
            _record(result, usage, statuses)
            if isinstance(result.output, ResearchReport):
                explorer_reports.append(result.output)
            else:
                explorer_failures.append(f"{result.stage_id} did not produce a valid report")

        if not explorer_reports:
            return _blocked(
                "All research explorers failed.",
                usage,
                statuses,
                *explorer_failures,
            )

        reports_json = json.dumps(
            [report.model_dump(mode="json") for report in explorer_reports],
            ensure_ascii=False,
        )
        docs_failure: str | None = None
        try:
            docs_outcome = await context.stages.run(
                stage_id="docs",
                task=(
                    f"Original objective:\n{request.task}\n\n"
                    f"Planner verification focus:\n"
                    f"{json.dumps(plan_outcome.output.verification_focus, ensure_ascii=False)}"
                    f"\n\nExplorer reports:\n{reports_json}"
                ),
                instructions=(
                    "Independently verify the important claims against official "
                    "documentation, standards, papers, repositories, or other primary "
                    "sources. Correct weak claims. Return direct URLs."
                ),
                output_type=ResearchReport,
                capabilities=("web_search",),
                model_tier="strong",
            )
        except CapabilityUnavailable as exc:
            return _blocked(str(exc), usage, statuses, str(exc))
        except Exception as exc:
            docs_failure = f"primary-source verification failed: {type(exc).__name__}: {exc}"
            statuses.append(StageStatus(stage_id="docs", status="failed", summary=docs_failure))
            docs_report = None
        else:
            _record(docs_outcome, usage, statuses)
            docs_report = (
                docs_outcome.output if isinstance(docs_outcome.output, ResearchReport) else None
            )
            if docs_report is None:
                docs_failure = (
                    docs_outcome.failure
                    or "primary-source verification did not produce a valid report"
                )

        reducer_input = {
            "objective": request.task,
            "plan": plan_outcome.output.model_dump(mode="json"),
            "explorers": [report.model_dump(mode="json") for report in explorer_reports],
            "primary_source_verification": (
                docs_report.model_dump(mode="json") if docs_report else None
            ),
            "stage_failures": [
                *explorer_failures,
                *([docs_failure] if docs_failure else []),
            ],
        }
        try:
            reduce_outcome = await context.stages.run(
                stage_id="reduce",
                task=json.dumps(reducer_input, ensure_ascii=False),
                instructions=(
                    "Synthesize a direct answer to the original objective. Cite direct "
                    "URLs from the supplied evidence, clearly label inference and "
                    "uncertainty, and retain unresolved conflicts. Never invent a source."
                ),
                output_type=ResearchSynthesis,
                capabilities=(),
                model_tier="strong",
            )
        except Exception as exc:
            reduce_failure = f"final reducer failed: {type(exc).__name__}: {exc}"
            statuses.append(StageStatus(stage_id="reduce", status="failed", summary=reduce_failure))
            fallback = docs_report or explorer_reports[0]
            return ExpertResult(
                status="partial",
                final_content=fallback.summary,
                evidence=fallback.evidence,
                unresolved=tuple(
                    [
                        *fallback.unresolved,
                        *explorer_failures,
                        *([docs_failure] if docs_failure else []),
                        reduce_failure,
                    ]
                ),
                stage_statuses=tuple(statuses),
                usage=usage,
            )
        _record(reduce_outcome, usage, statuses)
        if not isinstance(reduce_outcome.output, ResearchSynthesis):
            fallback = docs_report or explorer_reports[0]
            return ExpertResult(
                status="partial",
                final_content=fallback.summary,
                evidence=fallback.evidence,
                unresolved=tuple(
                    [
                        *fallback.unresolved,
                        *explorer_failures,
                        *([docs_failure] if docs_failure else []),
                        "The final reducer failed; returning the strongest available report.",
                    ]
                ),
                stage_statuses=tuple(statuses),
                usage=usage,
            )

        synthesis = reduce_outcome.output
        unresolved = tuple(
            [
                *synthesis.unresolved,
                *explorer_failures,
                *([docs_failure] if docs_failure else []),
            ]
        )
        return ExpertResult(
            status="partial" if unresolved or docs_report is None else "completed",
            final_content=synthesis.final_content,
            evidence=synthesis.evidence,
            unresolved=unresolved,
            stage_statuses=tuple(statuses),
            usage=usage,
        )


def _record(
    outcome: StageOutcome,
    usage: dict[str, int],
    statuses: list[StageStatus],
) -> None:
    accumulate_usage(usage, outcome.usage)
    statuses.append(
        StageStatus(
            stage_id=outcome.stage_id,
            status=outcome.status,
            summary=outcome.failure or _summary(outcome.output),
            usage=outcome.usage,
        )
    )


def _summary(output: object) -> str:
    for field in ("summary", "final_content", "objective"):
        value = getattr(output, field, None)
        if value:
            return str(value)[:1_000]
    return "Stage completed."


def _blocked(
    message: str,
    usage: dict[str, int],
    statuses: list[StageStatus],
    *details: str | None,
) -> ExpertResult:
    return ExpertResult(
        status="blocked",
        final_content=message,
        unresolved=tuple(detail for detail in details if detail),
        stage_statuses=tuple(statuses),
        usage=usage,
    )


__all__ = ["ResearchExpertRunner"]
