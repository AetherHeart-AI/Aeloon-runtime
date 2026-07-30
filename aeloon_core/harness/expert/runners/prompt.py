"""Generic single-agent runner for trusted custom ExpertSkill manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.harness.expert.base import (
    ExpertEvidence,
    ExpertFinding,
    ExpertResult,
    ExpertRunContext,
    ExpertRunRequest,
    StageStatus,
)


class PromptExpertReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "partial", "blocked"]
    final_content: str = Field(min_length=1, max_length=48_000)
    artifacts: tuple[str, ...] = Field(default=(), max_length=64)
    evidence: tuple[ExpertEvidence, ...] = Field(default=(), max_length=128)
    findings: tuple[ExpertFinding, ...] = Field(default=(), max_length=128)
    unresolved: tuple[str, ...] = Field(default=(), max_length=64)


class PromptExpertRunner:
    """Execute one ExpertSkill as a single isolated structured-output agent."""

    async def run(
        self,
        request: ExpertRunRequest,
        context: ExpertRunContext,
    ) -> ExpertResult:
        outcome = await context.stages.run(
            stage_id="run",
            task=request.task,
            instructions=(
                "Complete the requested outcome using the ExpertSkill instructions. "
                "Return an honest completed, partial, or blocked structured report."
            ),
            output_type=PromptExpertReport,
            capabilities=context.expert.capabilities,
            model_tier=context.expert.model_tier,
        )
        if not isinstance(outcome.output, PromptExpertReport):
            failure = outcome.failure or "custom ExpertSkill did not return a valid report"
            return ExpertResult(
                status="blocked",
                final_content=failure,
                unresolved=(failure,),
                stage_statuses=(
                    StageStatus(
                        stage_id=outcome.stage_id,
                        status=outcome.status,
                        summary=failure,
                        usage=outcome.usage,
                    ),
                ),
                usage=outcome.usage,
            )
        report = outcome.output
        return ExpertResult(
            status=report.status,
            final_content=report.final_content,
            artifacts=report.artifacts,
            evidence=report.evidence,
            findings=report.findings,
            unresolved=report.unresolved,
            stage_statuses=(
                StageStatus(
                    stage_id=outcome.stage_id,
                    status=outcome.status,
                    summary=report.final_content[:1_000],
                    usage=outcome.usage,
                ),
            ),
            usage=outcome.usage,
        )


__all__ = ["PromptExpertRunner"]
