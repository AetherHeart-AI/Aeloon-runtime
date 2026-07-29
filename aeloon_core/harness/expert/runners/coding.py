"""Built-in bounded coding ExpertSkill runner."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.harness.execution import accumulate_usage
from aeloon_core.harness.expert.base import (
    ExpertEvidence,
    ExpertFinding,
    ExpertResult,
    ExpertRunContext,
    ExpertRunRequest,
    StageOutcome,
    StageStatus,
)


class CodingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=8_000)
    steps: tuple[str, ...] = Field(min_length=1, max_length=24)
    acceptance_checks: tuple[str, ...] = Field(min_length=1, max_length=24)


class BuildReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=12_000)
    artifacts: tuple[str, ...] = Field(default=(), max_length=64)
    evidence: tuple[ExpertEvidence, ...] = Field(default=(), max_length=64)
    unresolved: tuple[str, ...] = Field(default=(), max_length=32)


class ReviewReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=12_000)
    findings: tuple[ExpertFinding, ...] = Field(default=(), max_length=64)
    evidence: tuple[ExpertEvidence, ...] = Field(default=(), max_length=64)


class CodingExpertRunner:
    """plan -> build -> review -> optional one fix -> one re-review."""

    async def run(
        self,
        request: ExpertRunRequest,
        context: ExpertRunContext,
    ) -> ExpertResult:
        usage: dict[str, int] = {}
        statuses: list[StageStatus] = []
        plan = await context.stages.run(
            stage_id="plan",
            task=request.task,
            instructions=(
                "Inspect the repository read-only. Produce the smallest implementation "
                "plan consistent with project instructions and explicit acceptance checks. "
                "Do not edit files."
            ),
            output_type=CodingPlan,
            capabilities=("filesystem_read", "repo_context", "planning"),
            model_tier="fast",
        )
        _record(plan, usage, statuses)
        if not isinstance(plan.output, CodingPlan):
            return _blocked("The coding planner did not produce a valid plan.", usage, statuses)

        build = await context.stages.run(
            stage_id="build",
            task=_json_task(request.task, plan=plan.output),
            instructions=(
                "Implement the requested repository change. Preserve unrelated user "
                "changes, follow project instructions, run proportionate checks, and "
                "return exact artifact paths and verification evidence."
            ),
            output_type=BuildReport,
            capabilities=("filesystem", "shell", "repo_context", "planning"),
            model_tier="strong",
        )
        _record(build, usage, statuses)
        if not isinstance(build.output, BuildReport):
            return _blocked("The builder did not produce a valid report.", usage, statuses)

        try:
            review = await context.stages.run(
                stage_id="review",
                task=_json_task(
                    request.task,
                    plan=plan.output,
                    build=build.output,
                ),
                instructions=(
                    "Independently inspect the implementation read-only. Report only "
                    "concrete, actionable correctness, safety, regression, or "
                    "acceptance-test findings. Use an empty findings list when no "
                    "material issue remains. Do not edit."
                ),
                output_type=ReviewReport,
                capabilities=("filesystem_read", "repo_context"),
                model_tier="strong",
            )
        except Exception as exc:
            failure = f"Independent review failed: {type(exc).__name__}: {exc}"
            statuses.append(StageStatus(stage_id="review", status="failed", summary=failure))
            return ExpertResult(
                status="partial",
                final_content=build.output.summary,
                artifacts=build.output.artifacts,
                evidence=build.output.evidence,
                unresolved=tuple([*build.output.unresolved, failure]),
                stage_statuses=tuple(statuses),
                usage=usage,
            )
        _record(review, usage, statuses)
        if not isinstance(review.output, ReviewReport):
            return ExpertResult(
                status="partial",
                final_content=build.output.summary,
                artifacts=build.output.artifacts,
                evidence=build.output.evidence,
                unresolved=tuple([*build.output.unresolved, "Independent review failed."]),
                stage_statuses=tuple(statuses),
                usage=usage,
            )
        if not review.output.findings:
            unresolved = build.output.unresolved
            return ExpertResult(
                status="partial" if unresolved else "completed",
                final_content=build.output.summary,
                artifacts=build.output.artifacts,
                evidence=tuple([*build.output.evidence, *review.output.evidence]),
                unresolved=unresolved,
                stage_statuses=tuple(statuses),
                usage=usage,
            )

        try:
            fix = await context.stages.run(
                stage_id="fix",
                task=_json_task(
                    request.task,
                    plan=plan.output,
                    build=build.output,
                    review=review.output,
                ),
                instructions=(
                    "Address the supplied review findings in one bounded corrective pass. "
                    "Re-run relevant verification and report any unresolved item. Do not "
                    "expand scope beyond the original request and concrete findings."
                ),
                output_type=BuildReport,
                capabilities=("filesystem", "shell", "repo_context", "planning"),
                model_tier="strong",
            )
        except Exception as exc:
            failure = f"Corrective pass failed: {type(exc).__name__}: {exc}"
            statuses.append(StageStatus(stage_id="fix", status="failed", summary=failure))
            return ExpertResult(
                status="partial",
                final_content=build.output.summary,
                artifacts=build.output.artifacts,
                evidence=build.output.evidence,
                findings=review.output.findings,
                unresolved=tuple([*build.output.unresolved, failure]),
                stage_statuses=tuple(statuses),
                usage=usage,
            )
        _record(fix, usage, statuses)
        if not isinstance(fix.output, BuildReport):
            return ExpertResult(
                status="partial",
                final_content=build.output.summary,
                artifacts=build.output.artifacts,
                evidence=build.output.evidence,
                findings=review.output.findings,
                unresolved=tuple([*build.output.unresolved, "Corrective pass failed."]),
                stage_statuses=tuple(statuses),
                usage=usage,
            )

        try:
            rereview = await context.stages.run(
                stage_id="re-review",
                task=_json_task(
                    request.task,
                    plan=plan.output,
                    build=build.output,
                    initial_review=review.output,
                    fix=fix.output,
                ),
                instructions=(
                    "Re-review the corrected workspace read-only. Confirm whether each "
                    "original finding is resolved and return any remaining concrete "
                    "findings. This is the final review pass; do not edit."
                ),
                output_type=ReviewReport,
                capabilities=("filesystem_read", "repo_context"),
                model_tier="strong",
            )
        except Exception as exc:
            failure = f"Final re-review failed: {type(exc).__name__}: {exc}"
            statuses.append(StageStatus(stage_id="re-review", status="failed", summary=failure))
            return ExpertResult(
                status="partial",
                final_content=fix.output.summary,
                artifacts=tuple(dict.fromkeys([*build.output.artifacts, *fix.output.artifacts])),
                evidence=tuple([*build.output.evidence, *fix.output.evidence]),
                findings=review.output.findings,
                unresolved=tuple([*fix.output.unresolved, failure]),
                stage_statuses=tuple(statuses),
                usage=usage,
            )
        _record(rereview, usage, statuses)
        remaining = (
            rereview.output.findings
            if isinstance(rereview.output, ReviewReport)
            else review.output.findings
        )
        review_evidence = (
            rereview.output.evidence if isinstance(rereview.output, ReviewReport) else ()
        )
        unresolved = tuple(
            [
                *fix.output.unresolved,
                *(
                    ["Final re-review did not complete."]
                    if not isinstance(rereview.output, ReviewReport)
                    else []
                ),
            ]
        )
        return ExpertResult(
            status="partial" if remaining or unresolved else "completed",
            final_content=fix.output.summary,
            artifacts=tuple(dict.fromkeys([*build.output.artifacts, *fix.output.artifacts])),
            evidence=tuple(
                [
                    *build.output.evidence,
                    *fix.output.evidence,
                    *review_evidence,
                ]
            ),
            findings=remaining,
            unresolved=unresolved,
            stage_statuses=tuple(statuses),
            usage=usage,
        )


def _json_task(objective: str, **reports: BaseModel) -> str:
    payload = {
        "objective": objective,
        **{key: report.model_dump(mode="json") for key, report in reports.items()},
    }
    return json.dumps(payload, ensure_ascii=False)


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
    value = getattr(output, "summary", None)
    return str(value)[:1_000] if value else "Stage completed."


def _blocked(
    message: str,
    usage: dict[str, int],
    statuses: list[StageStatus],
) -> ExpertResult:
    return ExpertResult(
        status="blocked",
        final_content=message,
        unresolved=(message,),
        stage_statuses=tuple(statuses),
        usage=usage,
    )


__all__ = ["CodingExpertRunner"]
