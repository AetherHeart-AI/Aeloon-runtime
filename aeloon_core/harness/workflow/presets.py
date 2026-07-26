"""Preset fixed Workflow Templates shipped with Aeloon Core."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.harness.workflow.base import (
    OutputCondition,
    WorkflowNode,
    WorkflowPlan,
    WorkflowTemplate,
)


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DelegateInput(_StrictInput):
    role_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    task: str = Field(min_length=1, max_length=32_000)


class DelegateTuning(_StrictInput):
    extra_instructions: str = Field(default="", max_length=8_000)


class DelegateWorkflow(WorkflowTemplate[DelegateInput, DelegateTuning]):
    id = "delegate"
    description = "Run one known role for one self-contained objective"
    tags = (
        "delegate",
        "single",
        "task",
        "implement",
        "fix",
        "inspect",
        "review",
        "research",
        "单角色",
        "委派",
        "任务",
        "实现",
        "修复",
        "检查",
        "审查",
        "研究",
    )
    when_to_use = (
        "Use for one self-contained task that clearly belongs to one available role."
    )
    avoid_when = "Avoid when work requires independent parallel branches or review gates."
    input_model = DelegateInput
    tuning_model = DelegateTuning

    def build(self, inputs: DelegateInput, tuning: DelegateTuning) -> WorkflowPlan:
        return WorkflowPlan(
            nodes=(
                WorkflowNode(
                    id="delegate",
                    role_id=inputs.role_id,
                    objective=_with_extra(inputs.task, tuning.extra_instructions),
                ),
            )
        )


class ParallelInvestigateInput(_StrictInput):
    role_id: str = Field(
        default="explorer",
        pattern=r"^[a-z][a-z0-9_-]{0,63}$",
    )
    tasks: tuple[str, ...] = Field(min_length=1, max_length=8)


class ParallelInvestigateTuning(_StrictInput):
    focus: str = Field(default="", max_length=8_000)


class ParallelInvestigateWorkflow(
    WorkflowTemplate[ParallelInvestigateInput, ParallelInvestigateTuning]
):
    id = "parallel-investigate"
    description = "Run independent read-only investigations concurrently"
    tags = (
        "parallel",
        "investigate",
        "inspect",
        "research",
        "并行",
        "调查",
        "检查",
        "研究",
        "分析",
    )
    when_to_use = (
        "Use when two or more investigation or research tasks are independent and "
        "can safely share a read-only workspace."
    )
    avoid_when = (
        "Avoid for mutating work, hidden dependencies, or tasks that need one result "
        "before the next objective can be authored."
    )
    input_model = ParallelInvestigateInput
    tuning_model = ParallelInvestigateTuning

    def build(
        self,
        inputs: ParallelInvestigateInput,
        tuning: ParallelInvestigateTuning,
    ) -> WorkflowPlan:
        return WorkflowPlan(
            nodes=tuple(
                WorkflowNode(
                    id=f"investigate-{index}",
                    role_id=inputs.role_id,
                    objective=_with_extra(task, tuning.focus),
                )
                for index, task in enumerate(inputs.tasks, start=1)
            )
        )


class ImplementInput(_StrictInput):
    objective: str = Field(min_length=1, max_length=32_000)
    acceptance: str = Field(default="", max_length=8_000)


class ImplementReviewTuning(_StrictInput):
    implementation_constraints: str = Field(default="", max_length=8_000)
    review_focus: str = Field(default="", max_length=8_000)


class ImplementReviewWorkflow(
    WorkflowTemplate[ImplementInput, ImplementReviewTuning]
):
    id = "implement-review"
    description = "Implement a scoped change and then run an independent review"
    tags = ("implement", "review", "code", "实现", "审查")
    when_to_use = (
        "Use for a well-scoped implementation that should be independently reviewed "
        "after verification."
    )
    avoid_when = (
        "Avoid when requirements are unknown, research must author the implementation "
        "objective, or review findings must be automatically fixed."
    )
    input_model = ImplementInput
    tuning_model = ImplementReviewTuning

    def build(
        self,
        inputs: ImplementInput,
        tuning: ImplementReviewTuning,
    ) -> WorkflowPlan:
        implementation = _implementation_objective(inputs, tuning)
        review = _review_objective(inputs, tuning.review_focus)
        return WorkflowPlan(
            nodes=(
                WorkflowNode(
                    id="implement",
                    role_id="builder",
                    objective=implementation,
                ),
                WorkflowNode(
                    id="review",
                    role_id="reviewer",
                    objective=review,
                    depends_on=("implement",),
                    include_reports=("implement",),
                ),
            )
        )


class ImplementReviewReviseTuning(ImplementReviewTuning):
    max_revision_rounds: int = Field(default=1, ge=1, le=2)


class ImplementReviewReviseWorkflow(
    WorkflowTemplate[ImplementInput, ImplementReviewReviseTuning]
):
    id = "implement-review-revise"
    description = "Implement, review, conditionally fix findings, and re-review"
    tags = (
        "implement",
        "review",
        "revise",
        "fix",
        "实现",
        "审查",
        "修复",
        "审查修复",
    )
    when_to_use = (
        "Use for a well-scoped implementation where actionable review findings should "
        "be fixed automatically for at most two bounded rounds."
    )
    avoid_when = (
        "Avoid when findings require product decisions, new authority, external input, "
        "or an unbounded exploratory repair loop."
    )
    input_model = ImplementInput
    tuning_model = ImplementReviewReviseTuning

    def build(
        self,
        inputs: ImplementInput,
        tuning: ImplementReviewReviseTuning,
    ) -> WorkflowPlan:
        nodes: list[WorkflowNode] = [
            WorkflowNode(
                id="implement",
                role_id="builder",
                objective=_implementation_objective(inputs, tuning),
            ),
            WorkflowNode(
                id="review-1",
                role_id="reviewer",
                objective=_review_objective(inputs, tuning.review_focus),
                depends_on=("implement",),
                include_reports=("implement",),
            ),
            WorkflowNode(
                id="fix-1",
                role_id="builder",
                objective=_fix_objective(inputs, round_number=1),
                depends_on=("review-1",),
                include_reports=("review-1",),
                condition=_findings_nonempty("review-1"),
            ),
            WorkflowNode(
                id="review-2",
                role_id="reviewer",
                objective=_review_objective(inputs, tuning.review_focus),
                depends_on=("fix-1",),
                include_reports=("fix-1",),
                condition=_status_completed("fix-1"),
            ),
        ]
        success_conditions = [
            _findings_empty("review-1"),
            _findings_empty("review-2"),
        ]
        if tuning.max_revision_rounds == 2:
            nodes.extend(
                (
                    WorkflowNode(
                        id="fix-2",
                        role_id="builder",
                        objective=_fix_objective(inputs, round_number=2),
                        depends_on=("review-2",),
                        include_reports=("review-2",),
                        condition=_findings_nonempty("review-2"),
                    ),
                    WorkflowNode(
                        id="review-3",
                        role_id="reviewer",
                        objective=_review_objective(inputs, tuning.review_focus),
                        depends_on=("fix-2",),
                        include_reports=("fix-2",),
                        condition=_status_completed("fix-2"),
                    ),
                )
            )
            success_conditions.append(_findings_empty("review-3"))
        return WorkflowPlan(
            nodes=tuple(nodes),
            success_when_any=tuple(success_conditions),
        )


def _with_extra(objective: str, extra: str) -> str:
    normalized = extra.strip()
    if not normalized:
        return objective
    return f"{objective}\n\nAdditional run-scoped instructions:\n{normalized}"


def _implementation_objective(
    inputs: ImplementInput,
    tuning: ImplementReviewTuning,
) -> str:
    sections = [inputs.objective]
    if inputs.acceptance:
        sections.append(f"Acceptance conditions:\n{inputs.acceptance}")
    if tuning.implementation_constraints:
        sections.append(
            "Additional run-scoped constraints:\n"
            + tuning.implementation_constraints
        )
    return "\n\n".join(sections)


def _review_objective(inputs: ImplementInput, review_focus: str) -> str:
    sections = [
        "Independently review the implementation for the objective below. "
        "Return an empty structured findings list only when no actionable issue remains.",
        f"Authoritative objective:\n{inputs.objective}",
    ]
    if inputs.acceptance:
        sections.append(f"Acceptance conditions:\n{inputs.acceptance}")
    if review_focus:
        sections.append(f"Additional review focus:\n{review_focus}")
    return "\n\n".join(sections)


def _fix_objective(inputs: ImplementInput, *, round_number: int) -> str:
    return (
        f"Revision round {round_number}: fix every actionable finding in the attached "
        "review report that is valid and in scope. Re-run affected verification. Preserve "
        "unrelated user work and report any finding that cannot be safely resolved.\n\n"
        f"Authoritative objective:\n{inputs.objective}"
    )


def _findings_nonempty(node_id: str) -> OutputCondition:
    return OutputCondition(
        node_id=node_id,
        field_path="findings",
        operator="non_empty",
    )


def _findings_empty(node_id: str) -> OutputCondition:
    return OutputCondition(
        node_id=node_id,
        field_path="findings",
        operator="empty",
    )


def _status_completed(node_id: str) -> OutputCondition:
    return OutputCondition(
        node_id=node_id,
        field_path="status",
        operator="equals",
        value="completed",
    )


BUILTIN_WORKFLOWS = (
    DelegateWorkflow,
    ParallelInvestigateWorkflow,
    ImplementReviewWorkflow,
    ImplementReviewReviseWorkflow,
)


__all__ = [
    "BUILTIN_WORKFLOWS",
    "DelegateWorkflow",
    "ImplementReviewReviseWorkflow",
    "ImplementReviewWorkflow",
    "ParallelInvestigateWorkflow",
]
