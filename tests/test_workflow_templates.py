from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.harness.catalog import Catalog
from aeloon_core.harness.workflow.base import (
    OutputCondition,
    WorkflowDefinitionError,
    WorkflowNode,
    WorkflowPlan,
)


def test_builtin_workflow_catalog_and_chinese_search(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)

    assert [template.id for template in catalog.workflows.list()] == [
        "delegate",
        "implement-review",
        "implement-review-revise",
        "parallel-investigate",
    ]
    matches = catalog.workflows.search("实现代码并审查修复")
    assert matches[0]["id"] == "implement-review-revise"
    assert "input_schema" in matches[0]
    assert "tuning_schema" in matches[0]
    assert catalog.workflows.search("unmatched vocabulary")[0]["id"] == "delegate"


def test_template_inputs_and_tuning_are_strict(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)
    template = catalog.workflows.get("implement-review-revise")

    with pytest.raises(WorkflowDefinitionError, match="extra"):
        template.compile(
            inputs={"objective": "do it", "extra": True},
            tuning={},
            roles=catalog.roles,
        )
    with pytest.raises(WorkflowDefinitionError, match="less than or equal to 2"):
        template.compile(
            inputs={"objective": "do it"},
            tuning={"max_revision_rounds": 3},
            roles=catalog.roles,
        )


def test_template_build_validation_is_wrapped_as_definition_error(
    tmp_path: Path,
) -> None:
    catalog = Catalog.discover(tmp_path)
    template = catalog.workflows.get("delegate")

    with pytest.raises(WorkflowDefinitionError, match="could not compile"):
        template.compile(
            inputs={"role_id": "builder", "task": "x" * 32_000},
            tuning={"extra_instructions": "additional"},
            roles=catalog.roles,
        )


def test_conditional_template_validates_referenced_role_output(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / ".aeloon-core" / "catalog.py"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        """
from aeloon_core.harness.agent import Role

class ProjectReviewer(Role):
    id = "reviewer"
    description = "Reviewer without findings"
    system_prompt = "Return a generic report."

ROLES = (ProjectReviewer,)
WORKFLOWS = ()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    catalog = Catalog.discover(tmp_path)
    template = catalog.workflows.get("implement-review-revise")

    with pytest.raises(WorkflowDefinitionError, match="references field 'findings'"):
        template.compile(
            inputs={"objective": "do it"},
            tuning={},
            roles=catalog.roles,
        )


def test_revision_template_compiles_bounded_conditional_nodes(
    tmp_path: Path,
) -> None:
    catalog = Catalog.discover(tmp_path)
    template = catalog.workflows.get("implement-review-revise")

    one_round = template.compile(
        inputs={"objective": "do it"},
        tuning={"max_revision_rounds": 1},
        roles=catalog.roles,
    )
    two_rounds = template.compile(
        inputs={"objective": "do it"},
        tuning={"max_revision_rounds": 2},
        roles=catalog.roles,
    )

    assert [node.id for node in one_round.nodes] == [
        "implement",
        "review-1",
        "fix-1",
        "review-2",
    ]
    assert [node.id for node in two_rounds.nodes][-2:] == ["fix-2", "review-3"]
    assert two_rounds.nodes[2].condition == OutputCondition(
        node_id="review-1",
        field_path="findings",
        operator="non_empty",
    )


def test_plan_rejects_unknown_dependencies_conditions_and_cycles() -> None:
    with pytest.raises(ValueError, match="unknown dependencies"):
        WorkflowPlan(
            nodes=(
                WorkflowNode(
                    id="one",
                    role_id="builder",
                    objective="x",
                    depends_on=("missing",),
                ),
            )
        )
    with pytest.raises(ValueError, match="condition must reference a dependency"):
        WorkflowPlan(
            nodes=(
                WorkflowNode(id="one", role_id="builder", objective="x"),
                WorkflowNode(
                    id="two",
                    role_id="reviewer",
                    objective="y",
                    condition=OutputCondition(
                        node_id="one",
                        field_path="findings",
                        operator="empty",
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="dependency cycle"):
        WorkflowPlan(
            nodes=(
                WorkflowNode(
                    id="one",
                    role_id="builder",
                    objective="x",
                    depends_on=("two",),
                ),
                WorkflowNode(
                    id="two",
                    role_id="reviewer",
                    objective="y",
                    depends_on=("one",),
                ),
            )
        )


def test_project_workflow_is_loaded_and_can_override_builtin(tmp_path: Path) -> None:
    catalog_path = tmp_path / ".aeloon-core" / "catalog.py"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        """
from pydantic import BaseModel
from aeloon_core.harness.workflow import (
    WorkflowNode,
    WorkflowPlan,
    WorkflowTemplate,
)

class Inputs(BaseModel):
    task: str

class ProjectDelegate(WorkflowTemplate):
    id = "delegate"
    description = "Project delegate"
    tags = ("project",)
    when_to_use = "Project requests."
    avoid_when = "Never for external requests."
    input_model = Inputs

    def build(self, inputs, tuning):
        return WorkflowPlan(nodes=(
            WorkflowNode(id="project", role_id="builder", objective=inputs.task),
        ))

ROLES = ()
WORKFLOWS = (ProjectDelegate,)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    catalog = Catalog.discover(tmp_path)

    assert catalog.workflows.get("delegate").description == "Project delegate"
