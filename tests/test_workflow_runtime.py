from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.customization.catalog import Catalog
from aeloon_core.customization.roles import ReviewFinding, ReviewReport, WorkerReport
from aeloon_core.customization.workflows import WorkflowNode, WorkflowPlan
from aeloon_core.harness.runner import WorkflowRunner
from aeloon_core.harness.workflow_tools import workflow_tools
from aeloon_core.tools.registry import ToolRegistry


class FakeRoleFactory:
    def __init__(self, outputs: dict[str, list[Any]] | None = None) -> None:
        self.progress = None
        self.outputs = {
            role_id: deque(values) for role_id, values in (outputs or {}).items()
        }
        self.tasks: list[tuple[str, str]] = []
        self.active = 0
        self.max_active = 0

    async def run(
        self,
        snapshot: Any,
        *,
        task: str,
        progress: Any = None,
        execution_key: str | None = None,
    ) -> Any:
        del progress, execution_key
        self.tasks.append((snapshot.id, task))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        queue = self.outputs.get(snapshot.id)
        if queue:
            value = queue.popleft()
            if isinstance(value, BaseException):
                raise value
            return value
        return WorkerReport(summary=f"{snapshot.id} complete")


def _config(path: Path, *, max_upstream_chars: int = 32_000) -> Config:
    return Config(
        workspace=path,
        agents={
            "templates": {
                "max_concurrency": 4,
                "max_upstream_chars": max_upstream_chars,
            }
        },
    ).normalized()


@pytest.mark.asyncio
async def test_parallel_safe_nodes_run_concurrently(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)
    factory = FakeRoleFactory()
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("parallel-investigate")
    plan = template.compile(
        inputs={"role_id": "explorer", "tasks": ["one", "two", "three"]},
        tuning={},
        roles=catalog.roles,
    )

    result = await runner.run(template, plan)

    assert result.status == "completed"
    assert factory.max_active == 3


@pytest.mark.asyncio
async def test_exclusive_nodes_never_overlap(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)
    factory = FakeRoleFactory()
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("delegate")
    plan = WorkflowPlan(
        nodes=(
            WorkflowNode(id="one", role_id="builder", objective="one"),
            WorkflowNode(id="two", role_id="builder", objective="two"),
        )
    )

    result = await runner.run(template, plan)

    assert result.status == "completed"
    assert factory.max_active == 1


@pytest.mark.asyncio
async def test_clean_review_skips_every_revision_node(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)
    factory = FakeRoleFactory(
        {
            "reviewer": [
                ReviewReport(summary="clean", findings=()),
            ]
        }
    )
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("implement-review-revise")
    plan = template.compile(
        inputs={"objective": "implement it"},
        tuning={"max_revision_rounds": 2},
        roles=catalog.roles,
    )

    result = await runner.run(template, plan)

    assert result.status == "completed"
    assert result.node_statuses["review-1"] == "completed"
    assert result.node_statuses["fix-1"] == "skipped"
    assert result.node_statuses["review-3"] == "skipped"


@pytest.mark.asyncio
async def test_revision_loop_stops_after_configured_rounds(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)
    finding = ReviewFinding(
        id="F-1",
        severity="high",
        location="module.py:1",
        impact="broken",
        reproduction="run test",
    )
    factory = FakeRoleFactory(
        {
            "reviewer": [
                ReviewReport(summary="issue", findings=(finding,)),
                ReviewReport(summary="still broken", findings=(finding,)),
                ReviewReport(summary="clean", findings=()),
            ]
        }
    )
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("implement-review-revise")
    plan = template.compile(
        inputs={"objective": "implement it"},
        tuning={"max_revision_rounds": 2},
        roles=catalog.roles,
    )

    result = await runner.run(template, plan)

    assert result.status == "completed"
    assert result.node_statuses["fix-1"] == "completed"
    assert result.node_statuses["fix-2"] == "completed"
    assert len([role for role, _ in factory.tasks if role == "reviewer"]) == 3
    assert len([role for role, _ in factory.tasks if role == "builder"]) == 3


@pytest.mark.asyncio
async def test_remaining_findings_return_partial(tmp_path: Path) -> None:
    catalog = Catalog.discover(tmp_path)
    finding = ReviewFinding(
        id="F-1",
        severity="medium",
        location="module.py:2",
        impact="risk",
        reproduction="inspect",
    )
    factory = FakeRoleFactory(
        {
            "reviewer": [
                ReviewReport(summary="issue", findings=(finding,)),
                ReviewReport(summary="still issue", findings=(finding,)),
            ]
        }
    )
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("implement-review-revise")
    plan = template.compile(
        inputs={"objective": "implement it"},
        tuning={"max_revision_rounds": 1},
        roles=catalog.roles,
    )

    result = await runner.run(template, plan)

    assert result.status == "partial"
    assert result.node_statuses["review-2"] == "completed"


@pytest.mark.asyncio
async def test_upstream_reports_are_marked_untrusted_and_bounded(
    tmp_path: Path,
) -> None:
    catalog = Catalog.discover(tmp_path)
    factory = FakeRoleFactory(
        {"builder": [WorkerReport(summary="x" * 5_000)]}
    )
    runner = WorkflowRunner(
        config=_config(tmp_path, max_upstream_chars=1_000),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("implement-review")
    plan = template.compile(
        inputs={"objective": "implement it"},
        tuning={},
        roles=catalog.roles,
    )

    await runner.run(template, plan)

    review_task = next(task for role, task in factory.tasks if role == "reviewer")
    assert "UNTRUSTED UPSTREAM REPORTS" in review_task
    assert "truncated by host" in review_task
    assert len(review_task) < 4_000


@pytest.mark.asyncio
async def test_failed_node_skips_dependents_and_reports_failure(
    tmp_path: Path,
) -> None:
    catalog = Catalog.discover(tmp_path)
    factory = FakeRoleFactory({"builder": [RuntimeError("boom")]})
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    template = catalog.workflows.get("implement-review")
    plan = template.compile(
        inputs={"objective": "implement it"},
        tuning={},
        roles=catalog.roles,
    )

    result = await runner.run(template, plan)

    assert result.status == "failed"
    assert result.node_statuses == {"implement": "failed", "review": "skipped"}
    assert "RuntimeError: boom" in result.failures["implement"]


@pytest.mark.asyncio
async def test_workflow_tools_search_describe_validate_and_execute(
    tmp_path: Path,
) -> None:
    catalog = Catalog.discover(tmp_path)
    factory = FakeRoleFactory()
    runner = WorkflowRunner(
        config=_config(tmp_path),
        roles=catalog.roles,
        role_factory=factory,  # type: ignore[arg-type]
    )
    registry = ToolRegistry()
    for tool in workflow_tools(
        config=_config(tmp_path),
        roles=catalog.roles,
        workflows=catalog.workflows,
        runner=runner,
    ):
        registry.register(tool)

    search = await registry.execute("workflow_search", {"query": "实现审查"})
    describe = await registry.execute(
        "workflow_describe",
        {"template_id": "implement-review"},
    )
    invalid = await registry.execute(
        "workflow_execute",
        {"template_id": "delegate", "inputs": {"missing": "task"}},
    )
    executed = await registry.execute(
        "workflow_execute",
        {
            "template_id": "delegate",
            "inputs": {"role_id": "explorer", "task": "inspect"},
        },
    )

    assert "implement-review" in search
    assert "input_schema" in describe
    assert "WORKFLOW_INPUT_INVALID" in invalid
    assert '"status":"completed"' in executed
