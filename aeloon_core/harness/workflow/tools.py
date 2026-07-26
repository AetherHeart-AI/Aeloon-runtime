"""Master tool adapters for discovering and executing fixed Workflow Templates."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.config import Config
from aeloon_core.harness.agent.base import RoleRegistry
from aeloon_core.harness.tool.base import Tool
from aeloon_core.harness.workflow.base import (
    WorkflowDefinitionError,
    WorkflowRegistry,
)
from aeloon_core.harness.workflow.runner import WorkflowRunner


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkflowSearchArgs(_StrictArgs):
    query: str = Field(min_length=1, max_length=8_000)
    tags: tuple[str, ...] = Field(default=(), max_length=16)
    limit: int = Field(default=5, ge=1, le=10)


class WorkflowDescribeArgs(_StrictArgs):
    template_id: str = Field(min_length=1, max_length=64)


class WorkflowExecuteArgs(_StrictArgs):
    template_id: str = Field(min_length=1, max_length=64)
    inputs: dict[str, Any]
    tuning: dict[str, Any] = Field(default_factory=dict)


class WorkflowSearchTool(Tool):
    name = "workflow_search"
    description = (
        "Search trusted fixed workflow templates by outcome, constraints, and tags. "
        "Use when the host-provided candidates are missing or ambiguous."
    )
    args_model = WorkflowSearchArgs
    concurrency_mode = "read_only"

    def __init__(self, workflows: WorkflowRegistry) -> None:
        self.workflows = workflows

    async def execute(
        self,
        query: str,
        tags: tuple[str, ...] = (),
        limit: int = 5,
    ) -> str:
        return json.dumps(
            self.workflows.search(query, tags=tags, limit=limit),
            ensure_ascii=False,
            sort_keys=True,
        )


class WorkflowDescribeTool(Tool):
    name = "workflow_describe"
    description = (
        "Inspect one fixed workflow template's purpose, limitations, input schema, "
        "and run-scoped tuning schema."
    )
    args_model = WorkflowDescribeArgs
    concurrency_mode = "read_only"

    def __init__(self, workflows: WorkflowRegistry) -> None:
        self.workflows = workflows

    async def execute(self, template_id: str) -> str:
        try:
            snapshot = self.workflows.get(template_id)
        except KeyError as exc:
            return f"Error [WORKFLOW_NOT_FOUND]: {exc}"
        return json.dumps(
            snapshot.descriptor(include_schema=True),
            ensure_ascii=False,
            sort_keys=True,
        )


class WorkflowExecuteTool(Tool):
    name = "workflow_execute"
    description = (
        "Validate and execute one trusted fixed workflow template in this turn. "
        "Inputs and tuning must match the template schemas exactly."
    )
    args_model = WorkflowExecuteArgs
    concurrency_mode = "exclusive"

    def __init__(
        self,
        *,
        config: Config,
        roles: RoleRegistry,
        workflows: WorkflowRegistry,
        runner: WorkflowRunner,
    ) -> None:
        self.config = config
        self.roles = roles
        self.workflows = workflows
        self.runner = runner

    async def execute(
        self,
        template_id: str,
        inputs: dict[str, Any],
        tuning: dict[str, Any] | None = None,
    ) -> str:
        try:
            template = self.workflows.get(template_id)
        except KeyError as exc:
            return f"Error [WORKFLOW_NOT_FOUND]: {exc}"
        try:
            plan = template.compile(
                inputs=inputs,
                tuning=tuning,
                roles=self.roles,
                max_nodes=min(
                    self.config.agents.templates.max_nodes,
                    self.config.agents.harness.max_agent_calls,
                ),
            )
        except WorkflowDefinitionError as exc:
            return (
                "Error [WORKFLOW_INPUT_INVALID]: "
                f"template={template_id!r}; detail={str(exc)!r}"
            )
        result = await self.runner.run(template, plan)
        return result.model_dump_json()


def workflow_tools(
    *,
    config: Config,
    roles: RoleRegistry,
    workflows: WorkflowRegistry,
    runner: WorkflowRunner,
) -> tuple[Tool, ...]:
    return (
        WorkflowSearchTool(workflows),
        WorkflowDescribeTool(workflows),
        WorkflowExecuteTool(
            config=config,
            roles=roles,
            workflows=workflows,
            runner=runner,
        ),
    )


__all__ = [
    "WorkflowDescribeTool",
    "WorkflowExecuteTool",
    "WorkflowSearchTool",
    "workflow_tools",
]
