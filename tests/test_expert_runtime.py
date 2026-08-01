"""Tests for Expert runtime budgets, registration, and optional adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.config import Config
from aeloon_core.harness.execution import HarnessAgentRuntime
from aeloon_core.harness.expert import (
    ExpertCatalogError,
    ExpertResult,
    ExpertRunnerRegistry,
    ExpertRuntime,
    LangGraphExpertRunner,
)
from aeloon_core.harness.model import ModelRouter
from aeloon_core.harness.provider import ScriptedPiModel
from aeloon_core.harness.skill import SkillRegistry


def _response(*parts: dict[str, object]) -> dict[str, object]:
    return {"content": list(parts)}


def _call(name: str, arguments: dict[str, object], call_id: str) -> dict[str, object]:
    return {"type": "toolCall", "name": name, "arguments": arguments, "id": call_id}


class FakeRunner:
    async def run(self, request, context) -> ExpertResult:
        del context
        return ExpertResult(
            status="completed",
            final_content=f"done: {request.task}",
            usage={"requests": 2},
        )


def _custom_expert(workspace: Path, *, max_calls: int = 1) -> None:
    directory = workspace / ".aeloon-core" / "skills" / "custom"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"""---
name: custom
description: Custom expert.
kind: expert
runner: project.custom
capabilities: []
max_calls_per_turn: {max_calls}
---
# Custom

Complete the assigned task.
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_runtime_enforces_per_expert_budget_and_accumulates_usage(
    tmp_path: Path,
) -> None:
    _custom_expert(tmp_path, max_calls=1)
    config = Config(
        workspace=tmp_path,
        experts={"enabled": ["workspace:custom"], "max_calls_per_turn": 4},
    ).normalized()
    skills = SkillRegistry.discover(config)
    runtime = ExpertRuntime(
        config=config,
        skills=skills,
        runners=ExpertRunnerRegistry({"project.custom": FakeRunner()}),
        model_router=ModelRouter(
            config,
            injected_model=ScriptedPiModel(({"text": "unused"},)),
        ),
        agent_runtime=HarnessAgentRuntime(),
    )

    result = await runtime.run("workspace:custom", "work")

    assert result.final_content == "done: work"
    assert runtime.usage == {"requests": 2}
    with pytest.raises(RuntimeError, match="per-turn call budget"):
        await runtime.run("workspace:custom", "again")


@pytest.mark.asyncio
async def test_default_research_backend_returns_blocked_without_exa_key(
    tmp_path: Path,
) -> None:
    config = Config(
        workspace=tmp_path,
        agents={"defaults": {"context_compaction": {"enabled": False}}},
        experts={"enabled": ["builtin:research"]},
    ).normalized()
    skills = SkillRegistry.discover(config)
    model = ScriptedPiModel(
        (
            _response(
                _call(
                    "final_result",
                    {
                        "objective": "answer",
                        "assignments": [
                            {"id": "one", "question": "question one"},
                            {"id": "two", "question": "question two"},
                        ],
                        "verification_focus": [],
                    },
                    "research-plan",
                )
            ),
        )
    )
    runtime = ExpertRuntime(
        config=config,
        skills=skills,
        runners=ExpertRunnerRegistry.discover(tmp_path),
        model_router=ModelRouter(config, injected_model=model),
        agent_runtime=HarnessAgentRuntime(),
    )

    result = await runtime.run("builtin:research", "Research it")

    assert result.status == "blocked"
    assert "EXA_API_KEY" in result.final_content
    assert result.usage["requests"] == 1


@pytest.mark.asyncio
async def test_langgraph_adapter_normalizes_mapping_result(tmp_path: Path) -> None:
    class Graph:
        async def ainvoke(self, state):
            assert state["request"]["task"] == "work"
            assert state["scope"] == ["builtin:coding"]
            return {
                "status": "completed",
                "final_content": "graph done",
            }

    config = Config(workspace=tmp_path).normalized()
    skills = SkillRegistry.discover(config)
    expert = skills.require("builtin:coding")
    from aeloon_core.harness.expert.base import ExpertRunContext, ExpertRunRequest

    context = ExpertRunContext(
        config=config,
        expert=expert,  # type: ignore[arg-type]
        skills=skills,
        scope=skills.expert_scope(expert),  # type: ignore[arg-type]
        stages=object(),  # type: ignore[arg-type]
    )

    result = await LangGraphExpertRunner(Graph()).run(
        ExpertRunRequest(expert_id="builtin:coding", task="work"),
        context,
    )

    assert result.final_content == "graph done"


def test_project_catalog_rejects_removed_role_workflow_entries(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / ".aeloon-core" / "catalog.py"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "from aeloon_core.harness.agent import Role\nROLES = (Role,)\n",
        encoding="utf-8",
    )

    with pytest.raises(ExpertCatalogError, match="migrate to EXPERT_RUNNERS"):
        ExpertRunnerRegistry.discover(tmp_path)
