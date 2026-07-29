"""Tests for bounded built-in ExpertSkill pipelines."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.harness.capabilities import CapabilityUnavailable
from aeloon_core.harness.expert.base import (
    ExpertFinding,
    ExpertRunContext,
    ExpertRunRequest,
    StageOutcome,
)
from aeloon_core.harness.expert.runners.coding import (
    BuildReport,
    CodingExpertRunner,
    CodingPlan,
    ReviewReport,
)
from aeloon_core.harness.expert.runners.research import (
    ResearchAssignment,
    ResearchExpertRunner,
    ResearchPlan,
    ResearchReport,
    ResearchSynthesis,
)
from aeloon_core.harness.skill import ExpertSkillSnapshot, SkillRegistry


class FakeStages:
    def __init__(self, outputs: dict[str, list[Any]]) -> None:
        self.outputs = defaultdict(list, outputs)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def run(
        self,
        *,
        stage_id: str,
        task: str,
        instructions: str,
        output_type: Any,
        capabilities: tuple[str, ...],
        model_tier: str | None = None,
    ) -> StageOutcome:
        del task, instructions, output_type, model_tier
        self.calls.append((stage_id, capabilities))
        output = self.outputs[stage_id].pop(0)
        if isinstance(output, BaseException):
            raise output
        return StageOutcome(
            stage_id=stage_id,
            status="completed",
            output=output,
            usage={"requests": 1},
        )


def _context(tmp_path: Path, expert_id: str, stages: FakeStages) -> ExpertRunContext:
    config = Config(workspace=tmp_path).normalized()
    skills = SkillRegistry.discover(config)
    expert = skills.require(expert_id)
    assert isinstance(expert, ExpertSkillSnapshot)
    return ExpertRunContext(
        config=config,
        expert=expert,
        skills=skills,
        scope=skills.expert_scope(expert),
        stages=stages,
    )


@pytest.mark.asyncio
async def test_coding_stops_after_clean_independent_review(tmp_path: Path) -> None:
    stages = FakeStages(
        {
            "plan": [
                CodingPlan(
                    summary="plan",
                    steps=("change",),
                    acceptance_checks=("tests",),
                )
            ],
            "build": [BuildReport(summary="built", artifacts=("a.py",))],
            "review": [ReviewReport(summary="clean")],
        }
    )

    result = await CodingExpertRunner().run(
        ExpertRunRequest(expert_id="builtin:coding", task="Implement it"),
        _context(tmp_path, "builtin:coding", stages),
    )

    assert result.status == "completed"
    assert result.artifacts == ("a.py",)
    assert [stage for stage, _ in stages.calls] == ["plan", "build", "review"]
    assert stages.calls[0][1] == ("filesystem_read", "repo_context", "planning")
    assert result.usage["requests"] == 3


@pytest.mark.asyncio
async def test_coding_allows_only_one_fix_and_one_rereview(tmp_path: Path) -> None:
    finding = ExpertFinding(
        id="bug-1",
        severity="high",
        location="a.py:1",
        impact="wrong result",
        reproduction="run test",
    )
    stages = FakeStages(
        {
            "plan": [
                CodingPlan(
                    summary="plan",
                    steps=("change",),
                    acceptance_checks=("tests",),
                )
            ],
            "build": [BuildReport(summary="built", artifacts=("a.py",))],
            "review": [ReviewReport(summary="bug", findings=(finding,))],
            "fix": [BuildReport(summary="fixed", artifacts=("a.py",))],
            "re-review": [ReviewReport(summary="still broken", findings=(finding,))],
        }
    )

    result = await CodingExpertRunner().run(
        ExpertRunRequest(expert_id="builtin:coding", task="Implement it"),
        _context(tmp_path, "builtin:coding", stages),
    )

    assert result.status == "partial"
    assert result.findings == (finding,)
    assert [stage for stage, _ in stages.calls] == [
        "plan",
        "build",
        "review",
        "fix",
        "re-review",
    ]


@pytest.mark.asyncio
async def test_research_fans_out_then_verifies_docs_and_reduces(tmp_path: Path) -> None:
    stages = FakeStages(
        {
            "plan": [
                ResearchPlan(
                    objective="answer",
                    assignments=(
                        ResearchAssignment(id="one", question="one"),
                        ResearchAssignment(id="two", question="two"),
                    ),
                )
            ],
            "explore-one": [ResearchReport(summary="one")],
            "explore-two": [ResearchReport(summary="two")],
            "docs": [ResearchReport(summary="verified")],
            "reduce": [ResearchSynthesis(final_content="answer")],
        }
    )

    result = await ResearchExpertRunner().run(
        ExpertRunRequest(expert_id="builtin:research", task="Research it"),
        _context(tmp_path, "builtin:research", stages),
    )

    assert result.status == "completed"
    assert result.final_content == "answer"
    assert [stage for stage, _ in stages.calls] == [
        "plan",
        "explore-one",
        "explore-two",
        "docs",
        "reduce",
    ]
    assert stages.calls[1][1] == ("web_search",)
    assert stages.calls[3][1] == ("web_search",)


@pytest.mark.asyncio
async def test_research_missing_web_capability_returns_blocked(tmp_path: Path) -> None:
    stages = FakeStages(
        {
            "plan": [
                ResearchPlan(
                    objective="answer",
                    assignments=(
                        ResearchAssignment(id="one", question="one"),
                        ResearchAssignment(id="two", question="two"),
                    ),
                )
            ],
            "explore-one": [CapabilityUnavailable("EXA_API_KEY missing")],
            "explore-two": [CapabilityUnavailable("EXA_API_KEY missing")],
        }
    )

    result = await ResearchExpertRunner().run(
        ExpertRunRequest(expert_id="builtin:research", task="Research it"),
        _context(tmp_path, "builtin:research", stages),
    )

    assert result.status == "blocked"
    assert "EXA_API_KEY" in result.final_content
