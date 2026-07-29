"""ExpertSkill execution contracts and normalized results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.config import Config
from aeloon_core.harness.skill import (
    ExpertSkillSnapshot,
    SkillRegistry,
    SkillScope,
)

ExpertStatus = Literal["completed", "partial", "blocked"]
StageState = Literal["completed", "partial", "blocked", "failed"]


class ExpertEvidence(BaseModel):
    """One bounded evidence item produced by an ExpertSkill."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    kind: Literal["file", "test", "lint", "typecheck", "runtime", "source"]
    locator: str = Field(min_length=1, max_length=2_000)
    claim: str = Field(min_length=1, max_length=2_000)
    status: Literal["passed", "failed", "observed", "not_applicable"]
    method: str | None = Field(default=None, max_length=2_000)


class ExpertFinding(BaseModel):
    """One actionable finding returned by an independent review stage."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=128)
    severity: Literal["critical", "high", "medium", "low"]
    location: str = Field(min_length=1, max_length=1_000)
    impact: str = Field(min_length=1, max_length=2_000)
    reproduction: str = Field(min_length=1, max_length=2_000)


class StageStatus(BaseModel):
    """Normalized status of one explicit Expert runner stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str
    status: StageState
    summary: str
    usage: dict[str, int] = Field(default_factory=dict)


class ExpertResult(BaseModel):
    """Typed completion value returned to Master by every ExpertSkill."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ExpertStatus
    final_content: str = Field(min_length=1, max_length=64_000)
    artifacts: tuple[str, ...] = Field(default=(), max_length=64)
    evidence: tuple[ExpertEvidence, ...] = Field(default=(), max_length=128)
    findings: tuple[ExpertFinding, ...] = Field(default=(), max_length=128)
    unresolved: tuple[str, ...] = Field(default=(), max_length=64)
    stage_statuses: tuple[StageStatus, ...] = Field(default=(), max_length=64)
    usage: dict[str, int] = Field(default_factory=dict)


class ExpertRunRequest(BaseModel):
    """One Master-authorized, current-turn Expert invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expert_id: str
    task: str = Field(min_length=1, max_length=64_000)


class StageOutcome(BaseModel):
    """Host execution result for one isolated model stage."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    stage_id: str
    status: StageState
    output: Any | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    tools_used: tuple[str, ...] = ()
    failure: str | None = None


@runtime_checkable
class ExpertStageExecutor(Protocol):
    """Run one isolated stage with a frozen ExpertSkill scope."""

    async def run(
        self,
        *,
        stage_id: str,
        task: str,
        instructions: str,
        output_type: Any,
        capabilities: tuple[str, ...],
        model_tier: Literal["fast", "strong"] | None = None,
    ) -> StageOutcome: ...


@dataclass(frozen=True, slots=True)
class ExpertRunContext:
    """Host-owned dependencies available to a trusted runner implementation."""

    config: Config
    expert: ExpertSkillSnapshot
    skills: SkillRegistry
    scope: SkillScope
    stages: ExpertStageExecutor


@runtime_checkable
class ExpertRunner(Protocol):
    """Minimal runner boundary; core does not know runner graph semantics."""

    async def run(
        self,
        request: ExpertRunRequest,
        context: ExpertRunContext,
    ) -> ExpertResult: ...


__all__ = [
    "ExpertEvidence",
    "ExpertFinding",
    "ExpertResult",
    "ExpertRunContext",
    "ExpertRunRequest",
    "ExpertRunner",
    "ExpertStageExecutor",
    "ExpertStatus",
    "StageOutcome",
    "StageStatus",
]
