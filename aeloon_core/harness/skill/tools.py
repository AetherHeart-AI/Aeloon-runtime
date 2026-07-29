"""Scope-enforced Skill tools."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.harness.skill.base import SkillScope
from aeloon_core.harness.skill.registry import SkillRegistry
from aeloon_core.harness.tool import FunctionTool


class SkillSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    limit: int = Field(default=10, ge=1, le=25)


class SkillLoadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1)


class SkillReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1)
    path: str = Field(min_length=1)


def skill_tools(
    *,
    registry: SkillRegistry,
    scope: SkillScope,
) -> tuple[FunctionTool, ...]:
    """Build the three lazy Skill tools for one immutable scope."""

    async def search(query: str = "", limit: int = 10) -> str:
        return json.dumps(
            registry.search(query, scope=scope, limit=limit),
            ensure_ascii=False,
        )

    async def load(skill_id: str) -> str:
        snapshot = registry.load(skill_id, scope=scope)
        return json.dumps(
            {
                "descriptor": snapshot.descriptor(),
                "instructions": snapshot.instructions,
            },
            ensure_ascii=False,
        )

    async def read(skill_id: str, path: str) -> str:
        return registry.read_resource(skill_id, path, scope=scope)

    return (
        FunctionTool(
            name="skill_search",
            description=(
                "Search only the Skills visible to this agent. Returns metadata, not instructions."
            ),
            args_model=SkillSearchArgs,
            handler=search,
            concurrency_mode="read_only",
        ),
        FunctionTool(
            name="skill_load",
            description="Load the instructions for one in-scope Skill or ExpertSkill.",
            args_model=SkillLoadArgs,
            handler=load,
            concurrency_mode="read_only",
        ),
        FunctionTool(
            name="skill_read",
            description="Read a relative text resource from one in-scope Skill.",
            args_model=SkillReadArgs,
            handler=read,
            concurrency_mode="read_only",
        ),
    )


__all__ = ["skill_tools"]
