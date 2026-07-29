"""Master-only ExpertSkill invocation tool."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.harness.expert.runtime import ExpertRuntime
from aeloon_core.harness.tool import FunctionTool


class ExpertRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert_id: str = Field(min_length=1)
    task: str = Field(min_length=1, max_length=64_000)


def expert_run_tool(runtime: ExpertRuntime) -> FunctionTool:
    """Build the only Master capability that can invoke an ExpertSkill."""

    async def run(expert_id: str, task: str) -> str:
        result = await runtime.run(expert_id, task)
        return result.model_dump_json()

    return FunctionTool(
        name="expert_run",
        description=(
            "Run one enabled ExpertSkill to completion in the current turn. The result "
            "is a typed completed, partial, or blocked report. Experts cannot nest."
        ),
        args_model=ExpertRunArgs,
        handler=run,
        # A generic invocation may select a mutating expert, so it is a Master-side
        # barrier. Runners can still fan out safe internal stages, and the runtime
        # enforces each ExpertSkill's own declaration for direct concurrent callers.
        concurrency_mode="exclusive",
    )


__all__ = ["expert_run_tool"]
