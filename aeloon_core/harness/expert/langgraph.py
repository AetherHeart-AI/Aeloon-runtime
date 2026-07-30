"""Optional adapter for trusted, precompiled LangGraph expert runners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aeloon_core.harness.expert.base import (
    ExpertResult,
    ExpertRunContext,
    ExpertRunRequest,
)


class LangGraphExpertRunner:
    """Wrap a trusted compiled graph without teaching core generic DAG semantics."""

    def __init__(
        self,
        graph: Any,
        *,
        result_adapter: Callable[[Any], ExpertResult] | None = None,
    ) -> None:
        if not callable(getattr(graph, "ainvoke", None)):
            raise TypeError("LangGraph expert runner requires a compiled graph with ainvoke()")
        self.graph = graph
        self.result_adapter = result_adapter

    async def run(
        self,
        request: ExpertRunRequest,
        context: ExpertRunContext,
    ) -> ExpertResult:
        state = {
            "request": request.model_dump(mode="json"),
            "expert": context.expert.descriptor(),
            "scope": sorted(context.scope.skill_ids),
            "workspace": str(context.config.workspace),
            "stage_executor": context.stages,
        }
        output = await self.graph.ainvoke(state)
        if self.result_adapter is not None:
            return self.result_adapter(output)
        if isinstance(output, ExpertResult):
            return output
        if isinstance(output, dict):
            candidate = output.get("expert_result", output)
            return ExpertResult.model_validate(candidate)
        raise TypeError(
            "LangGraph expert must return ExpertResult or an ExpertResult-compatible mapping"
        )


__all__ = ["LangGraphExpertRunner"]
