"""Turn-scoped ExpertSkill runtime with budgets, timeouts, and isolation."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from aeloon_core.config import Config
from aeloon_core.harness.capabilities import (
    CapabilityUnavailable,
    WebCapabilityFactory,
)
from aeloon_core.harness.execution import HarnessAgentRuntime, accumulate_usage
from aeloon_core.harness.expert.base import (
    ExpertResult,
    ExpertRunContext,
    ExpertRunRequest,
)
from aeloon_core.harness.expert.registry import ExpertRunnerRegistry
from aeloon_core.harness.expert.stage import HarnessExpertStageExecutor
from aeloon_core.harness.model import ModelRouter
from aeloon_core.harness.skill import ExpertSkillSnapshot, SkillRegistry


class ExpertRuntime:
    """Execute enabled ExpertSkills only within the current Master turn."""

    def __init__(
        self,
        *,
        config: Config,
        skills: SkillRegistry,
        runners: ExpertRunnerRegistry,
        model_router: ModelRouter,
        agent_runtime: HarnessAgentRuntime,
        progress: Any | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        web_capability_factory: WebCapabilityFactory | None = None,
    ) -> None:
        self.config = config
        self.skills = skills
        self.runners = runners
        self.model_router = model_router
        self.agent_runtime = agent_runtime
        self.progress = progress
        self.session_id = session_id
        self.turn_id = turn_id
        self.web_capability_factory = web_capability_factory
        self._enabled = {
            expert.id: expert for expert in skills.enabled_experts(config.experts.enabled)
        }
        self._calls = 0
        self._calls_by_expert: Counter[str] = Counter()
        self._semaphore = asyncio.Semaphore(config.experts.max_concurrency)
        self._concurrency_gate = _ExpertConcurrencyGate()
        self.usage: dict[str, int] = {}

    async def run(self, expert_id: str, task: str) -> ExpertResult:
        """Invoke one enabled ExpertSkill and normalize all recoverable failures."""

        expert = self._enabled.get(expert_id)
        if expert is None:
            available = ", ".join(sorted(self._enabled))
            raise PermissionError(f"ExpertSkill {expert_id!r} is not enabled; enabled: {available}")
        self._reserve(expert)
        runner = self.runners.require(expert.runner)
        scope = self.skills.expert_scope(expert)
        stages = HarnessExpertStageExecutor(
            config=self.config,
            expert=expert,
            skills=self.skills,
            scope=scope,
            model_router=self.model_router,
            agent_runtime=self.agent_runtime,
            runner_id=expert.runner,
            progress=self.progress,
            session_id=self.session_id,
            turn_id=self.turn_id,
            web_capability_factory=self.web_capability_factory,
        )
        context = ExpertRunContext(
            config=self.config,
            expert=expert,
            skills=self.skills,
            scope=scope,
            stages=stages,
        )
        request = ExpertRunRequest(expert_id=expert_id, task=task)

        async def invoke() -> ExpertResult:
            async with self._semaphore:
                async with self._concurrency_gate.enter(expert.concurrency_mode):
                    raw_result = await runner.run(request, context)
                    return ExpertResult.model_validate(raw_result)

        try:
            result = await asyncio.wait_for(
                invoke(),
                timeout=self.config.experts.timeout_seconds,
            )
        except TimeoutError:
            result = ExpertResult(
                status="blocked",
                final_content=(
                    f"ExpertSkill {expert_id!r} exceeded the "
                    f"{self.config.experts.timeout_seconds:g}s turn-local timeout."
                ),
                unresolved=("The expert timed out; no background work remains active.",),
            )
        except CapabilityUnavailable as exc:
            result = ExpertResult(
                status="blocked",
                final_content=str(exc),
                unresolved=(str(exc),),
            )
        except Exception as exc:
            result = ExpertResult(
                status="blocked",
                final_content=(f"ExpertSkill {expert_id!r} failed: {type(exc).__name__}: {exc}"),
                unresolved=(str(exc),),
            )
        accumulate_usage(self.usage, result.usage)
        return result

    def descriptors(self) -> tuple[dict[str, Any], ...]:
        return tuple(expert.descriptor() for expert in self._enabled.values())

    def _reserve(self, expert: ExpertSkillSnapshot) -> None:
        if self._calls >= self.config.experts.max_calls_per_turn:
            raise RuntimeError(
                "the Master exhausted the turn-wide ExpertSkill call budget "
                f"of {self.config.experts.max_calls_per_turn}"
            )
        used = self._calls_by_expert[expert.id]
        if used >= expert.max_calls_per_turn:
            raise RuntimeError(
                f"ExpertSkill {expert.id!r} exhausted its per-turn call budget "
                f"of {expert.max_calls_per_turn}"
            )
        self._calls += 1
        self._calls_by_expert[expert.id] += 1


class _ExpertConcurrencyGate:
    """Allow parallel-safe experts together while making exclusive experts solitary."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._parallel_runs = 0
        self._exclusive_running = False
        self._exclusive_waiters = 0

    def enter(self, mode: str) -> _ExpertConcurrencyLease:
        return _ExpertConcurrencyLease(self, exclusive=mode == "exclusive")

    async def _acquire(self, *, exclusive: bool) -> None:
        async with self._condition:
            if exclusive:
                self._exclusive_waiters += 1
                try:
                    await self._condition.wait_for(
                        lambda: not self._exclusive_running and self._parallel_runs == 0
                    )
                finally:
                    self._exclusive_waiters -= 1
                    self._condition.notify_all()
                self._exclusive_running = True
                return
            await self._condition.wait_for(
                lambda: not self._exclusive_running and self._exclusive_waiters == 0
            )
            self._parallel_runs += 1

    async def _release(self, *, exclusive: bool) -> None:
        async with self._condition:
            if exclusive:
                self._exclusive_running = False
            else:
                self._parallel_runs -= 1
            self._condition.notify_all()


class _ExpertConcurrencyLease:
    def __init__(self, gate: _ExpertConcurrencyGate, *, exclusive: bool) -> None:
        self.gate = gate
        self.exclusive = exclusive

    async def __aenter__(self) -> None:
        await self.gate._acquire(exclusive=self.exclusive)

    async def __aexit__(self, *_args: object) -> None:
        await self.gate._release(exclusive=self.exclusive)


__all__ = ["ExpertRuntime"]
