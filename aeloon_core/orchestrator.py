"""Conversation-scoped Ultra Master with isolated ExpertSkills."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from aeloon_core.config import Config
from aeloon_core.conversation import SessionStore
from aeloon_core.harness.agent.prompt import (
    MASTER_USER_REQUEST_MARKER,
    SYSTEM_PROMPT,
    master_system_prompt,
)
from aeloon_core.harness.capabilities import (
    WebCapabilityFactory,
    master_capabilities,
)
from aeloon_core.harness.execution import (
    AgentRunSpec,
    AgentRunStatus,
    CapabilityManifest,
    HarnessAgentRuntime,
    accumulate_usage,
    deserialize_messages,
    serialize_messages,
)
from aeloon_core.harness.expert import (
    ExpertRunnerRegistry,
    ExpertRuntime,
    expert_run_tool,
)
from aeloon_core.harness.model import ModelRouter
from aeloon_core.harness.skill import SkillRegistry, skill_tools
from aeloon_core.harness.tool import ToolRegistry


@dataclass
class TurnResult:
    """Result of one Master turn."""

    session_id: str
    final_content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    status: str | None = None
    turn_id: str | None = None


class AeloonCoreOrchestrator:
    """Own the Master conversation and all current-turn ExpertSkill work."""

    def __init__(
        self,
        config: Config,
        *,
        model: Model | None = None,
        model_settings: ModelSettings | None = None,
        web_capability_factory: WebCapabilityFactory | None = None,
    ) -> None:
        self.config = config
        self.model_router = ModelRouter(
            config,
            injected_model=model,
            injected_settings=model_settings,
        )
        master_binding = self.model_router.resolve_master()
        self.model_settings = dict(master_binding.settings)
        self.agent_runtime = HarnessAgentRuntime()
        self.skills = SkillRegistry.discover(config)
        self.runners = ExpertRunnerRegistry.discover(config.workspace)
        for expert in self.skills.enabled_experts(config.experts.enabled):
            self.runners.require(expert.runner)
        self.experts = self.skills.enabled_experts(config.experts.enabled)
        self.master_skill_scope = self.skills.master_scope(config)
        self.sessions = SessionStore(data_dir=config.data_dir)
        self.web_capability_factory = web_capability_factory

    @property
    def model(self) -> Model:
        """Return the currently resolved Master model."""

        return self.model_router.resolve_master().model

    @model.setter
    def model(self, value: Model) -> None:
        self.model_router.set_injected_model(value, settings=self.model_settings)

    async def run_turn(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        on_progress: Any | None = None,
    ) -> TurnResult:
        """Run one Master request and finish every ExpertSkill before returning."""

        actual_session_id = session_id or self.sessions.new_session()
        self.sessions.session_path(actual_session_id)
        turn_id = str(getattr(on_progress, "turn_id", "") or uuid.uuid4().hex[:12])
        stored_messages = await asyncio.to_thread(
            self.sessions.load_pydantic_messages,
            actual_session_id,
        )
        history = deserialize_messages(stored_messages)
        expert_runtime = ExpertRuntime(
            config=self.config,
            skills=self.skills,
            runners=self.runners,
            model_router=self.model_router,
            agent_runtime=self.agent_runtime,
            progress=on_progress,
            session_id=actual_session_id,
            turn_id=turn_id,
            web_capability_factory=self.web_capability_factory,
        )
        instructions = (
            SYSTEM_PROMPT.strip()
            + f"\n\nWorkspace: {self.config.workspace}\n\n"
            + master_system_prompt(
                expert_descriptors=list(expert_runtime.descriptors()),
                plain_skill_ids=sorted(
                    self.master_skill_scope.skill_ids - {expert.id for expert in self.experts}
                ),
            )
        )
        tools = ToolRegistry()
        for tool in skill_tools(
            registry=self.skills,
            scope=self.master_skill_scope,
        ):
            tools.register(tool)
        tools.register(expert_run_tool(expert_runtime))

        defaults = self.config.agents.defaults
        policy = defaults.runtime
        binding = self.model_router.resolve_master()
        outcome = await self.agent_runtime.run(
            AgentRunSpec(
                role="master",
                model=binding.model,
                model_settings=binding.settings,
                instructions=instructions,
                prompt=f"{MASTER_USER_REQUEST_MARKER}{prompt}",
                history=history,
                tools=tools,
                output_type=str,
                terminal_models={},
                capability_manifest=CapabilityManifest.from_registry(
                    tools,
                    namespace="master",
                ),
                request_limit=defaults.max_iterations,
                max_output_tokens=defaults.max_output_tokens,
                transition_trace_enabled=policy.transition_trace_enabled,
                stuck_detection_enabled=policy.stuck_detection_enabled,
                stuck_detection_threshold=policy.stuck_detection_threshold,
                session_id=actual_session_id,
                turn_id=turn_id,
                progress=on_progress,
                capabilities=master_capabilities(self.config),
                prompt_cache=binding.prompt_cache,
            )
        )
        if outcome.status is not AgentRunStatus.COMPLETED or not isinstance(
            outcome.output, str
        ):
            raise RuntimeError(
                "Master did not produce a final response: "
                + (outcome.failure or outcome.status.value)
            )

        usage = dict(outcome.usage)
        accumulate_usage(usage, expert_runtime.usage)
        messages = serialize_messages(outcome.messages)
        blocks = list(getattr(on_progress, "blocks", []) or [])
        result = TurnResult(
            session_id=actual_session_id,
            final_content=outcome.output,
            tools_used=list(outcome.tools_used),
            messages=messages,
            blocks=blocks,
            usage=usage,
            duration_ms=getattr(on_progress, "duration_ms", None),
            transitions=[record.to_dict() for record in outcome.transitions],
            status=outcome.status.value,
            turn_id=turn_id,
        )
        await asyncio.to_thread(
            self.sessions.append_turn_once,
            session_id=actual_session_id,
            user_prompt=prompt,
            final_content=result.final_content,
            tools_used=result.tools_used,
            messages=result.messages,
            blocks=result.blocks,
            usage=result.usage,
            duration_ms=result.duration_ms,
            turn_id=turn_id,
        )
        return result

    async def close(self) -> None:
        """Close the model clients owned by this orchestrator."""

        await self.model_router.close()


__all__ = ["AeloonCoreOrchestrator", "TurnResult"]
