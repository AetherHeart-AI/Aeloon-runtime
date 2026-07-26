"""Conversation-scoped Master orchestration powered by Pydantic AI Harness."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from aeloon_core.config import Config
from aeloon_core.context import SYSTEM_PROMPT
from aeloon_core.customization.catalog import Catalog
from aeloon_core.harness import (
    AgentRunSpec,
    AgentRunStatus,
    CapabilityManifest,
    HarnessAgentRuntime,
    ModelRouter,
    RoleAgentFactory,
    WorkflowRunner,
    deserialize_messages,
    master_harness_capabilities,
    serialize_messages,
    workflow_tools,
)
from aeloon_core.master_prompt import MASTER_USER_REQUEST_MARKER, master_system_prompt
from aeloon_core.session import SessionStore
from aeloon_core.tools.filesystem import ReadTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.tools.search_grep import GlobTool, GrepTool, ListTool


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
    """Own the Master conversation and its in-turn Harness child agents."""

    def __init__(
        self,
        config: Config,
        *,
        model: Model | None = None,
        model_settings: ModelSettings | None = None,
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
        self.catalog = Catalog.discover(config.workspace)
        self.worker_types = self.catalog.roles
        self.sessions = SessionStore(data_dir=config.data_dir)
        self.master_observation_tools = self._build_master_observation_tools()

    @property
    def model(self) -> Model:
        """Return the currently resolved Master model."""

        return self.model_router.resolve_master().model

    @model.setter
    def model(self, value: Model) -> None:
        self.model_router.set_injected_model(value, settings=self.model_settings)

    def _build_master_observation_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        protected = (self.config.data_dir,)
        for tool in (
            ListTool(workspace=self.config.workspace, denied_paths=protected),
            ReadTool(workspace=self.config.workspace, denied_paths=protected),
            GlobTool(workspace=self.config.workspace, denied_paths=protected),
            GrepTool(workspace=self.config.workspace, denied_paths=protected),
        ):
            registry.register(tool)
        return registry

    async def run_turn(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        on_progress: Any | None = None,
    ) -> TurnResult:
        """Run one user request and finish every child agent before returning."""

        actual_session_id = session_id or self.sessions.new_session()
        self.sessions.session_path(actual_session_id)
        turn_id = str(getattr(on_progress, "turn_id", "") or uuid.uuid4().hex[:12])
        stored_messages = await asyncio.to_thread(
            self.sessions.load_pydantic_messages,
            actual_session_id,
        )
        history = deserialize_messages(stored_messages)
        worker_types = [snapshot.descriptor() for snapshot in self.worker_types.list()]
        template_config = self.config.agents.templates
        workflow_candidates = (
            list(
                self.catalog.workflows.search(
                    prompt,
                    limit=template_config.presearch_limit,
                )
            )
            if template_config.enabled
            else []
        )
        instructions = (
            SYSTEM_PROMPT.strip()
            + f"\n\nWorkspace: {self.config.workspace}\n\n"
            + master_system_prompt(
                worker_types=worker_types,
                workflow_candidates=workflow_candidates,
                workflow_templates_enabled=template_config.enabled,
                worker_request_limit=self.config.agents.harness.sub_agent_request_limit,
                max_worker_continuations=(
                    self.config.agents.harness.max_worker_continuations
                ),
            )
        )
        tools = ToolRegistry()
        for name in ("list", "read", "glob", "grep"):
            tool = self.master_observation_tools.get(name)
            assert tool is not None
            tools.register(tool)
        role_factory = RoleAgentFactory(
            config=self.config,
            model_router=self.model_router,
            roles=self.worker_types,
            progress=on_progress,
            session_id=actual_session_id,
            turn_id=turn_id,
        )
        if template_config.enabled:
            workflow_runner = WorkflowRunner(
                config=self.config,
                roles=self.worker_types,
                role_factory=role_factory,
            )
            for tool in workflow_tools(
                config=self.config,
                roles=self.worker_types,
                workflows=self.catalog.workflows,
                runner=workflow_runner,
            ):
                tools.register(tool)

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
                capabilities=master_harness_capabilities(
                    config=self.config,
                    model_router=self.model_router,
                    worker_types=self.worker_types,
                    role_factory=role_factory,
                ),
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

        messages = serialize_messages(outcome.messages)
        blocks = list(getattr(on_progress, "blocks", []) or [])
        result = TurnResult(
            session_id=actual_session_id,
            final_content=outcome.output,
            tools_used=list(outcome.tools_used),
            messages=messages,
            blocks=blocks,
            usage=outcome.usage,
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
