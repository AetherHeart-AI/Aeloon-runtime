"""Thin orchestration layer around the state-machine runtime."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from aeloon_core.config import Config
from aeloon_core.context import (
    append_user_message,
    apply_skill_guidance,
    build_initial_messages,
    refresh_initial_system_message,
)
from aeloon_core.context_compaction import CompactionResult, maybe_compact_messages
from aeloon_core.default_profile import DEFAULT_PROFILE_ID, load_default_profile
from aeloon_core.model_metadata import resolve_context_window
from aeloon_core.profile_artifacts import CompatibilityPolicy, ProfileArtifactStore
from aeloon_core.providers.base import GenerationSettings
from aeloon_core.providers.custom_provider import CustomProvider
from aeloon_core.session import SessionStore
from aeloon_core.skills import SkillRegistry
from aeloon_core.state_machine import run_agent_loop
from aeloon_core.tools.filesystem import EditTool, ReadTool, WriteTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.tools.search_grep import GlobTool, GrepTool
from aeloon_core.tools.shell import ExecTool
from aeloon_core.tools.skill import SkillTool
from aeloon_core.tools.todo import TodoWriteTool
from aeloon_core.tools.web import WebFetchTool, WebSearchTool


@dataclass
class TurnResult:
    """Result of one orchestrated agent turn."""

    session_id: str
    final_content: str | None
    tools_used: list[str]
    messages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    usage: dict[str, Any] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    status: str | None = None
    turn_id: str | None = None
    profile: dict[str, Any] | None = None


class AeloonCoreOrchestrator:
    """Build messages, run the state machine, and persist turns."""

    def __init__(self, config: Config) -> None:
        self.config = config
        defaults = config.agents.defaults
        provider_config = config.providers.custom
        self.provider = CustomProvider(
            api_key=provider_config.api_key,
            api_base=provider_config.api_base,
            default_model=defaults.model,
            extra_headers=provider_config.extra_headers,
            proxy=provider_config.proxy,
            generation=GenerationSettings(
                temperature=defaults.temperature,
                reasoning_effort=defaults.reasoning_effort,
            ),
            chat_timeout=defaults.chat_timeout,
        )
        self.registry = ToolRegistry()
        self.profile_registry = ToolRegistry()
        self.skills = SkillRegistry.discover(config)
        workspace = config.workspace
        self.todo_tool = TodoWriteTool(data_dir=config.data_dir)
        for registry, protected_paths in (
            (self.registry, ()),
            (self.profile_registry, (config.data_dir,)),
        ):
            for tool in (
                ExecTool(
                    workspace=workspace,
                    timeout=config.tools.exec.timeout,
                    denied_paths=protected_paths,
                ),
                ReadTool(workspace=workspace, denied_paths=protected_paths),
                WriteTool(workspace=workspace, denied_paths=protected_paths),
                EditTool(workspace=workspace, denied_paths=protected_paths),
                GlobTool(workspace=workspace, denied_paths=protected_paths),
                GrepTool(workspace=workspace, denied_paths=protected_paths),
                WebFetchTool(config=config.tools.web),
                WebSearchTool(config=config.tools.web),
            ):
                registry.register(tool)
            # Only expose the tool when there is something to advertise, matching the
            # guidance text (which lists described skills only).
            if self.skills.enabled and self.skills.described():
                registry.register(SkillTool(registry=self.skills))
            registry.register(self.todo_tool)
        self.profile_store = ProfileArtifactStore(
            data_dir=config.data_dir,
            compatibility=CompatibilityPolicy(
                tool_schema_fingerprints=_tool_schema_fingerprints(self.registry)
            ),
        )
        self.sessions = SessionStore(data_dir=config.data_dir, workspace=config.workspace)

    async def run_turn(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        on_progress: Any | None = None,
    ) -> TurnResult:
        """Run one prompt through the agent loop."""

        actual_session_id = session_id or self.sessions.new_session()
        self.todo_tool.set_session_id(actual_session_id)
        defaults = self.config.agents.defaults
        if defaults.profile_id == DEFAULT_PROFILE_ID:
            profile = await load_default_profile(
                self.profile_store,
                workspace=self.config.workspace,
            )
        elif defaults.profile_id is not None:
            profile = self.profile_store.load_active(defaults.profile_id)
        else:
            profile = None
        messages = self.sessions.load_messages(
            actual_session_id,
            initial_messages=build_initial_messages(workspace=self.config.workspace),
        )
        messages = refresh_initial_system_message(messages, workspace=self.config.workspace)
        messages = apply_skill_guidance(messages, self.skills.format_guidance())
        messages = append_user_message(messages, prompt)
        turn_id = str(getattr(on_progress, "turn_id", "") or uuid.uuid4().hex[:12])
        prepare_model_input = None
        if defaults.context_compaction.enabled:
            context_window_tokens = await resolve_context_window(defaults.model)
            context_window_tokens = context_window_tokens or defaults.context_window_tokens

            async def prepare_model_input(
                current_messages: list[dict[str, Any]],
                current_tools: list[dict[str, Any]],
                additional_messages: list[dict[str, Any]],
            ) -> CompactionResult:
                compaction = await maybe_compact_messages(
                    provider=self.provider,
                    model=defaults.model,
                    messages=current_messages,
                    tools=current_tools,
                    additional_messages=additional_messages,
                    config=defaults.context_compaction,
                    context_window_tokens=context_window_tokens,
                )
                usage_hook = getattr(on_progress, "on_usage", None)
                if compaction.usage and usage_hook is not None:
                    hook_result = usage_hook(
                        compaction.usage,
                        node_kind="context_processing",
                    )
                    if inspect.isawaitable(hook_result):
                        await hook_result
                return compaction

        policy = defaults.uasm
        trace_write_tail: asyncio.Task[None] | None = None
        trace_write_failed = False

        async def write_transition(
            record: dict[str, Any],
            previous: asyncio.Task[None] | None,
        ) -> None:
            nonlocal trace_write_failed
            if previous is not None:
                await previous
            if trace_write_failed:
                return
            try:
                await asyncio.to_thread(
                    self.sessions.append_transition,
                    session_id=actual_session_id,
                    turn_id=turn_id,
                    transition=record,
                )
            except OSError as exc:
                trace_write_failed = True
                logger.warning(
                    "Disabling transition persistence after trace write failed: {}",
                    exc,
                )

        def persist_transition(record: Any) -> None:
            nonlocal trace_write_tail
            trace_write_tail = asyncio.create_task(
                write_transition(record.to_dict(), trace_write_tail)
            )

        try:
            state = await run_agent_loop(
                provider=self.provider,
                model=defaults.model,
                tools=self.profile_registry if profile is not None else self.registry,
                messages=messages,
                max_iterations=defaults.max_iterations,
                max_auto_continue_iterations=defaults.max_auto_continue_iterations,
                max_finalization_iterations=defaults.max_finalization_iterations,
                transition_trace_enabled=policy.transition_trace_enabled,
                minimal_context_recent_turns=policy.minimal_context_recent_turns,
                minimal_context_tool_result_chars=policy.minimal_context_tool_result_chars,
                session_id=actual_session_id,
                turn_id=turn_id,
                on_transition=(
                    persist_transition if policy.transition_trace_enabled else None
                ),
                on_progress=on_progress,
                prepare_model_input=prepare_model_input,
                profile=profile,
                max_handoffs=defaults.max_handoffs,
            )
        except BaseException:
            if trace_write_tail is not None:
                trace_write_tail.cancel()
                await asyncio.gather(trace_write_tail, return_exceptions=True)
            raise
        if trace_write_tail is not None:
            await trace_write_tail
        final_content = state.metadata.final_content
        tools_used = state.tools_used
        messages = state.messages
        usage = state.token_ledger.to_dict()
        transitions = [record.to_dict() for record in state.transitions]
        status = state.metadata.status.value
        profile_ref = state.profile_ref.to_dict() if state.profile_ref is not None else None
        blocks = list(getattr(on_progress, "blocks", []) or [])
        self.sessions.append_turn(
            session_id=actual_session_id,
            user_prompt=prompt,
            final_content=final_content,
            tools_used=tools_used,
            messages=messages,
            blocks=blocks,
            usage=usage,
            turn_id=turn_id,
            profile=profile_ref,
        )
        return TurnResult(
            session_id=actual_session_id,
            final_content=final_content,
            tools_used=tools_used,
            messages=messages,
            blocks=blocks,
            usage=usage,
            transitions=transitions,
            status=status,
            turn_id=turn_id,
            profile=profile_ref,
        )


def _tool_schema_fingerprints(registry: ToolRegistry) -> dict[str, str]:
    """Hash canonical host schemas for artifact compatibility checks."""

    fingerprints: dict[str, str] = {}
    for definition in registry.get_definitions():
        function = definition.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str):
            continue
        payload = json.dumps(
            definition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        fingerprints[name] = hashlib.sha256(payload).hexdigest()
    return fingerprints
