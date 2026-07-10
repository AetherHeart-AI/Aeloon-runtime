"""Thin orchestration layer around the state-machine runtime."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from aeloon_core.config import Config, UASMConfig
from aeloon_core.context import (
    append_user_message,
    apply_skill_guidance,
    build_initial_messages,
    refresh_initial_system_message,
)
from aeloon_core.context_compaction import CompactionResult, maybe_compact_messages
from aeloon_core.model_metadata import resolve_context_window
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
        self.skills = SkillRegistry.discover(config)
        workspace = config.workspace
        for tool in (
            ExecTool(workspace=workspace, timeout=config.tools.exec.timeout),
            ReadTool(workspace=workspace),
            WriteTool(workspace=workspace),
            EditTool(workspace=workspace),
            GlobTool(workspace=workspace),
            GrepTool(workspace=workspace),
            WebFetchTool(config=config.tools.web),
            WebSearchTool(config=config.tools.web),
        ):
            self.registry.register(tool)
        # Only expose the tool when there is something to advertise, matching the
        # guidance text (which lists described skills only).
        if self.skills.enabled and self.skills.described():
            self.registry.register(SkillTool(registry=self.skills))
        self.todo_tool = TodoWriteTool(data_dir=config.data_dir)
        self.registry.register(self.todo_tool)
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
                tools=self.registry,
                messages=messages,
                max_iterations=defaults.max_iterations,
                max_auto_continue_iterations=defaults.max_auto_continue_iterations,
                max_finalization_iterations=defaults.max_finalization_iterations,
                rule_engine_enabled=policy.rule_engine_enabled,
                temporary_guard_enabled=policy.temporary_guard_enabled,
                minimal_context_enabled=policy.minimal_context_enabled,
                transition_trace_enabled=policy.transition_trace_enabled,
                guard_decision_mode=policy.guard_decision_mode,
                minimal_context_recent_turns=policy.minimal_context_recent_turns,
                minimal_context_tool_result_chars=policy.minimal_context_tool_result_chars,
                session_id=actual_session_id,
                turn_id=turn_id,
                experiment_labels={"ablation_group": _ablation_group(policy)},
                on_transition=(
                    persist_transition if policy.transition_trace_enabled else None
                ),
                on_progress=on_progress,
                prepare_model_input=prepare_model_input,
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
        )


def _ablation_group(policy: UASMConfig) -> str:
    switches = (
        policy.rule_engine_enabled,
        policy.temporary_guard_enabled,
        policy.minimal_context_enabled,
    )
    return {
        (False, False, False): "A0",
        (True, False, False): "A1",
        (True, True, False): "A2",
        (True, True, True): "A3",
    }.get(switches, "custom")
