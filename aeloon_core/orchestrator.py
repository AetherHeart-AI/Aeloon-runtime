"""Thin orchestration layer around the standalone kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aeloon_core.config import Config
from aeloon_core.context import (
    append_user_message,
    apply_skill_guidance,
    build_initial_messages,
    refresh_initial_system_message,
)
from aeloon_core.context_compaction import maybe_compact_messages
from aeloon_core.kernel import run_agent_kernel
from aeloon_core.model_metadata import GENERIC_DEFAULT_MAX_TOKENS, ModelLimits, resolve_model_limits
from aeloon_core.providers.base import GenerationSettings
from aeloon_core.providers.custom_provider import CustomProvider
from aeloon_core.session import SessionStore
from aeloon_core.skills import SkillRegistry
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


class AeloonCoreOrchestrator:
    """Build messages, run the kernel, and persist turns."""

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
                max_tokens=defaults.max_tokens,
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
        if defaults.context_compaction.enabled:
            model_limits = await resolve_model_limits(defaults.model)
            compaction = await maybe_compact_messages(
                provider=self.provider,
                model=defaults.model,
                messages=messages,
                config=defaults.context_compaction,
                context_window_tokens=defaults.context_window_tokens,
                output_tokens=_output_token_budget(defaults.max_tokens, model_limits),
                model_limits=model_limits,
            )
            messages = compaction.messages
        final_content, tools_used, messages = await run_agent_kernel(
            provider=self.provider,
            model=defaults.model,
            tools=self.registry,
            messages=messages,
            max_iterations=defaults.max_iterations,
            max_auto_continue_iterations=defaults.max_auto_continue_iterations,
            max_finalization_iterations=defaults.max_finalization_iterations,
            on_progress=on_progress,
        )
        blocks = list(getattr(on_progress, "blocks", []) or [])
        self.sessions.append_turn(
            session_id=actual_session_id,
            user_prompt=prompt,
            final_content=final_content,
            tools_used=tools_used,
            messages=messages,
            blocks=blocks,
        )
        return TurnResult(
            session_id=actual_session_id,
            final_content=final_content,
            tools_used=tools_used,
            messages=messages,
            blocks=blocks,
        )


def _output_token_budget(
    configured_max_tokens: int | None,
    model_limits: ModelLimits | None,
) -> int:
    if configured_max_tokens is not None:
        return max(1, configured_max_tokens)
    if model_limits and model_limits.output_tokens is not None:
        return max(1, model_limits.output_tokens)
    return GENERIC_DEFAULT_MAX_TOKENS
