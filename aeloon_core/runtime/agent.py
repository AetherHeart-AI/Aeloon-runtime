"""Session-aware runtime adapter around the stateless core run API."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from aeloon_core.blocking import run_blocking
from aeloon_core.config import Config
from aeloon_core.core import (
    AgentMessage,
    ImageContent,
    InferencePort,
    Model,
    RunController,
    RunEvent,
    RunRequest,
    RunResult,
    StreamOptions,
    TextContent,
    Usage,
    UserMessage,
    message_from_dict,
    run_agent,
)
from aeloon_core.core.compaction import ContextPolicy, ContextUpdate, effective_context_policy
from aeloon_core.runtime.attachments import AttachmentAccessCallback, ResolvedAttachment
from aeloon_core.runtime.compaction import (
    CompactionResult,
    CompactionSettings,
    compact_preparation,
    prepare_compaction,
    summarize_branch,
)
from aeloon_core.runtime.prompt import build_system_prompt
from aeloon_core.runtime.providers import ProviderManager
from aeloon_core.runtime.resources import ResourceLoader
from aeloon_core.runtime.session import Session
from aeloon_core.runtime.tooling import RuntimeToolSet

RunObserver = Callable[[RunEvent], Awaitable[None] | None]


class SessionContextCompactor:
    """Implement core's compactor port using a durable runtime Session."""

    def __init__(
        self,
        *,
        session: Session | None,
        inference: InferencePort,
        model: Model,
        options: StreamOptions,
        settings: CompactionSettings,
    ) -> None:
        self.session = session
        self.inference = inference
        self.model = model
        self.options = options
        self.settings = settings
        self._cancellation = asyncio.Event()
        self._overflow_compactions = 0

    def cancel(self) -> None:
        self._cancellation.set()

    async def compact(
        self,
        _messages: tuple[AgentMessage, ...],
        *,
        reason: str,
    ) -> ContextUpdate:
        result, entry_id = await self.compact_session(reason=reason)
        context = await self.session.build_context() if self.session is not None else None
        return ContextUpdate(
            messages=context.messages if context is not None else (),
            summary=result.summary,
            tokens_before=result.tokens_before,
            first_kept_id=result.first_kept_entry_id,
            usage=result.usage,
            details={**result.details, "compactionEntryId": entry_id},
            compaction_boundary_ms=(
                context.compaction_boundary_ms if context is not None else None
            ),
            compaction_boundary_index=(
                context.compaction_boundary_index if context is not None else None
            ),
        )

    async def compact_session(
        self,
        *,
        reason: str,
        custom_instructions: str | None = None,
    ) -> tuple[CompactionResult, str]:
        settings = self.settings
        if reason == "overflow":
            divisor = 2**self._overflow_compactions
            settings = CompactionSettings(
                enabled=settings.enabled,
                reserve_tokens=settings.reserve_tokens,
                keep_recent_tokens=max(1, settings.keep_recent_tokens // divisor),
            )
            self._overflow_compactions += 1
        preparation = await prepare_compaction(
            self.session,
            settings,
            force=reason == "overflow",
        )
        if preparation is None:
            raise RuntimeError("Session does not need compaction")
        result = await compact_preparation(
            preparation,
            inference=self.inference,
            model=self.model,
            stream_options=self.options,
            settings=settings,
            custom_instructions=custom_instructions,
            cancellation_event=self._cancellation,
        )
        entry_id = await self.session.append_compaction(
            summary=result.summary,
            tokens_before=result.tokens_before,
            first_kept_entry_id=result.first_kept_entry_id,
            usage=result.usage,
            details={**result.details, "reason": reason},
        )
        return result, entry_id


class SessionAgent:
    """Runtime-owned operation object; one instance is used by one operation."""

    def __init__(
        self,
        *,
        config: Config,
        session: Session | None,
        provider_manager: ProviderManager,
        active_tool_names: tuple[str, ...] | None = None,
        attachments: tuple[ResolvedAttachment, ...] = (),
        on_attachment_access: AttachmentAccessCallback | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.provider_manager = provider_manager
        self._configured_active_tools = active_tool_names
        self.inference: InferencePort | None = None
        self.model: Model | None = None
        self.controller: RunController | None = None
        self._observers: list[RunObserver] = []
        self._resources = None
        self._resource_loader_instance: ResourceLoader | None = None
        self._pending_next_turn: list[UserMessage] = []
        self._tool_set = RuntimeToolSet(
            self.config.workspace,
            shell_path=self.config.tools.shell_path,
            auto_resize_images=self.config.tools.auto_resize_images,
            attachments=attachments,
            on_attachment_access=on_attachment_access,
        )

    def subscribe(self, observer: RunObserver) -> Callable[[], None]:
        self._observers.append(observer)

        def remove() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return remove

    async def prepare(self) -> None:
        if self.model is not None:
            return
        self.model = await self.provider_manager.model(self.config.agent.model)
        self.inference = self.provider_manager.inference(self.model)
        self._resource_loader_instance = self._resource_loader()
        self._resources = await run_blocking(self._resource_loader_instance.reload)

    async def prompt(
        self,
        text: str,
        *,
        images: Sequence[ImageContent] = (),
        run_id: str,
    ) -> RunResult:
        await self.prepare()
        assert self.inference is not None and self.model is not None and self._resources is not None
        context = await self.session.build_context() if self.session is not None else None
        content = text if not images else (TextContent(text), *tuple(images))
        active_tool_names = self._tool_set.active_names(
            self._configured_active_tools,
            context.active_tool_names if context is not None else None,
        )
        context_policy = effective_context_policy(
            ContextPolicy(
                enabled=self.config.agent.compaction.enabled,
                reserve_tokens=self.config.agent.compaction.reserve_tokens,
                keep_recent_tokens=self.config.agent.compaction.keep_recent_tokens,
            ),
            self.model.context_window,
        )
        request = RunRequest(
            run_id=run_id,
            inference=self.inference,
            model=self.model,
            messages=context.messages if context is not None else (),
            input=(*self._pending_next_turn, UserMessage(content)),
            system_prompt=build_system_prompt(
                cwd=str(self.config.workspace),
                tools=tuple(self._tool_set.by_name[name] for name in active_tool_names),
                resources=self._resources,
            ),
            tools=self._tool_set.tools,
            active_tool_names=active_tool_names,
            stream_options=self._stream_options(),
            context_policy=context_policy,
            steering_mode=self.config.agent.steering_mode,
            follow_up_mode=self.config.agent.follow_up_mode,
            context_id=self.session.id if self.session is not None else run_id,
            compaction_boundary_ms=(
                context.compaction_boundary_ms if context is not None else None
            ),
            compaction_boundary_index=(
                context.compaction_boundary_index if context is not None else None
            ),
        )
        self._pending_next_turn.clear()
        self.controller = RunController(
            steering_mode=self.config.agent.steering_mode,
            follow_up_mode=self.config.agent.follow_up_mode,
        )
        compactor = (
            SessionContextCompactor(
                session=self.session,
                inference=self.inference,
                model=self.model,
                options=request.stream_options,
                settings=CompactionSettings(
                    enabled=context_policy.enabled,
                    reserve_tokens=context_policy.reserve_tokens,
                    keep_recent_tokens=context_policy.keep_recent_tokens,
                ),
            )
            if self.session is not None
            else None
        )
        try:
            return await run_agent(
                request,
                controller=self.controller,
                emit=self._emit,
                compactor=compactor,
            )
        finally:
            self.controller = None

    async def skill(
        self,
        name: str,
        additional_instructions: str | None,
        *,
        images: Sequence[ImageContent] = (),
        run_id: str,
    ) -> RunResult:
        await self.prepare()
        assert self._resources is not None and self._resource_loader_instance is not None
        try:
            skill = await run_blocking(self._resource_loader_instance.load_skill, name)
        except KeyError:
            raise ValueError(f"Unknown skill: {name}") from None
        prompt = (
            f'<skill name="{skill.name}" location="{skill.file_path}">\n{skill.content}\n</skill>'
        )
        if additional_instructions:
            prompt += f"\n\n{additional_instructions}"
        return await self.prompt(prompt, images=images, run_id=run_id)

    async def prompt_template(
        self,
        name: str,
        arguments: Sequence[str],
        *,
        run_id: str,
    ) -> RunResult:
        await self.prepare()
        assert self._resources is not None
        template = next(
            (item for item in self._resources.prompt_templates if item.name == name),
            None,
        )
        if template is None:
            raise ValueError(f"Unknown prompt template: {name}")
        return await self.prompt(template.format(arguments), run_id=run_id)

    async def next_turn(self, text: str) -> None:
        self._pending_next_turn.append(UserMessage(text))

    async def steer(self, text: str) -> None:
        if self.controller is None:
            raise RuntimeError("Run is not active")
        await self.controller.steer(text)

    async def follow_up(self, text: str) -> None:
        if self.controller is None:
            raise RuntimeError("Run is not active")
        await self.controller.follow_up(text)

    async def abort(self) -> dict[str, list[dict[str, Any]]]:
        if self.controller is None:
            raise RuntimeError("Run is not active")
        return await self.controller.cancel()

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        await self.prepare()
        assert self.inference is not None and self.model is not None
        if self.session is None:
            raise RuntimeError("Compaction requires a persistent session")
        context_policy = effective_context_policy(
            ContextPolicy(
                enabled=True,
                reserve_tokens=self.config.agent.compaction.reserve_tokens,
                keep_recent_tokens=self.config.agent.compaction.keep_recent_tokens,
            ),
            self.model.context_window,
        )
        compactor = SessionContextCompactor(
            session=self.session,
            inference=self.inference,
            model=self.model,
            options=self._stream_options(),
            settings=CompactionSettings(
                enabled=context_policy.enabled,
                reserve_tokens=context_policy.reserve_tokens,
                keep_recent_tokens=context_policy.keep_recent_tokens,
            ),
        )
        result, _ = await compactor.compact_session(
            reason="explicit",
            custom_instructions=custom_instructions,
        )
        return result

    async def navigate_tree(
        self,
        target_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
        label: str | None = None,
    ) -> dict[str, Any]:
        await self.prepare()
        assert self.inference is not None and self.model is not None
        if self.session is None:
            raise RuntimeError("Tree navigation requires a persistent session")
        if await self.session.get_entry(target_id) is None:
            raise ValueError(f"Entry {target_id} not found")
        old_leaf = await self.session.get_leaf_id()
        old_branch = await self.session.get_branch(old_leaf)
        target_branch = await self.session.get_branch(target_id)
        target_ids = {entry["id"] for entry in target_branch}
        abandoned = [entry for entry in old_branch if entry["id"] not in target_ids]
        summary_entry: str | None = None
        summary_payload: tuple[str, Usage, Any] | None = None
        if summarize and abandoned:
            messages = tuple(
                message_from_dict(entry["message"])
                for entry in abandoned
                if entry.get("type") == "message"
            )
            summary_payload = await summarize_branch(
                messages,
                inference=self.inference,
                model=self.model,
                stream_options=self._stream_options(),
                custom_instructions=custom_instructions,
                replace_instructions=replace_instructions,
            )
        await self.session.set_leaf_id(target_id)
        if summary_payload is not None:
            summary, usage, details = summary_payload
            summary_entry = await self.session.append_branch_summary(
                from_id=str(old_leaf or target_id),
                summary=summary,
                usage=usage,
                details=details,
            )
        if label:
            await self.session.set_label(summary_entry or target_id, label)
        return {
            "cancelled": False,
            "oldLeafId": old_leaf,
            "newLeafId": await self.session.get_leaf_id(),
            "summaryEntryId": summary_entry,
        }

    async def close(self) -> None:
        await self.provider_manager.close()
        self.inference = None

    async def _emit(self, event: RunEvent) -> None:
        if (
            self.session is not None
            and event.type == "message_end"
            and not event.data.get("transient")
        ):
            raw = event.data.get("message")
            if isinstance(raw, dict):
                await self.session.append_message(message_from_dict(raw))
        for observer in tuple(self._observers):
            result = observer(event)
            if inspect.isawaitable(result):
                await result

    async def _summarization_retry(self, data: dict[str, Any]) -> None:
        event_type = "auto_retry_start" if data.get("stage") == "start" else "auto_retry_end"
        await self._emit(RunEvent(event_type, data))  # type: ignore[arg-type]

    def _stream_options(self) -> StreamOptions:
        retry = self.config.agent.retry
        return StreamOptions(
            timeout_ms=self.config.agent.timeout_ms,
            max_tokens=self.config.agent.max_tokens,
            temperature=self.config.agent.temperature,
            thinking_level=self.config.agent.thinking_level,
            max_retries=retry.max_retries if retry.enabled else 0,
            base_delay_ms=retry.base_delay_ms,
            max_retry_delay_ms=retry.max_retry_delay_ms,
            metadata={"on_retry": self._summarization_retry},
        )

    def _resource_loader(self) -> ResourceLoader:
        return ResourceLoader(
            cwd=Path(self.config.workspace),
            agent_dir=self.config.data_dir,
            additional_roots=tuple(self.config.resources.roots),
            enabled_skills=(
                tuple(self.config.resources.enabled_skills)
                if self.config.resources.enabled_skills is not None
                else None
            ),
            no_skills=self.config.resources.no_skills,
            no_prompt_templates=self.config.resources.no_prompt_templates,
            no_context_files=self.config.resources.no_context_files,
        )


SessionAgentFactory = Callable[
    [Config, Session | None, ProviderManager],
    SessionAgent | Awaitable[SessionAgent],
]

__all__ = [
    "SessionAgent",
    "SessionAgentFactory",
    "SessionContextCompactor",
]
