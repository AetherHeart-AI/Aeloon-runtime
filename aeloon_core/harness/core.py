"""Stateful Pi-compatible Python agent harness."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any, Literal

from aeloon_core.harness.compaction import (
    CompactionResult,
    CompactionSettings,
    compact_preparation,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
    summarize_branch,
)
from aeloon_core.harness.events import HarnessEventDispatcher
from aeloon_core.harness.input_queue import TurnInputQueues
from aeloon_core.harness.prompt import build_system_prompt
from aeloon_core.harness.provider_runtime import ProviderRuntime, model_to_dict
from aeloon_core.harness.resources import ResourceLoader
from aeloon_core.harness.session import Session
from aeloon_core.harness.tool_runtime import ToolConfigurationChange, ToolRuntime
from aeloon_core.harness.tools import DEFAULT_ACTIVE_TOOLS, create_all_tools
from aeloon_core.harness.transcript import ConversationTranscript
from aeloon_core.harness.types import (
    AgentMessage,
    AgentTool,
    AssistantMessage,
    EventListener,
    HarnessError,
    HarnessEventType,
    HookHandler,
    ImageContent,
    Model,
    Provider,
    QueueMode,
    Resources,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    ToolResultMessage,
    Usage,
    UserMessage,
    content_to_dict,
    message_from_dict,
    message_to_dict,
)

AgentHarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]


class AgentHarness:
    """Public facade that coordinates the harness runtime components."""

    def __init__(
        self,
        *,
        provider: Provider,
        model: Model,
        cwd: str,
        session: Session | None = None,
        tools: Iterable[AgentTool] | None = None,
        active_tool_names: Sequence[str] = DEFAULT_ACTIVE_TOOLS,
        resources: Resources | None = None,
        resource_loader: ResourceLoader | None = None,
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel = "off",
        stream_options: StreamOptions | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        compaction: CompactionSettings | None = None,
        shell_path: str | None = None,
        auto_resize_images: bool = True,
    ) -> None:
        self._model = model
        self.cwd = cwd
        default_tools = create_all_tools(
            cwd,
            shell_path=shell_path,
            auto_resize_images=auto_resize_images,
        )
        configured = list(tools) if tools is not None else list(default_tools.values())
        self._events = HarnessEventDispatcher()
        self._tool_runtime = ToolRuntime(configured, active_tool_names, self._events)
        self._provider_runtime = ProviderRuntime(provider, self._events)
        self._transcript = ConversationTranscript(session, self._events)
        self._queues = TurnInputQueues(
            self._events,
            steering_mode=steering_mode,
            follow_up_mode=follow_up_mode,
        )
        self.resource_loader = resource_loader
        self._resources = (
            resources
            if resources is not None
            else resource_loader.reload()
            if resource_loader is not None
            else Resources()
        )
        self._custom_system_prompt = system_prompt
        self._thinking_level = thinking_level
        base_options = stream_options or StreamOptions()
        self._stream_options = replace(base_options, thinking_level=thinking_level)
        self.compaction_settings = compaction or CompactionSettings()
        self._phase: AgentHarnessPhase = "idle"
        self._run_task: asyncio.Task[Any] | None = None
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._abort_requested = False

    @property
    def phase(self) -> AgentHarnessPhase:
        return self._phase

    @property
    def is_streaming(self) -> bool:
        return self._phase in {"turn", "retry"}

    @property
    def is_idle(self) -> bool:
        return self._phase == "idle"

    @property
    def model(self) -> Model:
        return self._model

    @property
    def provider(self) -> Provider:
        return self._provider_runtime.provider

    @provider.setter
    def provider(self, provider: Provider) -> None:
        self._provider_runtime.provider = provider

    @property
    def session(self) -> Session | None:
        return self._transcript.session

    @session.setter
    def session(self, session: Session | None) -> None:
        self._transcript.session = session

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return self._tool_runtime.tools

    @property
    def active_tools(self) -> tuple[AgentTool, ...]:
        return self._tool_runtime.active_tools

    @property
    def active_tool_names(self) -> tuple[str, ...]:
        return self._tool_runtime.active_names

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def stream_options(self) -> StreamOptions:
        return self._stream_options

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return self._transcript.messages

    @property
    def steering_mode(self) -> QueueMode:
        return self._queues.steering_mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._queues.follow_up_mode

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(
            cwd=self.cwd,
            tools=self.active_tools,
            resources=self._resources,
            custom_prompt=self._custom_system_prompt,
        )

    def subscribe(self, listener: EventListener) -> Any:
        return self._events.subscribe(listener)

    def on(self, event_type: str, handler: HookHandler) -> Any:
        return self._events.on(event_type, handler)

    async def prompt(
        self,
        text: str,
        *,
        images: Sequence[ImageContent] = (),
    ) -> AssistantMessage:
        if self._phase != "idle":
            raise HarnessError("busy", f"Harness is busy ({self._phase})")
        if not text.strip():
            raise HarnessError("invalid_argument", "Prompt must not be empty")
        self._run_task = asyncio.current_task()
        self._idle_event.clear()
        self._abort_requested = False
        self._set_phase("turn")
        try:
            await self._restore_context()
            await self._reload_resources()
            prompt_message = UserMessage(
                text if not images else (TextContent(text), *tuple(images))
            )
            initial = [*self._queues.take_next_turn(), prompt_message]
            hook = await self._hook(
                "before_agent_start",
                {
                    "prompt": text,
                    "images": [content_to_dict(image) for image in images],
                    "systemPrompt": self.system_prompt,
                    "resources": _resources_dict(self._resources),
                },
            )
            if "messages" in hook:
                initial = [
                    item if not isinstance(item, dict) else message_from_dict(item)
                    for item in hook["messages"]
                ]
            effective_system_prompt = str(hook.get("systemPrompt") or self.system_prompt)
            response = await self._run_loop(initial, effective_system_prompt)
            if self.session is not None and response.stop_reason not in {"error", "aborted"}:
                await self._auto_compact_if_needed()
            return response
        finally:
            self._run_task = None
            self._set_phase("idle")
            self._idle_event.set()
            await self._emit("settled", {"nextTurnCount": self._queues.next_turn_count})

    async def skill(
        self,
        name: str,
        additional_instructions: str | None = None,
    ) -> AssistantMessage:
        self._require_idle("invoke a skill")
        await self._reload_resources()
        skill = next((item for item in self._resources.skills if item.name == name), None)
        if skill is None:
            raise HarnessError("invalid_argument", f"Unknown skill: {name}")
        prompt = (
            f'<skill name="{skill.name}" location="{skill.file_path}">\n{skill.content}\n</skill>'
        )
        if additional_instructions:
            prompt += f"\n\n{additional_instructions}"
        return await self.prompt(prompt)

    async def prompt_from_template(
        self,
        name: str,
        args: Sequence[str] = (),
    ) -> AssistantMessage:
        self._require_idle("invoke a prompt template")
        await self._reload_resources()
        template = next(
            (item for item in self._resources.prompt_templates if item.name == name), None
        )
        if template is None:
            raise HarnessError("invalid_argument", f"Unknown prompt template: {name}")
        return await self.prompt(template.format(args))

    async def steer(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        if self._phase not in {"turn", "retry"}:
            raise HarnessError("invalid_state", "steer() requires an active turn")
        await self._queues.enqueue("steer", text, images)

    async def follow_up(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        if self._phase not in {"turn", "retry"}:
            raise HarnessError("invalid_state", "follow_up() requires an active turn")
        await self._queues.enqueue("follow_up", text, images)

    async def next_turn(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        await self._queues.enqueue("next_turn", text, images)

    async def append_message(self, message: AgentMessage) -> None:
        if self._phase != "idle":
            raise HarnessError("busy", "Cannot append a message while the harness is busy")
        await self._append_message(message, emit_events=False)

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        if self._phase != "idle":
            raise HarnessError("busy", f"Harness is busy ({self._phase})")
        if self.session is None:
            raise HarnessError("compaction", "Compaction requires a persistent session")
        self._idle_event.clear()
        self._set_phase("compaction")
        try:
            try:
                return await self._compact_session(custom_instructions, reason="manual")
            except Exception as exc:
                await self._emit(
                    "compaction_end",
                    {
                        "reason": "manual",
                        "aborted": isinstance(exc, asyncio.CancelledError),
                        "willRetry": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if isinstance(exc, HarnessError):
                    raise
                raise HarnessError("compaction", f"Compaction failed: {exc}", cause=exc) from exc
        finally:
            self._set_phase("idle")
            self._idle_event.set()
            await self._emit("settled", {"operation": "compaction"})

    async def navigate_tree(
        self,
        target_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
        label: str | None = None,
    ) -> dict[str, Any]:
        if self._phase != "idle":
            raise HarnessError("busy", f"Harness is busy ({self._phase})")
        if self.session is None:
            raise HarnessError("session", "Tree navigation requires a persistent session")
        if await self.session.get_entry(target_id) is None:
            raise HarnessError("session", f"Entry {target_id} not found")
        self._idle_event.clear()
        self._set_phase("branch_summary")
        old_leaf = await self.session.get_leaf_id()
        try:
            old_branch = await self.session.get_branch(old_leaf)
            target_branch = await self.session.get_branch(target_id)
            target_ids = {entry["id"] for entry in target_branch}
            abandoned = [entry for entry in old_branch if entry["id"] not in target_ids]
            preparation = {
                "oldLeafId": old_leaf,
                "targetId": target_id,
                "abandonedEntryIds": [entry["id"] for entry in abandoned],
            }
            hook = await self._hook("session_before_tree", {"preparation": preparation})
            if hook.get("cancel"):
                return {"cancelled": True}
            summary_entry: str | None = None
            summary_payload: tuple[str, Usage, Any, bool] | None = None
            if summarize and abandoned:
                messages = tuple(
                    message_from_dict(entry["message"])
                    for entry in abandoned
                    if entry.get("type") == "message"
                )
                if hook.get("summary"):
                    summary_data = hook["summary"]
                    summary = str(summary_data["summary"])
                    usage = Usage.from_dict(summary_data.get("usage"))
                    details = summary_data.get("details")
                    from_hook = True
                else:
                    summary, usage, details = await summarize_branch(
                        messages,
                        provider=self.provider,
                        model=self._model,
                        stream_options=self._stream_options,
                        custom_instructions=(
                            str(hook.get("customInstructions"))
                            if hook.get("customInstructions")
                            else custom_instructions
                        ),
                        replace_instructions=bool(
                            hook.get("replaceInstructions", replace_instructions)
                        ),
                    )
                    from_hook = False
                summary_payload = (summary, usage, details, from_hook)
            await self.session.set_leaf_id(target_id)
            if summary_payload is not None:
                summary, usage, details, from_hook = summary_payload
                summary_entry = await self.session.append_branch_summary(
                    from_id=str(old_leaf or target_id),
                    summary=summary,
                    usage=usage,
                    details=details,
                    from_hook=from_hook,
                )
            if label:
                await self.session.set_label(summary_entry or target_id, label)
            await self._restore_context()
            result = {
                "cancelled": False,
                "oldLeafId": old_leaf,
                "newLeafId": await self.session.get_leaf_id(),
                "summaryEntryId": summary_entry,
            }
            await self._emit("session_tree", result)
            return result
        finally:
            self._set_phase("idle")
            self._idle_event.set()
            await self._emit("settled", {"operation": "tree"})

    async def set_model(self, model: Model) -> None:
        self._require_idle("set model")
        previous = self._model
        self._model = model
        if self.session is not None:
            await self.session.append_model_change(model.provider, model.id)
        await self._emit(
            "model_update",
            {
                "model": model_to_dict(model),
                "previousModel": model_to_dict(previous),
                "source": "set",
            },
        )

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._require_idle("set thinking level")
        previous = self._thinking_level
        self._thinking_level = level
        self._stream_options = replace(self._stream_options, thinking_level=level)
        if self.session is not None:
            await self.session.append_thinking_level_change(level)
        await self._emit("thinking_level_update", {"level": level, "previousLevel": previous})

    async def set_tools(
        self,
        tools: Iterable[AgentTool],
        active_tool_names: Sequence[str] | None = None,
    ) -> None:
        self._require_idle("set tools")
        change = self._tool_runtime.configure(tools, active_tool_names)
        await self._record_tools_update(change)

    async def set_active_tools(self, names: Sequence[str]) -> None:
        self._require_idle("set active tools")
        change = self._tool_runtime.activate(names)
        await self._record_tools_update(change)

    async def set_resources(self, resources: Resources) -> None:
        self._require_idle("set resources")
        previous = self._resources
        self._resources = resources
        await self._emit(
            "resources_update",
            {
                "resources": _resources_dict(resources),
                "previousResources": _resources_dict(previous),
            },
        )

    async def set_stream_options(self, options: StreamOptions) -> None:
        self._require_idle("set stream options")
        self._stream_options = replace(options, thinking_level=self._thinking_level)

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._queues.set_steering_mode(mode)

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._queues.set_follow_up_mode(mode)

    async def abort(self) -> dict[str, list[dict[str, Any]]]:
        result = self._queues.clear_interactive()
        self._abort_requested = True
        self._provider_runtime.cancel()
        self._tool_runtime.cancel()
        await self._emit("abort", result)
        if asyncio.current_task() is not self._run_task:
            await self.wait_for_idle()
        return result

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def close(self) -> None:
        await self._provider_runtime.close()

    async def _run_loop(
        self,
        initial: list[AgentMessage],
        system_prompt: str,
    ) -> AssistantMessage:
        new_messages: list[AgentMessage] = []
        await self._emit("agent_start")
        await self._emit("turn_start")
        for message in initial:
            await self._append_message(message)
            new_messages.append(message)
        pending = await self._queues.drain_steering()
        first_turn = True
        overflow_attempted = False
        final: AssistantMessage | None = None
        while True:
            has_more_tools = True
            while has_more_tools or pending:
                if not first_turn:
                    await self._emit("turn_start")
                first_turn = False
                for message in pending:
                    await self._append_message(message)
                    new_messages.append(message)
                pending = []
                final = await self._stream_response(system_prompt)
                if (
                    final.stop_reason == "error"
                    and not overflow_attempted
                    and self.session is not None
                    and self.compaction_settings.enabled
                    and _is_context_overflow(final.error_message)
                ):
                    overflow_attempted = True
                    await self._emit(
                        "message_end",
                        {"message": message_to_dict(final), "transient": True},
                    )
                    await self._emit(
                        "turn_end",
                        {"message": message_to_dict(final), "toolResults": []},
                    )
                    self._set_phase("compaction")
                    try:
                        await self._compact_session(None, reason="overflow")
                    except Exception as exc:
                        self._set_phase("turn")
                        await self._transcript.append(final, emit_events=False)
                        await self._emit(
                            "compaction_end",
                            {
                                "reason": "overflow",
                                "aborted": False,
                                "willRetry": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        await self._emit(
                            "agent_end",
                            {
                                "messages": [
                                    message_to_dict(item) for item in (*new_messages, final)
                                ]
                            },
                        )
                        return final
                    else:
                        self._set_phase("turn")
                        continue
                await self._finish_assistant_message(final)
                new_messages.append(final)
                if final.stop_reason in {"error", "aborted"}:
                    await self._emit(
                        "turn_end", {"message": message_to_dict(final), "toolResults": []}
                    )
                    await self._emit(
                        "agent_end", {"messages": [message_to_dict(item) for item in new_messages]}
                    )
                    return final
                tool_results: list[ToolResultMessage] = []
                terminate = False
                if final.tool_calls:
                    if final.stop_reason == "length":
                        tool_results = await self._tool_runtime.fail_truncated_calls(
                            final.tool_calls
                        )
                    else:
                        tool_results, terminate = await self._tool_runtime.execute_calls(
                            final.tool_calls,
                            is_aborted=lambda: self._abort_requested,
                        )
                    for result in tool_results:
                        await self._append_message(result)
                        new_messages.append(result)
                has_more_tools = bool(final.tool_calls) and not terminate
                await self._emit(
                    "turn_end",
                    {
                        "message": message_to_dict(final),
                        "toolResults": [message_to_dict(item) for item in tool_results],
                    },
                )
                if self._abort_requested:
                    return final
                pending = await self._queues.drain_steering()
            follow_up = await self._queues.drain_follow_up()
            if follow_up:
                pending = follow_up
                continue
            break
        assert final is not None
        await self._emit(
            "agent_end", {"messages": [message_to_dict(item) for item in new_messages]}
        )
        return final

    async def _stream_response(self, system_prompt: str) -> AssistantMessage:
        return await self._provider_runtime.request(
            model=self._model,
            messages=self._transcript.messages,
            system_prompt=system_prompt,
            tools=self.active_tools,
            session_id=self.session.id if self.session else "ephemeral",
            stream_options=self._stream_options,
            on_retry=self._provider_retry,
        )

    async def _finish_assistant_message(self, message: AssistantMessage) -> None:
        # The provider stream already emitted message_start.
        await self._transcript.append(message, message_started=True)

    async def _append_message(
        self,
        message: AgentMessage,
        *,
        emit_events: bool = True,
    ) -> None:
        await self._transcript.append(message, emit_events=emit_events)

    async def _auto_compact_if_needed(self) -> None:
        context_tokens = estimate_context_tokens(self._transcript.messages)
        if not should_compact(context_tokens, self._model.context_window, self.compaction_settings):
            return
        self._set_phase("compaction")
        try:
            try:
                await self._compact_session(None, reason="threshold")
            except Exception as exc:
                await self._emit(
                    "compaction_end",
                    {
                        "reason": "threshold",
                        "aborted": isinstance(exc, asyncio.CancelledError),
                        "willRetry": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
        finally:
            self._set_phase("turn")

    async def _compact_session(
        self,
        custom_instructions: str | None,
        *,
        reason: str,
    ) -> CompactionResult:
        assert self.session is not None
        await self._emit("compaction_start", {"reason": reason})
        preparation = await prepare_compaction(self.session, self.compaction_settings)
        if preparation is None:
            raise HarnessError("compaction", "Session does not need compaction")
        hook = await self._hook(
            "session_before_compact",
            {
                "preparation": {
                    "firstKeptEntryId": preparation.first_kept_entry_id,
                    "tokensBefore": preparation.tokens_before,
                },
                "customInstructions": custom_instructions,
            },
        )
        if hook.get("cancel"):
            raise HarnessError("compaction", "Compaction cancelled by hook")
        hook_result = hook.get("compaction")
        if hook_result:
            result = CompactionResult(
                summary=str(hook_result["summary"]),
                first_kept_entry_id=str(
                    hook_result.get("firstKeptEntryId") or preparation.first_kept_entry_id
                ),
                tokens_before=int(hook_result.get("tokensBefore") or preparation.tokens_before),
                retained_tail=preparation.retained_tail,
                usage=Usage.from_dict(hook_result.get("usage")),
                details=hook_result.get("details") or {},
            )
            from_hook = True
        else:
            result = await compact_preparation(
                preparation,
                provider=self.provider,
                model=self._model,
                stream_options=self._stream_options,
                settings=self.compaction_settings,
                custom_instructions=custom_instructions,
            )
            from_hook = False
        entry_id = await self.session.append_compaction(
            summary=result.summary,
            tokens_before=result.tokens_before,
            first_kept_entry_id=result.first_kept_entry_id,
            usage=result.usage,
            details=result.details,
            from_hook=from_hook,
        )
        await self._restore_context()
        await self._emit(
            "session_compact",
            {
                "compactionEntryId": entry_id,
                "summary": result.summary,
                "fromHook": from_hook,
            },
        )
        await self._emit(
            "compaction_end",
            {"reason": reason, "aborted": False, "willRetry": reason == "overflow"},
        )
        return result

    async def _restore_context(self) -> None:
        context = await self._transcript.restore()
        if context is None:
            return
        if context.thinking_level in {"off", "minimal", "low", "medium", "high", "max"}:
            self._thinking_level = context.thinking_level  # type: ignore[assignment]
            self._stream_options = replace(
                self._stream_options, thinking_level=self._thinking_level
            )
        if context.active_tool_names is not None:
            self._tool_runtime.restore_active(context.active_tool_names)

    async def _reload_resources(self) -> None:
        if self.resource_loader is None:
            return
        previous = self._resources
        current = self.resource_loader.reload()
        if current != previous:
            self._resources = current
            await self._emit(
                "resources_update",
                {
                    "resources": _resources_dict(current),
                    "previousResources": _resources_dict(previous),
                },
            )

    async def _record_tools_update(self, change: ToolConfigurationChange) -> None:
        if self.session is not None:
            await self.session.append_active_tools_change(self.active_tool_names)
        await self._emit(
            "tools_update",
            {
                "toolNames": list(self._tool_runtime.tool_names),
                "previousToolNames": list(change.previous_tool_names),
                "activeToolNames": list(self.active_tool_names),
                "previousActiveToolNames": list(change.previous_active_names),
                "source": "set",
            },
        )

    async def _provider_retry(self, data: dict[str, Any]) -> None:
        if data.get("stage") == "start":
            self._set_phase("retry")
            await self._emit("auto_retry_start", data)
            return
        await self._emit("auto_retry_end", data)
        if not self._abort_requested:
            self._set_phase("turn")

    async def _emit(self, event_type: HarnessEventType, data: dict[str, Any] | None = None) -> None:
        await self._events.emit(event_type, data)

    async def _hook(self, event_type: HarnessEventType, data: dict[str, Any]) -> dict[str, Any]:
        return await self._events.hook(event_type, data)

    def _set_phase(self, phase: AgentHarnessPhase) -> None:
        self._phase = phase

    def _require_idle(self, operation: str) -> None:
        if self._phase != "idle":
            raise HarnessError("busy", f"Cannot {operation} while harness is {self._phase}")


def _resources_dict(resources: Resources) -> dict[str, Any]:
    return {
        "skills": [
            {
                "name": skill.name,
                "description": skill.description,
                "filePath": skill.file_path,
            }
            for skill in resources.skills
        ],
        "promptTemplates": [
            {"name": prompt.name, "description": prompt.description}
            for prompt in resources.prompt_templates
        ],
        "contextFiles": [
            {"path": path, "content": content} for path, content in resources.context_files
        ],
    }


def _is_context_overflow(error_message: str | None) -> bool:
    message = (error_message or "").lower()
    return any(
        marker in message
        for marker in (
            "context length",
            "context window",
            "maximum context",
            "max context",
            "too many tokens",
        )
    )


__all__ = ["AgentHarness", "AgentHarnessPhase"]
