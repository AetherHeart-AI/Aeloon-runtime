"""Stateful Pi-compatible Python agent harness."""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from jsonschema import Draft202012Validator

from aeloon_core.harness.compaction import (
    CompactionResult,
    CompactionSettings,
    compact_preparation,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
    summarize_branch,
)
from aeloon_core.harness.prompt import build_system_prompt
from aeloon_core.harness.resources import ResourceLoader
from aeloon_core.harness.session import Session
from aeloon_core.harness.tools import DEFAULT_ACTIVE_TOOLS, create_all_tools
from aeloon_core.harness.types import (
    AgentMessage,
    AgentTool,
    AssistantMessage,
    AssistantStreamEvent,
    EventListener,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HookHandler,
    ImageContent,
    Model,
    Provider,
    ProviderContext,
    QueueMode,
    Resources,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
    content_from_dict,
    content_to_dict,
    message_from_dict,
    message_to_dict,
)

AgentHarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]


class AgentHarness:
    """Own model, tools, queues, session state, and the full agent loop."""

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
        self.provider = provider
        self._model = model
        self.cwd = cwd
        self.session = session
        default_tools = create_all_tools(
            cwd,
            shell_path=shell_path,
            auto_resize_images=auto_resize_images,
        )
        configured = list(tools) if tools is not None else list(default_tools.values())
        self._tools = _unique_tools(configured)
        self._active_tool_names = _validate_active_tools(self._tools, active_tool_names)
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
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self.compaction_settings = compaction or CompactionSettings()
        self._phase: AgentHarnessPhase = "idle"
        self._messages: list[AgentMessage] = []
        self._steer_queue: list[UserMessage] = []
        self._follow_up_queue: list[UserMessage] = []
        self._next_turn_queue: list[UserMessage] = []
        self._listeners: list[EventListener] = []
        self._handlers: dict[str, list[HookHandler]] = defaultdict(list)
        self._provider_task: asyncio.Task[AssistantMessage] | None = None
        self._tool_tasks: set[asyncio.Task[Any]] = set()
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
    def thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools.values())

    @property
    def active_tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools[name] for name in self._active_tool_names)

    @property
    def active_tool_names(self) -> tuple[str, ...]:
        return self._active_tool_names

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def stream_options(self) -> StreamOptions:
        return self._stream_options

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(
            cwd=self.cwd,
            tools=self.active_tools,
            resources=self._resources,
            custom_prompt=self._custom_system_prompt,
        )

    def subscribe(self, listener: EventListener) -> Any:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def on(self, event_type: str, handler: HookHandler) -> Any:
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers[event_type]
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

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
            initial = [*self._next_turn_queue, prompt_message]
            self._next_turn_queue.clear()
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
            self._provider_task = None
            self._tool_tasks.clear()
            self._run_task = None
            self._set_phase("idle")
            self._idle_event.set()
            await self._emit("settled", {"nextTurnCount": len(self._next_turn_queue)})

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
        self._steer_queue.append(UserMessage(text if not images else (TextContent(text), *images)))
        await self._emit_queue_update()

    async def follow_up(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        if self._phase not in {"turn", "retry"}:
            raise HarnessError("invalid_state", "follow_up() requires an active turn")
        self._follow_up_queue.append(
            UserMessage(text if not images else (TextContent(text), *images))
        )
        await self._emit_queue_update()

    async def next_turn(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        self._next_turn_queue.append(
            UserMessage(text if not images else (TextContent(text), *images))
        )
        await self._emit_queue_update()

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
            {"model": _model_dict(model), "previousModel": _model_dict(previous), "source": "set"},
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
        previous_tools = tuple(self._tools)
        previous_active = self._active_tool_names
        self._tools = _unique_tools(list(tools))
        selected = active_tool_names if active_tool_names is not None else tuple(self._tools)
        self._active_tool_names = _validate_active_tools(self._tools, selected)
        await self._record_tools_update(previous_tools, previous_active)

    async def set_active_tools(self, names: Sequence[str]) -> None:
        self._require_idle("set active tools")
        previous_tools = tuple(self._tools)
        previous_active = self._active_tool_names
        self._active_tool_names = _validate_active_tools(self._tools, names)
        await self._record_tools_update(previous_tools, previous_active)

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
        self._steering_mode = _queue_mode(mode)

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_mode = _queue_mode(mode)

    async def abort(self) -> dict[str, list[dict[str, Any]]]:
        cleared_steer = [message_to_dict(message) for message in self._steer_queue]
        cleared_follow_up = [message_to_dict(message) for message in self._follow_up_queue]
        self._steer_queue.clear()
        self._follow_up_queue.clear()
        self._abort_requested = True
        if self._provider_task is not None and not self._provider_task.done():
            self._provider_task.cancel()
        for task in tuple(self._tool_tasks):
            if not task.done():
                task.cancel()
        result = {"clearedSteer": cleared_steer, "clearedFollowUp": cleared_follow_up}
        await self._emit("abort", result)
        if asyncio.current_task() is not self._run_task:
            await self.wait_for_idle()
        return result

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

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
        pending = await self._drain_queue(self._steer_queue, self._steering_mode)
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
                        self._messages.append(final)
                        await self.session.append_message(final)
                        await self._emit("save_point", {"hadPendingMutations": True})
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
                        tool_results = await self._fail_truncated_calls(final.tool_calls)
                    else:
                        tool_results, terminate = await self._execute_tool_calls(
                            final, final.tool_calls
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
                pending = await self._drain_queue(self._steer_queue, self._steering_mode)
            follow_up = await self._drain_queue(self._follow_up_queue, self._follow_up_mode)
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
        hook = await self._hook(
            "context", {"messages": [message_to_dict(message) for message in self._messages]}
        )
        messages = self._messages
        if "messages" in hook:
            messages = [
                item if not isinstance(item, dict) else message_from_dict(item)
                for item in hook["messages"]
            ]
        request_hook = await self._hook(
            "before_provider_request",
            {
                "model": _model_dict(self._model),
                "sessionId": self.session.id if self.session else "ephemeral",
                "streamOptions": _stream_options_dict(self._stream_options),
            },
        )
        options = replace(
            self._stream_options,
            metadata={**self._stream_options.metadata, "on_retry": self._provider_retry},
        )
        patch = request_hook.get("streamOptions")
        if isinstance(patch, dict):
            options = _patch_stream_options(options, patch)
            options = replace(
                options,
                metadata={**options.metadata, "on_retry": self._provider_retry},
            )
        context = ProviderContext(
            system_prompt=system_prompt,
            messages=tuple(messages),
            tools=tuple(tool.definition() for tool in self.active_tools),
            session_id=self.session.id if self.session else "ephemeral",
        )
        payload_hook = await self._hook(
            "before_provider_payload",
            {
                "model": _model_dict(self._model),
                "payload": {
                    "systemPrompt": system_prompt,
                    "messages": [message_to_dict(message) for message in messages],
                    "tools": list(context.tools),
                },
            },
        )
        payload_patch = payload_hook.get("payload")
        if isinstance(payload_patch, Mapping):
            patched_system = payload_patch.get("systemPrompt", context.system_prompt)
            patched_messages = payload_patch.get("messages")
            patched_tools = payload_patch.get("tools", context.tools)
            if isinstance(patched_messages, Sequence) and not isinstance(
                patched_messages, str | bytes
            ):
                messages = [
                    item if not isinstance(item, Mapping) else message_from_dict(item)
                    for item in patched_messages
                ]
            if not isinstance(patched_tools, Sequence) or isinstance(patched_tools, str | bytes):
                patched_tools = context.tools
            context = ProviderContext(
                system_prompt=str(patched_system),
                messages=tuple(messages),
                tools=tuple(dict(item) for item in patched_tools if isinstance(item, Mapping)),
                session_id=context.session_id,
            )
        self._provider_task = asyncio.create_task(self._collect_provider_stream(context, options))
        try:
            return await self._provider_task
        except asyncio.CancelledError:
            return AssistantMessage(
                content=(),
                provider=self._model.provider,
                model=self._model.id,
                stop_reason="aborted",
                error_message="Operation aborted",
            )
        except Exception as exc:
            return AssistantMessage(
                content=(),
                provider=self._model.provider,
                model=self._model.id,
                stop_reason="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._provider_task = None

    async def _collect_provider_stream(
        self,
        context: ProviderContext,
        options: StreamOptions,
    ) -> AssistantMessage:
        final: AssistantMessage | None = None
        started = False
        async for event in self.provider.stream(self._model, context, options):
            if event.type == "start":
                started = True
                await self._emit(
                    "message_start",
                    {
                        "message": message_to_dict(
                            AssistantMessage(
                                content=(),
                                provider=self._model.provider,
                                model=self._model.id,
                            )
                        )
                    },
                )
            elif event.type in {"text_delta", "thinking_delta", "toolcall_delta"}:
                await self._emit(
                    "message_update", {"assistantMessageEvent": _stream_event_dict(event)}
                )
            elif event.type in {"done", "error"}:
                final = event.message
        if final is None:
            raise HarnessError("provider", "Provider stream ended without a final message")
        if not started:
            await self._emit("message_start", {"message": message_to_dict(final)})
        return final

    async def _finish_assistant_message(self, message: AssistantMessage) -> None:
        self._messages.append(message)
        if self.session is not None:
            await self.session.append_message(message)
        await self._emit("message_end", {"message": message_to_dict(message)})
        if self.session is not None:
            await self._emit("save_point", {"hadPendingMutations": True})

    async def _append_message(
        self,
        message: AgentMessage,
        *,
        emit_events: bool = True,
    ) -> None:
        if emit_events:
            await self._emit("message_start", {"message": message_to_dict(message)})
        self._messages.append(message)
        if self.session is not None:
            await self.session.append_message(message)
        if emit_events:
            await self._emit("message_end", {"message": message_to_dict(message)})
        if self.session is not None:
            await self._emit("save_point", {"hadPendingMutations": True})

    async def _execute_tool_calls(
        self,
        assistant: AssistantMessage,
        calls: tuple[ToolCall, ...],
    ) -> tuple[list[ToolResultMessage], bool]:
        sequential = any(
            self._tools.get(call.name) is not None
            and self._tools[call.name].execution_mode == "sequential"
            for call in calls
        )
        results: list[tuple[ToolResultMessage, bool]] = []
        if sequential:
            for call in calls:
                results.append(await self._execute_tool_call(assistant, call))
                if self._abort_requested:
                    break
        else:
            tasks = [
                asyncio.create_task(self._execute_tool_call(assistant, call)) for call in calls
            ]
            self._tool_tasks.update(tasks)
            try:
                results = list(await asyncio.gather(*tasks))
            finally:
                for task in tasks:
                    self._tool_tasks.discard(task)
        terminate = any(item[1] for item in results)
        return [item[0] for item in results], terminate

    async def _execute_tool_call(
        self,
        assistant: AssistantMessage,
        call: ToolCall,
    ) -> tuple[ToolResultMessage, bool]:
        del assistant
        await self._emit(
            "tool_execution_start",
            {"toolCallId": call.id, "toolName": call.name, "args": call.arguments},
        )
        tool = self._tools.get(call.name)
        result: ToolResult
        args = dict(call.arguments)
        if tool is None or call.name not in self._active_tool_names:
            result = ToolResult.text(f"Tool {call.name} not found", is_error=True)
        else:
            try:
                if tool.prepare_arguments is not None:
                    args = tool.prepare_arguments(args)
                Draft202012Validator(tool.parameters).validate(args)
                hook = await self._hook(
                    "tool_call",
                    {
                        "toolCallId": call.id,
                        "toolName": call.name,
                        "input": args,
                    },
                )
                if hook.get("block"):
                    result = ToolResult.text(
                        str(hook.get("reason") or "Tool call blocked"), is_error=True
                    )
                else:

                    async def on_update(update: ToolResult) -> None:
                        await self._emit(
                            "tool_execution_update",
                            {
                                "toolCallId": call.id,
                                "toolName": call.name,
                                "partialResult": _tool_result_dict(update),
                            },
                        )

                    result = await tool.execute(call.id, args, on_update)
            except asyncio.CancelledError:
                result = ToolResult.text("Operation aborted", is_error=True)
            except Exception as exc:
                result = ToolResult.text(f"{type(exc).__name__}: {exc}", is_error=True)
        hook = await self._hook(
            "tool_result",
            {
                "toolCallId": call.id,
                "toolName": call.name,
                "input": args,
                **_tool_result_dict(result),
            },
        )
        if hook:
            raw_content = hook.get("content", result.content)
            if isinstance(raw_content, str):
                raw_content = (TextContent(raw_content),)
            content = tuple(
                content_from_dict(part) if isinstance(part, Mapping) else part
                for part in raw_content
            )
            raw_usage = hook.get("usage", result.usage)
            result = ToolResult(
                content=content,
                details=hook.get("details", result.details),
                is_error=bool(hook.get("isError", result.is_error)),
                terminate=bool(hook.get("terminate", result.terminate)),
                usage=(Usage.from_dict(raw_usage) if isinstance(raw_usage, Mapping) else raw_usage),
            )
        await self._emit(
            "tool_execution_end",
            {
                "toolCallId": call.id,
                "toolName": call.name,
                "result": _tool_result_dict(result),
                "isError": result.is_error,
            },
        )
        message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            is_error=result.is_error,
            usage=result.usage,
        )
        return message, result.terminate

    async def _fail_truncated_calls(self, calls: tuple[ToolCall, ...]) -> list[ToolResultMessage]:
        results: list[ToolResultMessage] = []
        for call in calls:
            await self._emit(
                "tool_execution_start",
                {"toolCallId": call.id, "toolName": call.name, "args": call.arguments},
            )
            text = (
                f'Tool call "{call.name}" was not executed: the response hit the output token '
                "limit, so its arguments may be truncated. Re-issue the tool call "
                "with complete arguments."
            )
            result = ToolResult.text(text, is_error=True)
            await self._emit(
                "tool_execution_end",
                {
                    "toolCallId": call.id,
                    "toolName": call.name,
                    "result": _tool_result_dict(result),
                    "isError": True,
                },
            )
            results.append(ToolResultMessage(call.id, call.name, result.content, is_error=True))
        return results

    async def _auto_compact_if_needed(self) -> None:
        context_tokens = estimate_context_tokens(self._messages)
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
        if self.session is None:
            return
        context = await self.session.build_context()
        self._messages = list(context.messages)
        if context.thinking_level in {"off", "minimal", "low", "medium", "high", "max"}:
            self._thinking_level = context.thinking_level  # type: ignore[assignment]
            self._stream_options = replace(
                self._stream_options, thinking_level=self._thinking_level
            )
        if context.active_tool_names is not None:
            available = tuple(name for name in context.active_tool_names if name in self._tools)
            self._active_tool_names = available

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

    async def _record_tools_update(
        self,
        previous_tools: tuple[str, ...],
        previous_active: tuple[str, ...],
    ) -> None:
        if self.session is not None:
            await self.session.append_active_tools_change(self._active_tool_names)
        await self._emit(
            "tools_update",
            {
                "toolNames": list(self._tools),
                "previousToolNames": list(previous_tools),
                "activeToolNames": list(self._active_tool_names),
                "previousActiveToolNames": list(previous_active),
                "source": "set",
            },
        )

    async def _emit_queue_update(self) -> None:
        await self._emit(
            "queue_update",
            {
                "steer": [message_to_dict(message) for message in self._steer_queue],
                "followUp": [message_to_dict(message) for message in self._follow_up_queue],
                "nextTurn": [message_to_dict(message) for message in self._next_turn_queue],
            },
        )

    async def _drain_queue(self, queue: list[UserMessage], mode: QueueMode) -> list[UserMessage]:
        if not queue:
            return []
        count = len(queue) if mode == "all" else 1
        drained = queue[:count]
        del queue[:count]
        await self._emit_queue_update()
        return drained

    async def _provider_retry(self, data: dict[str, Any]) -> None:
        if data.get("stage") == "start":
            self._set_phase("retry")
            await self._emit("auto_retry_start", data)
            return
        await self._emit("auto_retry_end", data)
        if not self._abort_requested:
            self._set_phase("turn")

    async def _emit(self, event_type: HarnessEventType, data: dict[str, Any] | None = None) -> None:
        event = HarnessEvent(event_type, data or {})
        for listener in tuple(self._listeners):
            try:
                result = listener(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue

    async def _hook(self, event_type: HarnessEventType, data: dict[str, Any]) -> dict[str, Any]:
        event = HarnessEvent(event_type, data)
        await self._emit(event_type, data)
        merged: dict[str, Any] = {}
        for handler in tuple(self._handlers.get(event_type, ())):
            result = handler(event)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict):
                merged.update(result)
        return merged

    def _set_phase(self, phase: AgentHarnessPhase) -> None:
        self._phase = phase

    def _require_idle(self, operation: str) -> None:
        if self._phase != "idle":
            raise HarnessError("busy", f"Cannot {operation} while harness is {self._phase}")


def _unique_tools(tools: Sequence[AgentTool]) -> dict[str, AgentTool]:
    result: dict[str, AgentTool] = {}
    for tool in tools:
        if tool.name in result:
            raise HarnessError("invalid_argument", f"Duplicate tool name: {tool.name}")
        result[tool.name] = tool
    return result


def _validate_active_tools(tools: dict[str, AgentTool], names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(names))
    unknown = [name for name in result if name not in tools]
    if unknown:
        raise HarnessError("invalid_argument", f"Unknown active tools: {', '.join(unknown)}")
    return result


def _queue_mode(mode: str) -> QueueMode:
    if mode not in {"all", "one-at-a-time"}:
        raise HarnessError("invalid_argument", f"Invalid queue mode: {mode}")
    return mode  # type: ignore[return-value]


def _model_dict(model: Model) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "api": model.api,
        "baseUrl": model.base_url,
        "reasoning": model.reasoning,
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
    }


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


def _stream_options_dict(options: StreamOptions) -> dict[str, Any]:
    return {
        "timeoutMs": options.timeout_ms,
        "maxTokens": options.max_tokens,
        "temperature": options.temperature,
        "thinkingLevel": options.thinking_level,
        "maxRetries": options.max_retries,
        "baseDelayMs": options.base_delay_ms,
        "maxRetryDelayMs": options.max_retry_delay_ms,
        "headers": dict(options.headers),
        "metadata": dict(options.metadata),
    }


def _patch_stream_options(options: StreamOptions, patch: dict[str, Any]) -> StreamOptions:
    values: dict[str, Any] = {}
    aliases = {
        "timeoutMs": "timeout_ms",
        "maxTokens": "max_tokens",
        "thinkingLevel": "thinking_level",
        "maxRetries": "max_retries",
        "baseDelayMs": "base_delay_ms",
        "maxRetryDelayMs": "max_retry_delay_ms",
    }
    for key, value in patch.items():
        values[aliases.get(key, key)] = value
    return replace(options, **values)


def _stream_event_dict(event: AssistantStreamEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "delta": event.delta,
        "contentIndex": event.content_index,
        "toolCallIndex": event.tool_call_index,
        "toolCallId": event.tool_call_id,
        "toolName": event.tool_name,
    }


def _tool_result_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "content": [content_to_dict(part) for part in result.content],
        "details": result.details,
        "isError": result.is_error,
        "terminate": result.terminate,
        "usage": result.usage.to_dict() if result.usage else None,
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
