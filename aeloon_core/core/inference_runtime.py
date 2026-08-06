"""Inference request preparation and stream lifecycle for one run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from aeloon_core.core.events import RunEventDispatcher
from aeloon_core.core.types import (
    AgentMessage,
    AssistantMessage,
    AssistantStreamEvent,
    InferenceContext,
    InferenceError,
    InferencePort,
    Model,
    StreamOptions,
    Tool,
    message_from_dict,
    message_to_dict,
)

RetryCallback = Callable[[dict[str, Any]], Awaitable[None]]


class InferenceRuntime:
    """Build requests and own the active inference task for one run."""

    def __init__(self, inference: InferencePort, events: RunEventDispatcher) -> None:
        self._inference = inference
        self._events = events
        self._task: asyncio.Task[AssistantMessage] | None = None

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def request(
        self,
        *,
        model: Model,
        messages: Sequence[AgentMessage],
        system_prompt: str,
        tools: Sequence[Tool],
        session_id: str,
        stream_options: StreamOptions,
        on_retry: RetryCallback,
    ) -> AssistantMessage:
        context_messages = await self._context_messages(messages)
        options = await self._request_options(
            model=model,
            session_id=session_id,
            stream_options=stream_options,
            on_retry=on_retry,
        )
        context = InferenceContext(
            system_prompt=system_prompt,
            messages=tuple(context_messages),
            tools=tuple(tool.definition() for tool in tools),
            session_id=session_id,
        )
        context = await self._patch_context(model, context)
        self._task = asyncio.create_task(
            collect_assistant(self._inference, model, context, options, events=self._events)
        )
        try:
            return await self._task
        except asyncio.CancelledError:
            return AssistantMessage(
                content=(),
                provider=model.provider,
                model=model.id,
                stop_reason="aborted",
                error_message="Operation aborted",
            )
        except Exception as exc:
            return AssistantMessage(
                content=(),
                provider=model.provider,
                model=model.id,
                stop_reason="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._task = None

    async def _context_messages(self, messages: Sequence[AgentMessage]) -> list[AgentMessage]:
        hook = await self._events.hook(
            "context",
            {"messages": [message_to_dict(message) for message in messages]},
        )
        if "messages" not in hook:
            return list(messages)
        return [
            item if not isinstance(item, Mapping) else message_from_dict(item)
            for item in hook["messages"]
        ]

    async def _request_options(
        self,
        *,
        model: Model,
        session_id: str,
        stream_options: StreamOptions,
        on_retry: RetryCallback,
    ) -> StreamOptions:
        hook = await self._events.hook(
            "before_inference_request",
            {
                "model": model_to_dict(model),
                "sessionId": session_id,
                "streamOptions": stream_options_to_dict(stream_options),
            },
        )
        options = _with_retry_callback(stream_options, on_retry)
        patch = hook.get("streamOptions")
        if isinstance(patch, dict):
            options = patch_stream_options(options, patch)
            options = _with_retry_callback(options, on_retry)
        return options

    async def _patch_context(
        self,
        model: Model,
        context: InferenceContext,
    ) -> InferenceContext:
        hook = await self._events.hook(
            "before_inference_context",
            {
                "model": model_to_dict(model),
                "context": {
                    "systemPrompt": context.system_prompt,
                    "messages": [message_to_dict(message) for message in context.messages],
                    "tools": list(context.tools),
                },
            },
        )
        patch = hook.get("context")
        if not isinstance(patch, Mapping):
            return context
        raw_messages = patch.get("messages")
        messages = list(context.messages)
        if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, str | bytes):
            messages = [
                item if not isinstance(item, Mapping) else message_from_dict(item)
                for item in raw_messages
            ]
        raw_tools = patch.get("tools", context.tools)
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, str | bytes):
            raw_tools = context.tools
        return InferenceContext(
            system_prompt=str(patch.get("systemPrompt", context.system_prompt)),
            messages=tuple(messages),
            tools=tuple(dict(item) for item in raw_tools if isinstance(item, Mapping)),
            session_id=context.session_id,
        )


async def collect_assistant(
    inference: InferencePort,
    model: Model,
    context: InferenceContext,
    options: StreamOptions,
    *,
    events: RunEventDispatcher | None = None,
) -> AssistantMessage:
    """Collect a vendor-neutral stream into its final assistant message."""

    final: AssistantMessage | None = None
    started = False
    async for event in inference.stream(model, context, options):
        if event.type == "start":
            started = True
            if events is not None:
                await events.emit(
                    "message_start",
                    {
                        "message": message_to_dict(
                            AssistantMessage(content=(), provider=model.provider, model=model.id)
                        )
                    },
                )
        elif event.type in {"text_delta", "thinking_delta", "toolcall_delta"}:
            if events is not None:
                await events.emit(
                    "message_update",
                    {"assistantMessageEvent": stream_event_to_dict(event)},
                )
        elif event.type in {"done", "error"}:
            final = event.message
    if final is None:
        raise InferenceError("invalid_response", "Inference stream ended without a final message")
    if events is not None and not started:
        await events.emit("message_start", {"message": message_to_dict(final)})
    return final


def model_to_dict(model: Model) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "reasoning": model.reasoning,
        "input": list(model.input),
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
    }


def stream_options_to_dict(options: StreamOptions) -> dict[str, Any]:
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


def patch_stream_options(options: StreamOptions, patch: dict[str, Any]) -> StreamOptions:
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


def stream_event_to_dict(event: AssistantStreamEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "delta": event.delta,
        "contentIndex": event.content_index,
        "toolCallIndex": event.tool_call_index,
        "toolCallId": event.tool_call_id,
        "toolName": event.tool_name,
    }


def _with_retry_callback(options: StreamOptions, on_retry: RetryCallback) -> StreamOptions:
    return replace(options, metadata={**options.metadata, "on_retry": on_retry})


__all__ = ["InferenceRuntime", "collect_assistant", "model_to_dict"]
