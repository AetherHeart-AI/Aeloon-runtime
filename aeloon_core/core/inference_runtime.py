"""Inference request preparation and stream lifecycle for one run."""

from __future__ import annotations

import asyncio
import re
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
    TextContent,
    Tool,
    ToolResultMessage,
    message_from_dict,
    message_to_dict,
)

_NON_RETRYABLE_LIMIT_PATTERN = re.compile(
    r"insufficient_quota|quota exceeded|usage limit|out of budget|billing|available balance",
    re.IGNORECASE,
)
_RETRYABLE_ERROR_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|\b(?:408|409|429|500|502|503|504|524)\b|"
    r"service.?unavailable|server.?error|internal.?error|network.?error|connection|"
    r"fetch failed|enotfound|eai_again|socket|timed? out|timeout|terminated|"
    r"stream ended|ended without|http2 request did not get a response|please retry",
    re.IGNORECASE,
)

RetryCallback = Callable[[dict[str, Any]], Awaitable[None]]


class InferenceRuntime:
    """Build requests and own the active inference task for one run."""

    def __init__(self, inference: InferencePort, events: RunEventDispatcher) -> None:
        self._inference = inference
        self._events = events
        self._task: asyncio.Task[Any] | None = None
        self.last_attempt = 0
        self._next_attempt_id = 0

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
        context = replace(
            context,
            messages=normalize_inference_messages(context.messages),
        )
        max_retries = 3 if options.max_retries is None else max(0, options.max_retries)
        attempt_options = replace(options, max_retries=0)
        attempted_retry = False
        attempt_base = self._next_attempt_id
        for attempt in range(max_retries + 1):
            attempt_id = attempt_base + attempt
            failure: Exception | None = None
            attempt_state = {"started": False}
            try:
                self._task = asyncio.create_task(
                    collect_assistant(
                        self._inference,
                        model,
                        context,
                        attempt_options,
                        events=self._events,
                        attempt=attempt_id,
                        attempt_state=attempt_state,
                    )
                )
                message = await self._task
            except asyncio.CancelledError:
                message = AssistantMessage(
                    content=(),
                    provider=model.provider,
                    model=model.id,
                    stop_reason="aborted",
                    error_message="Operation aborted",
                )
            except Exception as exc:
                failure = exc
                message = AssistantMessage(
                    content=(),
                    provider=model.provider,
                    model=model.id,
                    stop_reason="error",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            finally:
                self._task = None

            if not attempt_state["started"]:
                await self._events.emit(
                    "message_start",
                    {"message": message_to_dict(message), "attempt": attempt_id},
                )
                attempt_state["started"] = True

            retryable = _is_retryable_failure(message, failure, model.context_window)
            if message.stop_reason != "error" or not retryable or attempt >= max_retries:
                self.last_attempt = attempt_id
                self._next_attempt_id = attempt_id + 1
                if attempted_retry:
                    await on_retry(
                        {
                            "stage": "end",
                            "attempt": attempt,
                            "maxAttempts": max_retries + 1,
                            "delayMs": 0,
                            "error": message.error_message,
                        }
                    )
                return message

            attempted_retry = True
            await self._events.emit(
                "message_end",
                {
                    "message": message_to_dict(message),
                    "attempt": attempt_id,
                    "willRetry": True,
                },
            )
            delay_ms = _retry_delay_ms(failure, attempt, options)
            await on_retry(
                {
                    "stage": "start",
                    "attempt": attempt + 1,
                    "maxAttempts": max_retries + 1,
                    "delayMs": delay_ms,
                    "error": message.error_message,
                }
            )
            try:
                self._task = asyncio.create_task(asyncio.sleep(delay_ms / 1000))
                await self._task
            except asyncio.CancelledError:
                message = AssistantMessage(
                    content=(),
                    provider=model.provider,
                    model=model.id,
                    stop_reason="aborted",
                    error_message="Operation aborted",
                )
                self.last_attempt = attempt_id + 1
                self._next_attempt_id = attempt_id + 2
                await self._events.emit(
                    "message_start",
                    {"message": message_to_dict(message), "attempt": attempt_id + 1},
                )
                await on_retry(
                    {
                        "stage": "end",
                        "attempt": attempt + 1,
                        "maxAttempts": max_retries + 1,
                        "delayMs": 0,
                        "error": message.error_message,
                    }
                )
                return message
            finally:
                self._task = None

        raise AssertionError("unreachable inference retry state")

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
    attempt: int = 0,
    attempt_state: dict[str, bool] | None = None,
) -> AssistantMessage:
    """Collect a vendor-neutral stream into its final assistant message."""

    final: AssistantMessage | None = None
    started = False
    async for event in inference.stream(model, context, options):
        if event.type == "start":
            started = True
            if attempt_state is not None:
                attempt_state["started"] = True
            if events is not None:
                await events.emit(
                    "message_start",
                    {
                        "message": message_to_dict(
                            AssistantMessage(content=(), provider=model.provider, model=model.id)
                        ),
                        "attempt": attempt,
                    },
                )
        elif event.type in {"text_delta", "thinking_delta", "toolcall_delta"}:
            if events is not None:
                await events.emit(
                    "message_update",
                    {
                        "assistantMessageEvent": {
                            **stream_event_to_dict(event),
                            "attempt": attempt,
                        }
                    },
                )
        elif event.type in {"done", "error"}:
            final = event.message
    if final is None:
        raise InferenceError(
            "invalid_response",
            "Inference stream ended without a final message",
            retryable=True,
        )
    if events is not None and not started:
        if attempt_state is not None:
            attempt_state["started"] = True
        await events.emit(
            "message_start",
            {"message": message_to_dict(final), "attempt": attempt},
        )
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


def _is_retryable_failure(
    message: AssistantMessage,
    failure: Exception | None,
    context_window: int,
) -> bool:
    if message.stop_reason != "error":
        return False
    from aeloon_core.core.compaction import is_context_overflow

    if is_context_overflow(message, context_window):
        return False
    if isinstance(failure, InferenceError) and failure.retryable:
        return True
    error_message = message.error_message or ""
    if _NON_RETRYABLE_LIMIT_PATTERN.search(error_message):
        return False
    return bool(_RETRYABLE_ERROR_PATTERN.search(error_message))


def _retry_delay_ms(
    failure: Exception | None,
    attempt: int,
    options: StreamOptions,
) -> int:
    if isinstance(failure, InferenceError) and failure.retry_after_ms is not None:
        return min(max(0, failure.retry_after_ms), options.max_retry_delay_ms)
    return min(options.base_delay_ms * (2**attempt), options.max_retry_delay_ms)


def normalize_inference_messages(
    messages: Sequence[AgentMessage],
) -> tuple[AgentMessage, ...]:
    """Return a provider-safe projection without mutating durable history.

    Failed assistant attempts remain in the Session for display and auditing, but
    replaying them can produce invalid provider payloads. Tool calls also need a
    result for every id before another assistant/user message starts.
    """

    projected, _boundary_index = project_inference_messages(messages)
    return projected


def project_inference_messages(
    messages: Sequence[AgentMessage],
    *,
    boundary_index: int | None = None,
) -> tuple[tuple[AgentMessage, ...], int | None]:
    """Project durable messages and rebase an optional compaction boundary.

    Every projected message retains the index of the durable message that produced
    it. Synthetic tool results inherit the assistant tool-call index, so filtering
    failed attempts cannot make the boundary drift into the fresh context.
    """

    result: list[AgentMessage] = []
    origins: list[int] = []
    pending_calls: dict[str, tuple[str, int]] = {}
    resolved_calls: set[str] = set()

    def flush_orphans() -> None:
        nonlocal pending_calls, resolved_calls
        for call_id, (tool_name, source_index) in pending_calls.items():
            if call_id not in resolved_calls:
                result.append(
                    ToolResultMessage(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        content=(TextContent("No result provided"),),
                        is_error=True,
                    )
                )
                origins.append(source_index)
        pending_calls = {}
        resolved_calls = set()

    for source_index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            flush_orphans()
            if message.stop_reason in {"error", "aborted"}:
                continue
            result.append(message)
            origins.append(source_index)
            pending_calls = {
                call.id: (call.name, source_index) for call in message.tool_calls
            }
            continue
        if isinstance(message, ToolResultMessage):
            resolved_calls.add(message.tool_call_id)
            result.append(message)
            origins.append(source_index)
            continue
        flush_orphans()
        result.append(message)
        origins.append(source_index)
    flush_orphans()
    projected_boundary_index = None
    if boundary_index is not None:
        projected_boundary_index = max(
            (
                index
                for index, source_index in enumerate(origins)
                if source_index <= boundary_index
            ),
            default=-1,
        )
    return tuple(result), projected_boundary_index


__all__ = [
    "InferenceRuntime",
    "collect_assistant",
    "model_to_dict",
    "normalize_inference_messages",
    "project_inference_messages",
]
