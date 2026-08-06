"""OpenAI-compatible inference implementation shared by runtime providers."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from aeloon_core.config import is_sensitive_header
from aeloon_core.core.types import (
    AgentMessage,
    AssistantMessage,
    AssistantStreamEvent,
    ImageContent,
    InferenceContext,
    InferenceError,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from aeloon_core.runtime.providers.base import BaseProvider


def _request_model_id(model: Model) -> str:
    prefix = f"{model.provider}/"
    return model.id.removeprefix(prefix)


class OpenAICompatibleProvider(BaseProvider):
    """Reusable OpenAI-compatible streaming inference implementation."""

    driver = "openai-compatible"

    def __init__(
        self,
        *,
        provider_id: str,
        name: str,
        endpoint: str,
        models: tuple[Model, ...] = (),
        enabled: bool = True,
        api_key: str | None = None,
        proxy: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        chat_path: str = "/chat/completions",
        requires_api_key: bool = False,
        request_model_id: Callable[[Model], str] | None = None,
        prepare_payload: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        thinking_level_map: Mapping[str, str | None] | None = None,
        requires_reasoning_content: bool = False,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            name=name,
            endpoint=endpoint,
            enabled=enabled,
        )
        self._models = {model.id: model for model in models}
        self.api_key = api_key
        self.proxy = proxy
        self.headers = dict(headers or {})
        self.chat_path = "/" + chat_path.lstrip("/")
        self.requires_api_key = requires_api_key
        self.request_model_id = request_model_id or _request_model_id
        self.prepare_payload = prepare_payload
        self.thinking_level_map = dict(thinking_level_map or {})
        self.requires_reasoning_content = requires_reasoning_content
        self._client = client
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def models(self) -> dict[str, Model]:
        if self._models:
            return dict(self._models)
        discovered = await self._discover_models()
        return {model.id: model for model in discovered}

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "authenticated": bool(self.api_key) if self.requires_api_key else None,
            "credential_configured": bool(self.api_key),
        }

    async def discover_models(self) -> list[Model]:
        return await self._discover_models()

    def stream(
        self,
        model: Model,
        context: InferenceContext,
        options: StreamOptions,
    ) -> AsyncIterator[AssistantStreamEvent]:
        return self._stream(model, context, options)

    async def _stream(
        self,
        model: Model,
        context: InferenceContext,
        options: StreamOptions,
    ) -> AsyncIterator[AssistantStreamEvent]:
        if self.requires_api_key and not self.api_key:
            raise InferenceError("auth", "An API key is required for the selected provider")

        payload = _openai_payload(
            model,
            context,
            options,
            thinking_level_map=self.thinking_level_map,
            requires_reasoning_content=self.requires_reasoning_content,
        )
        payload["model"] = self.request_model_id(model)
        if self.prepare_payload is not None:
            payload = self.prepare_payload(payload)
        headers = {"content-type": "application/json", **self.headers, **options.headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        max_retries = 3 if options.max_retries is None else max(0, options.max_retries)
        timeout = None if options.timeout_ms is None else options.timeout_ms / 1000
        client = await self._get_client()
        url = f"{self.endpoint}{self.chat_path}"

        response: httpx.Response | None = None
        retrying = False
        for attempt in range(max_retries + 1):
            try:
                extensions = (
                    {"timeout": httpx.Timeout(timeout).as_dict()} if timeout is not None else None
                )
                request = client.build_request(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    extensions=extensions,
                )
                response = await client.send(request, stream=True)
                if response.status_code < 400:
                    if retrying:
                        await _notify_retry(
                            options,
                            stage="end",
                            attempt=attempt,
                            delay=0,
                            error=None,
                        )
                    break
                body = self._sanitize((await response.aread()).decode(errors="replace")[:4_000])
                if not _retryable_status(response.status_code) or attempt >= max_retries:
                    await response.aclose()
                    if retrying:
                        await _notify_retry(
                            options,
                            stage="end",
                            attempt=attempt,
                            delay=0,
                            error=body,
                        )
                    raise InferenceError(
                        "http_error",
                        f"{self.name} returned HTTP {response.status_code}: {body}",
                    )
                delay = _retry_delay(
                    response,
                    attempt,
                    options.base_delay_ms,
                    options.max_retry_delay_ms,
                )
                await response.aclose()
                await _notify_retry(
                    options,
                    stage="start",
                    attempt=attempt + 1,
                    delay=delay,
                    error=f"HTTP {response.status_code}: {body}",
                )
                retrying = True
                await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= max_retries:
                    if retrying:
                        await _notify_retry(
                            options,
                            stage="end",
                            attempt=attempt,
                            delay=0,
                            error=str(exc),
                        )
                    raise InferenceError(
                        "transport", f"{self.name} request failed: {exc}", cause=exc
                    ) from exc
                delay = min(
                    options.base_delay_ms / 1000 * (2**attempt),
                    options.max_retry_delay_ms / 1000,
                )
                await _notify_retry(
                    options,
                    stage="start",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(exc),
                )
                retrying = True
                await asyncio.sleep(delay)
        if response is None:
            raise InferenceError("transport", f"{self.name} request produced no response")

        text_parts: dict[int, str] = {}
        thinking_parts: dict[int, str] = {}
        calls: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage = Usage()
        saw_chunk = False
        yield AssistantStreamEvent("start")
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise InferenceError(
                        "invalid_response",
                        f"{self.name} emitted invalid SSE JSON",
                        cause=exc,
                    ) from exc
                saw_chunk = True
                if isinstance(chunk.get("error"), Mapping):
                    error = chunk["error"]
                    raise InferenceError(
                        "provider_error",
                        self._sanitize(
                            str(
                                error.get("message")
                                or error.get("code")
                                or f"{self.name} stream error"
                            )
                        ),
                    )
                if isinstance(chunk.get("usage"), Mapping):
                    usage = _price_usage(Usage.from_dict(chunk["usage"]), model)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts[0] = text_parts.get(0, "") + content
                    yield AssistantStreamEvent("text_delta", delta=content, content_index=0)
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    thinking_parts[0] = thinking_parts.get(0, "") + reasoning
                    yield AssistantStreamEvent("thinking_delta", delta=reasoning, content_index=0)
                for raw_call in delta.get("tool_calls") or []:
                    index = int(raw_call.get("index") or 0)
                    state = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    state["id"] += str(raw_call.get("id") or "")
                    function = raw_call.get("function") or {}
                    state["name"] += str(function.get("name") or "")
                    argument_delta = str(function.get("arguments") or "")
                    state["arguments"] += argument_delta
                    yield AssistantStreamEvent(
                        "toolcall_delta",
                        delta=argument_delta,
                        tool_call_index=index,
                        tool_call_id=state["id"] or None,
                        tool_name=state["name"] or None,
                    )
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
        except asyncio.CancelledError:
            raise
        except InferenceError:
            raise
        except httpx.HTTPError as exc:
            raise InferenceError(
                "transport", f"{self.name} stream failed: {exc}", cause=exc
            ) from exc
        finally:
            await response.aclose()

        if not saw_chunk:
            raise InferenceError(
                "invalid_response",
                f"{self.name} stream ended without any chunks",
            )

        assistant_content: list[TextContent | ThinkingContent | ToolCall] = []
        if thinking_parts:
            assistant_content.append(
                ThinkingContent("".join(thinking_parts[index] for index in sorted(thinking_parts)))
            )
        if text_parts:
            assistant_content.append(
                TextContent("".join(text_parts[index] for index in sorted(text_parts)))
            )
        for index in sorted(calls):
            state = calls[index]
            assistant_content.append(
                ToolCall(
                    id=state["id"] or f"call_{uuid.uuid4().hex[:12]}",
                    name=state["name"],
                    arguments=_parse_tool_arguments(state["arguments"]),
                )
            )
        stop_reason = {
            "tool_calls": "toolUse",
            "length": "length",
            "content_filter": "error",
            "stop": "stop",
        }.get(finish_reason, "stop")
        message = AssistantMessage(
            content=tuple(assistant_content),
            provider=model.provider,
            model=model.id,
            usage=usage,
            stop_reason=stop_reason,  # type: ignore[arg-type]
        )
        yield AssistantStreamEvent("done", message=message)

    async def _discover_models(self) -> list[Model]:
        headers = dict(self.headers)
        if self.api_key:
            headers.setdefault("authorization", f"Bearer {self.api_key}")
        client = await self._get_client()
        try:
            response = await client.get(f"{self.endpoint}/models", headers=headers, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise InferenceError(
                "model_discovery",
                f"Could not load models from {self.name}: {exc}",
                cause=exc,
            ) from exc
        values = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(values, list):
            raise InferenceError("model_discovery", f"{self.name} returned an invalid model list")
        models: list[Model] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, Mapping):
                continue
            raw_id = str(value.get("id") or "").strip().lstrip("/")
            if not raw_id:
                continue
            prefix = f"{self.id}/"
            local_id = raw_id.removeprefix(prefix)
            model_id = f"{prefix}{local_id}"
            if model_id in seen:
                continue
            seen.add(model_id)
            context_window = max(1, int(value.get("context_window") or 128_000))
            models.append(
                Model(
                    id=model_id,
                    name=str(value.get("name") or raw_id),
                    provider=self.id,
                    reasoning=bool(value.get("reasoning", False)),
                    input=("text", "image") if value.get("supports_image") else ("text",),
                    context_window=context_window,
                    max_tokens=min(
                        max(1, int(value.get("max_tokens") or 32_768)),
                        context_window,
                    ),
                    cost=dict(value.get("cost") or {}),
                )
            )
        if not models:
            raise InferenceError("model_discovery", f"{self.name} returned no usable models")
        return models

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                proxy=self.proxy,
                timeout=None,
            )
        return self._client

    def _sanitize(self, value: str) -> str:
        secrets = [self.api_key]
        secrets.extend(
            header_value for name, header_value in self.headers.items() if is_sensitive_header(name)
        )
        for secret in secrets:
            if secret:
                value = value.replace(secret, "***")
        return value


def _openai_payload(
    model: Model,
    context: InferenceContext,
    options: StreamOptions,
    *,
    thinking_level_map: Mapping[str, str | None],
    requires_reasoning_content: bool,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if context.system_prompt:
        messages.append({"role": "system", "content": context.system_prompt})
    messages.extend(
        _openai_message(
            message,
            model,
            requires_reasoning_content=requires_reasoning_content,
        )
        for message in context.messages
    )
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if context.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
            for tool in context.tools
        ]
    payload["max_tokens"] = options.max_tokens or model.max_tokens
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    mapped_thinking = thinking_level_map.get(options.thinking_level)
    if mapped_thinking:
        payload["reasoning_effort"] = mapped_thinking
    return payload


def _openai_message(
    message: AgentMessage,
    model: Model,
    *,
    requires_reasoning_content: bool,
) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return {"role": "user", "content": message.content}
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextContent):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent) and "image" in model.input:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mime_type};base64,{part.data}"},
                    }
                )
        return {"role": "user", "content": parts}
    if isinstance(message, ToolResultMessage):
        text = "\n".join(part.text for part in message.content if isinstance(part, TextContent))
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": text}
    text = "".join(part.text for part in message.content if isinstance(part, TextContent))
    thinking = "".join(
        part.thinking for part in message.content if isinstance(part, ThinkingContent)
    )
    calls = [part for part in message.content if isinstance(part, ToolCall)]
    value: dict[str, Any] = {"role": "assistant", "content": text or None}
    if thinking or requires_reasoning_content:
        value["reasoning_content"] = thinking
    if calls:
        value["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
            for call in calls
        ]
    return value


def _parse_tool_arguments(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _price_usage(usage: Usage, model: Model) -> Usage:
    rates = model.cost
    costs = {
        "input": usage.input * float(rates.get("input", 0)) / 1_000_000,
        "output": usage.output * float(rates.get("output", 0)) / 1_000_000,
        "cacheRead": usage.cache_read * float(rates.get("cacheRead", 0)) / 1_000_000,
        "cacheWrite": usage.cache_write * float(rates.get("cacheWrite", 0)) / 1_000_000,
    }
    costs["total"] = sum(costs.values())
    return replace(usage, cost=costs)


def _retryable_status(status: int) -> bool:
    return status in {408, 409, 429} or status >= 500


def _retry_delay(
    response: httpx.Response,
    attempt: int,
    base_delay_ms: int,
    max_delay_ms: int,
) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            parsed = float(retry_after)
            if math.isfinite(parsed) and parsed >= 0:
                return min(parsed, max_delay_ms / 1000)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                return min(
                    max(0.0, (retry_at - datetime.now(UTC)).total_seconds()),
                    max_delay_ms / 1000,
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return min(base_delay_ms / 1000 * (2**attempt), max_delay_ms / 1000)


async def _notify_retry(
    options: StreamOptions,
    *,
    stage: str,
    attempt: int,
    delay: float,
    error: str | None,
) -> None:
    callback = options.metadata.get("on_retry")
    if not callable(callback):
        return
    result = callback(
        {
            "stage": stage,
            "attempt": attempt,
            "delayMs": round(delay * 1000),
            "error": error,
        }
    )
    if inspect.isawaitable(result):
        await result


__all__ = ["OpenAICompatibleProvider"]
