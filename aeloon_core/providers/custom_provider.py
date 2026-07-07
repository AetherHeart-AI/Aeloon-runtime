"""Direct OpenAI-compatible provider."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import json_repair
from openai import AsyncOpenAI

from aeloon_core.model_metadata import resolve_max_tokens_for_model
from aeloon_core.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ResponseFormat,
    ToolCallRequest,
)

_UNSUPPORTED_TOOL_MARKERS = (
    "tool choice requires",
    "enable-auto-tool-choice",
    "tool-call-parser",
)


def _is_tooling_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _UNSUPPORTED_TOOL_MARKERS)


class CustomProvider(LLMProvider):
    """OpenAI-compatible chat provider."""

    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
        extra_headers: dict[str, str] | None = None,
        proxy: str | None = None,
        generation: GenerationSettings | None = None,
        chat_timeout: int = 3600,
    ) -> None:
        super().__init__(
            api_key,
            api_base,
            proxy=proxy,
            generation=generation,
            chat_timeout=chat_timeout,
        )
        self.default_model = default_model
        default_headers = {
            "x-session-affinity": uuid.uuid4().hex,
            **(extra_headers or {}),
        }
        self._http_client = httpx.AsyncClient(proxy=proxy) if proxy else None
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": api_base,
            "default_headers": default_headers,
        }
        if self._http_client is not None:
            client_kwargs["http_client"] = self._http_client
        self._client = AsyncOpenAI(**client_kwargs)

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    async def _resolve_max_tokens(self, model: str, max_tokens: int | None) -> int:
        return await resolve_max_tokens_for_model(
            model,
            max_tokens,
            api_base=self.api_base,
            http_client=self._http_client,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        resolved_model = model or self.default_model
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=resolved_model,
            max_tokens=await self._resolve_max_tokens(resolved_model, max_tokens),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as exc:
            if tools and _is_tooling_unsupported_error(exc):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("tools", None)
                fallback_kwargs.pop("tool_choice", None)
                try:
                    return self._parse(
                        await self._client.chat.completions.create(**fallback_kwargs)
                    )
                except Exception as fallback_error:
                    return LLMResponse(
                        content=f"Error: {fallback_error}",
                        finish_reason="error",
                    )
            return LLMResponse(content=f"Error: {exc}", finish_reason="error")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        resolved_model = model or self.default_model
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=resolved_model,
            max_tokens=await self._resolve_max_tokens(resolved_model, max_tokens),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            if not hasattr(stream, "__aiter__"):
                return self._parse(stream)
            return await self._collect_stream(
                stream,
                on_delta=on_delta,
                on_reasoning_delta=on_reasoning_delta,
            )
        except Exception as exc:
            if tools and _is_tooling_unsupported_error(exc):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("tools", None)
                fallback_kwargs.pop("tool_choice", None)
                try:
                    stream = await self._client.chat.completions.create(**fallback_kwargs)
                    if not hasattr(stream, "__aiter__"):
                        return self._parse(stream)
                    return await self._collect_stream(
                        stream,
                        on_delta=on_delta,
                        on_reasoning_delta=on_reasoning_delta,
                    )
                except Exception as fallback_error:
                    return LLMResponse(
                        content=f"Error: {fallback_error}",
                        finish_reason="error",
                    )
            return LLMResponse(content=f"Error: {exc}", finish_reason="error")

    async def _collect_stream(
        self,
        stream: Any,
        *,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        try:
            async for chunk in stream:
                usage = self._usage_dict(getattr(chunk, "usage", None)) or usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    if on_delta is not None:
                        await on_delta(content)

                reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                    if on_reasoning_delta is not None:
                        await on_reasoning_delta(reasoning)

                for tool_call_delta in getattr(delta, "tool_calls", None) or []:
                    self._accumulate_tool_call_delta(tool_call_parts, tool_call_delta)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=self._stream_tool_calls(tool_call_parts),
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
        )

    @staticmethod
    def _accumulate_tool_call_delta(
        parts: dict[int, dict[str, str]],
        tool_call_delta: Any,
    ) -> None:
        index = getattr(tool_call_delta, "index", len(parts))
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = len(parts)
        entry = parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
        tool_call_id = getattr(tool_call_delta, "id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            entry["id"] = tool_call_id

        function = getattr(tool_call_delta, "function", None)
        if function is None:
            return
        name = getattr(function, "name", None)
        if isinstance(name, str) and name:
            entry["name"] += name
        arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str) and arguments:
            entry["arguments"] += arguments

    @staticmethod
    def _stream_tool_calls(parts: dict[int, dict[str, str]]) -> list[ToolCallRequest]:
        tool_calls: list[ToolCallRequest] = []
        for index in sorted(parts):
            entry = parts[index]
            name = entry.get("name") or ""
            if not name:
                continue
            raw_args = (entry.get("arguments") or "").strip()
            try:
                arguments = json_repair.loads(raw_args) if raw_args else {}
            except Exception:
                arguments = {"_raw": raw_args}
            tool_calls.append(
                ToolCallRequest(
                    id=entry.get("id") or uuid.uuid4().hex[:9],
                    name=name,
                    arguments=arguments,
                )
            )
        return tool_calls

    @staticmethod
    def _usage_dict(usage: Any) -> dict[str, int]:
        if not usage:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _parse(self, response: Any) -> LLMResponse:
        if not response.choices:
            return LLMResponse(
                content=(
                    "Error: API returned empty choices. This may indicate a temporary "
                    "service issue or an invalid model response."
                ),
                finish_reason="error",
            )
        choice = response.choices[0]
        msg = choice.message
        tool_calls: list[ToolCallRequest] = []
        for tc in msg.tool_calls or []:
            raw_arguments = tc.function.arguments
            arguments: dict[str, Any] | list[Any] | None
            if isinstance(raw_arguments, str):
                loaded_arguments = json_repair.loads(raw_arguments)
                arguments = loaded_arguments if isinstance(loaded_arguments, dict | list) else None
            elif isinstance(raw_arguments, dict | list):
                arguments = raw_arguments
            else:
                arguments = None
            tool_calls.append(
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )
        usage = response.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )
