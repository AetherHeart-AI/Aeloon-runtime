"""Direct OpenAI-compatible provider."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from openai import AsyncOpenAI

from aeloon_core.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ResponseFormat,
    ToolArgumentsError,
    ToolCallRequest,
)

_UNSUPPORTED_TOOL_MARKERS = (
    "tool choice requires",
    "enable-auto-tool-choice",
    "tool-call-parser",
)
_COMPLETE_FINISH_REASONS = frozenset({"stop", "tool_calls"})
_MISSING_FINISH_REASON = "unknown"


def _raw_argument_chars(raw_arguments: Any) -> int:
    return len(raw_arguments) if isinstance(raw_arguments, str) else 0


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _decode_tool_arguments(
    raw_arguments: Any,
) -> tuple[dict[str, Any], ToolArgumentsError | None]:
    """Strictly decode one JSON-object argument payload without retaining invalid input."""

    raw_chars = _raw_argument_chars(raw_arguments)
    if not isinstance(raw_arguments, str):
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message="Tool arguments must be a JSON string encoding an object.",
            position=None,
            raw_chars=raw_chars,
        )
    try:
        decoded = json.loads(raw_arguments, parse_constant=_reject_non_json_constant)
    except json.JSONDecodeError as exc:
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message=f"Tool arguments are not valid JSON: {exc.msg}.",
            position=exc.pos,
            raw_chars=raw_chars,
        )
    except ValueError as exc:
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message=f"Tool arguments are not valid JSON: {exc}.",
            position=None,
            raw_chars=raw_chars,
        )
    if not isinstance(decoded, dict):
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_NOT_OBJECT",
            message=(
                "Tool arguments must decode to a JSON object, "
                f"not {type(decoded).__name__}."
            ),
            position=None,
            raw_chars=raw_chars,
        )
    return decoded, None


def _generation_incomplete_error(
    finish_reason: str,
    raw_arguments: Any,
) -> ToolArgumentsError:
    return ToolArgumentsError(
        code="GENERATION_INCOMPLETE",
        message=(
            f"Tool-call generation ended with finish_reason={finish_reason!r}; "
            "the call was not executed."
        ),
        position=None,
        raw_chars=_raw_argument_chars(raw_arguments),
    )


def _tool_call_request(
    *,
    call_id: Any,
    name: Any,
    raw_arguments: Any,
    finish_reason: str,
) -> ToolCallRequest:
    valid_call_id = (
        isinstance(call_id, str) and bool(call_id) and call_id == call_id.strip()
    )
    valid_name = isinstance(name, str) and bool(name) and name == name.strip()
    resolved_call_id = call_id if valid_call_id else f"invalid-{uuid.uuid4().hex[:9]}"
    resolved_name = name if valid_name else "invalid_tool_call"
    if finish_reason not in _COMPLETE_FINISH_REASONS:
        arguments = {}
        arguments_error = _generation_incomplete_error(finish_reason, raw_arguments)
    elif not valid_call_id or not valid_name:
        missing = ", ".join(
            field
            for field, valid in (("id", valid_call_id), ("function name", valid_name))
            if not valid
        )
        arguments = {}
        arguments_error = ToolArgumentsError(
            code="TOOL_CALL_INCOMPLETE",
            message=f"Tool call is missing a non-empty {missing}.",
            position=None,
            raw_chars=_raw_argument_chars(raw_arguments),
        )
    else:
        arguments, arguments_error = _decode_tool_arguments(raw_arguments)
    return ToolCallRequest(
        id=resolved_call_id,
        name=resolved_name,
        arguments=arguments,
        arguments_error=arguments_error,
    )


def _reject_invalid_tool_batch(
    tool_calls: list[ToolCallRequest],
    raw_char_counts: list[int],
) -> list[ToolCallRequest]:
    """Reject otherwise-valid calls when any call in a complete batch is malformed."""

    first_error = next(
        (
            call.arguments_error
            for call in tool_calls
            if call.arguments_error is not None
            and call.arguments_error.code
            in {
                "TOOL_ARGUMENTS_INVALID_JSON",
                "TOOL_ARGUMENTS_NOT_OBJECT",
                "TOOL_CALL_INCOMPLETE",
            }
        ),
        None,
    )
    if first_error is None:
        return tool_calls
    for call, raw_chars in zip(tool_calls, raw_char_counts, strict=True):
        if call.arguments_error is not None:
            continue
        call.arguments = {}
        call.arguments_error = ToolArgumentsError(
            code="TOOL_BATCH_REJECTED",
            message=(
                "Tool batch was rejected because another call failed with "
                f"{first_error.code}."
            ),
            position=None,
            raw_chars=raw_chars,
        )
    return tool_calls


def _is_tooling_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _UNSUPPORTED_TOOL_MARKERS)


class CustomProvider(LLMProvider):
    """OpenAI-compatible chat provider."""

    supports_concurrent_calls = True

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
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max(1, max_tokens)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    async def _create_with_tool_fallback(
        self,
        kwargs: dict[str, Any],
        run: Callable[[dict[str, Any]], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        """Run a completion, retrying without tools if the route rejects tooling."""

        try:
            return await run(kwargs)
        except Exception as exc:
            if kwargs.get("tools") and _is_tooling_unsupported_error(exc):
                fallback = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
                try:
                    return await run(fallback)
                except Exception as fallback_error:
                    return LLMResponse(content=f"Error: {fallback_error}", finish_reason="error")
            return LLMResponse(content=f"Error: {exc}", finish_reason="error")

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
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        async def _run(kw: dict[str, Any]) -> LLMResponse:
            return self._parse(await self._client.chat.completions.create(**kw))

        return await self._create_with_tool_fallback(kwargs, _run)

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
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        async def _run(kw: dict[str, Any]) -> LLMResponse:
            stream = await self._client.chat.completions.create(**kw)
            if not hasattr(stream, "__aiter__"):
                return self._parse(stream)
            return await self._collect_stream(
                stream,
                on_delta=on_delta,
                on_reasoning_delta=on_reasoning_delta,
            )

        return await self._create_with_tool_fallback(kwargs, _run)

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
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        stream_error: Exception | None = None
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
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stream_error = exc
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if stream_error is None:
                        stream_error = exc

        if stream_error is not None:
            detail = f"{type(stream_error).__name__}: {stream_error}"
            return LLMResponse(
                content=f"Error calling LLM: stream ended before completion ({detail})",
                tool_calls=self._stream_tool_calls(tool_call_parts, "error"),
                finish_reason="error",
                usage=usage,
            )

        resolved_finish_reason = finish_reason or _MISSING_FINISH_REASON
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=self._stream_tool_calls(tool_call_parts, resolved_finish_reason),
            finish_reason=resolved_finish_reason,
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
    def _stream_tool_calls(
        parts: dict[int, dict[str, str]],
        finish_reason: str = "tool_calls",
    ) -> list[ToolCallRequest]:
        tool_calls: list[ToolCallRequest] = []
        raw_char_counts: list[int] = []
        for index in sorted(parts):
            entry = parts[index]
            name = entry.get("name") or ""
            raw_arguments = entry.get("arguments") or ""
            tool_calls.append(
                _tool_call_request(
                    call_id=entry.get("id"),
                    name=name,
                    raw_arguments=raw_arguments,
                    finish_reason=finish_reason,
                )
            )
            raw_char_counts.append(_raw_argument_chars(raw_arguments))
        return _reject_invalid_tool_batch(tool_calls, raw_char_counts)

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
        finish_reason = choice.finish_reason or _MISSING_FINISH_REASON
        tool_calls: list[ToolCallRequest] = []
        raw_char_counts: list[int] = []
        for tc in msg.tool_calls or []:
            function = getattr(tc, "function", None)
            raw_arguments = getattr(function, "arguments", None)
            tool_calls.append(
                _tool_call_request(
                    call_id=getattr(tc, "id", None),
                    name=getattr(function, "name", None),
                    raw_arguments=raw_arguments,
                    finish_reason=finish_reason,
                )
            )
            raw_char_counts.append(_raw_argument_chars(raw_arguments))
        _reject_invalid_tool_batch(tool_calls, raw_char_counts)
        usage = response.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )
