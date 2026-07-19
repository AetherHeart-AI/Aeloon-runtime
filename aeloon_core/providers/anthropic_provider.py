"""Direct Anthropic Messages API provider."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from anthropic import AsyncAnthropic

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
    "tools are not supported",
    "tool use is not supported",
)
_THINKING_TOOL_CHOICE_MARKER = "incompatible with thinking enabled"
_COMPLETE_TOOL_STOP_REASONS = frozenset({"tool_use"})
_MISSING_STOP_REASON = "unknown"
_DEFAULT_MAX_TOKENS = 8_192


def _raw_argument_chars(raw_arguments: Any) -> int:
    if isinstance(raw_arguments, str):
        return len(raw_arguments)
    if isinstance(raw_arguments, dict):
        return len(json.dumps(raw_arguments, ensure_ascii=False, default=str))
    return 0


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _decode_tool_arguments(
    raw_arguments: Any,
) -> tuple[dict[str, Any], ToolArgumentsError | None]:
    """Strictly decode one Anthropic tool input without retaining invalid input."""

    raw_chars = _raw_argument_chars(raw_arguments)
    if isinstance(raw_arguments, dict):
        return raw_arguments, None
    if not isinstance(raw_arguments, str):
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message="Tool input must be a JSON object.",
            position=None,
            raw_chars=raw_chars,
        )
    try:
        decoded = json.loads(raw_arguments, parse_constant=_reject_non_json_constant)
    except json.JSONDecodeError as exc:
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message=f"Tool input is not valid JSON: {exc.msg}.",
            position=exc.pos,
            raw_chars=raw_chars,
        )
    except ValueError as exc:
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message=f"Tool input is not valid JSON: {exc}.",
            position=None,
            raw_chars=raw_chars,
        )
    if not isinstance(decoded, dict):
        return {}, ToolArgumentsError(
            code="TOOL_ARGUMENTS_NOT_OBJECT",
            message=(
                "Tool input must decode to a JSON object, "
                f"not {type(decoded).__name__}."
            ),
            position=None,
            raw_chars=raw_chars,
        )
    return decoded, None


def _generation_incomplete_error(
    stop_reason: str,
    raw_arguments: Any,
) -> ToolArgumentsError:
    return ToolArgumentsError(
        code="GENERATION_INCOMPLETE",
        message=(
            f"Tool-use generation ended with stop_reason={stop_reason!r}; "
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
    stop_reason: str,
) -> ToolCallRequest:
    valid_call_id = (
        isinstance(call_id, str) and bool(call_id) and call_id == call_id.strip()
    )
    valid_name = isinstance(name, str) and bool(name) and name == name.strip()
    resolved_call_id = call_id if valid_call_id else f"invalid-{uuid.uuid4().hex[:9]}"
    resolved_name = name if valid_name else "invalid_tool_call"
    if stop_reason not in _COMPLETE_TOOL_STOP_REASONS:
        arguments = {}
        arguments_error = _generation_incomplete_error(stop_reason, raw_arguments)
    elif not valid_call_id or not valid_name:
        missing = ", ".join(
            field
            for field, valid in (("id", valid_call_id), ("tool name", valid_name))
            if not valid
        )
        arguments = {}
        arguments_error = ToolArgumentsError(
            code="TOOL_CALL_INCOMPLETE",
            message=f"Tool use is missing a non-empty {missing}.",
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
    """Reject otherwise-valid uses when any use in a complete batch is malformed."""

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
                "Tool batch was rejected because another use failed with "
                f"{first_error.code}."
            ),
            position=None,
            raw_chars=raw_chars,
        )
    return tool_calls


def _is_tooling_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _UNSUPPORTED_TOOL_MARKERS)


class AnthropicProvider(LLMProvider):
    """Anthropic-compatible Messages API provider."""

    supports_concurrent_calls = True

    def __init__(
        self,
        api_key: str = "no-key",
        base_url: str = "https://api.anthropic.com",
        default_model: str = "claude-sonnet-4-6",
        extra_headers: dict[str, str] | None = None,
        proxy: str | None = None,
        generation: GenerationSettings | None = None,
        chat_timeout: int = 3600,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
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
            "base_url": base_url,
            "default_headers": default_headers,
        }
        if self._http_client is not None:
            client_kwargs["http_client"] = self._http_client
        self._client = AsyncAnthropic(**client_kwargs)

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
        system, conversation = self._prepare_messages(messages)
        if response_format == {"type": "json_object"}:
            system.append(
                {
                    "type": "text",
                    "text": "Return exactly one valid JSON object and no surrounding text.",
                }
            )
        configured_model = model or self.default_model
        resolved_model = _api_model_id(configured_model)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": conversation,
            "max_tokens": max(1, max_tokens or _DEFAULT_MAX_TOKENS),
        }
        # Current Claude models reject sampling overrides. Anthropic-compatible
        # third-party models such as Kimi still accept the existing setting.
        if not resolved_model.startswith("claude-"):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if reasoning_effort:
            kwargs["output_config"] = {"effort": reasoning_effort}
        if tools:
            kwargs["tools"] = tools
        resolved_tool_choice = self._tool_choice(tool_choice)
        if resolved_tool_choice is not None and tools:
            kwargs["tool_choice"] = resolved_tool_choice
        return kwargs

    @staticmethod
    def _prepare_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        system: list[dict[str, str]] = []
        conversation: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if isinstance(content, str) and content:
                    system.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    system.extend(
                        {"type": "text", "text": str(block.get("text") or "")}
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and block.get("text")
                    )
                continue
            if role not in {"user", "assistant"}:
                continue
            if isinstance(content, str):
                clean_content: str | list[dict[str, Any]] = content or "(empty)"
            elif isinstance(content, list):
                clean_content = [
                    {key: value for key, value in block.items() if key != "_meta"}
                    for block in content
                    if isinstance(block, dict)
                ] or [{"type": "text", "text": "(empty)"}]
            else:
                clean_content = "(empty)"
            conversation.append({"role": role, "content": clean_content})
        if not conversation:
            conversation.append({"role": "user", "content": "(empty)"})
        return system, conversation

    @staticmethod
    def _tool_choice(
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            choice = {
                "auto": "auto",
                "none": "none",
                "any": "any",
                "required": "any",
            }.get(tool_choice)
            return {"type": choice} if choice else None
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none", "any"}:
            return dict(tool_choice)
        if choice_type == "tool" and isinstance(tool_choice.get("name"), str):
            return dict(tool_choice)
        function = tool_choice.get("function")
        if choice_type == "function" and isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return {"type": "tool", "name": name}
        return None

    async def _create_with_tool_fallback(
        self,
        kwargs: dict[str, Any],
        run: Callable[[dict[str, Any]], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        """Run a message request, retrying without tools when unsupported."""

        try:
            return await run(kwargs)
        except Exception as exc:
            if kwargs.get("tool_choice") and _THINKING_TOOL_CHOICE_MARKER in str(exc).lower():
                fallback = dict(kwargs)
                fallback.pop("tool_choice", None)
                try:
                    return await run(fallback)
                except Exception as fallback_error:
                    return LLMResponse(
                        content=f"Error: {fallback_error}",
                        finish_reason="error",
                    )
            if kwargs.get("tools") and _is_tooling_unsupported_error(exc):
                fallback = {key: value for key, value in kwargs.items() if key != "tools"}
                fallback.pop("tool_choice", None)
                try:
                    return await run(fallback)
                except Exception as fallback_error:
                    return LLMResponse(
                        content=f"Error: {fallback_error}",
                        finish_reason="error",
                    )
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
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        async def _run(kw: dict[str, Any]) -> LLMResponse:
            return self._parse(await self._client.messages.create(**kw))

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
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        kwargs["stream"] = True

        async def _run(kw: dict[str, Any]) -> LLMResponse:
            stream = await self._client.messages.create(**kw)
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
        tool_parts: dict[int, dict[str, Any]] = {}
        thinking_parts: dict[int, dict[str, Any]] = {}
        stop_reason: str | None = None
        usage: dict[str, int] = {}

        stream_error: Exception | None = None
        try:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "message_start":
                    usage = self._usage_dict(getattr(event.message, "usage", None)) or usage
                    continue
                if event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", None) or stop_reason
                    usage = self._merge_usage(
                        usage,
                        self._usage_dict(getattr(event, "usage", None)),
                    )
                    continue
                if event_type == "content_block_start":
                    self._start_content_block(
                        tool_parts,
                        thinking_parts,
                        int(getattr(event, "index", 0) or 0),
                        getattr(event, "content_block", None),
                    )
                    continue
                if event_type != "content_block_delta":
                    continue
                index = int(getattr(event, "index", 0) or 0)
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", None)
                if delta_type == "text_delta":
                    text = getattr(delta, "text", None)
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                        if on_delta is not None:
                            await on_delta(text)
                elif delta_type == "thinking_delta":
                    thinking = getattr(delta, "thinking", None)
                    if isinstance(thinking, str) and thinking:
                        reasoning_parts.append(thinking)
                        entry = thinking_parts.setdefault(
                            index,
                            {"type": "thinking", "thinking": "", "signature": ""},
                        )
                        entry["thinking"] += thinking
                        if on_reasoning_delta is not None:
                            await on_reasoning_delta(thinking)
                elif delta_type == "signature_delta":
                    signature = getattr(delta, "signature", None)
                    if isinstance(signature, str):
                        entry = thinking_parts.setdefault(
                            index,
                            {"type": "thinking", "thinking": "", "signature": ""},
                        )
                        entry["signature"] += signature
                elif delta_type == "input_json_delta":
                    partial_json = getattr(delta, "partial_json", None)
                    if isinstance(partial_json, str):
                        tool_parts.setdefault(
                            index,
                            {"id": "", "name": "", "input": ""},
                        )["input"] += partial_json
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
                tool_calls=self._stream_tool_calls(tool_parts, "error"),
                finish_reason="error",
                usage=usage,
                thinking_blocks=self._thinking_blocks(thinking_parts),
            )

        resolved_stop_reason = stop_reason or _MISSING_STOP_REASON
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=self._stream_tool_calls(tool_parts, resolved_stop_reason),
            finish_reason=resolved_stop_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
            thinking_blocks=self._thinking_blocks(thinking_parts),
        )

    @staticmethod
    def _start_content_block(
        tool_parts: dict[int, dict[str, Any]],
        thinking_parts: dict[int, dict[str, Any]],
        index: int,
        block: Any,
    ) -> None:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            initial_input = getattr(block, "input", None)
            tool_parts[index] = {
                "id": getattr(block, "id", "") or "",
                "name": getattr(block, "name", "") or "",
                "input": (
                    json.dumps(initial_input, ensure_ascii=False)
                    if isinstance(initial_input, dict) and initial_input
                    else ""
                ),
            }
        elif block_type in {"thinking", "redacted_thinking"}:
            thinking_parts[index] = _model_dump(block)

    @staticmethod
    def _thinking_blocks(parts: dict[int, dict[str, Any]]) -> list[dict[str, Any]] | None:
        blocks = [parts[index] for index in sorted(parts)]
        return blocks or None

    @staticmethod
    def _stream_tool_calls(
        parts: dict[int, dict[str, Any]],
        stop_reason: str,
    ) -> list[ToolCallRequest]:
        tool_calls: list[ToolCallRequest] = []
        raw_char_counts: list[int] = []
        for index in sorted(parts):
            entry = parts[index]
            raw_arguments = entry.get("input") or ""
            tool_calls.append(
                _tool_call_request(
                    call_id=entry.get("id"),
                    name=entry.get("name"),
                    raw_arguments=raw_arguments,
                    stop_reason=stop_reason,
                )
            )
            raw_char_counts.append(_raw_argument_chars(raw_arguments))
        return _reject_invalid_tool_batch(tool_calls, raw_char_counts)

    @staticmethod
    def _usage_dict(usage: Any) -> dict[str, int]:
        if not usage:
            return {}
        input_tokens = int(_field(usage, "input_tokens", 0) or 0)
        output_tokens = int(_field(usage, "output_tokens", 0) or 0)
        cache_creation = int(_field(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(_field(usage, "cache_read_input_tokens", 0) or 0)
        prompt_tokens = input_tokens + cache_creation + cache_read
        result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }
        if cache_creation:
            result["cache_creation_input_tokens"] = cache_creation
        if cache_read:
            result["cache_read_input_tokens"] = cache_read
        return result

    @staticmethod
    def _merge_usage(current: dict[str, int], update: dict[str, int]) -> dict[str, int]:
        if not update:
            return current
        merged = dict(current)
        for key, value in update.items():
            if value or key not in merged:
                merged[key] = value
        prompt_tokens = merged.get("prompt_tokens", 0)
        completion_tokens = merged.get("completion_tokens", 0)
        merged["total_tokens"] = prompt_tokens + completion_tokens
        return merged

    def _parse(self, response: Any) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        thinking_blocks: list[dict[str, Any]] = []
        raw_tool_uses: list[tuple[Any, Any, Any]] = []
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    content_parts.append(text)
            elif block_type == "thinking":
                thinking = getattr(block, "thinking", None)
                if isinstance(thinking, str):
                    reasoning_parts.append(thinking)
                thinking_blocks.append(_model_dump(block))
            elif block_type == "redacted_thinking":
                thinking_blocks.append(_model_dump(block))
            elif block_type == "tool_use":
                raw_tool_uses.append(
                    (
                        getattr(block, "id", None),
                        getattr(block, "name", None),
                        getattr(block, "input", None),
                    )
                )
        stop_reason = getattr(response, "stop_reason", None) or _MISSING_STOP_REASON
        tool_calls = [
            _tool_call_request(
                call_id=call_id,
                name=name,
                raw_arguments=arguments,
                stop_reason=stop_reason,
            )
            for call_id, name, arguments in raw_tool_uses
        ]
        _reject_invalid_tool_batch(
            tool_calls,
            [_raw_argument_chars(arguments) for _, _, arguments in raw_tool_uses],
        )
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=stop_reason,
            usage=self._usage_dict(getattr(response, "usage", None)),
            reasoning_content="".join(reasoning_parts) or None,
            thinking_blocks=thinking_blocks or None,
        )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _api_model_id(model: str) -> str:
    """Translate Claude Code's Kimi context suffix to the actual API model ID."""

    return "k3" if model == "k3[1m]" else model


def _model_dump(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: child for key, child in value.items() if child is not None}
    return {
        key: child
        for key, child in vars(value).items()
        if child is not None and not key.startswith("_")
    }
