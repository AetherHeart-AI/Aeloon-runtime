"""Base LLM provider interface."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from loguru import logger

from aeloon_core.transitions import accumulate_usage

ResponseFormat = dict[str, str]

_TOOL_ARGUMENTS_ERROR_CODE_MAX_CHARS = 64
_TOOL_ARGUMENTS_ERROR_MESSAGE_MAX_CHARS = 240


@dataclass(frozen=True)
class ToolArgumentsError:
    """Bounded metadata describing why tool arguments cannot be executed."""

    code: str
    message: str
    position: int | None
    raw_chars: int

    def __post_init__(self) -> None:
        raw_chars = max(0, self.raw_chars)
        position = self.position
        if position is not None:
            position = min(max(0, position), raw_chars)
        object.__setattr__(
            self,
            "code",
            self.code[:_TOOL_ARGUMENTS_ERROR_CODE_MAX_CHARS],
        )
        object.__setattr__(
            self,
            "message",
            self.message[:_TOOL_ARGUMENTS_ERROR_MESSAGE_MAX_CHARS],
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "raw_chars", raw_chars)

    def to_dict(self) -> dict[str, str | int | None]:
        """Return JSON-compatible error metadata without raw arguments."""

        return {
            "code": self.code,
            "message": self.message,
            "position": self.position,
            "raw_chars": self.raw_chars,
        }


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    arguments_error: ToolArgumentsError | None = None

    def __post_init__(self) -> None:
        if self.arguments_error is not None:
            self.arguments = {}

    def to_anthropic_tool_use(
        self,
        *,
        input_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize to an Anthropic ``tool_use`` content block."""

        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.arguments if input_override is None else input_override,
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None
    thinking_blocks: list[dict] | None = None


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls."""

    temperature: float = 0.7
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    supports_concurrent_calls: ClassVar[bool] = False

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = (
        "rate limit",
        "overloaded",
        "connection",
        "server error",
        "temporarily unavailable",
    )
    _TRANSIENT_STATUS_RE = re.compile(
        r"\b(?:error code|status(?: code)?|http)\s*[:=]?\s*(?:429|500|502|503|504)\b"
    )
    _SENTINEL = object()

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        proxy: str | None = None,
        generation: GenerationSettings | None = None,
        chat_timeout: int = 3600,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.proxy = proxy
        self.generation = generation or GenerationSettings()
        self.chat_timeout = chat_timeout

    @abstractmethod
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
        """Send an Anthropic Messages API request."""

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return bool(cls._TRANSIENT_STATUS_RE.search(err)) or any(
            marker in err for marker in cls._TRANSIENT_ERROR_MARKERS
        )

    async def _safe_call(self, coro: Awaitable[LLMResponse]) -> LLMResponse:
        """Await an LLM coroutine with a timeout, converting failures to errors."""

        try:
            return await asyncio.wait_for(coro, timeout=self.chat_timeout)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return LLMResponse(
                content=f"Error calling LLM: request timed out after {self.chat_timeout}s",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    def _resolved_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: object,
        reasoning_effort: object,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Fill unset generation params from provider defaults."""

        return dict(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=(
                self.generation.temperature if temperature is self._SENTINEL else temperature
            ),
            reasoning_effort=(
                self.generation.reasoning_effort
                if reasoning_effort is self._SENTINEL
                else reasoning_effort
            ),
            tool_choice=tool_choice,
        )

    async def _retry(
        self,
        attempt_call: Callable[[int], Awaitable[LLMResponse]],
        *,
        label: str,
    ) -> LLMResponse:
        """Invoke attempt_call with retry on transient provider failures."""

        accumulated_usage: dict[str, int] = {}
        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await attempt_call(attempt)
            accumulate_usage(accumulated_usage, response.usage)
            if response.finish_reason != "error" or not self._is_transient_error(response.content):
                response.usage = accumulated_usage
                return response
            logger.warning(
                "{} transient error (attempt {}/{}), retrying in {}s: {}",
                label,
                attempt,
                len(self._CHAT_RETRY_DELAYS),
                delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(delay)
        response = await attempt_call(len(self._CHAT_RETRY_DELAYS) + 1)
        accumulate_usage(accumulated_usage, response.usage)
        response.usage = accumulated_usage
        return response

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
        """Stream an Anthropic Messages API response when supported."""

        del on_delta, on_reasoning_delta
        return await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient provider failures."""

        kw = self._resolved_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if response_format is not None:
            kw["response_format"] = response_format

        return await self._retry(lambda _attempt: self._safe_call(self.chat(**kw)), label="LLM")

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient provider failures."""

        kw = self._resolved_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

        async def _attempt(attempt: int) -> LLMResponse:
            # Only stream deltas on the first try; retries collect silently.
            return await self._safe_call(
                self.chat_stream(
                    **kw,
                    on_delta=on_delta if attempt == 1 else None,
                    on_reasoning_delta=on_reasoning_delta if attempt == 1 else None,
                )
            )

        return await self._retry(_attempt, label="Streaming LLM")
