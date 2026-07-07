"""Base LLM provider interface."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

ResponseFormat = dict[str, str]


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] | list[Any] | None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.arguments, dict):
            return
        if self.arguments is None:
            self.arguments = {}
            return
        if isinstance(self.arguments, list):
            if len(self.arguments) == 1 and isinstance(self.arguments[0], dict):
                self.arguments = self.arguments[0]
                return
            if self.arguments and all(isinstance(item, dict) for item in self.arguments):
                merged: dict[str, Any] = {}
                for item in self.arguments:
                    merged.update(item)
                self.arguments = merged

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to an OpenAI-style tool_call payload."""

        tool_call: dict[str, Any] = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = (
                self.function_provider_specific_fields
            )
        return tool_call


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None
    thinking_blocks: list[dict] | None = None


class ProviderAuthenticationError(RuntimeError):
    """Provider credentials are missing or expired."""


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls."""

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "connection",
        "server error",
        "temporarily unavailable",
    )
    _SENTINEL = object()

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        proxy: str | None = None,
        generation: GenerationSettings | None = None,
        chat_timeout: int = 3600,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.proxy = proxy
        self.generation = generation or GenerationSettings()
        self.chat_timeout = chat_timeout

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = (
                    None
                    if (msg.get("role") == "assistant" and msg.get("tool_calls"))
                    else "(empty)"
                )
                result.append(clean)
                continue
            if isinstance(content, list):
                new_items: list[Any] = []
                changed = False
                for item in content:
                    if isinstance(item, dict) and "_meta" in item:
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(item)
                if changed:
                    clean = dict(msg)
                    clean["content"] = new_items or "(empty)"
                    result.append(clean)
                    continue
            result.append(msg)
        return result

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Send a chat completion request."""

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    async def _safe_chat(self, **kwargs: Any) -> LLMResponse:
        try:
            return await asyncio.wait_for(self.chat(**kwargs), timeout=self.chat_timeout)
        except asyncio.CancelledError:
            raise
        except ProviderAuthenticationError:
            raise
        except TimeoutError:
            return LLMResponse(
                content=f"Error calling LLM: request timed out after {self.chat_timeout}s",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream a chat completion when a provider supports it."""

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

    async def _safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        try:
            return await asyncio.wait_for(self.chat_stream(**kwargs), timeout=self.chat_timeout)
        except asyncio.CancelledError:
            raise
        except ProviderAuthenticationError:
            raise
        except TimeoutError:
            return LLMResponse(
                content=f"Error calling LLM: request timed out after {self.chat_timeout}s",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient provider failures."""

        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
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

        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat(**kw)
            if response.finish_reason != "error":
                return response
            if not self._is_transient_error(response.content):
                return response
            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {}s: {}",
                attempt,
                len(self._CHAT_RETRY_DELAYS),
                delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(delay)
        return await self._safe_chat(**kw)

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient provider failures."""

        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat_stream(
                **kw,
                on_delta=on_delta if attempt == 1 else None,
                on_reasoning_delta=on_reasoning_delta if attempt == 1 else None,
            )
            if response.finish_reason != "error":
                return response
            if not self._is_transient_error(response.content):
                return response
            logger.warning(
                "Streaming LLM transient error (attempt {}/{}), retrying in {}s: {}",
                attempt,
                len(self._CHAT_RETRY_DELAYS),
                delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(delay)
        return await self._safe_chat_stream(**kw, on_delta=None, on_reasoning_delta=None)
