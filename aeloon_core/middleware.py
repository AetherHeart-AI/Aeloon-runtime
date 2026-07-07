"""Middleware contracts for the standalone agent kernel."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from aeloon_core.providers.base import LLMResponse


class AgentMiddleware(Protocol):
    """Protocol for around-style middleware hooks in the agent kernel."""

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable[LLMResponse]],
    ) -> LLMResponse: ...

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str: ...


class BaseAgentMiddleware:
    """No-op base class for middleware implementations."""

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        return await call_llm(messages, tool_defs)

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str:
        return await execute()
