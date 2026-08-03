"""Event publication and hook dispatch for the agent harness."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

from aeloon_core.harness.types import (
    EventListener,
    HarnessEvent,
    HarnessEventType,
    HookHandler,
)


class HarnessEventDispatcher:
    """Own event subscribers and composable lifecycle hooks.

    Listeners observe events and are intentionally isolated from the harness:
    a broken listener cannot interrupt a run. Hooks participate in the run and
    therefore propagate failures to their caller.
    """

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._handlers: dict[str, list[HookHandler]] = defaultdict(list)

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

    async def emit(
        self,
        event_type: HarnessEventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = HarnessEvent(event_type, data or {})
        for listener in tuple(self._listeners):
            try:
                result = listener(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue

    async def hook(
        self,
        event_type: HarnessEventType,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        event = HarnessEvent(event_type, data)
        await self.emit(event_type, data)
        merged: dict[str, Any] = {}
        for handler in tuple(self._handlers.get(event_type, ())):
            result = handler(event)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict):
                merged.update(result)
        return merged


__all__ = ["HarnessEventDispatcher"]
