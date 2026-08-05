"""Event publication and hook dispatch for one agent run."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

from aeloon_core.core.types import (
    RunEvent,
    RunEventSink,
    RunEventType,
    RunHook,
)


class RunEventDispatcher:
    """Own one authoritative event sink and composable lifecycle hooks.

    Listeners observe events and are intentionally isolated from the run:
    a broken listener cannot interrupt a run. Hooks participate in the run and
    therefore propagate failures to their caller.
    """

    def __init__(self, sink: RunEventSink | None = None) -> None:
        self._sink = sink
        self._listeners: list[RunEventSink] = []
        self._handlers: dict[str, list[RunHook]] = defaultdict(list)

    def subscribe(self, listener: RunEventSink) -> Any:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def on(self, event_type: str, handler: RunHook) -> Any:
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers[event_type]
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def emit(
        self,
        event_type: RunEventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = RunEvent(event_type, data or {})
        if self._sink is not None:
            result = self._sink(event)
            if inspect.isawaitable(result):
                await result
        for listener in tuple(self._listeners):
            try:
                result = listener(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue

    async def hook(
        self,
        event_type: RunEventType,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        event = RunEvent(event_type, data)
        await self.emit(event_type, data)
        merged: dict[str, Any] = {}
        for handler in tuple(self._handlers.get(event_type, ())):
            result = handler(event)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict):
                merged.update(result)
        return merged


__all__ = ["RunEventDispatcher"]
