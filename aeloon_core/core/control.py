"""Ephemeral steering, follow-up, and cancellation control for one run."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from aeloon_core.core.events import RunEventDispatcher
from aeloon_core.core.types import (
    ImageContent,
    QueueMode,
    RunError,
    TextContent,
    UserMessage,
    message_to_dict,
)

RunInputKind = Literal["steer", "follow_up"]


class RunController:
    """Control one active ``run_agent`` invocation.

    A controller can be created before the task starts, but its public methods
    are valid only while it is bound to an active run. Queue contents and the
    cancellation flag are reset when that invocation settles.
    """

    def __init__(
        self,
        *,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
    ) -> None:
        self._steering_mode = self.validate_mode(steering_mode)
        self._follow_up_mode = self.validate_mode(follow_up_mode)
        self._queues: dict[RunInputKind, list[UserMessage]] = {
            "steer": [],
            "follow_up": [],
        }
        self._events: RunEventDispatcher | None = None
        self._cancel: Callable[[], None] | None = None
        self._abort_requested = False

    @property
    def active(self) -> bool:
        return self._events is not None

    @property
    def abort_requested(self) -> bool:
        return self._abort_requested

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    async def steer(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        await self._enqueue("steer", text, images)

    async def follow_up(self, text: str, *, images: Sequence[ImageContent] = ()) -> None:
        await self._enqueue("follow_up", text, images)

    async def cancel(self) -> dict[str, list[dict[str, Any]]]:
        self._require_active("cancel")
        self._abort_requested = True
        cleared = self.clear()
        if self._cancel is not None:
            self._cancel()
        assert self._events is not None
        await self._events.emit("abort", cleared)
        return cleared

    def clear(self) -> dict[str, list[dict[str, Any]]]:
        steer = [message_to_dict(message) for message in self._queues["steer"]]
        follow_up = [message_to_dict(message) for message in self._queues["follow_up"]]
        self._queues["steer"].clear()
        self._queues["follow_up"].clear()
        return {"clearedSteer": steer, "clearedFollowUp": follow_up}

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "steer": [message_to_dict(message) for message in self._queues["steer"]],
            "followUp": [message_to_dict(message) for message in self._queues["follow_up"]],
            "nextTurn": [],
        }

    async def _bind(
        self,
        events: RunEventDispatcher,
        cancel: Callable[[], None],
    ) -> None:
        if self.active:
            raise RunError("busy", "RunController is already bound to an active run")
        self._events = events
        self._cancel = cancel
        self._abort_requested = False

    def _release(self) -> None:
        self._events = None
        self._cancel = None
        self._queues["steer"].clear()
        self._queues["follow_up"].clear()
        self._abort_requested = False

    async def _enqueue(
        self,
        kind: RunInputKind,
        text: str,
        images: Sequence[ImageContent],
    ) -> None:
        self._require_active(kind)
        if not text.strip() and not images:
            raise RunError("invalid_argument", "Queued input must not be empty")
        content = text if not images else (TextContent(text), *tuple(images))
        self._queues[kind].append(UserMessage(content))
        await self._emit_update()

    async def _drain_steering(self) -> list[UserMessage]:
        return await self._drain("steer", self._steering_mode)

    async def _drain_follow_up(self) -> list[UserMessage]:
        return await self._drain("follow_up", self._follow_up_mode)

    async def _drain(self, kind: RunInputKind, mode: QueueMode) -> list[UserMessage]:
        queue = self._queues[kind]
        if not queue:
            return []
        count = len(queue) if mode == "all" else 1
        drained = queue[:count]
        del queue[:count]
        await self._emit_update()
        return drained

    async def _emit_update(self) -> None:
        if self._events is not None:
            await self._events.emit("queue_update", self.snapshot())

    def _require_active(self, operation: str) -> None:
        if not self.active:
            raise RunError("invalid_state", f"Cannot {operation}: run is not active")

    @staticmethod
    def validate_mode(mode: str) -> QueueMode:
        if mode not in {"all", "one-at-a-time"}:
            raise RunError("invalid_argument", f"Invalid queue mode: {mode}")
        return mode  # type: ignore[return-value]


__all__ = ["RunController", "RunInputKind"]
