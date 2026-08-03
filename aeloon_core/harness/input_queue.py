"""Steering, follow-up, and next-turn input queues."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from aeloon_core.harness.events import HarnessEventDispatcher
from aeloon_core.harness.types import (
    HarnessError,
    ImageContent,
    QueueMode,
    TextContent,
    UserMessage,
    message_to_dict,
)

InputQueueKind = Literal["steer", "follow_up", "next_turn"]


class TurnInputQueues:
    """Manage all deferred user input and its queueing policy."""

    def __init__(
        self,
        events: HarnessEventDispatcher,
        *,
        steering_mode: QueueMode,
        follow_up_mode: QueueMode,
    ) -> None:
        self._events = events
        self._queues: dict[InputQueueKind, list[UserMessage]] = {
            "steer": [],
            "follow_up": [],
            "next_turn": [],
        }
        self._steering_mode = self.validate_mode(steering_mode)
        self._follow_up_mode = self.validate_mode(follow_up_mode)

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    @property
    def next_turn_count(self) -> int:
        return len(self._queues["next_turn"])

    def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_mode = self.validate_mode(mode)

    def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_mode = self.validate_mode(mode)

    async def enqueue(
        self,
        kind: InputQueueKind,
        text: str,
        images: Sequence[ImageContent] = (),
    ) -> None:
        content = text if not images else (TextContent(text), *tuple(images))
        self._queues[kind].append(UserMessage(content))
        await self._emit_update()

    async def drain_steering(self) -> list[UserMessage]:
        return await self._drain("steer", self._steering_mode)

    async def drain_follow_up(self) -> list[UserMessage]:
        return await self._drain("follow_up", self._follow_up_mode)

    def take_next_turn(self) -> list[UserMessage]:
        queued = list(self._queues["next_turn"])
        self._queues["next_turn"].clear()
        return queued

    def clear_interactive(self) -> dict[str, list[dict[str, Any]]]:
        steer = [message_to_dict(message) for message in self._queues["steer"]]
        follow_up = [message_to_dict(message) for message in self._queues["follow_up"]]
        self._queues["steer"].clear()
        self._queues["follow_up"].clear()
        return {"clearedSteer": steer, "clearedFollowUp": follow_up}

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "steer": [message_to_dict(message) for message in self._queues["steer"]],
            "followUp": [message_to_dict(message) for message in self._queues["follow_up"]],
            "nextTurn": [message_to_dict(message) for message in self._queues["next_turn"]],
        }

    async def _drain(self, kind: InputQueueKind, mode: QueueMode) -> list[UserMessage]:
        queue = self._queues[kind]
        if not queue:
            return []
        count = len(queue) if mode == "all" else 1
        drained = queue[:count]
        del queue[:count]
        await self._emit_update()
        return drained

    async def _emit_update(self) -> None:
        await self._events.emit("queue_update", self.snapshot())

    @staticmethod
    def validate_mode(mode: str) -> QueueMode:
        if mode not in {"all", "one-at-a-time"}:
            raise HarnessError("invalid_argument", f"Invalid queue mode: {mode}")
        return mode  # type: ignore[return-value]


__all__ = ["InputQueueKind", "TurnInputQueues"]
