"""Typed runtime DTOs and failures shared by non-transport callers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aeloon_core.core import RunError


class RuntimeFailure(RunError):
    """Stable application-layer failure independent of any wire protocol."""


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    name: str
    time: str
    workspace: str | None
    session_id: str | None
    operation_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "name": self.name,
            "workspace": self.workspace,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    workspace: str
    created_at: str
    title: str | None
    schema_version: int = 3

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SessionInfo:
        return cls(
            session_id=str(value["session_id"]),
            workspace=str(value["workspace"]),
            created_at=str(value["created_at"]),
            title=str(value["title"]) if value.get("title") is not None else None,
            schema_version=int(value.get("schema_version", 3)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "title": self.title,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    metadata: dict[str, Any]
    state: dict[str, Any]
    stats: dict[str, Any]
    timeline: tuple[dict[str, Any], ...]
    active_operations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "state": self.state,
            "stats": self.stats,
            "timeline": list(self.timeline),
            "active_operations": list(self.active_operations),
        }


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    operation_id: str
    turn_id: str
    queue_position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "turn_id": self.turn_id,
            "queue_position": self.queue_position,
        }


@dataclass(frozen=True, slots=True)
class TurnInput:
    kind: str
    text: str = ""
    name: str | None = None
    additional_instructions: str | None = None
    arguments: tuple[str, ...] = ()
    attachments: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind}
        if self.kind == "prompt":
            value.update({"text": self.text, "attachments": list(self.attachments)})
        elif self.kind == "skill":
            value.update(
                {
                    "name": self.name,
                    "additional_instructions": self.additional_instructions,
                }
            )
        else:
            value.update({"name": self.name, "arguments": list(self.arguments)})
        return value


RuntimeEventListener = Callable[[RuntimeEvent], Awaitable[None] | None]

__all__ = [
    "OperationSnapshot",
    "RuntimeEvent",
    "RuntimeEventListener",
    "RuntimeFailure",
    "SessionInfo",
    "SessionSnapshot",
    "TurnInput",
]
