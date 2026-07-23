"""Canonical PydanticAI message persistence contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

MESSAGE_SCHEMA_VERSION = 2
MESSAGE_FORMAT = "pydantic-ai-v1"


class LegacySessionError(RuntimeError):
    """Raised when execution is requested for pre-PydanticAI history."""


def serialize_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")


def deserialize_messages(messages: Any) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(messages)


__all__ = [
    "LegacySessionError",
    "MESSAGE_FORMAT",
    "MESSAGE_SCHEMA_VERSION",
    "deserialize_messages",
    "serialize_messages",
]
