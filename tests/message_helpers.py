from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart

from aeloon_core.conversation.history import (
    MESSAGE_FORMAT,
    MESSAGE_SCHEMA_VERSION,
    serialize_messages,
)


def checkpoint(content: str) -> dict[str, Any]:
    return {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "message_format": MESSAGE_FORMAT,
        "messages": serialize_messages([ModelResponse(parts=[TextPart(content)])]),
    }
