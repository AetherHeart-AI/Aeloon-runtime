from __future__ import annotations

from typing import Any

from aeloon_core.conversation.history import (
    MESSAGE_FORMAT,
    MESSAGE_SCHEMA_VERSION,
    serialize_messages,
)


def checkpoint(content: str) -> dict[str, Any]:
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "api": "openai-completions",
        "provider": "aeloon-test",
        "model": "scripted",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        "stopReason": "stop",
        "timestamp": 0,
    }
    return {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "message_format": MESSAGE_FORMAT,
        "messages": serialize_messages([message]),
    }
