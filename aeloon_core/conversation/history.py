"""Canonical pi-ai conversation-message persistence contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, TypeAlias

PiMessage: TypeAlias = dict[str, Any]

MESSAGE_SCHEMA_VERSION = 3
MESSAGE_FORMAT = "pi-ai-v1"
_MESSAGE_ROLES = frozenset({"user", "assistant", "toolResult"})
LEGACY_PYDANTIC_AI_SCHEMA_VERSION = 2
LEGACY_PYDANTIC_AI_MESSAGE_FORMAT = "pydantic-ai-v1"


class LegacySessionError(RuntimeError):
    """Raised when execution is requested for an older runtime's history."""


def serialize_messages(messages: Sequence[Mapping[str, Any]]) -> list[PiMessage]:
    """Validate and return an isolated JSON-safe pi-ai transcript."""

    return _validate_messages(messages)


def deserialize_messages(messages: Any) -> list[PiMessage]:
    """Validate persisted pi-ai messages without changing their wire shape."""

    if not isinstance(messages, list):
        raise ValueError("Pi session history must be a message array")
    return _validate_messages(messages)


def migrate_pydantic_messages(messages: Any) -> list[PiMessage]:
    """Translate the former persisted Pydantic AI wire format into pi-ai messages."""

    if not isinstance(messages, list):
        raise ValueError("Pydantic AI session history must be a message array")
    migrated: list[PiMessage] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Pydantic AI session history contains a non-object message")
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        if message.get("kind") == "request":
            migrated.extend(_migrate_request_parts(parts))
        elif message.get("kind") == "response":
            migrated.append(_migrate_response(message, parts))
    return _validate_messages(migrated)


def _migrate_request_parts(parts: list[Any]) -> list[PiMessage]:
    migrated: list[PiMessage] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        kind = part.get("part_kind")
        if kind == "user-prompt":
            migrated.append(
                {
                    "role": "user",
                    "content": _text_content(part.get("content")),
                    "timestamp": _timestamp_ms(part.get("timestamp")),
                }
            )
        elif kind in {"tool-return", "retry-prompt"} and part.get("tool_call_id"):
            migrated.append(
                {
                    "role": "toolResult",
                    "toolCallId": str(part["tool_call_id"]),
                    "toolName": str(part.get("tool_name") or "tool"),
                    "content": [
                        {"type": "text", "text": _render_content(part.get("content"))}
                    ],
                    "isError": kind == "retry-prompt" or part.get("outcome") == "failed",
                    "timestamp": _timestamp_ms(part.get("timestamp")),
                }
            )
    return migrated


def _migrate_response(message: Mapping[str, Any], parts: list[Any]) -> PiMessage:
    content: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        kind = part.get("part_kind")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("content") or "")})
        elif kind == "thinking":
            content.append(
                {"type": "thinking", "thinking": str(part.get("content") or "")}
            )
        elif kind == "tool-call":
            arguments = part.get("args")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            content.append(
                {
                    "type": "toolCall",
                    "id": str(part.get("tool_call_id") or part.get("id") or "tool-call"),
                    "name": str(part.get("tool_name") or "tool"),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
    raw_usage = message.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) else {}
    has_tool_call = any(part.get("type") == "toolCall" for part in content)
    return {
        "role": "assistant",
        "content": content,
        "api": "openai-completions",
        "provider": str(message.get("provider_name") or "deepseek"),
        "model": str(message.get("model_name") or "unknown"),
        "usage": {
            "input": int(usage.get("input_tokens") or 0),
            "output": int(usage.get("output_tokens") or 0),
            "cacheRead": int(usage.get("cache_read_tokens") or 0),
            "cacheWrite": int(usage.get("cache_write_tokens") or 0),
            "totalTokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "total": 0,
            },
        },
        "stopReason": "toolUse" if has_tool_call else "stop",
        "timestamp": _timestamp_ms(message.get("timestamp")),
    }


def _text_content(value: Any) -> str | list[dict[str, str]]:
    if isinstance(value, str):
        return value
    return [{"type": "text", "text": _render_content(value)}]


def _render_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(parsed.timestamp() * 1_000)
        except ValueError:
            pass
    return 0


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[PiMessage]:
    validated: list[PiMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"Pi message {index} must be an object")
        role = message.get("role")
        if role not in _MESSAGE_ROLES:
            raise ValueError(f"Pi message {index} has unsupported role {role!r}")
        if "content" not in message:
            raise ValueError(f"Pi message {index} has no content")
        try:
            normalized = json.loads(json.dumps(message, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pi message {index} is not JSON serializable") from exc
        validated.append(copy.deepcopy(normalized))
    return validated


__all__ = [
    "LegacySessionError",
    "MESSAGE_FORMAT",
    "MESSAGE_SCHEMA_VERSION",
    "LEGACY_PYDANTIC_AI_MESSAGE_FORMAT",
    "LEGACY_PYDANTIC_AI_SCHEMA_VERSION",
    "PiMessage",
    "deserialize_messages",
    "migrate_pydantic_messages",
    "serialize_messages",
]
