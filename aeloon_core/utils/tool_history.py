"""Helpers for detecting duplicate tool calls in message history."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def tool_call_fingerprint(name: str, arguments: Any) -> str:
    """Create a stable fingerprint for a tool name and argument payload."""

    payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{name}:{payload}".encode()).hexdigest()
    return digest


def collect_tool_call_fingerprints(messages: list[dict[str, Any]]) -> set[str]:
    """Collect fingerprints for assistant tool calls already answered by tools."""

    answered = {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str)
    }
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("id") not in answered:
                continue
            function = call.get("function") or {}
            name = function.get("name")
            raw_args = function.get("arguments")
            if not isinstance(name, str):
                continue
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                arguments = raw_args
            seen.add(tool_call_fingerprint(name, arguments))
    return seen


def duplicate_tool_result(name: str) -> str:
    """Return the standard duplicate-call tool result."""

    return (
        f"Skipped duplicate call to {name}: the same tool with identical arguments "
        "already ran in this conversation. Use the existing result or change the arguments."
    )
