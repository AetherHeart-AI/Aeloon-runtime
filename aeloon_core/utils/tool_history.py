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


def collect_successful_tool_call_fingerprints(
    messages: list[dict[str, Any]],
) -> set[str]:
    """Collect fingerprints only for calls with successful tool results."""

    answered = {
        block["tool_use_id"]
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_use_id"), str)
        and not _tool_result_failed(block.get("content"))
    }
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                not isinstance(block, dict)
                or block.get("type") != "tool_use"
                or block.get("id") not in answered
            ):
                continue
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(name, str):
                continue
            seen.add(tool_call_fingerprint(name, arguments))
    return seen


def _tool_result_failed(content: Any) -> bool:
    text = str(content or "").lstrip().lower()
    return text.startswith("error") or text.startswith("skipped duplicate call")
