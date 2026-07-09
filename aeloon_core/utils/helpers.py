"""Small helper functions used by the standalone kernel."""

from __future__ import annotations

from typing import Any


def build_assistant_message(
    content: str | None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible assistant message."""

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if content == "":
            message["content"] = None
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if thinking_blocks:
        message["thinking_blocks"] = thinking_blocks
    return message
