"""Small helper functions used by the standalone kernel."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
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


def detect_image_mime(raw: bytes) -> str | None:
    """Best-effort image MIME detection."""

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def safe_filename(name: str, default: str = "file") -> str:
    """Return a conservative filesystem-safe filename."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return cleaned or default


def guess_mime(path: str | Path) -> str:
    """Guess a MIME type for a local path."""

    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"
