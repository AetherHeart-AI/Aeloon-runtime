"""Prompt and message construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are Aeloon Core, a compact local coding and research assistant.

Use tools when they help verify facts or inspect the workspace. Prefer small,
reversible file edits. Keep replies concise and mention commands or files that
matter. The current runtime intentionally has no channels, MCP, memory, skills,
cron, billing, subagents, or plugins.
"""


def build_initial_messages(*, workspace: Path) -> list[dict[str, Any]]:
    """Build the initial system messages for a session."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip() + f"\n\nWorkspace: {workspace}",
        }
    ]


def append_user_message(messages: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """Return a copy of messages with a user prompt appended."""

    return [*messages, {"role": "user", "content": prompt}]
