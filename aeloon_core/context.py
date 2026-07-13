"""Prompt and message construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are Aeloon Core, a compact local coding and research assistant.

Use tools when they help verify facts or inspect the workspace. Prefer small,
reversible file edits. Keep replies concise and mention commands or files that
matter. The current runtime intentionally has no channels, MCP, memory, cron,
billing, subagents, or plugins. Skills may be available as on-demand instructions
through the skill tool when the system context lists them.

For file changes, read existing files before changing them and prefer edit for
small changes to existing files. Use JSON write only for small new files or small
intentional replacements. When the runtime provides an
`[aeloon-core:write-protocol-v1]` system message, use its framed WRITE protocol for
large or multi-file output; never put a large file body in JSON tool arguments.

When you decide to use a tool, include a concise public thinking note in your
assistant content before the tool call. Explain what you need to verify or
inspect in one or two short sentences. If the provider exposes a separate
reasoning/thinking field, the terminal UI may display that field directly.
"""

SKILL_GUIDANCE_MARKER = "[aeloon-core:skill-guidance]"


def build_initial_messages(*, workspace: Path) -> list[dict[str, Any]]:
    """Build the initial system messages for a session."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip() + f"\n\nWorkspace: {workspace}",
        }
    ]


def refresh_initial_system_message(
    messages: list[dict[str, Any]],
    *,
    workspace: Path,
) -> list[dict[str, Any]]:
    """Return messages with the current runtime system prompt."""

    current = build_initial_messages(workspace=workspace)[0]
    if messages and messages[0].get("role") == "system":
        return [current, *messages[1:]]
    return [current, *messages]


def apply_skill_guidance(
    messages: list[dict[str, Any]],
    guidance: str | None,
) -> list[dict[str, Any]]:
    """Insert or replace the skill guidance system message."""

    without_old = [
        message
        for message in messages
        if not (
            message.get("role") == "system"
            and str(message.get("content") or "").startswith(SKILL_GUIDANCE_MARKER)
        )
    ]
    if not guidance:
        return without_old

    message = {
        "role": "system",
        "content": f"{SKILL_GUIDANCE_MARKER}\n{guidance}",
    }
    if without_old and without_old[0].get("role") == "system":
        return [without_old[0], message, *without_old[1:]]
    return [message, *without_old]


def append_user_message(messages: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """Return a copy of messages with a user prompt appended."""

    return [*messages, {"role": "user", "content": prompt}]
