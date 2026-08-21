"""Runtime-generated session titles that never enter the transcript."""

from __future__ import annotations

import re
from dataclasses import replace

from aeloon_runtime.core.events import RunEventDispatcher
from aeloon_runtime.core.inference_runtime import InferenceRuntime
from aeloon_runtime.core.types import (
    InferencePort,
    Model,
    StreamOptions,
    UserMessage,
)
from aeloon_runtime.runtime.session import Session

GENERIC_SESSION_TITLES = frozenset(
    {
        "",
        "new chat",
        "new session",
        "new task",
        "new thread",
        "untitled",
    }
)

_SYSTEM_PROMPT = """Create a concise title for this coding-assistant session.
Summarize the user's actual goal, not the wording of the first sentence.
Use the same language as the user's request. Prefer 2-7 words and at most 60 characters.
Return only the title: no quotes, prefix, markdown, or trailing punctuation."""


def is_generic_session_title(title: str | None) -> bool:
    """Return whether a title is still an untouched placeholder."""

    return (title or "").strip().casefold() in GENERIC_SESSION_TITLES


def normalize_session_title(value: str, *, maximum: int = 60) -> str | None:
    """Normalize a model response into a safe, single-line session title."""

    title = value.strip().splitlines()[0].strip() if value.strip() else ""
    title = title.lstrip(" \t*_#>")
    title = re.sub(r"^(?:title|session title|标题|会话标题)\s*[:：]\s*", "", title, flags=re.I)
    title = title.strip(" \t\"'`*_#<>《》「」『』“”‘’")
    title = re.sub(r"[.!?。！？:：;,，；]+$", "", title).strip()
    if not title:
        return None
    characters = list(title)
    if len(characters) <= maximum:
        return title
    return "".join(characters[: maximum - 1]).rstrip() + "…"


def fallback_session_title(user_prompt: str, *, maximum: int = 60) -> str | None:
    """Derive a stable title when the optional naming request cannot complete."""

    title = normalize_session_title(user_prompt, maximum=maximum)
    if title is None or is_generic_session_title(title):
        return None
    return title


async def rename_session(
    *,
    session: Session,
    inference: InferencePort,
    model: Model,
    user_prompt: str,
    assistant_text: str,
    stream_options: StreamOptions,
) -> str | None:
    """Generate and persist the first semantic title for ``session``.

    The provider call is deliberately made outside the main agent run so the
    naming request and response never become part of the user's context window.
    """

    current_title = await session.get_name() or session.metadata.metadata.get("title")
    if not is_generic_session_title(str(current_title or "")):
        return None

    prompt = (
        "User request:\n"
        f"{user_prompt.strip()[:4_000]}\n\n"
        "Assistant outcome:\n"
        f"{assistant_text.strip()[:4_000]}"
    )

    async def on_retry(_data: dict[str, object]) -> None:
        return None

    title: str | None = None
    try:
        response = await InferenceRuntime(inference, RunEventDispatcher()).request(
            model=model,
            messages=(UserMessage(prompt),),
            system_prompt=_SYSTEM_PROMPT,
            tools=(),
            session_id=f"{session.id}:rename",
            stream_options=replace(
                stream_options,
                max_tokens=80,
                temperature=0.2,
                thinking_level="off",
            ),
            on_retry=on_retry,
        )
        if response.stop_reason not in {"error", "aborted"}:
            title = normalize_session_title(response.text)
    except Exception:
        # Naming must remain available when the separate best-effort inference
        # request fails before it can return a normal error response.
        pass
    if title is None or is_generic_session_title(title):
        title = fallback_session_title(user_prompt)
    if title is None:
        return None

    # A manual rename may have landed while the model was generating the title.
    latest_title = await session.get_name() or session.metadata.metadata.get("title")
    if not is_generic_session_title(str(latest_title or "")):
        return None
    await session.set_name(title)
    return title


__all__ = [
    "GENERIC_SESSION_TITLES",
    "fallback_session_title",
    "is_generic_session_title",
    "normalize_session_title",
    "rename_session",
]
