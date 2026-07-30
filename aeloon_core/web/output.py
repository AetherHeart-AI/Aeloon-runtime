"""Sanitizers for bounded, operator-visible Web output."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
_INTERNAL_ID = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|[0-9a-fA-F]{20,64})(?![A-Za-z0-9])"
)
_AUTHORIZATION_SECRET = re.compile(
    r"(?im)(\b(?:proxy-)?authorization\s*[:=]\s*)[^\r\n]*"
)
_ASSIGNMENT_SECRET = re.compile(
    r"""(?ix)
    (
        (?<![A-Za-z0-9_])
        ["']?
        (?:
            AWS_ACCESS_KEY_ID
            |
            API[_-]?KEY
            |
            CLIENT[_-]?SECRET
            |
            TOKEN
            |
            PASSWORD
            |
            SECRET
            |
            COOKIE
            |
            [A-Za-z_][A-Za-z0-9_.-]*
            (?:API[_-]?KEY|CLIENT[_-]?SECRET|TOKEN|PASSWORD|SECRET|COOKIE)
            [A-Za-z0-9_.-]*
        )
        ["']?
        \s*[:=]\s*
    )
    (?:
        "(?:\\.|[^"\\\r\n])*"
        |
        '(?:\\.|[^'\\\r\n])*'
        |
        [^\r\n]*
    )
    """
)
_NAMED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|x[_-]?auth[_-]?token|"
    r"password|secret|cookie)"
    r"(\s*[:=]\s*)[^\s,;]+"
)
_URL_CREDENTIAL = re.compile(r"(://[^:/@\s]+:)[^@\s]+(@)")
_KEY_LIKE_SECRET = re.compile(
    r"(?i)\b(?:(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}|"
    r"(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{8,}|"
    r"xox[baprs]-[A-Za-z0-9-]{12,})\b"
)


def redact_sensitive_text(value: Any) -> str:
    """Best-effort redaction for common credential-bearing text formats."""

    message = _normalize_for_redaction(value)
    # Authorization schemes can contain structured, comma-delimited values,
    # so redact the entire header rather than trying to enumerate schemes.
    message = _AUTHORIZATION_SECRET.sub(r"\1[redacted]", message)
    # Quoted assignments stop at their closing quote. Unquoted assignments
    # consume the line because passphrases and cookie values may contain spaces.
    message = _ASSIGNMENT_SECRET.sub(r"\1[redacted]", message)
    message = _NAMED_SECRET.sub(r"\1\2[redacted]", message)
    message = _URL_CREDENTIAL.sub(r"\1[redacted]\2", message)
    message = _KEY_LIKE_SECRET.sub("[redacted]", message)
    return _INTERNAL_ID.sub("[id]", message)


def sanitize_operator_output(value: Any, *, limit: int) -> str:
    """Normalize and bound a local operator preview, preserving head and tail."""

    if limit <= 0:
        return ""
    text = redact_sensitive_text(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    if not text or len(text) <= limit:
        return text
    return _bounded_head_tail(text, limit=limit)


def _normalize_for_redaction(value: Any) -> str:
    text = _ANSI_ESCAPE.sub("", str(value))
    return "".join(
        char
        if char in {"\r", "\n", "\t"}
        else ""
        if unicodedata.category(char).startswith("C")
        else char
        for char in text
    )


def _bounded_head_tail(text: str, *, limit: int) -> str:
    if limit < 32:
        return f"{text[: limit - 1]}…" if limit > 1 else "…"[:limit]

    hidden = len(text) - limit
    for _ in range(4):
        marker = f"\n… [{hidden} chars hidden] …\n"
        visible_chars = limit - len(marker)
        next_hidden = len(text) - visible_chars
        if next_hidden == hidden:
            break
        hidden = next_hidden

    marker = f"\n… [{hidden} chars hidden] …\n"
    visible_chars = limit - len(marker)
    head_chars = (visible_chars * 2) // 3
    tail_chars = visible_chars - head_chars
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"
