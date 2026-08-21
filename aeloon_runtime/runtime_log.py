"""Small private Runtime lifecycle log with bounded rotation.

The log is intentionally independent from the trace recorder: traces are
opt-in equivalence artifacts, while this file is a short operational record
that makes a detached Runtime diagnosable without retaining unbounded output.

``diagnostics.logs`` reads only these two files. Request bodies, tokens,
enrollment codes, attachment bytes and session text are never persisted or
returned.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_FIELD_WHITELIST = frozenset(
    {
        "transport",
        "device_id",
        "source",
        "duration_ms",
        "reason",
        "thread_id",
        "pending_frames",
        "pending_bytes",
        "after_seq",
        "current_seq",
        "replay_complete",
        "replayed_events",
        "pid",
        "server_instance_id",
        "socket",
        "host",
        "port",
        "tls",
        "scheme",
        "name",
        "revoked",
    }
)
_FORBIDDEN_FIELD_TOKENS = (
    "token",
    "secret",
    "password",
    "api_key",
    "authorization",
    "pairing",
    "enroll",
    "code",
    "payload",
    "content",
    "data_base64",
    "prompt",
    "message",
    "body",
    "text",
    "attachment",
)
_SCALAR_TYPES = (str, int, float, bool)


def _field_name_forbidden(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _FORBIDDEN_FIELD_TOKENS)


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Keep only explicit diagnostic scalars. Never persist secrets or bodies."""

    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        if key in {"at", "event"}:
            continue
        if key not in LOG_FIELD_WHITELIST or _field_name_forbidden(key):
            continue
        if value is None or isinstance(value, _SCALAR_TYPES):
            if isinstance(value, bool) or value is None or isinstance(value, str):
                sanitized[key] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                sanitized[key] = int(value)
            elif isinstance(value, float):
                sanitized[key] = float(value)
    return sanitized


def read_runtime_logs(directory: Path, *, limit: int = 200) -> dict[str, Any]:
    """Return newest-first filtered records from ``runtime.log.1`` then ``runtime.log``."""

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    root = directory.expanduser().resolve(strict=False)
    entries: list[dict[str, Any]] = []
    for path in (root / "runtime.log.1", root / "runtime.log"):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            at = record.get("at")
            event = record.get("event")
            if not isinstance(at, str) or not isinstance(event, str):
                continue
            entries.append(
                {
                    "at": at,
                    "event": event,
                    "fields": sanitize_log_fields(record),
                }
            )
    entries.reverse()
    truncated = len(entries) > limit
    return {"entries": entries[:limit], "truncated": truncated}


class RuntimeLog:
    """Append lifecycle records to a mode-0600 log, retaining one old file."""

    def __init__(self, directory: Path, *, max_bytes: int = 1 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.directory = directory.expanduser().resolve(strict=False)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.path = self.directory / "runtime.log"
        self.previous_path = self.directory / "runtime.log.1"
        self.max_bytes = max_bytes
        self._rotate_if_needed()
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(self._fd, 0o600)
        self._closed = False

    def write(self, event: str, **fields: Any) -> None:
        if self._closed:
            return
        record = {
            "at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event": event,
            **sanitize_log_fields(fields),
        }
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(self._fd, payload[offset:])
        except OSError:
            # Diagnostics must never take the Runtime down.
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)

    def _rotate_if_needed(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < self.max_bytes:
            return
        try:
            self.previous_path.unlink(missing_ok=True)
            os.replace(self.path, self.previous_path)
            self.previous_path.chmod(0o600)
        except OSError:
            # The current process can still append to the existing log if
            # another filesystem policy prevents rotation.
            return


__all__ = ["LOG_FIELD_WHITELIST", "RuntimeLog", "read_runtime_logs", "sanitize_log_fields"]
