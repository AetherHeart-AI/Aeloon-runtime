"""Opt-in, local JSONL recording for Runtime boundary equivalence work.

Traces are deliberately kept out of the normal Runtime data path.  A caller
must pass ``--record-trace DIRECTORY`` to enable this recorder; the default is
therefore no trace file at all.  Secrets are removed before they reach disk,
and base64 payloads are written as content-addressed, mode-0600 blobs.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 1
_SECRET_KEY = re.compile(
    r"(?:password|passwd|api[_-]?key|authorization|auth[_-]?header|cookie|secret|token|credential)",
    re.IGNORECASE,
)
_BASE64_KEY = re.compile(r"(?:^|[_-])(?:data|content|payload)[_-]?base64$", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SECRET_TEXT = re.compile(
    r"(password|passwd|api[_-]?key|authorization|auth[_-]?header|cookie|secret|token|credential)"
    r"\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)


class TraceRecorder:
    """Synchronously append an ordered, redacted Runtime boundary trace."""

    def __init__(self, directory: Path, *, process_name: str = "aeloon-runtime") -> None:
        self.directory = directory.expanduser().resolve(strict=False)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.blob_directory = self.directory / "blobs"
        self.blob_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blob_directory.chmod(0o700)
        stamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            ":", "-"
        )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", process_name)
        self.path = self.directory / f"{stamp}-{safe_name}-{os.getpid()}.jsonl"
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(self._fd, 0o600)
        self._sequence = 0
        self._closed = False
        self._write("checkpoint", checkpoint="trace.started")

    def request(self, request_id: Any, method: str, params: Any) -> None:
        self._write(
            "request",
            request_id=request_id,
            method=method,
            params=self.capture(params),
        )

    def response(self, request_id: Any, method: str, result: Any) -> None:
        self._write(
            "response",
            request_id=request_id,
            method=method,
            result=self.capture(result),
        )
        if method in {"system.snapshot", "thread.get", "git.status"}:
            self._write("checkpoint", request_id=request_id, method=method, checkpoint=method)

    def error(self, request_id: Any, method: str, code: str, message: str) -> None:
        self._write(
            "error",
            request_id=request_id,
            method=method,
            error={"code": code, "message": self.capture(message)},
        )

    def event(self, event: Any, payload: Any = None) -> None:
        if isinstance(event, str):
            value = {"name": event, "payload": payload}
        elif isinstance(event, dict):
            value = event
        else:
            value = dict(event)
        self._write(
            "event",
            event=value.get("name") or value.get("event"),
            payload=self.capture(value),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)

    def capture(self, value: Any, key: str = "") -> Any:
        if _SECRET_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, str):
            return _redact_text(value)
        if isinstance(value, list):
            return [self.capture(item, key) for item in value]
        if isinstance(value, tuple):
            return [self.capture(item, key) for item in value]
        if not isinstance(value, dict):
            return value
        captured: dict[str, Any] = {}
        header_name = value.get("name") or value.get("key")
        for child_key, child_value in value.items():
            child_name = str(child_key)
            if (
                isinstance(header_name, str)
                and _SECRET_KEY.search(header_name)
                and child_name.lower() in {"value", "header_value", "content"}
            ) or _SECRET_KEY.search(child_name):
                captured[child_name] = "[REDACTED]"
            elif _BASE64_KEY.search(child_name) and isinstance(child_value, str):
                captured[child_name] = self._capture_blob(child_value)
            else:
                captured[child_name] = self.capture(child_value, child_name)
        return captured

    def _capture_blob(self, encoded: str) -> dict[str, Any]:
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return {"sha256": "invalid", "size_bytes": 0, "path": ""}
        digest = hashlib.sha256(data).hexdigest()
        path = self.blob_directory / f"{digest}.bin"
        if not path.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                try:
                    _write_all(fd, data)
                    os.fchmod(fd, 0o600)
                finally:
                    os.close(fd)
        return {"sha256": digest, "size_bytes": len(data), "path": f"blobs/{path.name}"}

    def _write(self, kind: str, **fields: Any) -> None:
        if self._closed:
            return
        self._sequence += 1
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": self._sequence,
            "at": datetime.now(UTC).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "kind": kind,
            **fields,
        }
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        _write_all(self._fd, payload.encode())
        os.fsync(self._fd)


def _redact_text(value: str) -> str:
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _SECRET_TEXT.sub(r"\1=[REDACTED]", value)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
