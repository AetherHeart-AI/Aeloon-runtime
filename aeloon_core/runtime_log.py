"""Small private Runtime lifecycle log with bounded rotation.

The log is intentionally independent from the trace recorder: traces are
opt-in equivalence artifacts, while this file is a short operational record
that makes a detached Runtime diagnosable without retaining unbounded output.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
            **fields,
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


__all__ = ["RuntimeLog"]
