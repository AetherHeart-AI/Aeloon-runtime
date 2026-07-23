"""Lightweight JSONL session persistence."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeloon_core.context import build_initial_messages
from aeloon_core.message_history import (
    MESSAGE_FORMAT,
    MESSAGE_SCHEMA_VERSION,
    LegacySessionError,
    deserialize_messages,
)

_ENCODED_SESSION_PREFIX = "~"
_JSONL_SUFFIX = ".jsonl"
_MAX_PATH_COMPONENT_BYTES = 255
_DIRECT_SESSION_ID_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


@dataclass(frozen=True)
class SessionSummary:
    """Summary for one persisted session."""

    session_id: str
    title: str
    updated_at: str
    turns: int


class SessionStore:
    """Persist session turns as one JSON object per line."""

    def __init__(self, *, data_dir: Path, workspace: Path) -> None:
        self.data_dir = data_dir
        self.workspace = workspace
        self.sessions_dir = data_dir / "sessions"
        self.traces_dir = data_dir / "traces"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def new_session(self) -> str:
        """Create a session id."""

        session_id = uuid.uuid4().hex[:12]
        self._safe_session_id(session_id)
        return session_id

    def session_path(self, session_id: str) -> Path:
        """Return the JSONL path for a session."""

        return self._canonical_path(self.sessions_dir, session_id)

    def trace_path(self, session_id: str) -> Path:
        """Return the independent transition-trace JSONL path for a session."""

        return self._canonical_path(self.traces_dir, session_id)

    def load_messages(
        self,
        session_id: str,
        *,
        initial_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Load the latest message array for a session."""

        records = self._read_records(session_id)
        for record in reversed(records):
            messages = record.get("messages")
            if isinstance(messages, list):
                return messages
        return initial_messages or build_initial_messages(workspace=self.workspace)

    def load_pydantic_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load executable v2 history, rejecting but never altering legacy data."""

        records = self._read_records(session_id)
        if not records:
            return []
        record = records[-1]
        if (
            record.get("schema_version") != MESSAGE_SCHEMA_VERSION
            or record.get("message_format") != MESSAGE_FORMAT
        ):
            raise LegacySessionError(
                f"Session {session_id!r} uses the legacy message format; "
                "create a new session to continue. Existing data was not modified."
            )
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError("PydanticAI session record has no message array")
        return messages

    def append_turn(
        self,
        *,
        session_id: str,
        user_prompt: str,
        final_content: str | None,
        tools_used: list[str],
        messages: list[dict[str, Any]],
        blocks: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Append one completed turn."""

        deserialize_messages(messages)
        path = self._writable_path(self.sessions_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "turn",
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "message_format": MESSAGE_FORMAT,
            "session_id": session_id,
            "turn_id": turn_id,
            "created_at": datetime.now(UTC).isoformat(),
            "user_prompt": user_prompt,
            "final_content": final_content,
            "tools_used": tools_used,
            "messages": messages,
            "blocks": blocks or [],
            "usage": usage or {},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_turn_once(
        self,
        *,
        session_id: str,
        user_prompt: str,
        final_content: str | None,
        tools_used: list[str],
        messages: list[dict[str, Any]],
        blocks: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        turn_id: str,
    ) -> bool:
        """Durably append one turn once, repairing a crash-truncated tail.

        The per-session file lock is intentionally independent of FlowStore's
        SQLite writer lock. A crash after a partial write leaves no newline; the
        next recovery truncates that fragment, writes the canonical record, and
        fsyncs it before the caller marks the durable commit as projected.
        """

        deserialize_messages(messages)
        path = self._writable_path(self.sessions_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = {
            "type": "turn",
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "message_format": MESSAGE_FORMAT,
            "session_id": session_id,
            "turn_id": turn_id,
            "user_prompt": user_prompt,
            "final_content": final_content,
            "tools_used": tools_used,
            "messages": messages,
            "blocks": blocks or [],
            "usage": usage or {},
        }
        with self._locked_session_file(path) as handle:
            self._repair_partial_tail(handle)
            records = self._records_for_session(
                self._records_from_locked_file(handle),
                session_id,
            )
            matching = [record for record in records if record.get("turn_id") == turn_id]
            if matching:
                persisted = matching[-1]
                fields = tuple(expected)
                if any(persisted.get(field) != expected[field] for field in fields):
                    raise ValueError("persisted Master turn differs from its durable commit")
                return False

            record = {
                **expected,
                "created_at": datetime.now(UTC).isoformat(),
            }
            handle.seek(0, os.SEEK_END)
            handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
            return True

    def append_transition(
        self,
        *,
        session_id: str,
        turn_id: str,
        transition: dict[str, Any],
    ) -> None:
        """Append one transition to the session's independent trace stream."""

        path = self._writable_path(self.traces_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "transition",
            "schema_version": 1,
            "session_id": session_id,
            "turn_id": turn_id,
            **transition,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def history(self, session_id: str) -> list[dict[str, Any]]:
        """Return persisted turn records for a session."""

        return self._read_records(session_id)

    def transition_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return persisted transition records without affecting turn history."""

        return self._read_scoped_records(self.traces_dir, session_id)

    def list_sessions(self) -> list[SessionSummary]:
        """List persisted sessions, newest first."""

        session_ids: set[str] = set()
        for path in self.sessions_dir.glob("*.jsonl"):
            for record in self._read_jsonl(path):
                session_id = record.get("session_id")
                if not isinstance(session_id, str):
                    continue
                try:
                    self._safe_session_id(session_id)
                except ValueError:
                    continue
                session_ids.add(session_id)

        summaries: list[SessionSummary] = []
        for session_id in session_ids:
            records = self._read_records(session_id)
            if not records:
                continue
            last = records[-1]
            title_lines = str(last.get("user_prompt") or session_id).strip().splitlines()
            title = (title_lines[0] if title_lines else "Untitled session")[:80]
            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    title=title,
                    updated_at=str(last.get("created_at") or ""),
                    turns=len(records),
                )
            )
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def _read_records(self, session_id: str) -> list[dict[str, Any]]:
        return self._read_scoped_records(self.sessions_dir, session_id)

    def _read_scoped_records(
        self,
        directory: Path,
        session_id: str,
    ) -> list[dict[str, Any]]:
        canonical = self._canonical_path(directory, session_id)
        if canonical.exists():
            records = self._records_for_session(self._read_jsonl(canonical), session_id)
            if records:
                return records

        legacy = self._legacy_path(directory, session_id)
        if legacy != canonical and legacy.exists():
            return self._records_for_session(self._read_jsonl(legacy), session_id)
        return []

    def _writable_path(self, directory: Path, session_id: str) -> Path:
        """Return the canonical path, importing only owned legacy records once."""

        canonical = self._canonical_path(directory, session_id)
        legacy = self._legacy_path(directory, session_id)
        if legacy == canonical or not legacy.exists():
            return canonical

        with self._locked_session_file(canonical) as destination:
            destination.seek(0, os.SEEK_END)
            if destination.tell() > 0:
                return canonical
            records = self._records_for_session(self._read_jsonl(legacy), session_id)
            if records:
                for record in records:
                    destination.write(
                        (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode()
                    )
                destination.flush()
                os.fsync(destination.fileno())
        return canonical

    @classmethod
    def _canonical_path(cls, directory: Path, session_id: str) -> Path:
        return directory / f"{cls._safe_session_id(session_id)}{_JSONL_SUFFIX}"

    @classmethod
    def _legacy_path(cls, directory: Path, session_id: str) -> Path:
        legacy = cls._legacy_safe_session_id(session_id) or "default"
        return directory / f"{legacy}{_JSONL_SUFFIX}"

    @staticmethod
    def _records_for_session(
        records: list[dict[str, Any]],
        session_id: str,
    ) -> list[dict[str, Any]]:
        return [record for record in records if record.get("session_id") == session_id]

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    @contextmanager
    def _locked_session_file(path: Path) -> Iterator[Any]:
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _repair_partial_tail(handle: Any) -> None:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        data = handle.read()
        newline = data.rfind(b"\n")
        handle.seek(0)
        handle.truncate(newline + 1 if newline >= 0 else 0)

    @staticmethod
    def _records_from_locked_file(handle: Any) -> list[dict[str, Any]]:
        handle.seek(0)
        records: list[dict[str, Any]] = []
        for line in handle.read().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        """Encode one logical id as a single collision-free path component."""

        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string")
        try:
            encoded = session_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("session_id must contain valid Unicode") from exc

        if session_id and all(
            character in _DIRECT_SESSION_ID_CHARACTERS for character in session_id
        ):
            stem = session_id
        else:
            stem = f"{_ENCODED_SESSION_PREFIX}{encoded.hex()}"
        if len(f"{stem}{_JSONL_SUFFIX}".encode()) > _MAX_PATH_COMPONENT_BYTES:
            raise ValueError("session_id is too long for a session path")
        return stem

    @staticmethod
    def _legacy_safe_session_id(session_id: str) -> str:
        return "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
