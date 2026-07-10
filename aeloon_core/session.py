"""Lightweight JSONL session persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeloon_core.context import build_initial_messages


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

        return uuid.uuid4().hex[:12]

    def session_path(self, session_id: str) -> Path:
        """Return the JSONL path for a session."""

        safe = self._safe_session_id(session_id)
        return self.sessions_dir / f"{safe or 'default'}.jsonl"

    def trace_path(self, session_id: str) -> Path:
        """Return the independent transition-trace JSONL path for a session."""

        return self.traces_dir / f"{self._safe_session_id(session_id) or 'default'}.jsonl"

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

        path = self.session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "turn",
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

    def append_transition(
        self,
        *,
        session_id: str,
        turn_id: str,
        transition: dict[str, Any],
    ) -> None:
        """Append one transition to the session's independent trace stream."""

        path = self.trace_path(session_id)
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

        path = self.trace_path(session_id)
        if not path.exists():
            return []
        return self._read_jsonl(path)

    def list_sessions(self) -> list[SessionSummary]:
        """List persisted sessions, newest first."""

        summaries: list[SessionSummary] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            records = self._read_records(path.stem)
            if not records:
                continue
            last = records[-1]
            title = str(last.get("user_prompt") or path.stem).strip().splitlines()[0][:80]
            summaries.append(
                SessionSummary(
                    session_id=path.stem,
                    title=title,
                    updated_at=str(last.get("created_at") or ""),
                    turns=len(records),
                )
            )
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def _read_records(self, session_id: str) -> list[dict[str, Any]]:
        path = self.session_path(session_id)
        if not path.exists():
            return []
        return self._read_jsonl(path)

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
    def _safe_session_id(session_id: str) -> str:
        return "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
