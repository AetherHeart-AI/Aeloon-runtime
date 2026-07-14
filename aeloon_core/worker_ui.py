"""Operator-only, privacy-preserving Worker timeline storage and queries.

This module is intentionally not part of :class:`WorkerControlService`. Base
agents keep their existing bounded control surface, while the local operator UI
can inspect a durable projection of Worker activity without reading private
transcripts. Tool output is limited to bounded exec output and failure previews.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any

from loguru import logger

from aeloon_core.operator_output import sanitize_operator_output
from aeloon_core.worker_sessions import WorkerRunRecord, WorkerStore

_MAX_EVENTS_PER_RUN = 500
_MAX_TOTAL_EVENTS = 50_000
_MAX_PENDING_EVENTS = 256
_GLOBAL_PRUNE_BATCH = 1_000
_FLUSH_TIMEOUT_SECONDS = 0.5
_MAX_STEP_CHARS = 160
_MAX_SUMMARY_CHARS = 600
_MAX_TOOL_COMMAND_CHARS = 160
_MAX_TOOL_RESULT_PREVIEW_CHARS = 4_000
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
_INTERNAL_ID = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|[0-9a-fA-F]{20,64})(?![A-Za-z0-9])"
)
_ACTIVITY_PHASES = {
    "analyzing",
    "planning",
    "drafting",
    "using_tool",
    "processing",
    "working_step",
    "finalizing",
    "delegating",
    "handoff",
    "branch_running",
    "branch_done",
    "synthesizing",
}
_LIFECYCLE_EVENTS = {
    "created",
    "running",
    "completed",
    "partial",
    "failed",
    "cancelled",
    "timed_out",
}
_TOOL_STATUSES = {"done", "error", "cancelled"}
_LOW_SIGNAL_TOOLS = {
    "discover_profiles",
    "glob",
    "grep",
    "inspect_worker",
    "list_workers",
    "read",
    "skill",
    "webfetch",
    "websearch",
}
_TOOL_METRICS = {
    "exit_code",
    "input_chars",
    "item_count",
    "new_chars",
    "old_chars",
    "result_chars",
    "result_lines",
    "todo_completed",
}


class WorkerUiEventKind(StrEnum):
    LIFECYCLE = "lifecycle"
    PHASE = "phase"
    TOOL = "tool"
    GUARD = "guard"


@dataclass(frozen=True, slots=True)
class _BufferedJournalRecord:
    run_id: str
    kind: WorkerUiEventKind
    payload: dict[str, Any]
    priority: int


class WorkerUiJournal:
    """Persist only a strict display projection of Worker activity."""

    writes_are_buffered = True

    def __init__(self, store: WorkerStore) -> None:
        self.path = Path(store.path)
        self.available = False
        self._active_write = False
        self._closing = False
        self._condition = threading.Condition()
        self._dropped_events = 0
        self._pending: deque[_BufferedJournalRecord] = deque()
        self._writer_thread: threading.Thread | None = None
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS worker_ui_events (
                      sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                      run_id TEXT NOT NULL REFERENCES worker_runs(run_id) ON DELETE CASCADE,
                      kind TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS worker_ui_events_run_idx
                      ON worker_ui_events(run_id, sequence);
                    CREATE TABLE IF NOT EXISTS worker_ui_state (
                      singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                      event_count INTEGER NOT NULL
                    );
                    """
                )
                # Reconcile once on startup so cascades or an interrupted older
                # version cannot leave the shared counter stale.
                row = connection.execute(
                    "INSERT INTO worker_ui_state(singleton, event_count) "
                    "VALUES (1, (SELECT COUNT(*) FROM worker_ui_events)) "
                    "ON CONFLICT(singleton) DO UPDATE SET "
                    "event_count = excluded.event_count "
                    "RETURNING event_count"
                ).fetchone()
                event_count = int(row["event_count"] if row else 0)
                if event_count > _MAX_TOTAL_EVENTS:
                    self._prune_global(connection)
        except sqlite3.Error as exc:
            # A concurrent WorkerStore writer can briefly own SQLite's schema
            # lock. The operator timeline is optional observability and must not
            # prevent the runtime from starting when that happens.
            logger.warning("Worker UI timeline disabled: {}", exc)
        else:
            self.available = True

    def record_lifecycle(self, event: str, run: WorkerRunRecord) -> None:
        if event not in _LIFECYCLE_EVENTS:
            return
        result = run.result
        report = result.report if result is not None else None
        summary = _safe_text(
            report.summary if report is not None else None,
            limit=_MAX_SUMMARY_CHARS,
        )
        status = str(run.status.value)
        payload: dict[str, Any] = {
            "phase": event,
            "status": status,
        }
        if result is not None:
            payload["duration_ms"] = max(0, int(result.duration_ms))
        if summary:
            payload["summary"] = summary
            if status == "failed":
                payload["error_summary"] = summary
        terminal = event in {"completed", "partial", "failed", "cancelled", "timed_out"}
        self._enqueue(
            _BufferedJournalRecord(
                run_id=run.run_id,
                kind=WorkerUiEventKind.LIFECYCLE,
                payload=payload,
                priority=3 if terminal else 1,
            )
        )

    def record_activity(
        self,
        *,
        run_id: str,
        phase: str,
        role_id: str | None = None,
        tool_names: tuple[str, ...] = (),
        current_step: str | None = None,
        todo_completed: int | None = None,
        todo_total: int | None = None,
        detail_source: str = "host",
    ) -> None:
        del role_id, detail_source
        if phase not in _ACTIVITY_PHASES:
            return
        safe_tools = [
            name
            for name in (_safe_identifier(item) for item in tool_names[:4])
            if name is not None
        ]
        payload: dict[str, Any] = {"phase": phase}
        if safe_tools:
            payload["tool_names"] = safe_tools
        safe_step = _safe_text(current_step, limit=_MAX_STEP_CHARS)
        if safe_step:
            payload["current_step"] = safe_step
        if todo_completed is not None:
            payload["todo_completed"] = max(0, int(todo_completed))
        if todo_total is not None:
            payload["todo_total"] = max(0, int(todo_total))
        self._enqueue(
            _BufferedJournalRecord(
                run_id=run_id,
                kind=WorkerUiEventKind.PHASE,
                payload=payload,
                priority=0,
            )
        )

    def record_tool(
        self,
        *,
        run_id: str,
        tool_name: str,
        status: str,
        metrics: Mapping[str, Any],
        duration_ms: int | None,
    ) -> None:
        name = _safe_identifier(tool_name) or "tool"
        safe_status = status if status in _TOOL_STATUSES else "error"
        safe_metrics: dict[str, Any] = {
            key: _bounded_int(value)
            for key, value in metrics.items()
            if key in _TOOL_METRICS and _bounded_int(value) is not None
        }
        command = _safe_text(metrics.get("command"), limit=_MAX_TOOL_COMMAND_CHARS)
        if command:
            safe_metrics["command"] = command
        result_preview = _safe_multiline_text(
            metrics.get("result_preview"),
            limit=_MAX_TOOL_RESULT_PREVIEW_CHARS,
        )
        if result_preview and (name == "exec" or safe_status != "done"):
            safe_metrics["result_preview"] = result_preview
        payload: dict[str, Any] = {
            "tool_name": name,
            "status": safe_status,
            "signal": (
                "low" if name in _LOW_SIGNAL_TOOLS and safe_status == "done" else "high"
            ),
            "metrics": safe_metrics,
        }
        if duration_ms is not None:
            payload["duration_ms"] = max(0, int(duration_ms))
        self._enqueue(
            _BufferedJournalRecord(
                run_id=run_id,
                kind=WorkerUiEventKind.TOOL,
                payload=payload,
                priority=0 if payload["signal"] == "low" else 2,
            )
        )

    def record_guard(self, *, run_id: str, resolution: Any) -> None:
        record = resolution.to_record()
        event = _safe_identifier(record.get("event")) or "runtime_error"
        action = _safe_identifier(record.get("action")) or "finalize"
        source = _safe_identifier(record.get("source")) or "guard"
        # Deliberately omit evidence, usage, causes, and model-authored text.
        self._enqueue(
            _BufferedJournalRecord(
                run_id=run_id,
                kind=WorkerUiEventKind.GUARD,
                payload={"event": event, "action": action, "source": source},
                priority=2,
            )
        )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        """Return safe journal rows in append order, without storage identities."""

        if not self.available:
            return []
        # Reads are outside Worker execution. Waiting briefly here preserves a
        # fresh operator view without charging SQLite lock latency to the Worker.
        self.flush()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT kind, payload_json, created_at FROM worker_ui_events "
                "WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if not isinstance(payload, dict):
                continue
            events.append(
                {
                    "kind": str(row["kind"]),
                    **payload,
                    "ts": str(row["created_at"]),
                }
            )
        return events

    @property
    def dropped_events(self) -> int:
        with self._condition:
            return self._dropped_events

    @property
    def pending_events(self) -> int:
        with self._condition:
            return len(self._pending)

    def flush(self, timeout: float = _FLUSH_TIMEOUT_SECONDS) -> bool:
        """Wait briefly for buffered writes; never wait indefinitely on SQLite."""

        deadline = monotonic() + max(0.0, timeout)
        with self._condition:
            while self._pending or self._active_write:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    def close(self, timeout: float = _FLUSH_TIMEOUT_SECONDS) -> bool:
        """Stop accepting records and make a bounded attempt to drain the queue."""

        with self._condition:
            self._closing = True
            writer = self._writer_thread
        drained = self.flush(timeout)
        if writer is not None and writer is not threading.current_thread():
            writer.join(max(0.0, timeout))
        return drained

    def _enqueue(self, record: _BufferedJournalRecord) -> None:
        if not self.available:
            return
        with self._condition:
            if self._closing:
                return
            if len(self._pending) >= _MAX_PENDING_EVENTS:
                if record.kind is WorkerUiEventKind.PHASE:
                    self._remove_latest_phase(record.run_id)
                if len(self._pending) >= _MAX_PENDING_EVENTS:
                    victim = next(
                        (
                            index
                            for index, queued in enumerate(self._pending)
                            if queued.priority < record.priority
                        ),
                        None,
                    )
                    if victim is not None:
                        del self._pending[victim]
                        self._dropped_events += 1
                    elif record.priority == 3:
                        self._pending.popleft()
                        self._dropped_events += 1
                    elif record.priority > 0:
                        equal_priority = next(
                            (
                                index
                                for index, queued in enumerate(self._pending)
                                if queued.priority == record.priority
                            ),
                            None,
                        )
                        if equal_priority is None:
                            self._dropped_events += 1
                            return
                        del self._pending[equal_priority]
                        self._dropped_events += 1
                    else:
                        self._dropped_events += 1
                        return
            self._pending.append(record)
            self._ensure_writer_locked()
            self._condition.notify_all()

    def _remove_latest_phase(self, run_id: str) -> None:
        for index in range(len(self._pending) - 1, -1, -1):
            queued = self._pending[index]
            if queued.kind is WorkerUiEventKind.PHASE and queued.run_id == run_id:
                del self._pending[index]
                self._dropped_events += 1
                return

    def _ensure_writer_locked(self) -> None:
        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            return
        self._writer_thread = threading.Thread(
            target=self._drain,
            name="aeloon-worker-ui-journal",
            daemon=True,
        )
        self._writer_thread.start()

    def _drain(self) -> None:
        while True:
            with self._condition:
                if not self._pending:
                    self._writer_thread = None
                    self._condition.notify_all()
                    return
                record = self._pending.popleft()
                self._active_write = True
            try:
                self._write_record(record)
            except sqlite3.Error as exc:
                # A busy or damaged optional journal must not affect Worker work.
                logger.warning("Ignoring Worker UI journal write failure: {}", exc)
            except Exception as exc:
                logger.warning("Ignoring Worker UI journal failure: {}", exc)
            finally:
                with self._condition:
                    self._active_write = False
                    self._condition.notify_all()

    def _write_record(self, record: _BufferedJournalRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO worker_ui_events(run_id, kind, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    record.run_id,
                    record.kind.value,
                    json.dumps(record.payload, ensure_ascii=False, separators=(",", ":")),
                    _now(),
                ),
            )
            deleted = connection.execute(
                "DELETE FROM worker_ui_events WHERE run_id = ? AND sequence NOT IN ("
                "SELECT sequence FROM worker_ui_events WHERE run_id = ? "
                "ORDER BY sequence DESC LIMIT ?)",
                (record.run_id, record.run_id, _MAX_EVENTS_PER_RUN),
            ).rowcount
            row = connection.execute(
                "UPDATE worker_ui_state SET event_count = MAX(0, event_count + ?) "
                "WHERE singleton = 1 RETURNING event_count",
                (1 - max(0, deleted),),
            ).fetchone()
            event_count = int(row["event_count"] if row else 0)
            if event_count > _MAX_TOTAL_EVENTS:
                self._prune_global(connection)

    @staticmethod
    def _prune_global(connection: sqlite3.Connection) -> int:
        # COUNT is intentionally paid only at the batch boundary. It repairs a
        # high shared counter after foreign-key cascades while keeping the normal
        # per-event path O(1) across processes and journal instances.
        row = connection.execute(
            "SELECT COUNT(*) AS event_count FROM worker_ui_events"
        ).fetchone()
        event_count = int(row["event_count"] if row else 0)
        max_events = max(1, int(_MAX_TOTAL_EVENTS))
        if event_count > max_events:
            batch = max(1, min(int(_GLOBAL_PRUNE_BATCH), max_events))
            target = max(0, max_events - batch)
            delete_count = max(1, event_count - target)
            deleted = connection.execute(
                "DELETE FROM worker_ui_events WHERE sequence IN ("
                "SELECT sequence FROM worker_ui_events "
                "ORDER BY sequence LIMIT ?)",
                (delete_count,),
            ).rowcount
            event_count = max(0, event_count - max(0, deleted))
        connection.execute(
            "UPDATE worker_ui_state SET event_count = ? WHERE singleton = 1",
            (event_count,),
        )
        return event_count

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=0.1)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=100")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            connection.close()
            raise
        return connection

class WorkerUiQueryService:
    """Operator capability for safe Worker details; never register as a Base tool."""

    def __init__(self, *, manager: Any, journal: WorkerUiJournal) -> None:
        self.manager = manager
        self.journal = journal

    def inspect_worker(self, worker_id: str) -> dict[str, Any]:
        worker, runs = self.manager.inspect_worker(worker_id)
        profile_id = worker.profile.profile_id
        timeline: list[dict[str, Any]] = []
        run_views: list[dict[str, Any]] = []
        current_phase = "queued"
        current_step: str | None = None
        todo_completed: int | None = None
        todo_total: int | None = None
        phase_history: list[str] = []
        for run in runs:
            run_views.append(_run_view(run))
            safe_events = self.journal.list_events(run.run_id)
            run_phase = run.status.value
            run_step: str | None = None
            run_todo_completed: int | None = None
            run_todo_total: int | None = None
            run_phase_history: list[str] = []
            for event in safe_events:
                if event.get("kind") == WorkerUiEventKind.PHASE:
                    run_phase = str(event.get("phase") or run_phase)
                    step = event.get("current_step")
                    if isinstance(step, str) and step:
                        run_step = step
                    completed = event.get("todo_completed")
                    total = event.get("todo_total")
                    if isinstance(completed, int):
                        run_todo_completed = completed
                    if isinstance(total, int):
                        run_todo_total = total
                    if run_phase not in run_phase_history:
                        run_phase_history.append(run_phase)
            if run is runs[-1]:
                current_phase = run_phase
                current_step = run_step
                todo_completed = run_todo_completed
                todo_total = run_todo_total
                phase_history = run_phase_history
            timeline.extend(
                _project_timeline(safe_events, run_sequence=run.run_sequence)
            )
        latest = runs[-1] if runs else None
        if latest is not None and latest.status.terminal:
            current_phase = latest.status.value
            current_step = None
        return {
            "worker_id": worker.worker_id,
            "label": f"{profile_id}#{worker.worker_id[:4]}",
            "profile_id": profile_id,
            "status": worker.status.value,
            "created_at": worker.created_at,
            "phase": current_phase,
            "phases": phase_history,
            "current_step": current_step,
            "todo_completed": todo_completed,
            "todo_total": todo_total,
            "runs": run_views,
            "timeline": timeline,
            "timeline_available": bool(timeline),
        }


def _run_view(run: WorkerRunRecord) -> dict[str, Any]:
    result = run.result
    report = result.report if result is not None else None
    summary = _safe_text(
        report.summary if report is not None else None,
        limit=_MAX_SUMMARY_CHARS,
    )
    view: dict[str, Any] = {
        "run_id": run.run_id,
        "worker_id": run.worker_id,
        "run_sequence": run.run_sequence,
        "status": run.status.value,
        "goal": _safe_text(run.context.goal, limit=1_200),
        "created_at": run.created_at,
        "summary": summary or None,
        "duration_ms": result.duration_ms if result is not None else None,
        "tool_outcome": result.tool_outcome if result is not None else None,
        "usage": _safe_usage(result.usage if result is not None else {}),
    }
    if run.status.value == "failed" and summary:
        view["error_summary"] = summary
    return view


def _project_timeline(
    events: list[dict[str, Any]],
    *,
    run_sequence: int,
) -> list[dict[str, Any]]:
    """Aggregate routine tools while preserving high-signal ordering."""

    projected: list[dict[str, Any]] = []
    aggregate: dict[str, Any] | None = None
    seen_phases: set[tuple[Any, ...]] = set()
    for event in events:
        row = {
            "run_number": run_sequence,
            "run_sequence": run_sequence,
            **event,
        }
        if row.get("kind") == WorkerUiEventKind.PHASE:
            phase = str(row.get("phase") or "")
            # Tool rows already describe these high-frequency transitions. Keep
            # them as current state above the timeline, not as scrollback noise.
            if phase in {"using_tool", "processing"}:
                continue
            fingerprint = (
                phase,
                row.get("current_step"),
                row.get("todo_completed"),
                row.get("todo_total"),
            )
            if fingerprint in seen_phases:
                continue
            seen_phases.add(fingerprint)
        if (
            row.get("kind") == WorkerUiEventKind.TOOL
            and row.get("signal") == "low"
            and row.get("status") == "done"
        ):
            name = str(row.get("tool_name") or "tool")
            if aggregate is None:
                aggregate = {
                    "kind": "tools",
                    "signal": "low",
                    "status": "done",
                    "count": 0,
                    "tool_counts": {},
                    "duration_ms": 0,
                    "run_number": run_sequence,
                    "run_sequence": run_sequence,
                    "ts": row.get("ts"),
                }
                projected.append(aggregate)
            aggregate["count"] += 1
            counts = aggregate["tool_counts"]
            counts[name] = counts.get(name, 0) + 1
            duration = row.get("duration_ms")
            if isinstance(duration, int):
                aggregate["duration_ms"] += duration
            aggregate["ended_at"] = row.get("ts")
            continue
        projected.append(row)
        # Any visible non-routine row is an ordering boundary. Without this,
        # reads on opposite sides of a phase transition collapse into one row
        # positioned before the phase and misrepresent the execution timeline.
        aggregate = None
    return projected


def _safe_usage(value: Mapping[str, Any]) -> dict[str, int]:
    """Keep numeric accounting only; never retain provider metadata."""

    safe: dict[str, int] = {}
    token_keys = (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    )
    for key in token_keys:
        number = _bounded_int(value.get(key))
        if number is not None:
            safe[key] = max(0, number)
    total = value.get("totals") or value.get("total")
    if isinstance(total, Mapping):
        for key in token_keys:
            number = _bounded_int(total.get(key))
            if number is not None:
                safe[key] = max(0, number)
    return safe


def _bounded_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return max(-1_000_000_000, min(1_000_000_000, int(value)))


def _safe_identifier(value: Any) -> str | None:
    text = str(value or "")
    return text if _SAFE_IDENTIFIER.fullmatch(text) else None


def _safe_text(value: Any, *, limit: int) -> str:
    text = _ANSI_ESCAPE.sub("", str(value or ""))
    text = _INTERNAL_ID.sub("[id]", text)
    text = "".join(
        " "
        if char.isspace()
        else ""
        if unicodedata.category(char).startswith("C")
        else char
        for char in text
    )
    return " ".join(text.split())[:limit]


def _safe_multiline_text(value: Any, *, limit: int) -> str:
    return sanitize_operator_output(value, limit=limit)


def _now() -> str:
    return datetime.now(UTC).isoformat()
