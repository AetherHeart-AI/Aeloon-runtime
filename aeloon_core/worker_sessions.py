"""Durable control-plane records for Base-owned Worker sessions.

The UASM state machine remains the execution engine.  This module deliberately
models its long-lived ownership separately: a WorkerSession holds immutable
profile provenance and private context, while each WorkerRun owns one task and
one terminal result.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkerRunStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_CONTEXT = "waiting_for_context"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"

    @property
    def terminal(self) -> bool:
        return self in {
            self.COMPLETED,
            self.PARTIAL,
            self.FAILED,
            self.CANCELLED,
            self.ARCHIVED,
        }


class WorkerSessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    ARCHIVED = "archived"


class RunnerAttemptStatus(StrEnum):
    LEASED = "leased"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    INTERRUPTED = "interrupted"
    LOST = "lost"


class WorkerOperation(StrEnum):
    SPAWN = "spawn"
    SEND = "send"


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different scheduling request."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ProfileHandle(_FrozenModel):
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    artifact_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    activation_audit_id: str = Field(min_length=1)
    contract_hash: str = Field(min_length=1)


class PermissionSnapshot(_FrozenModel):
    tool_names: tuple[str, ...] = ()
    workspace_paths: tuple[str, ...] = ()
    network_hosts: tuple[str, ...] = ()
    sensitivity: str = "normal"


class BudgetGrant(_FrozenModel):
    max_tokens: int = Field(ge=0)
    max_seconds: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)


class ContextEnvelope(_FrozenModel):
    envelope_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal: str = Field(min_length=1)
    constraints: tuple[str, ...] = ()
    excerpts: tuple[dict[str, Any], ...] = ()
    artifact_refs: tuple[dict[str, Any], ...] = ()
    permissions: PermissionSnapshot
    budget: BudgetGrant
    expected_output: dict[str, Any] = Field(default_factory=dict)


class WorkerReport(_FrozenModel):
    """Model-authored data.  Lifecycle and accounting are host-owned."""

    summary: str = Field(min_length=1)
    evidence: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()


class ResultEnvelope(_FrozenModel):
    worker_id: str
    run_id: str
    status: WorkerRunStatus
    profile: ProfileHandle
    report: WorkerReport | None = None
    tool_outcome: Literal["known", "unknown", "none"] = "none"
    usage: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class WorkerSessionRecord:
    worker_id: str
    base_session_id: str
    profile: ProfileHandle
    status: WorkerSessionStatus
    created_at: str


@dataclass(frozen=True)
class WorkerRunRecord:
    run_id: str
    worker_id: str
    base_turn_id: str | None
    status: WorkerRunStatus
    context: ContextEnvelope
    idempotency_key: str
    created_at: str
    result: ResultEnvelope | None = None


class WorkerStore:
    """SQLite authority for Worker lifecycle and idempotent scheduling."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "worker-control.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS worker_sessions (
                  worker_id TEXT PRIMARY KEY,
                  base_session_id TEXT NOT NULL,
                  profile_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_runs (
                  run_id TEXT PRIMARY KEY,
                  worker_id TEXT NOT NULL REFERENCES worker_sessions(worker_id),
                  base_turn_id TEXT,
                  status TEXT NOT NULL,
                  context_json TEXT NOT NULL,
                  result_json TEXT,
                  operation_type TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(worker_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS worker_runs_worker_idx
                  ON worker_runs(worker_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS worker_checkpoints (
                  run_id TEXT PRIMARY KEY REFERENCES worker_runs(run_id),
                  checkpoint_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            self._migrate_worker_run_operation_type(connection)

    def create_worker(
        self,
        *,
        base_session_id: str,
        profile: ProfileHandle,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None = None,
    ) -> tuple[WorkerSessionRecord, WorkerRunRecord, bool]:
        """Create a Worker and its first queued Run, or return an idempotent match."""

        now = _now()
        worker_id = uuid.uuid4().hex
        run_id = uuid.uuid4().hex
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT r.* FROM worker_runs r JOIN worker_sessions s USING(worker_id) "
                "WHERE s.base_session_id = ? AND r.idempotency_key = ?",
                (base_session_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                session = self._session_row(connection, existing["worker_id"])
                self._validate_idempotent_replay(
                    existing,
                    operation=WorkerOperation.SPAWN,
                    base_turn_id=base_turn_id,
                    context=context,
                    stored_profile=session.profile,
                    requested_profile=profile,
                )
                return session, self._run_from_row(existing), False
            connection.execute(
                "INSERT INTO worker_sessions VALUES (?, ?, ?, ?, ?)",
                (worker_id, base_session_id, _dump(profile), WorkerSessionStatus.IDLE, now),
            )
            connection.execute(
                "INSERT INTO worker_runs("
                "run_id, worker_id, base_turn_id, status, context_json, result_json, "
                "operation_type, idempotency_key, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                (
                    run_id,
                    worker_id,
                    base_turn_id,
                    WorkerRunStatus.QUEUED,
                    _dump(context),
                    WorkerOperation.SPAWN,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            return (
                WorkerSessionRecord(
                    worker_id,
                    base_session_id,
                    profile,
                    WorkerSessionStatus.IDLE,
                    now,
                ),
                WorkerRunRecord(run_id, worker_id, base_turn_id, WorkerRunStatus.QUEUED,
                                context, idempotency_key, now),
                True,
            )

    def list_workers(self, base_session_id: str) -> list[WorkerSessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM worker_sessions WHERE base_session_id = ? ORDER BY created_at",
                (base_session_id,),
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def get_worker(self, worker_id: str) -> WorkerSessionRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM worker_sessions WHERE worker_id = ?", (worker_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown worker: {worker_id}")
        return self._session_from_row(row)

    def get_run(self, run_id: str) -> WorkerRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown worker run: {run_id}")
        return self._run_from_row(row)

    def list_runs(self, worker_id: str) -> list[WorkerRunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM worker_runs WHERE worker_id = ? ORDER BY created_at", (worker_id,)
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def create_run(
        self,
        *,
        worker_id: str,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None = None,
    ) -> tuple[WorkerRunRecord, bool]:
        """Queue a follow-up Run without reopening any prior terminal Run."""

        now = _now()
        run_id = uuid.uuid4().hex
        with self._transaction() as connection:
            self._session_row(connection, worker_id)
            existing = connection.execute(
                "SELECT * FROM worker_runs WHERE worker_id = ? AND idempotency_key = ?",
                (worker_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                self._validate_idempotent_replay(
                    existing,
                    operation=WorkerOperation.SEND,
                    base_turn_id=base_turn_id,
                    context=context,
                )
                return self._run_from_row(existing), False
            connection.execute(
                "INSERT INTO worker_runs("
                "run_id, worker_id, base_turn_id, status, context_json, result_json, "
                "operation_type, idempotency_key, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                (
                    run_id,
                    worker_id,
                    base_turn_id,
                    WorkerRunStatus.QUEUED,
                    _dump(context),
                    WorkerOperation.SEND,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            return (
                WorkerRunRecord(
                    run_id,
                    worker_id,
                    base_turn_id,
                    WorkerRunStatus.QUEUED,
                    context,
                    idempotency_key,
                    now,
                ),
                True,
            )

    def list_queued_runs(self) -> list[WorkerRunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM worker_runs WHERE status = ? ORDER BY created_at",
                (WorkerRunStatus.QUEUED,),
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def request_cancel(self, run_id: str) -> WorkerRunRecord:
        """Cancel a not-yet-running Run; active execution is cancelled by its manager."""

        return self.try_cancel_run(run_id)[0]

    def try_start_run(self, run_id: str) -> tuple[WorkerRunRecord, bool]:
        """Atomically claim one queued Run across in-process and detached runners."""

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown worker run: {run_id}")
            current = self._run_from_row(row)
            if current.status is not WorkerRunStatus.QUEUED:
                return current, False
            other_running = connection.execute(
                "SELECT 1 FROM worker_runs "
                "WHERE worker_id = ? AND run_id != ? AND status = ? LIMIT 1",
                (current.worker_id, run_id, WorkerRunStatus.RUNNING),
            ).fetchone()
            if other_running is not None:
                return current, False
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, updated_at = ? "
                "WHERE run_id = ? AND status = ?",
                (
                    WorkerRunStatus.RUNNING,
                    _now(),
                    run_id,
                    WorkerRunStatus.QUEUED,
                ),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                assert row is not None
                return self._run_from_row(row), False
            connection.execute(
                "UPDATE worker_sessions SET status = ? WHERE worker_id = ?",
                (WorkerSessionStatus.RUNNING, current.worker_id),
            )
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            return self._run_from_row(row), True

    def try_cancel_run(self, run_id: str) -> tuple[WorkerRunRecord, bool]:
        """Atomically cancel one active Run without overwriting a terminal outcome."""

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown worker run: {run_id}")
            current = self._run_from_row(row)
            if current.status.terminal:
                return current, False
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, updated_at = ? "
                "WHERE run_id = ? AND status = ?",
                (WorkerRunStatus.CANCELLED, _now(), run_id, current.status),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                assert row is not None
                return self._run_from_row(row), False
            self._refresh_session_status(connection, current.worker_id)
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            return self._run_from_row(row), True

    def archive_worker(self, worker_id: str) -> WorkerSessionRecord:
        """Archive only after every Run is terminal, retaining all audit records."""

        with self._transaction() as connection:
            session = self._session_row(connection, worker_id)
            active = connection.execute(
                "SELECT 1 FROM worker_runs WHERE worker_id = ? AND status IN (?, ?, ?, ?)",
                (
                    worker_id,
                    WorkerRunStatus.CREATED,
                    WorkerRunStatus.QUEUED,
                    WorkerRunStatus.RUNNING,
                    WorkerRunStatus.WAITING_FOR_CONTEXT,
                ),
            ).fetchone()
            if active is not None:
                raise ValueError("cancel or finish active worker runs before archiving")
            connection.execute(
                "UPDATE worker_sessions SET status = ? WHERE worker_id = ?",
                (WorkerSessionStatus.ARCHIVED, worker_id),
            )
            return WorkerSessionRecord(
                session.worker_id,
                session.base_session_id,
                session.profile,
                WorkerSessionStatus.ARCHIVED,
                session.created_at,
            )

    def save_checkpoint(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        """Persist a safe-point checkpoint; callers never resume from trace digests."""

        self.get_run(run_id)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO worker_checkpoints(run_id, checkpoint_json, created_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET checkpoint_json = excluded.checkpoint_json, "
                "created_at = excluded.created_at",
                (run_id, json.dumps(checkpoint, ensure_ascii=False, sort_keys=True), _now()),
            )

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT checkpoint_json FROM worker_checkpoints WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["checkpoint_json"]) if row is not None else None

    def append_transcript(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Append private Worker execution data; Base history never reads this file."""

        run = self.get_run(run_id)
        path = (
            self.data_dir
            / "worker-sessions"
            / run.worker_id
            / "runs"
            / run.run_id
            / "transcript.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return path

    def transition_run(
        self,
        run_id: str,
        *,
        expected: WorkerRunStatus,
        status: WorkerRunStatus,
        result: ResultEnvelope | None = None,
    ) -> WorkerRunRecord:
        """Compare-and-swap a Run state; terminal Runs cannot be reopened."""

        if expected.terminal:
            raise ValueError("terminal worker runs cannot transition")
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, result_json = COALESCE(?, result_json), "
                "updated_at = ? "
                "WHERE run_id = ? AND status = ?",
                (status, _dump(result) if result else None, _now(), run_id, expected),
            )
            if cursor.rowcount != 1:
                raise ValueError("worker run state changed concurrently")
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            if status is WorkerRunStatus.RUNNING:
                connection.execute(
                    "UPDATE worker_sessions SET status = ? WHERE worker_id = ?",
                    (WorkerSessionStatus.RUNNING, row["worker_id"]),
                )
            elif status is not WorkerRunStatus.QUEUED:
                active = connection.execute(
                    "SELECT 1 FROM worker_runs WHERE worker_id = ? AND status = ?",
                    (row["worker_id"], WorkerRunStatus.RUNNING),
                ).fetchone()
                if active is None:
                    connection.execute(
                        "UPDATE worker_sessions SET status = ? WHERE worker_id = ?",
                        (WorkerSessionStatus.IDLE, row["worker_id"]),
                    )
            return self._run_from_row(row)

    def complete_run(self, run_id: str, result: ResultEnvelope) -> WorkerRunRecord:
        return self.try_finalize_run(run_id, result)[0]

    def try_finalize_run(
        self,
        run_id: str,
        result: ResultEnvelope,
    ) -> tuple[WorkerRunRecord, bool]:
        """Atomically commit one terminal result; the first terminal writer wins."""

        if result.run_id != run_id:
            raise ValueError("result run_id does not match the finalized WorkerRun")
        if result.status not in {
            WorkerRunStatus.COMPLETED,
            WorkerRunStatus.PARTIAL,
            WorkerRunStatus.FAILED,
        }:
            raise ValueError("Worker finalization requires completed, partial, or failed status")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown worker run: {run_id}")
            current = self._run_from_row(row)
            if current.worker_id != result.worker_id:
                raise ValueError("result worker_id does not match the finalized WorkerRun")
            if current.status.terminal:
                return current, False
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, result_json = ?, updated_at = ? "
                "WHERE run_id = ? AND status = ?",
                (result.status, _dump(result), _now(), run_id, current.status),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                assert row is not None
                return self._run_from_row(row), False
            self._refresh_session_status(connection, current.worker_id)
            row = connection.execute(
                "SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            return self._run_from_row(row), True

    @staticmethod
    def _refresh_session_status(
        connection: sqlite3.Connection,
        worker_id: str,
    ) -> None:
        running = connection.execute(
            "SELECT 1 FROM worker_runs WHERE worker_id = ? AND status = ? LIMIT 1",
            (worker_id, WorkerRunStatus.RUNNING),
        ).fetchone()
        connection.execute(
            "UPDATE worker_sessions SET status = ? "
            "WHERE worker_id = ? AND status != ?",
            (
                WorkerSessionStatus.RUNNING if running is not None else WorkerSessionStatus.IDLE,
                worker_id,
                WorkerSessionStatus.ARCHIVED,
            ),
        )

    @staticmethod
    def _migrate_worker_run_operation_type(connection: sqlite3.Connection) -> None:
        """Add operation provenance to databases created before strict replay checks."""

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(worker_runs)").fetchall()
        }
        if "operation_type" in columns:
            return
        connection.execute(
            "ALTER TABLE worker_runs ADD COLUMN operation_type "
            "TEXT NOT NULL DEFAULT 'send'"
        )
        # Every Worker is born with exactly one Run, so the first inserted Run
        # is the historical spawn operation. All later Runs are sends.
        connection.execute(
            "UPDATE worker_runs SET operation_type = ? WHERE rowid IN ("
            "SELECT MIN(rowid) FROM worker_runs GROUP BY worker_id"
            ")",
            (WorkerOperation.SPAWN,),
        )

    @staticmethod
    def _validate_idempotent_replay(
        row: sqlite3.Row,
        *,
        operation: WorkerOperation,
        base_turn_id: str | None,
        context: ContextEnvelope,
        stored_profile: ProfileHandle | None = None,
        requested_profile: ProfileHandle | None = None,
    ) -> None:
        conflicts: list[str] = []
        if row["operation_type"] != operation:
            conflicts.append("operation type")
        if row["base_turn_id"] != base_turn_id:
            conflicts.append("base turn")
        stored_context = ContextEnvelope.model_validate_json(row["context_json"])
        if _normalized_context(stored_context) != _normalized_context(context):
            conflicts.append("context envelope")
        if operation is WorkerOperation.SPAWN and stored_profile != requested_profile:
            conflicts.append("profile")
        if conflicts:
            fields = ", ".join(conflicts)
            raise IdempotencyConflictError(
                f"idempotency conflict for key {row['idempotency_key']!r}: "
                f"request differs in {fields}"
            )

    def _session_row(self, connection: sqlite3.Connection, worker_id: str) -> WorkerSessionRecord:
        row = connection.execute(
            "SELECT * FROM worker_sessions WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        assert row is not None
        return self._session_from_row(row)

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> WorkerSessionRecord:
        return WorkerSessionRecord(
            row["worker_id"],
            row["base_session_id"],
            ProfileHandle.model_validate_json(row["profile_json"]),
            WorkerSessionStatus(row["status"]),
            row["created_at"],
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> WorkerRunRecord:
        return WorkerRunRecord(
            row["run_id"], row["worker_id"], row["base_turn_id"], WorkerRunStatus(row["status"]),
            ContextEnvelope.model_validate_json(row["context_json"]), row["idempotency_key"],
            row["created_at"], ResultEnvelope.model_validate_json(row["result_json"])
            if row["result_json"] else None,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _dump(value: BaseModel | None) -> str:
    return json.dumps(
        value.model_dump(mode="json") if value else None,
        ensure_ascii=False,
        sort_keys=True,
    )


def _normalized_context(context: ContextEnvelope) -> str:
    """Canonical scheduling input, excluding the envelope's random identity."""

    payload = context.model_dump(mode="json")
    payload.pop("envelope_id", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
