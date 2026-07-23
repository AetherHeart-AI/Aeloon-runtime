"""SQLite authority for durable WorkerSession and WorkerRun state."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import uuid
import weakref
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aeloon_core.flow_activation_fence import (
    flow_activation_fence,
    flow_id_from_turn_id,
)
from aeloon_core.message_history import (
    MESSAGE_FORMAT,
    MESSAGE_SCHEMA_VERSION,
    LegacySessionError,
)
from aeloon_core.worker_state import (
    BudgetGrant,
    BudgetIncrease,
    ContextEnvelope,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    IdempotencyConflictError,
    PermissionSnapshot,
    RelatedContextSection,
    RelatedWorkerContext,
    ReportItem,
    ReportText,
    ResultEnvelope,
    WaitingRequest,
    WorkerOperation,
    WorkerReport,
    WorkerRunFencedError,
    WorkerRunRecord,
    WorkerRunStatus,
    WorkerSessionRecord,
    WorkerSessionStatus,
)
from aeloon_core.workers import WorkerSnapshot

SCHEMA_VERSION = 5
_OWNER_STORES: weakref.WeakSet[Any] = weakref.WeakSet()
_OWNER_AT_FORK_REGISTERED = False


class WorkerStore:
    """SQLite authority for Worker lifecycle, snapshots, and continuation order."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "worker-control.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_pid = 0
        self._owner_token = ""
        self._owner_lock: Any | None = None
        self._initialize()
        self._ensure_execution_owner()
        _register_execution_owner(self)

    @property
    def execution_owner_token(self) -> str:
        """Return this process epoch's durable WorkerRun owner identity."""

        self._ensure_execution_owner()
        return self._owner_token

    def _ensure_execution_owner(self) -> None:
        """Hold a kernel-released lease that proves this exact owner is alive."""

        pid = os.getpid()
        if (
            self._owner_pid == pid
            and self._owner_token
            and self._owner_lock is not None
            and not self._owner_lock.closed
        ):
            return

        # A fork inherits the original open-file description. Drop the child's
        # reference and create a new epoch; it must never impersonate its parent.
        if self._owner_lock is not None:
            self._owner_lock.close()

        token = uuid.uuid4().hex
        lease_dir = self.data_dir / ".worker-owner-leases"
        lease_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle = (lease_dir / f"{token}.lock").open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={pid}\n".encode())
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._owner_pid = pid
        self._owner_token = token
        self._owner_lock = handle

    def _discard_inherited_execution_owner(self) -> None:
        """Ensure a forked child cannot keep its parent's owner lease alive."""

        if self._owner_lock is not None:
            self._owner_lock.close()
        self._owner_pid = 0
        self._owner_token = ""
        self._owner_lock = None

    @contextmanager
    def _verified_dead_execution_owner(self, token: str | None) -> Iterator[bool]:
        """Yield true only while holding proof that an exact owner exited.

        The token names a unique lock file created before the Run claim commits.
        A live or merely stalled process keeps its exclusive flock indefinitely;
        the kernel releases it on process exit, including SIGKILL. Missing,
        malformed, or inaccessible legacy ownership fails closed.
        """

        self._ensure_execution_owner()
        if token is None or token == self._owner_token or not _valid_owner_token(token):
            yield False
            return
        path = self.data_dir / ".worker-owner-leases" / f"{token}.lock"
        try:
            handle = path.open("rb")
        except OSError:
            yield False
            return
        acquired = False
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError):
                yield False
                return
            yield True
        finally:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"worker store schema v{version} is newer than supported v{SCHEMA_VERSION}"
                )
            if version < 2:
                self._archive_legacy_schema(connection)
                self._create_schema(connection)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version in {2, 3, 4}:
                self._migrate_durable_execution(connection)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._validate_schema(connection)

    @staticmethod
    def _migrate_durable_execution(connection: sqlite3.Connection) -> None:
        """Preserve v2/v3 data while adding durable execution fencing."""

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(worker_runs)").fetchall()
        }
        if "activated_at" not in columns:
            connection.execute("ALTER TABLE worker_runs ADD COLUMN activated_at TEXT")
        if "active_tool_count" not in columns:
            connection.execute(
                "ALTER TABLE worker_runs ADD COLUMN active_tool_count INTEGER "
                "NOT NULL DEFAULT 0 CHECK(active_tool_count >= 0)"
            )
        if "execution_owner_token" not in columns:
            connection.execute(
                "ALTER TABLE worker_runs ADD COLUMN execution_owner_token TEXT"
            )

    @staticmethod
    def _archive_legacy_schema(connection: sqlite3.Connection) -> None:
        """Move v1 tables aside intact before creating the current authority tables."""

        for table in (
            "worker_ui_events",
            "worker_ui_state",
            "worker_checkpoints",
            "worker_runs",
            "worker_sessions",
        ):
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if exists is None:
                continue
            archived = f"legacy_v1_{table}"
            archived_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (archived,),
            ).fetchone()
            if archived_exists is not None:
                raise RuntimeError(f"cannot archive legacy table: {archived} already exists")
            connection.execute(f"ALTER TABLE {table} RENAME TO {archived}")
            for index in connection.execute(f"PRAGMA index_list({archived})").fetchall():
                name = str(index["name"])
                if not name.startswith("sqlite_autoindex"):
                    connection.execute(f'DROP INDEX "{name}"')

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        statements = (
            """CREATE TABLE worker_sessions (
              worker_id TEXT PRIMARY KEY,
              base_session_id TEXT NOT NULL,
              snapshot_json TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            )""",
            """CREATE INDEX worker_sessions_base_idx
              ON worker_sessions(base_session_id, created_at)""",
            """CREATE TABLE worker_runs (
              run_id TEXT PRIMARY KEY,
              worker_id TEXT NOT NULL REFERENCES worker_sessions(worker_id) ON DELETE CASCADE,
              run_sequence INTEGER NOT NULL,
              source_run_id TEXT REFERENCES worker_runs(run_id),
              base_turn_id TEXT,
              activated_at TEXT,
              active_tool_count INTEGER NOT NULL DEFAULT 0
                CHECK(active_tool_count >= 0),
              execution_owner_token TEXT,
              status TEXT NOT NULL,
              context_json TEXT NOT NULL,
              result_json TEXT,
              waiting_request_json TEXT,
              cancel_requested_at TEXT,
              operation_type TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(worker_id, run_sequence),
              UNIQUE(worker_id, idempotency_key)
            )""",
            """CREATE INDEX worker_runs_worker_idx
              ON worker_runs(worker_id, run_sequence DESC)""",
            """CREATE INDEX worker_runs_status_idx
              ON worker_runs(status, created_at)""",
            """CREATE TABLE worker_checkpoints (
              run_id TEXT PRIMARY KEY REFERENCES worker_runs(run_id) ON DELETE CASCADE,
              checkpoint_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )""",
            """CREATE TABLE worker_ui_events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL REFERENCES worker_runs(run_id) ON DELETE CASCADE,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )""",
            """CREATE INDEX worker_ui_events_run_idx
              ON worker_ui_events(run_id, sequence)""",
            """CREATE TABLE worker_ui_state (
              singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
              event_count INTEGER NOT NULL
            )""",
            "INSERT INTO worker_ui_state(singleton, event_count) VALUES (1, 0)",
        )
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        required = {
            "worker_sessions": {
                "worker_id",
                "base_session_id",
                "snapshot_json",
                "status",
                "created_at",
            },
            "worker_runs": {
                "run_id",
                "worker_id",
                "run_sequence",
                "source_run_id",
                "base_turn_id",
                "activated_at",
                "active_tool_count",
                "execution_owner_token",
                "status",
                "context_json",
                "result_json",
                "waiting_request_json",
                "cancel_requested_at",
                "operation_type",
                "idempotency_key",
                "created_at",
                "updated_at",
            },
            "worker_checkpoints": {"run_id", "checkpoint_json", "created_at"},
            "worker_ui_events": {
                "sequence",
                "run_id",
                "kind",
                "payload_json",
                "created_at",
            },
            "worker_ui_state": {"singleton", "event_count"},
        }
        table_info: dict[str, list[sqlite3.Row]] = {}
        for table, columns in required.items():
            info = connection.execute(f"PRAGMA table_info({table})").fetchall()
            table_info[table] = info
            actual = {str(row["name"]) for row in info}
            missing = columns - actual
            if missing:
                raise RuntimeError(
                    f"worker store v{SCHEMA_VERSION} is invalid: {table} missing {sorted(missing)}"
                )

        required_primary_keys = {
            "worker_sessions": ("worker_id",),
            "worker_runs": ("run_id",),
            "worker_checkpoints": ("run_id",),
            "worker_ui_events": ("sequence",),
            "worker_ui_state": ("singleton",),
        }
        for table, expected in required_primary_keys.items():
            actual = tuple(
                str(row["name"])
                for row in sorted(table_info[table], key=lambda row: int(row["pk"]))
                if int(row["pk"]) > 0
            )
            if actual != expected:
                raise RuntimeError(
                    f"worker store v{SCHEMA_VERSION} is invalid: {table} primary key "
                    f"is {actual!r}, expected {expected!r}"
                )

        unique_indexes = {
            tuple(
                str(column["name"])
                for column in connection.execute(f"PRAGMA index_info({index['name']})").fetchall()
            )
            for index in connection.execute("PRAGMA index_list(worker_runs)").fetchall()
            if bool(index["unique"])
        }
        required_unique_indexes = {
            ("worker_id", "run_sequence"),
            ("worker_id", "idempotency_key"),
        }
        missing_indexes = required_unique_indexes - unique_indexes
        if missing_indexes:
            raise RuntimeError(
                f"worker store v{SCHEMA_VERSION} is invalid: worker_runs missing unique "
                f"indexes {sorted(missing_indexes)}"
            )

        required_foreign_keys = {
            "worker_runs": {
                ("worker_id", "worker_sessions", "worker_id"),
                ("source_run_id", "worker_runs", "run_id"),
            },
            "worker_checkpoints": {("run_id", "worker_runs", "run_id")},
            "worker_ui_events": {("run_id", "worker_runs", "run_id")},
        }
        for table, expected in required_foreign_keys.items():
            actual = {
                (str(row["from"]), str(row["table"]), str(row["to"]))
                for row in connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
            }
            missing = expected - actual
            if missing:
                raise RuntimeError(
                    f"worker store v{SCHEMA_VERSION} is invalid: {table} missing foreign "
                    f"keys {sorted(missing)}"
                )

    def create_worker(
        self,
        *,
        base_session_id: str,
        snapshot: WorkerSnapshot,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None = None,
    ) -> tuple[WorkerSessionRecord, WorkerRunRecord, bool]:
        """Create a WorkerSession and its first queued Run idempotently."""

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
                session = self._session_row(connection, str(existing["worker_id"]))
                self._validate_idempotent_replay(
                    existing,
                    operation=WorkerOperation.SPAWN,
                    base_turn_id=base_turn_id,
                    context=context,
                    source_run_id=None,
                    stored_snapshot=session.snapshot,
                    requested_snapshot=snapshot,
                )
                return session, self._run_from_row(existing), False

            connection.execute(
                "INSERT INTO worker_sessions("
                "worker_id, base_session_id, snapshot_json, status, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    worker_id,
                    base_session_id,
                    _dump(snapshot),
                    WorkerSessionStatus.IDLE,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO worker_runs("
                "run_id, worker_id, run_sequence, source_run_id, base_turn_id, status, "
                "context_json, result_json, waiting_request_json, operation_type, "
                "idempotency_key, created_at, updated_at"
                ") VALUES (?, ?, 1, NULL, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)",
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
            session = WorkerSessionRecord(
                worker_id=worker_id,
                base_session_id=base_session_id,
                snapshot=snapshot,
                status=WorkerSessionStatus.IDLE,
                created_at=now,
            )
            run = WorkerRunRecord(
                run_id=run_id,
                worker_id=worker_id,
                base_turn_id=base_turn_id,
                status=WorkerRunStatus.QUEUED,
                context=context,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            return session, run, True

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
                "SELECT * FROM worker_runs WHERE worker_id = ? ORDER BY run_sequence",
                (worker_id,),
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
        """Create an explicit reuse Run from the latest completed checkpoint."""

        return self._create_followup(
            worker_id=worker_id,
            source_run_id=None,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
            operation=WorkerOperation.REUSE,
        )

    def create_flow_reuse_run(
        self,
        *,
        source_run_id: str,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str,
    ) -> tuple[WorkerRunRecord, bool]:
        """Create a Flow-owned reuse Run from one exact latest source Run."""

        source = self.get_run(source_run_id)
        return self._create_followup(
            worker_id=source.worker_id,
            source_run_id=source_run_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
            operation=WorkerOperation.REUSE,
            exact_reuse_source=True,
        )

    def create_resume_run(
        self,
        *,
        source_run_id: str,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None = None,
    ) -> tuple[WorkerRunRecord, bool]:
        """Atomically create one continuation from the latest waiting Run."""

        source = self.get_run(source_run_id)
        return self._create_followup(
            worker_id=source.worker_id,
            source_run_id=source_run_id,
            context=context,
            idempotency_key=idempotency_key,
            base_turn_id=base_turn_id,
            operation=WorkerOperation.RESUME,
        )

    def _create_followup(
        self,
        *,
        worker_id: str,
        source_run_id: str | None,
        context: ContextEnvelope,
        idempotency_key: str,
        base_turn_id: str | None,
        operation: WorkerOperation,
        exact_reuse_source: bool = False,
    ) -> tuple[WorkerRunRecord, bool]:
        now = _now()
        run_id = uuid.uuid4().hex
        with self._transaction() as connection:
            session = self._session_row(connection, worker_id)
            if session.status is WorkerSessionStatus.ARCHIVED:
                raise ValueError("archived Workers cannot be reused")

            existing = connection.execute(
                "SELECT * FROM worker_runs WHERE worker_id = ? AND idempotency_key = ?",
                (worker_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                self._validate_idempotent_replay(
                    existing,
                    operation=operation,
                    base_turn_id=base_turn_id,
                    context=context,
                    source_run_id=source_run_id,
                )
                if exact_reuse_source and existing["source_run_id"] != source_run_id:
                    raise IdempotencyConflictError(
                        f"idempotency conflict for key {idempotency_key!r}: "
                        "request differs in source run"
                    )
                return self._run_from_row(existing), False

            latest_row = connection.execute(
                "SELECT * FROM worker_runs WHERE worker_id = ? ORDER BY run_sequence DESC LIMIT 1",
                (worker_id,),
            ).fetchone()
            if latest_row is None:
                raise ValueError("the Worker has no prior Run to continue")
            latest = self._run_from_row(latest_row)

            if _permissions_expand(latest.context.permissions, context.permissions):
                raise ValueError("a continuation cannot expand Worker permissions")
            if operation is WorkerOperation.RESUME:
                if latest.run_id != source_run_id:
                    raise ValueError("the Worker already moved to a newer run; inspect it again")
                if latest.status is not WorkerRunStatus.WAITING_FOR_CONTEXT:
                    raise ValueError("resume_worker requires the latest waiting_for_context Run")
                checkpoint = connection.execute(
                    "SELECT checkpoint_json FROM worker_checkpoints WHERE run_id = ?",
                    (source_run_id,),
                ).fetchone()
                if checkpoint is None:
                    raise ValueError("the waiting WorkerRun has no resumable checkpoint")
                _require_current_checkpoint(checkpoint, source_run_id)
                source_id = source_run_id
            elif exact_reuse_source:
                if latest.run_id != source_run_id:
                    raise ValueError(
                        "the WorkerSession context advanced beyond the requested source Run"
                    )
                if latest.status not in {
                    WorkerRunStatus.COMPLETED,
                    WorkerRunStatus.PARTIAL,
                    WorkerRunStatus.FAILED,
                    WorkerRunStatus.CANCELLED,
                }:
                    raise ValueError("the source WorkerRun is not safely reusable")
                if (
                    latest.status is WorkerRunStatus.FAILED
                    and (
                        latest.result is None
                        or latest.result.tool_outcome == "unknown"
                    )
                ):
                    raise ValueError("a WorkerRun with unknown outcome cannot be reused")
                if (
                    latest.result is not None
                    and latest.result.tool_outcome == "unknown"
                ):
                    raise ValueError("a WorkerRun with unknown outcome cannot be reused")
                checkpoint = connection.execute(
                    "SELECT checkpoint_json FROM worker_checkpoints WHERE run_id = ?",
                    (latest.run_id,),
                ).fetchone()
                if (
                    latest.status
                    in {WorkerRunStatus.COMPLETED, WorkerRunStatus.PARTIAL}
                    and checkpoint is None
                ):
                    raise ValueError("the source WorkerRun has no reusable checkpoint")
                if checkpoint is not None:
                    _require_current_checkpoint(checkpoint, latest.run_id)
                source_id = latest.run_id
            else:
                if latest.status not in {
                    WorkerRunStatus.COMPLETED,
                    WorkerRunStatus.PARTIAL,
                }:
                    raise ValueError("reuse_worker requires the latest completed or partial Run")
                checkpoint = connection.execute(
                    "SELECT checkpoint_json FROM worker_checkpoints WHERE run_id = ?",
                    (latest.run_id,),
                ).fetchone()
                if checkpoint is None:
                    raise ValueError("the latest WorkerRun has no reusable checkpoint")
                _require_current_checkpoint(checkpoint, latest.run_id)
                source_id = latest.run_id

            run_sequence = latest.run_sequence + 1
            connection.execute(
                "INSERT INTO worker_runs("
                "run_id, worker_id, run_sequence, source_run_id, base_turn_id, status, "
                "context_json, result_json, waiting_request_json, operation_type, "
                "idempotency_key, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)",
                (
                    run_id,
                    worker_id,
                    run_sequence,
                    source_id,
                    base_turn_id,
                    WorkerRunStatus.QUEUED,
                    _dump(context),
                    operation,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            return (
                WorkerRunRecord(
                    run_id=run_id,
                    worker_id=worker_id,
                    base_turn_id=base_turn_id,
                    status=WorkerRunStatus.QUEUED,
                    context=context,
                    idempotency_key=idempotency_key,
                    created_at=now,
                    run_sequence=run_sequence,
                    source_run_id=source_id,
                ),
                True,
            )

    def list_queued_runs(
        self,
        *,
        include_flow_owned: bool = False,
    ) -> list[WorkerRunRecord]:
        """List runnable queue entries; Flow reservations require durable activation."""

        query = "SELECT * FROM worker_runs WHERE status = ?"
        parameters: list[Any] = [WorkerRunStatus.QUEUED]
        if not include_flow_owned:
            query += (
                " AND (base_turn_id IS NULL OR base_turn_id NOT LIKE 'flow:%' "
                "OR activated_at IS NOT NULL)"
            )
        query += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._run_from_row(row) for row in rows]

    def activate_run(self, run_id: str) -> WorkerRunRecord:
        """Make one reserved queued Run durably claimable by any runner."""

        with self._transaction() as connection:
            current = self._run_from_row(self._required_run_row(connection, run_id))
            if current.status is WorkerRunStatus.QUEUED and current.activated_at is None:
                now = _now()
                connection.execute(
                    "UPDATE worker_runs SET activated_at = ?, updated_at = ? "
                    "WHERE run_id = ? AND status = ? AND activated_at IS NULL",
                    (
                        now,
                        now,
                        run_id,
                        WorkerRunStatus.QUEUED,
                    ),
                )
            return self._run_from_row(self._required_run_row(connection, run_id))

    def try_start_run(self, run_id: str) -> tuple[WorkerRunRecord, bool]:
        """Atomically claim a queued Run while serializing each WorkerSession."""

        current = self.get_run(run_id)
        flow_id = flow_id_from_turn_id(current.base_turn_id)
        if flow_id is None:
            return self._try_start_run(run_id)
        if current.status is not WorkerRunStatus.QUEUED or current.activated_at is None:
            return current, False
        with flow_activation_fence(self.data_dir, flow_id):
            current = self.get_run(run_id)
            if current.status is not WorkerRunStatus.QUEUED:
                return current, False
            if current.activated_at is None:
                return current, False
            claimable = self._flow_run_is_current(flow_id, current)
            if claimable is None:
                # A transient Flow-store read failure is not evidence that work is
                # obsolete. Leave the reservation queued for a later runner pass.
                return current, False
            if not claimable:
                return self.try_cancel_run(run_id)[0], False
            return self._try_start_run(run_id)

    def _try_start_run(self, run_id: str) -> tuple[WorkerRunRecord, bool]:
        """Claim a Run after any Flow ownership fence has been satisfied."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            row = self._required_run_row(connection, run_id)
            current = self._run_from_row(row)
            if current.status is not WorkerRunStatus.QUEUED:
                return current, False
            if (
                current.base_turn_id is not None
                and current.base_turn_id.startswith("flow:")
                and current.activated_at is None
            ):
                return current, False
            other_running = connection.execute(
                "SELECT 1 FROM worker_runs WHERE worker_id = ? AND run_id != ? "
                "AND status = ? LIMIT 1",
                (current.worker_id, run_id, WorkerRunStatus.RUNNING),
            ).fetchone()
            if other_running is not None:
                return current, False
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, execution_owner_token = ?, "
                "updated_at = ? WHERE run_id = ? AND status = ?",
                (
                    WorkerRunStatus.RUNNING,
                    owner_token,
                    _now(),
                    run_id,
                    WorkerRunStatus.QUEUED,
                ),
            )
            if cursor.rowcount != 1:
                return self._run_from_row(self._required_run_row(connection, run_id)), False
            connection.execute(
                "UPDATE worker_sessions SET status = ? WHERE worker_id = ?",
                (WorkerSessionStatus.RUNNING, current.worker_id),
            )
            return self._run_from_row(self._required_run_row(connection, run_id)), True

    def _flow_run_is_current(
        self,
        flow_id: str,
        run: WorkerRunRecord,
    ) -> bool | None:
        """Check the durable Flow binding while its activation fence is held."""

        try:
            from aeloon_core.flow_state import FlowStatus
            from aeloon_core.flows import FlowStore

            flow = FlowStore(self.data_dir).get_flow(flow_id)
        except KeyError:
            return False
        except (RuntimeError, sqlite3.Error):
            return None
        if flow.status is not FlowStatus.OPEN:
            return False
        worker = self.get_worker(run.worker_id)
        if flow.base_session_id != worker.base_session_id:
            return False
        for node in flow.nodes:
            if (
                node.current_run_id != run.run_id
                or node.worker_id != run.worker_id
                or not node.status.active
            ):
                continue
            binding = next(
                (
                    candidate
                    for candidate in reversed(node.runs)
                    if candidate.run_id == run.run_id
                ),
                None,
            )
            return bool(
                binding is not None
                and binding.worker_id == run.worker_id
                and binding.generation == node.generation
                and binding.attempt == node.attempt
            )
        return False

    def refresh_run_lease(self, run_id: str) -> bool:
        """Refresh one live owner's durable lease while it still owns the Run."""

        current = self.get_run(run_id)
        flow_id = flow_id_from_turn_id(current.base_turn_id)
        if flow_id is None:
            return self._refresh_run_lease(run_id)
        with flow_activation_fence(self.data_dir, flow_id):
            current = self.get_run(run_id)
            authorized = self._flow_run_is_current(flow_id, current)
            if authorized is None:
                # Keep retrying the heartbeat without extending the durable
                # lease. A transient Flow read is not cancellation authority.
                return True
            if not authorized:
                self.try_cancel_run(run_id)
                return False
            return self._refresh_run_lease(run_id)

    def _refresh_run_lease(self, run_id: str) -> bool:
        """Refresh after any Flow execution fence has been satisfied."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            cursor = connection.execute(
                "UPDATE worker_runs SET updated_at = ? WHERE run_id = ? AND status = ? "
                "AND execution_owner_token = ?",
                (_now(), run_id, WorkerRunStatus.RUNNING, owner_token),
            )
            return cursor.rowcount == 1

    def refresh_run_teardown_lease(self, run_id: str) -> bool:
        """Keep a revoked owner's lease alive only while it performs teardown."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            cursor = connection.execute(
                "UPDATE worker_runs SET updated_at = ? WHERE run_id = ? AND status = ? "
                "AND cancel_requested_at IS NOT NULL AND execution_owner_token = ?",
                (_now(), run_id, WorkerRunStatus.RUNNING, owner_token),
            )
            return cursor.rowcount == 1

    def require_run_execution_authority(self, run_id: str) -> None:
        """Fence stale owners before and after every Worker tool boundary."""

        current = self.get_run(run_id)
        flow_id = flow_id_from_turn_id(current.base_turn_id)
        if flow_id is None:
            self._require_run_execution_authority(run_id)
            return
        with flow_activation_fence(self.data_dir, flow_id):
            current = self.get_run(run_id)
            authorized = self._flow_run_is_current(flow_id, current)
            if authorized is None:
                raise WorkerRunFencedError(
                    "Flow execution authority is temporarily unavailable"
                )
            if not authorized:
                self.try_cancel_run(run_id)
                raise WorkerRunFencedError(
                    "Flow execution authority was revoked; no further tools may run"
                )
            self._require_run_execution_authority(run_id)

    def _require_run_execution_authority(self, run_id: str) -> None:
        """Check Worker authority after any Flow fence has been satisfied."""

        owner_token = self.execution_owner_token
        with self._connect() as connection:
            row = self._required_run_row(connection, run_id)
            status = WorkerRunStatus(str(row["status"]))
            cancel_requested_at = row["cancel_requested_at"]
            execution_owner_token = row["execution_owner_token"]
        if (
            status is not WorkerRunStatus.RUNNING
            or cancel_requested_at is not None
            or execution_owner_token != owner_token
        ):
            raise WorkerRunFencedError(
                "WorkerRun execution authority was revoked; no further tools may run"
            )

    def begin_tool_execution(self, run_id: str) -> None:
        """Durably mark one in-flight tool call before it can produce side effects."""

        current = self.get_run(run_id)
        flow_id = flow_id_from_turn_id(current.base_turn_id)
        if flow_id is None:
            self._begin_tool_execution(run_id)
            return
        with flow_activation_fence(self.data_dir, flow_id):
            current = self.get_run(run_id)
            authorized = self._flow_run_is_current(flow_id, current)
            if authorized is None:
                raise WorkerRunFencedError(
                    "Flow execution authority is temporarily unavailable"
                )
            if not authorized:
                self.try_cancel_run(run_id)
                raise WorkerRunFencedError(
                    "Flow execution authority was revoked; no further tools may run"
                )
            self._begin_tool_execution(run_id)

    def _begin_tool_execution(self, run_id: str) -> None:
        """Begin after any Flow execution fence has been satisfied."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            row = self._required_run_row(connection, run_id)
            if (
                WorkerRunStatus(str(row["status"])) is not WorkerRunStatus.RUNNING
                or row["cancel_requested_at"] is not None
                or row["execution_owner_token"] != owner_token
            ):
                raise WorkerRunFencedError(
                    "WorkerRun execution authority was revoked; no further tools may run"
                )
            cursor = connection.execute(
                "UPDATE worker_runs SET active_tool_count = active_tool_count + 1, "
                "updated_at = ? WHERE run_id = ? AND status = ? "
                "AND cancel_requested_at IS NULL AND execution_owner_token = ?",
                (_now(), run_id, WorkerRunStatus.RUNNING, owner_token),
            )
            if cursor.rowcount != 1:
                raise WorkerRunFencedError(
                    "WorkerRun execution authority was revoked; no further tools may run"
                )

    def end_tool_execution(self, run_id: str) -> None:
        """Release one durable in-flight marker after tool teardown completes."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            cursor = connection.execute(
                "UPDATE worker_runs SET active_tool_count = active_tool_count - 1, "
                "updated_at = ? WHERE run_id = ? AND active_tool_count > 0 "
                "AND status = ? AND execution_owner_token = ?",
                (_now(), run_id, WorkerRunStatus.RUNNING, owner_token),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("WorkerRun tool execution marker is unbalanced")

    def expire_stale_running_runs(self, *, stale_before: str) -> list[WorkerRunRecord]:
        """Settle expired leases without replaying side effects of uncertain Runs."""

        expired: list[WorkerRunRecord] = []
        # Keep every acquired dead-owner flock until the SQLite transaction has
        # committed. This makes proof, marker cleanup, and terminal transition
        # one indivisible recovery operation.
        with ExitStack() as owner_proofs:
            with self._transaction() as connection:
                rows = connection.execute(
                    "SELECT * FROM worker_runs WHERE status = ? AND updated_at <= ? "
                    "ORDER BY updated_at",
                    (WorkerRunStatus.RUNNING, stale_before),
                ).fetchall()
                now = _now()
                touched_workers: set[str] = set()
                dead_owners: dict[str | None, bool] = {}
                for row in rows:
                    current = self._run_from_row(row)
                    tool_owner_died = current.active_tool_count > 0
                    if tool_owner_died:
                        token = current.execution_owner_token
                        if token not in dead_owners:
                            dead_owners[token] = owner_proofs.enter_context(
                                self._verified_dead_execution_owner(token)
                            )
                        if not dead_owners[token]:
                            # A live owner may be stalled, or this may be a legacy
                            # marker with unknowable ownership. Both fail closed.
                            continue

                    if tool_owner_died:
                        result = ResultEnvelope(
                            worker_id=current.worker_id,
                            run_id=current.run_id,
                            status=WorkerRunStatus.FAILED,
                            report=WorkerReport(
                                summary=(
                                    "Worker owner process exited while a tool was in "
                                    "flight; execution outcome is unknown. External or "
                                    "descendant side effects may still be running or may "
                                    "already have completed. Clearing the control-plane "
                                    "in-flight marker does not roll them back; inspect side "
                                    "effects before retrying."
                                )
                            ),
                            tool_outcome="unknown",
                        )
                        cursor = connection.execute(
                            "UPDATE worker_runs SET status = ?, result_json = ?, "
                            "waiting_request_json = NULL, active_tool_count = 0, "
                            "updated_at = ? WHERE run_id = ? AND status = ? "
                            "AND updated_at <= ? AND active_tool_count = ? "
                            "AND execution_owner_token IS ?",
                            (
                                WorkerRunStatus.FAILED,
                                _dump(result),
                                now,
                                current.run_id,
                                WorkerRunStatus.RUNNING,
                                stale_before,
                                current.active_tool_count,
                                current.execution_owner_token,
                            ),
                        )
                    elif current.cancel_requested_at is not None:
                        cursor = connection.execute(
                            "UPDATE worker_runs SET status = ?, active_tool_count = 0, "
                            "updated_at = ? WHERE run_id = ? AND status = ? "
                            "AND updated_at <= ? AND active_tool_count = ? "
                            "AND execution_owner_token IS ?",
                            (
                                WorkerRunStatus.CANCELLED,
                                now,
                                current.run_id,
                                WorkerRunStatus.RUNNING,
                                stale_before,
                                current.active_tool_count,
                                current.execution_owner_token,
                            ),
                        )
                    else:
                        result = ResultEnvelope(
                            worker_id=current.worker_id,
                            run_id=current.run_id,
                            status=WorkerRunStatus.FAILED,
                            report=WorkerReport(
                                summary=(
                                    "Worker owner lease expired; execution outcome is unknown. "
                                    "Retry explicitly if it is safe."
                                )
                            ),
                            tool_outcome="unknown",
                        )
                        cursor = connection.execute(
                            "UPDATE worker_runs SET status = ?, result_json = ?, "
                            "waiting_request_json = NULL, active_tool_count = 0, "
                            "updated_at = ? WHERE run_id = ? AND status = ? "
                            "AND updated_at <= ? AND active_tool_count = ? "
                            "AND execution_owner_token IS ?",
                            (
                                WorkerRunStatus.FAILED,
                                _dump(result),
                                now,
                                current.run_id,
                                WorkerRunStatus.RUNNING,
                                stale_before,
                                current.active_tool_count,
                                current.execution_owner_token,
                            ),
                        )
                    if cursor.rowcount != 1:
                        continue
                    touched_workers.add(current.worker_id)
                    expired.append(
                        self._run_from_row(
                            self._required_run_row(connection, current.run_id)
                        )
                    )
                for worker_id in touched_workers:
                    self._refresh_session_status(connection, worker_id)
        return expired

    def try_cancel_run(self, run_id: str) -> tuple[WorkerRunRecord, bool]:
        """Cancel queued work or durably request teardown from its running owner."""

        with self._transaction() as connection:
            current = self._run_from_row(self._required_run_row(connection, run_id))
            if (
                current.status.settled
                and current.status is not WorkerRunStatus.WAITING_FOR_CONTEXT
            ):
                return current, False
            now = _now()
            if current.status is WorkerRunStatus.RUNNING:
                cursor = connection.execute(
                    "UPDATE worker_runs SET cancel_requested_at = ?, updated_at = ? "
                    "WHERE run_id = ? AND status = ? AND cancel_requested_at IS NULL",
                    (now, now, run_id, WorkerRunStatus.RUNNING),
                )
                return self._run_from_row(
                    self._required_run_row(connection, run_id)
                ), cursor.rowcount == 1
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, cancel_requested_at = ?, updated_at = ? "
                "WHERE run_id = ? AND status IN (?, ?)",
                (
                    WorkerRunStatus.CANCELLED,
                    now,
                    now,
                    run_id,
                    WorkerRunStatus.QUEUED,
                    WorkerRunStatus.WAITING_FOR_CONTEXT,
                ),
            )
            if cursor.rowcount != 1:
                return self._run_from_row(self._required_run_row(connection, run_id)), False
            self._refresh_session_status(connection, current.worker_id)
            return self._run_from_row(self._required_run_row(connection, run_id)), True

    def acknowledge_cancel_run(self, run_id: str) -> tuple[WorkerRunRecord, bool]:
        """Mark cancellation settled after the owning executor has torn down."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            current = self._run_from_row(self._required_run_row(connection, run_id))
            if current.status.settled:
                return current, False
            if (
                current.status is WorkerRunStatus.RUNNING
                and current.execution_owner_token != owner_token
            ):
                return current, False
            now = _now()
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, cancel_requested_at = "
                "COALESCE(cancel_requested_at, ?), updated_at = ? "
                "WHERE run_id = ? AND active_tool_count = 0 AND (status = ? OR "
                "(status = ? AND execution_owner_token = ?))",
                (
                    WorkerRunStatus.CANCELLED,
                    now,
                    now,
                    run_id,
                    WorkerRunStatus.QUEUED,
                    WorkerRunStatus.RUNNING,
                    owner_token,
                ),
            )
            if cursor.rowcount != 1:
                return self._run_from_row(self._required_run_row(connection, run_id)), False
            self._refresh_session_status(connection, current.worker_id)
            return self._run_from_row(self._required_run_row(connection, run_id)), True

    def archive_worker(self, worker_id: str) -> WorkerSessionRecord:
        """Soft-delete an idle WorkerSession while retaining its audit records."""

        with self._transaction() as connection:
            session = self._session_row(connection, worker_id)
            active = connection.execute(
                "SELECT 1 FROM worker_runs WHERE worker_id = ? AND status IN (?, ?) LIMIT 1",
                (worker_id, WorkerRunStatus.QUEUED, WorkerRunStatus.RUNNING),
            ).fetchone()
            if active is not None:
                raise ValueError("cancel or finish active worker runs before archiving")
            connection.execute(
                "UPDATE worker_sessions SET status = ? WHERE worker_id = ?",
                (WorkerSessionStatus.ARCHIVED, worker_id),
            )
            return WorkerSessionRecord(
                worker_id=session.worker_id,
                base_session_id=session.base_session_id,
                snapshot=session.snapshot,
                status=WorkerSessionStatus.ARCHIVED,
                created_at=session.created_at,
            )

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT checkpoint_json FROM worker_checkpoints WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["checkpoint_json"]) if row is not None else None

    def append_transcript(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Append private execution data; the Master never reads this transcript."""

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

    def complete_run(
        self,
        run_id: str,
        result: ResultEnvelope,
        *,
        checkpoint: dict[str, Any] | None = None,
        waiting_request: WaitingRequest | None = None,
    ) -> WorkerRunRecord:
        return self.try_finalize_run(
            run_id,
            result,
            checkpoint=checkpoint,
            waiting_request=waiting_request,
        )[0]

    def try_finalize_run(
        self,
        run_id: str,
        result: ResultEnvelope,
        *,
        checkpoint: dict[str, Any] | None = None,
        waiting_request: WaitingRequest | None = None,
    ) -> tuple[WorkerRunRecord, bool]:
        """Commit status, result, waiting request, and checkpoint in one transaction."""

        if result.run_id != run_id:
            raise ValueError("result run_id does not match the finalized WorkerRun")
        allowed = {
            WorkerRunStatus.COMPLETED,
            WorkerRunStatus.PARTIAL,
            WorkerRunStatus.WAITING_FOR_CONTEXT,
            WorkerRunStatus.FAILED,
        }
        if result.status not in allowed:
            raise ValueError("Worker finalization requires a settled execution status")
        if (result.status is WorkerRunStatus.WAITING_FOR_CONTEXT) != (waiting_request is not None):
            raise ValueError("waiting_for_context requires exactly one structured request")
        if (
            result.status
            in {
                WorkerRunStatus.COMPLETED,
                WorkerRunStatus.PARTIAL,
                WorkerRunStatus.WAITING_FOR_CONTEXT,
            }
            and checkpoint is None
        ):
            raise ValueError(f"{result.status.value} requires an atomic checkpoint")

        current = self.get_run(run_id)
        flow_id = flow_id_from_turn_id(current.base_turn_id)
        if flow_id is None or current.status.settled:
            return self._try_finalize_run(
                run_id,
                result,
                checkpoint=checkpoint,
                waiting_request=waiting_request,
            )
        with flow_activation_fence(self.data_dir, flow_id):
            current = self.get_run(run_id)
            authorized = self._flow_run_is_current(flow_id, current)
            if authorized is None:
                raise WorkerRunFencedError(
                    "Flow execution authority is temporarily unavailable"
                )
            if not authorized:
                # Publish cancellation before attempting terminal settlement.
                # _try_finalize_run will reject the model result and, once tool
                # teardown is complete, settle this exact Run as cancelled.
                self.try_cancel_run(run_id)
            return self._try_finalize_run(
                run_id,
                result,
                checkpoint=checkpoint,
                waiting_request=waiting_request,
            )

    def _try_finalize_run(
        self,
        run_id: str,
        result: ResultEnvelope,
        *,
        checkpoint: dict[str, Any] | None = None,
        waiting_request: WaitingRequest | None = None,
    ) -> tuple[WorkerRunRecord, bool]:
        """Finalize after any Flow execution fence has been satisfied."""

        owner_token = self.execution_owner_token
        with self._transaction() as connection:
            current = self._run_from_row(self._required_run_row(connection, run_id))
            if current.worker_id != result.worker_id:
                raise ValueError("result worker_id does not match the finalized WorkerRun")
            if current.status.settled:
                return current, False
            if current.status is not WorkerRunStatus.RUNNING:
                raise ValueError("only a running WorkerRun can be finalized")
            if current.execution_owner_token != owner_token:
                raise WorkerRunFencedError(
                    "WorkerRun execution authority belongs to another owner"
                )
            if current.active_tool_count:
                raise WorkerRunFencedError(
                    "cannot finalize a WorkerRun while tool execution remains in flight"
                )
            if current.cancel_requested_at is not None:
                now = _now()
                connection.execute(
                    "UPDATE worker_runs SET status = ?, updated_at = ? WHERE run_id = ? "
                    "AND status = ? AND execution_owner_token = ?",
                    (
                        WorkerRunStatus.CANCELLED,
                        now,
                        run_id,
                        WorkerRunStatus.RUNNING,
                        owner_token,
                    ),
                )
                self._refresh_session_status(connection, current.worker_id)
                return self._run_from_row(self._required_run_row(connection, run_id)), True
            cursor = connection.execute(
                "UPDATE worker_runs SET status = ?, result_json = ?, "
                "waiting_request_json = ?, updated_at = ? WHERE run_id = ? AND status = ? "
                "AND execution_owner_token = ?",
                (
                    result.status,
                    _dump(result),
                    _dump(waiting_request),
                    _now(),
                    run_id,
                    WorkerRunStatus.RUNNING,
                    owner_token,
                ),
            )
            if cursor.rowcount != 1:
                return self._run_from_row(self._required_run_row(connection, run_id)), False
            if checkpoint is not None:
                self._upsert_checkpoint(connection, run_id, checkpoint)
            self._refresh_session_status(connection, current.worker_id)
            return self._run_from_row(self._required_run_row(connection, run_id)), True

    @staticmethod
    def _upsert_checkpoint(
        connection: sqlite3.Connection,
        run_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO worker_checkpoints(run_id, checkpoint_json, created_at) "
            "VALUES (?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
            "checkpoint_json = excluded.checkpoint_json, created_at = excluded.created_at",
            (run_id, json.dumps(checkpoint, ensure_ascii=False, sort_keys=True), _now()),
        )

    @staticmethod
    def _refresh_session_status(connection: sqlite3.Connection, worker_id: str) -> None:
        running = connection.execute(
            "SELECT 1 FROM worker_runs WHERE worker_id = ? AND status = ? LIMIT 1",
            (worker_id, WorkerRunStatus.RUNNING),
        ).fetchone()
        connection.execute(
            "UPDATE worker_sessions SET status = ? WHERE worker_id = ? AND status != ?",
            (
                WorkerSessionStatus.RUNNING if running is not None else WorkerSessionStatus.IDLE,
                worker_id,
                WorkerSessionStatus.ARCHIVED,
            ),
        )

    @staticmethod
    def _validate_idempotent_replay(
        row: sqlite3.Row,
        *,
        operation: WorkerOperation,
        base_turn_id: str | None,
        context: ContextEnvelope,
        source_run_id: str | None,
        stored_snapshot: WorkerSnapshot | None = None,
        requested_snapshot: WorkerSnapshot | None = None,
    ) -> None:
        conflicts: list[str] = []
        if row["operation_type"] != operation:
            conflicts.append("operation type")
        if row["base_turn_id"] != base_turn_id:
            conflicts.append("base turn")
        if operation is WorkerOperation.RESUME and row["source_run_id"] != source_run_id:
            conflicts.append("source run")
        stored_context = ContextEnvelope.model_validate_json(row["context_json"])
        if _normalized_context(stored_context) != _normalized_context(context):
            conflicts.append("context envelope")
        if operation is WorkerOperation.SPAWN and stored_snapshot != requested_snapshot:
            conflicts.append("worker snapshot")
        if conflicts:
            raise IdempotencyConflictError(
                f"idempotency conflict for key {row['idempotency_key']!r}: "
                f"request differs in {', '.join(conflicts)}"
            )

    @staticmethod
    def _required_run_row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM worker_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown worker run: {run_id}")
        return row

    def _session_row(self, connection: sqlite3.Connection, worker_id: str) -> WorkerSessionRecord:
        row = connection.execute(
            "SELECT * FROM worker_sessions WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown worker: {worker_id}")
        return self._session_from_row(row)

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> WorkerSessionRecord:
        return WorkerSessionRecord(
            worker_id=str(row["worker_id"]),
            base_session_id=str(row["base_session_id"]),
            snapshot=WorkerSnapshot.model_validate_json(row["snapshot_json"]),
            status=WorkerSessionStatus(row["status"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> WorkerRunRecord:
        return WorkerRunRecord(
            run_id=str(row["run_id"]),
            worker_id=str(row["worker_id"]),
            base_turn_id=row["base_turn_id"],
            status=WorkerRunStatus(row["status"]),
            context=ContextEnvelope.model_validate_json(row["context_json"]),
            idempotency_key=str(row["idempotency_key"]),
            created_at=str(row["created_at"]),
            result=(
                ResultEnvelope.model_validate_json(row["result_json"])
                if row["result_json"]
                else None
            ),
            run_sequence=int(row["run_sequence"]),
            source_run_id=row["source_run_id"],
            waiting_request=(
                WaitingRequest.model_validate_json(row["waiting_request_json"])
                if row["waiting_request_json"]
                else None
            ),
            cancel_requested_at=row["cancel_requested_at"],
            activated_at=row["activated_at"],
            active_tool_count=int(row["active_tool_count"]),
            execution_owner_token=row["execution_owner_token"],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN EXCLUSIVE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _register_execution_owner(store: WorkerStore) -> None:
    global _OWNER_AT_FORK_REGISTERED

    _OWNER_STORES.add(store)
    if _OWNER_AT_FORK_REGISTERED or not hasattr(os, "register_at_fork"):
        return
    os.register_at_fork(after_in_child=_discard_inherited_execution_owners)
    _OWNER_AT_FORK_REGISTERED = True


def _discard_inherited_execution_owners() -> None:
    for store in tuple(_OWNER_STORES):
        store._discard_inherited_execution_owner()


def _dump(value: BaseModel | None) -> str | None:
    return value.model_dump_json() if value is not None else None


def _require_current_checkpoint(row: sqlite3.Row, run_id: str) -> dict[str, Any]:
    payload = json.loads(str(row["checkpoint_json"]))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != MESSAGE_SCHEMA_VERSION
        or payload.get("message_format") != MESSAGE_FORMAT
    ):
        raise LegacySessionError(
            f"Worker checkpoint {run_id!r} uses the legacy message format; "
            "create a new WorkerSession. Existing data was not modified."
        )
    return payload


def _normalized_context(context: ContextEnvelope) -> str:
    return json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _permissions_expand(
    previous: PermissionSnapshot,
    requested: PermissionSnapshot,
) -> bool:
    return not set(requested.tool_names).issubset(previous.tool_names) or (
        requested.skills_enabled and not previous.skills_enabled
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _valid_owner_token(token: str) -> bool:
    return len(token) == 32 and all(character in "0123456789abcdef" for character in token)


__all__ = [
    "BudgetIncrease",
    "BudgetGrant",
    "ContextEnvelope",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceStatus",
    "IdempotencyConflictError",
    "PermissionSnapshot",
    "ReportItem",
    "ReportText",
    "RelatedContextSection",
    "RelatedWorkerContext",
    "ResultEnvelope",
    "SCHEMA_VERSION",
    "WaitingRequest",
    "WorkerOperation",
    "WorkerReport",
    "WorkerRunRecord",
    "WorkerRunFencedError",
    "WorkerRunStatus",
    "WorkerSessionRecord",
    "WorkerSessionStatus",
    "WorkerStore",
]
