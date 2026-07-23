"""Durable first-class dynamic flows owned by a Master session."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aeloon_core import session_events
from aeloon_core.flow_state import (
    DEFAULT_FLOW_TURN_LEASE_SECONDS,
    FLOW_SCHEMA_VERSION,
    FLOW_STATE_VERSION,
    MAX_FLOW_GOAL_CHARS,
    MAX_FLOW_OBJECTIVE_CHARS,
    MAX_FLOW_RUN_BINDINGS,
    MAX_FLOW_SUMMARY_CHARS,
    DependencyPolicy,
    FlowAdvanceMode,
    FlowCompletion,
    FlowContextRef,
    FlowId,
    FlowIdempotencyConflictError,
    FlowNode,
    FlowNodeId,
    FlowNodeSpec,
    FlowNodeStatus,
    FlowRunBinding,
    FlowSessionSealedError,
    FlowStatus,
    FlowTurnCommit,
    FlowTurnConflictError,
    MasterFlow,
    WorkerSessionAction,
    WorkerSessionPolicy,
    _validate_graph,
    add_flow_nodes,
    cancel_flow_state,
    finish_flow,
    flow_node_spec_payload,
    flow_run_telemetry_payload,
    pause_flow,
    reopen_flow,
    retry_flow_node,
    revise_flow_node,
    skip_flow_node,
)
from aeloon_core.session_events import SessionEvent, SessionHead

FlowMutation = Callable[[MasterFlow], None]


class FlowStore:
    """SQLite authority for dynamic Flow state and idempotent decisions."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "flow-control.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_flow(
        self,
        *,
        base_session_id: str,
        goal: str,
        nodes: Sequence[FlowNodeSpec],
        idempotency_key: str,
        max_nodes: int = 64,
        max_rounds: int = 12,
        advance_mode: FlowAdvanceMode = FlowAdvanceMode.CHECKPOINTED,
        auto_advance_max_frontiers: int = 4,
        turn_id: str | None = None,
    ) -> tuple[MasterFlow, bool]:
        base_session_id = _normalized_text(base_session_id, "base_session_id")
        idempotency_key = _normalized_text(idempotency_key, "idempotency_key")
        goal = _normalized_text(goal, "goal")
        if not nodes:
            raise ValueError("a Flow requires at least one initial node")
        if len(nodes) > max_nodes:
            raise ValueError("initial nodes exceed max_nodes")
        _validate_graph((), nodes)
        now = _now()
        request = {
            "goal": goal,
            "nodes": [flow_node_spec_payload(node) for node in nodes],
            "max_nodes": max_nodes,
            "max_rounds": max_rounds,
        }
        if advance_mode is not FlowAdvanceMode.CHECKPOINTED:
            request["advance_mode"] = advance_mode.value
        if auto_advance_max_frontiers != 4:
            request["auto_advance_max_frontiers"] = auto_advance_max_frontiers
        request_digest = _digest(request)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM flows WHERE base_session_id = ? AND idempotency_key = ?",
                (base_session_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if str(existing["request_digest"]) != request_digest:
                    raise FlowIdempotencyConflictError(
                        "Flow idempotency key was reused for a different create request"
                    )
                return _flow_from_row(existing), False
            self._require_unsealed_session(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            flow = MasterFlow(
                flow_id=uuid.uuid4().hex,
                base_session_id=base_session_id,
                idempotency_key=idempotency_key,
                goal=goal,
                max_nodes=max_nodes,
                max_rounds=max_rounds,
                advance_mode=advance_mode,
                auto_advance_max_frontiers=auto_advance_max_frontiers,
                nodes=[FlowNode.from_spec(spec) for spec in nodes],
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO flows(flow_id, base_session_id, idempotency_key, "
                "request_digest, status, state_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    flow.flow_id,
                    flow.base_session_id,
                    flow.idempotency_key,
                    request_digest,
                    flow.status.value,
                    _dump_flow(flow),
                    now,
                    now,
                ),
            )
            return flow, True

    def get_flow(
        self,
        flow_id: str,
        *,
        base_session_id: str | None = None,
    ) -> MasterFlow:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM flows WHERE flow_id = ?", (flow_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Flow: {flow_id}")
        flow = _flow_from_row(row)
        _validate_owner(flow, base_session_id)
        return flow

    def list_flows(
        self,
        base_session_id: str,
        *,
        include_terminal: bool = True,
    ) -> list[MasterFlow]:
        query = "SELECT * FROM flows WHERE base_session_id = ?"
        parameters: list[Any] = [base_session_id]
        if not include_terminal:
            query += " AND status IN (?, ?, ?)"
            parameters.extend(
                (
                    FlowStatus.OPEN.value,
                    FlowStatus.PAUSED.value,
                    FlowStatus.CANCELLING.value,
                )
            )
        query += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_flow_from_row(row) for row in rows]

    def mutate(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        operation: str,
        idempotency_key: str,
        payload: Any,
        mutation: FlowMutation,
        turn_id: str | None = None,
    ) -> tuple[MasterFlow, bool]:
        """Apply one model-authored state decision exactly once."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        operation = _normalized_text(operation, "operation")
        idempotency_key = _normalized_text(idempotency_key, "idempotency_key")
        payload_digest = _digest(payload)
        with self._transaction() as connection:
            row = self._required_flow_row(connection, flow_id)
            flow = _flow_from_row(row)
            _validate_owner(flow, base_session_id)
            existing = connection.execute(
                "SELECT * FROM flow_operations WHERE flow_id = ? AND idempotency_key = ?",
                (flow_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["operation"]) != operation
                    or str(existing["payload_digest"]) != payload_digest
                ):
                    raise FlowIdempotencyConflictError(
                        "Flow idempotency key was reused for a different operation"
                    )
                return flow, False
            self._require_unsealed_session(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            mutation(flow)
            flow.updated_at = _now()
            connection.execute(
                "INSERT INTO flow_operations(flow_id, idempotency_key, operation, "
                "payload_digest, created_at) VALUES (?, ?, ?, ?, ?)",
                (flow_id, idempotency_key, operation, payload_digest, flow.updated_at),
            )
            self._save(connection, flow)
            return flow, True

    def replay_operation(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        operation: str,
        idempotency_key: str,
        payload: Any,
        turn_id: str | None = None,
    ) -> MasterFlow | None:
        """Return current state for an exact completed operation replay."""

        flow, replayed = self.operation_state(
            flow_id,
            base_session_id=base_session_id,
            operation=operation,
            idempotency_key=idempotency_key,
            payload=payload,
            turn_id=turn_id,
        )
        return flow if replayed else None

    def operation_state(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        operation: str,
        idempotency_key: str,
        payload: Any,
        turn_id: str | None = None,
    ) -> tuple[MasterFlow, bool]:
        """Atomically read current Flow state and one operation replay decision."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        operation = _normalized_text(operation, "operation")
        idempotency_key = _normalized_text(idempotency_key, "idempotency_key")
        payload_digest = _digest(payload)
        # BEGIN IMMEDIATE prevents a writer from committing the operation
        # between reading Flow state and checking its idempotency record.
        with self._transaction() as connection:
            row = self._required_flow_row(connection, flow_id)
            flow = _flow_from_row(row)
            _validate_owner(flow, base_session_id)
            existing = connection.execute(
                "SELECT * FROM flow_operations WHERE flow_id = ? AND idempotency_key = ?",
                (flow_id, idempotency_key),
            ).fetchone()
            if existing is None:
                self._require_unsealed_session(
                    connection,
                    base_session_id,
                    turn_id=turn_id,
                )
                return flow, False
            if (
                str(existing["operation"]) != operation
                or str(existing["payload_digest"]) != payload_digest
            ):
                raise FlowIdempotencyConflictError(
                    "Flow idempotency key was reused for a different operation"
                )
            return flow, True

    def update_runtime(
        self,
        flow_id: str,
        *,
        base_session_id: str,
        mutation: FlowMutation,
        turn_id: str | None = None,
    ) -> MasterFlow:
        """Persist a deterministic runtime projection without a model decision key."""

        with self._transaction() as connection:
            row = self._required_flow_row(connection, flow_id)
            flow = _flow_from_row(row)
            _validate_owner(flow, base_session_id)
            self._require_unsealed_session(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            mutation(flow)
            flow.updated_at = _now()
            self._save(connection, flow)
            return flow

    def begin_turn(
        self,
        base_session_id: str,
        turn_id: str,
        *,
        lease_seconds: float = DEFAULT_FLOW_TURN_LEASE_SECONDS,
    ) -> None:
        """Acquire one durable Master-turn lease without admitting duplicates."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM flow_session_state WHERE base_session_id = ?",
                (base_session_id,),
            ).fetchone()
            if row is not None and row["active_turn_id"] is not None:
                active_turn_id = str(row["active_turn_id"])
                expired = _lease_expired(row["lease_expires_at"], now_text)
                if not expired:
                    raise FlowTurnConflictError(
                        "another Master turn is already active for this session"
                    )
                if active_turn_id == turn_id:
                    raise FlowTurnConflictError(
                        "the Master turn lease expired and cannot be revived"
                    )
            connection.execute(
                "INSERT INTO flow_session_state(base_session_id, sealed, "
                "active_turn_id, lease_expires_at, updated_at) VALUES (?, 0, ?, ?, ?) "
                "ON CONFLICT(base_session_id) DO UPDATE SET sealed = 0, "
                "active_turn_id = excluded.active_turn_id, "
                "lease_expires_at = excluded.lease_expires_at, "
                "updated_at = excluded.updated_at",
                (base_session_id, turn_id, lease_expires_at, now_text),
            )

    def refresh_turn_lease(
        self,
        base_session_id: str,
        turn_id: str,
        *,
        lease_seconds: float = DEFAULT_FLOW_TURN_LEASE_SECONDS,
    ) -> None:
        """Renew the current turn without allowing an expired owner to revive."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE flow_session_state SET lease_expires_at = ?, updated_at = ? "
                "WHERE base_session_id = ? AND active_turn_id = ? "
                "AND lease_expires_at > ?",
                (
                    lease_expires_at,
                    now_text,
                    base_session_id,
                    turn_id,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                raise FlowTurnConflictError("the Master turn lease was lost or expired")

    def end_turn(self, base_session_id: str, turn_id: str) -> None:
        """Release only the caller's turn ownership after response persistence."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        with self._transaction() as connection:
            connection.execute(
                "UPDATE flow_session_state SET active_turn_id = NULL, "
                "lease_expires_at = NULL, updated_at = ? "
                "WHERE base_session_id = ? AND active_turn_id = ?",
                (_now(), base_session_id, turn_id),
            )

    def commit_turn_response(
        self,
        base_session_id: str,
        turn_id: str,
        payload: dict[str, Any],
    ) -> tuple[FlowTurnCommit, bool]:
        """Linearize one terminal response while the caller still owns the lease.

        The response and the final quiescence check live in the same transaction.
        A turn that loses its lease before this transaction commits cannot persist
        or return a terminal response. Exact replays return the durable commit so a
        process crash after the commit does not require rerunning the model.
        """

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        if payload.get("session_id") != base_session_id:
            raise ValueError("turn response session_id does not match its owner")
        if payload.get("turn_id") != turn_id:
            raise ValueError("turn response turn_id does not match its owner")
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_digest = hashlib.sha256(payload_json.encode()).hexdigest()

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
                (base_session_id, turn_id),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_digest"]) != payload_digest:
                    raise FlowIdempotencyConflictError(
                        "Master turn commit was replayed with a different response"
                    )
                session_events.require_turn_committed_event(connection, existing)
                return self._turn_commit_from_row(existing), False

            session_events.require_turn_commit_coverage(connection, base_session_id)
            self._require_active_turn_owner(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            rows = connection.execute(
                "SELECT flow_id FROM flows WHERE base_session_id = ? "
                "AND status IN (?, ?) ORDER BY created_at",
                (
                    base_session_id,
                    FlowStatus.OPEN.value,
                    FlowStatus.CANCELLING.value,
                ),
            ).fetchall()
            open_flow_ids = [str(row["flow_id"]) for row in rows]
            if open_flow_ids:
                raise ValueError(
                    f"cannot commit the Master response while Flows remain open: {open_flow_ids}"
                )

            now = _now()
            connection.execute(
                "UPDATE flow_session_state SET sealed = 1, updated_at = ? "
                "WHERE base_session_id = ?",
                (now, base_session_id),
            )
            connection.execute(
                "INSERT INTO flow_turn_commits(base_session_id, turn_id, "
                "payload_json, payload_digest, created_at) VALUES (?, ?, ?, ?, ?)",
                (base_session_id, turn_id, payload_json, payload_digest, now),
            )
            row = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
                (base_session_id, turn_id),
            ).fetchone()
            assert row is not None
            session_events.append_turn_committed_event(connection, row)
            return self._turn_commit_from_row(row), True

    def get_turn_commit(
        self,
        base_session_id: str,
        turn_id: str,
    ) -> FlowTurnCommit | None:
        """Return one already-linearized response for exact retry recovery."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
                (base_session_id, turn_id),
            ).fetchone()
        return self._turn_commit_from_row(row) if row is not None else None

    def get_session_head(self, base_session_id: str) -> SessionHead | None:
        """Return the durable conversation head for one Master session."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._connect() as connection:
            return session_events.get_session_head(connection, base_session_id)

    def get_session_event(self, event_id: str) -> SessionEvent | None:
        """Return one immutable durable conversation event."""

        event_id = _normalized_text(event_id, "event_id")
        with self._connect() as connection:
            return session_events.get_session_event(connection, event_id)

    def session_event_ancestry(self, base_session_id: str) -> list[SessionEvent]:
        """Return the selected conversation ancestry in root-to-head order."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._connect() as connection:
            head = session_events.get_session_head(connection, base_session_id)
            if head is None:
                return []
            return session_events.list_event_ancestry(connection, head.head_event_id)

    def materialize_session_head_commit(
        self,
        base_session_id: str,
    ) -> FlowTurnCommit | None:
        """Resolve a session head to its nearest immutable full turn snapshot."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._connect() as connection:
            head = session_events.get_session_head(connection, base_session_id)
            if head is None:
                return None
            row = connection.execute(
                "SELECT commits.* FROM session_events AS event "
                "JOIN flow_turn_commits AS commits "
                "ON commits.commit_sequence = event.turn_commit_sequence "
                "WHERE event.event_id = ?",
                (head.head_event_id,),
            ).fetchone()
            if row is not None:
                return self._turn_commit_from_row(row)

            # All v1 session events are committed-turn snapshots, so the common
            # path above is a direct unique-key lookup.  Keep ancestry fallback
            # semantics ready for future non-snapshot event kinds.
            row = connection.execute(
                """WITH RECURSIVE ancestry(
                  event_id, parent_event_id, turn_commit_sequence, depth
                ) AS (
                  SELECT event_id, parent_event_id, turn_commit_sequence, 0
                  FROM session_events WHERE event_id = ?
                  UNION ALL
                  SELECT parent.event_id, parent.parent_event_id,
                         parent.turn_commit_sequence, child.depth + 1
                  FROM session_events AS parent
                  JOIN ancestry AS child ON parent.event_id = child.parent_event_id
                )
                SELECT commits.*
                FROM ancestry
                JOIN flow_turn_commits AS commits
                  ON commits.commit_sequence = ancestry.turn_commit_sequence
                ORDER BY ancestry.depth
                LIMIT 1""",
                (head.head_event_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("durable session head has no committed turn snapshot")
        return self._turn_commit_from_row(row)

    def _fork_conversation_only_session_head(
        self,
        source_session_id: str,
        fork_session_id: str,
    ) -> SessionHead:
        """Share one event head without inheriting Flow or Worker ownership.

        This storage-only primitive cannot inspect JSONL or WorkerStore state;
        callers must validate those external authorities first. The orchestrator
        facade performs that check. Flow and Worker ownership are not copied.
        """

        source_session_id = _normalized_text(source_session_id, "source_session_id")
        fork_session_id = _normalized_text(fork_session_id, "fork_session_id")
        if source_session_id == fork_session_id:
            raise ValueError("a conversation fork requires a distinct session id")
        with self._transaction() as connection:
            for table in ("flow_turn_commits", "flows", "flow_session_state"):
                occupied = connection.execute(
                    f"SELECT 1 FROM {table} WHERE base_session_id = ? LIMIT 1",
                    (fork_session_id,),
                ).fetchone()
                if occupied is not None:
                    raise ValueError("fork target is not a pristine Master session")
            return session_events.fork_conversation_only_head(
                connection,
                source_session_id=source_session_id,
                fork_session_id=fork_session_id,
                created_at=_now(),
            )

    def list_unpersisted_turn_commits(
        self,
        base_session_id: str,
    ) -> list[FlowTurnCommit]:
        """Return durable responses still awaiting the JSONL history projection."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM flow_turn_commits WHERE base_session_id = ? "
                "AND persisted_at IS NULL ORDER BY commit_sequence",
                (base_session_id,),
            ).fetchall()
        return [self._turn_commit_from_row(row) for row in rows]

    def persist_turn_commit(
        self,
        base_session_id: str,
        turn_id: str,
        projector: Callable[[FlowTurnCommit], None],
    ) -> FlowTurnCommit:
        """Project one commit idempotently, then record a short durable marker.

        The projector must itself be idempotent by ``turn_id``. If the process
        stops after appending the JSONL record but before this transaction commits,
        recovery invokes it again and the existing record is verified instead of
        duplicated.
        """

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        turn_id = _normalized_text(turn_id, "turn_id")
        commit = self.get_turn_commit(base_session_id, turn_id)
        if commit is None:
            raise KeyError(f"unknown Master turn commit: {turn_id}")
        if commit.persisted_at is not None:
            return commit

        # File projection is deliberately outside the SQLite writer transaction.
        # The projector serializes per session and is idempotent by turn_id, so a
        # crash or concurrent recovery can safely retry before this short marker CAS.
        projector(commit)
        with self._transaction() as connection:
            connection.execute(
                "UPDATE flow_turn_commits SET persisted_at = ? "
                "WHERE base_session_id = ? AND turn_id = ? "
                "AND persisted_at IS NULL",
                (_now(), base_session_id, turn_id),
            )
            row = self._required_turn_commit_row(
                connection,
                base_session_id,
                turn_id,
            )
            return self._turn_commit_from_row(row)

    def seal_session_if_quiescent(
        self,
        base_session_id: str,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Atomically prevent new Flow work once no live Flow remains."""

        base_session_id = _normalized_text(base_session_id, "base_session_id")
        with self._transaction() as connection:
            self._require_turn_owner(
                connection,
                base_session_id,
                turn_id=turn_id,
            )
            rows = connection.execute(
                "SELECT flow_id FROM flows WHERE base_session_id = ? "
                "AND status IN (?, ?) ORDER BY created_at",
                (
                    base_session_id,
                    FlowStatus.OPEN.value,
                    FlowStatus.CANCELLING.value,
                ),
            ).fetchall()
            open_flow_ids = [str(row["flow_id"]) for row in rows]
            if open_flow_ids:
                raise ValueError(
                    f"cannot finish the Master turn while Flows remain open: {open_flow_ids}"
                )
            now = _now()
            connection.execute(
                "INSERT INTO flow_session_state(base_session_id, sealed, "
                "active_turn_id, lease_expires_at, updated_at) "
                "VALUES (?, 1, NULL, NULL, ?) ON CONFLICT(base_session_id) "
                "DO UPDATE SET sealed = 1, updated_at = excluded.updated_at",
                (base_session_id, now),
            )

    def _initialize(self) -> None:
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > FLOW_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Flow store schema v{version} is newer than supported v{FLOW_SCHEMA_VERSION}"
                )
            if version == 0:
                connection.execute(
                    """CREATE TABLE flows (
                      flow_id TEXT PRIMARY KEY,
                      base_session_id TEXT NOT NULL,
                      idempotency_key TEXT NOT NULL,
                      request_digest TEXT NOT NULL,
                      status TEXT NOT NULL,
                      state_json TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL,
                      UNIQUE(base_session_id, idempotency_key)
                    )"""
                )
                connection.execute(
                    "CREATE INDEX flows_session_idx ON flows(base_session_id, created_at)"
                )
                connection.execute(
                    """CREATE TABLE flow_operations (
                      flow_id TEXT NOT NULL REFERENCES flows(flow_id) ON DELETE CASCADE,
                      idempotency_key TEXT NOT NULL,
                      operation TEXT NOT NULL,
                      payload_digest TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      PRIMARY KEY(flow_id, idempotency_key)
                    )"""
                )
                self._create_session_state_table(connection)
                self._create_turn_commits_table(connection)
            elif version == 1:
                self._create_session_state_table(connection)
                self._create_turn_commits_table(connection)
            elif version == 2:
                connection.execute("ALTER TABLE flow_session_state ADD COLUMN active_turn_id TEXT")
                connection.execute(
                    "ALTER TABLE flow_session_state ADD COLUMN lease_expires_at TEXT"
                )
                self._create_turn_commits_table(connection)
            elif version == 3:
                self._create_turn_commits_table(connection)
            if version < FLOW_SCHEMA_VERSION:
                session_events.create_session_event_schema(connection)
                session_events.backfill_turn_commit_events(connection)
                connection.execute(f"PRAGMA user_version={FLOW_SCHEMA_VERSION}")
            self._validate_schema(connection)

    @staticmethod
    def _create_session_state_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS flow_session_state (
              base_session_id TEXT PRIMARY KEY,
              sealed INTEGER NOT NULL CHECK(sealed IN (0, 1)),
              active_turn_id TEXT,
              lease_expires_at TEXT,
              updated_at TEXT NOT NULL
            )"""
        )

    @staticmethod
    def _create_turn_commits_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS flow_turn_commits (
              commit_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              base_session_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              payload_digest TEXT NOT NULL,
              created_at TEXT NOT NULL,
              persisted_at TEXT,
              UNIQUE(base_session_id, turn_id)
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS flow_turn_commits_pending_idx "
            "ON flow_turn_commits(base_session_id, persisted_at, commit_sequence)"
        )

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        required = {
            "flows": {
                "flow_id",
                "base_session_id",
                "idempotency_key",
                "request_digest",
                "status",
                "state_json",
                "created_at",
                "updated_at",
            },
            "flow_operations": {
                "flow_id",
                "idempotency_key",
                "operation",
                "payload_digest",
                "created_at",
            },
            "flow_session_state": {
                "base_session_id",
                "sealed",
                "active_turn_id",
                "lease_expires_at",
                "updated_at",
            },
            "flow_turn_commits": {
                "commit_sequence",
                "base_session_id",
                "turn_id",
                "payload_json",
                "payload_digest",
                "created_at",
                "persisted_at",
            },
        }
        for table, columns in required.items():
            actual = {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            missing = columns - actual
            if missing:
                raise RuntimeError(
                    f"Flow store v{FLOW_SCHEMA_VERSION} is invalid: "
                    f"{table} missing {sorted(missing)}"
                )
        session_events.validate_session_event_schema(
            connection,
            store_version=FLOW_SCHEMA_VERSION,
        )

    @staticmethod
    def _required_flow_row(
        connection: sqlite3.Connection,
        flow_id: str,
    ) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM flows WHERE flow_id = ?", (flow_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Flow: {flow_id}")
        return row

    @staticmethod
    def _save(connection: sqlite3.Connection, flow: MasterFlow) -> None:
        connection.execute(
            "UPDATE flows SET status = ?, state_json = ?, updated_at = ? WHERE flow_id = ?",
            (flow.status.value, _dump_flow(flow), flow.updated_at, flow.flow_id),
        )

    @staticmethod
    def _required_turn_commit_row(
        connection: sqlite3.Connection,
        base_session_id: str,
        turn_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM flow_turn_commits WHERE base_session_id = ? AND turn_id = ?",
            (base_session_id, turn_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Master turn commit: {turn_id}")
        return row

    @staticmethod
    def _turn_commit_from_row(row: sqlite3.Row) -> FlowTurnCommit:
        return FlowTurnCommit(
            commit_sequence=int(row["commit_sequence"]),
            base_session_id=str(row["base_session_id"]),
            turn_id=str(row["turn_id"]),
            payload=json.loads(str(row["payload_json"])),
            payload_digest=str(row["payload_digest"]),
            created_at=str(row["created_at"]),
            persisted_at=(str(row["persisted_at"]) if row["persisted_at"] is not None else None),
        )

    @staticmethod
    def _require_unsealed_session(
        connection: sqlite3.Connection,
        base_session_id: str,
        *,
        turn_id: str | None,
    ) -> None:
        FlowStore._require_turn_owner(
            connection,
            base_session_id,
            turn_id=turn_id,
        )
        row = connection.execute(
            "SELECT sealed FROM flow_session_state WHERE base_session_id = ?",
            (base_session_id,),
        ).fetchone()
        if row is not None and bool(row["sealed"]):
            raise FlowSessionSealedError(
                "this Master turn is sealed; start a new turn before mutating Flows"
            )

    @staticmethod
    def _require_turn_owner(
        connection: sqlite3.Connection,
        base_session_id: str,
        *,
        turn_id: str | None,
    ) -> None:
        row = connection.execute(
            "SELECT active_turn_id, lease_expires_at FROM flow_session_state "
            "WHERE base_session_id = ?",
            (base_session_id,),
        ).fetchone()
        active_turn_id = (
            str(row["active_turn_id"])
            if row is not None and row["active_turn_id"] is not None
            else None
        )
        if active_turn_id is None:
            return
        if turn_id != active_turn_id:
            raise FlowTurnConflictError("this Flow mutation does not own the active Master turn")
        if _lease_expired(row["lease_expires_at"], _now()):
            raise FlowTurnConflictError("the active Master turn lease expired")

    @staticmethod
    def _require_active_turn_owner(
        connection: sqlite3.Connection,
        base_session_id: str,
        *,
        turn_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT active_turn_id, lease_expires_at FROM flow_session_state "
            "WHERE base_session_id = ?",
            (base_session_id,),
        ).fetchone()
        if row is None or row["active_turn_id"] is None:
            raise FlowTurnConflictError("the Master turn no longer owns this session")
        if str(row["active_turn_id"]) != turn_id:
            raise FlowTurnConflictError(
                "this terminal response does not own the active Master turn"
            )
        if _lease_expired(row["lease_expires_at"], _now()):
            raise FlowTurnConflictError("the active Master turn lease expired")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


def _validate_owner(flow: MasterFlow, base_session_id: str | None) -> None:
    if base_session_id is not None and flow.base_session_id != base_session_id:
        raise ValueError("cannot access a Flow owned by another Master session")


def _dump_flow(flow: MasterFlow) -> str:
    return flow.model_dump_json()


def _flow_from_row(row: sqlite3.Row) -> MasterFlow:
    payload = json.loads(str(row["state_json"]))
    version = int(payload.get("schema_version", 1))
    if version not in {1, 2, 3, 4, FLOW_STATE_VERSION}:
        raise RuntimeError(f"unsupported Flow state version: {version}")
    if version in {1, 2}:
        for node in payload.get("nodes", []):
            node.setdefault("worker_session_policy", WorkerSessionPolicy.AUTO.value)
            node.setdefault("reuse_source_run_id", None)
            node.setdefault("pending_fresh_reason", None)
            node.setdefault("pending_budget", None)
            for binding in node.get("runs", []):
                binding.setdefault("budget", None)
                binding.setdefault(
                    "session_action",
                    (
                        WorkerSessionAction.RESUME.value
                        if binding.get("source_run_id") is not None
                        else WorkerSessionAction.NEW.value
                    ),
                )
                binding.setdefault("session_reason", "legacy_binding")
                binding.setdefault(
                    "requested_session_policy",
                    WorkerSessionPolicy.AUTO.value,
                )
    if version in {1, 2, 3, 4}:
        payload.setdefault("advance_mode", FlowAdvanceMode.CHECKPOINTED.value)
        payload.setdefault("auto_advance_max_frontiers", 4)
        payload["auto_advance_max_frontiers"] = min(
            4, int(payload["auto_advance_max_frontiers"])
        )
        payload.setdefault("frontier_widths", [])
        payload.setdefault("auto_advanced_frontiers", 0)
        payload.setdefault("last_stop_reason", None)
        for node in payload.get("nodes", []):
            current_run_id = node.get("current_run_id")
            current_result = node.get("result")
            for binding in node.get("runs", []):
                telemetry: dict[str, Any] = {}
                if (
                    binding.get("run_id") == current_run_id
                    and isinstance(current_result, dict)
                ):
                    telemetry = flow_run_telemetry_payload(current_result)
                binding.setdefault("telemetry", telemetry)
        payload["schema_version"] = FLOW_STATE_VERSION
    return MasterFlow.model_validate(payload)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be nonempty")
    return normalized


def _lease_expired(value: Any, now: str) -> bool:
    return value is None or str(value) <= now


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DependencyPolicy",
    "DEFAULT_FLOW_TURN_LEASE_SECONDS",
    "FLOW_SCHEMA_VERSION",
    "FLOW_STATE_VERSION",
    "FlowAdvanceMode",
    "FlowCompletion",
    "FlowContextRef",
    "FlowId",
    "FlowIdempotencyConflictError",
    "FlowSessionSealedError",
    "FlowTurnCommit",
    "FlowTurnConflictError",
    "MAX_FLOW_GOAL_CHARS",
    "MAX_FLOW_OBJECTIVE_CHARS",
    "MAX_FLOW_RUN_BINDINGS",
    "MAX_FLOW_SUMMARY_CHARS",
    "FlowNode",
    "FlowNodeId",
    "FlowNodeSpec",
    "FlowNodeStatus",
    "FlowRunBinding",
    "FlowStatus",
    "FlowStore",
    "MasterFlow",
    "WorkerSessionAction",
    "WorkerSessionPolicy",
    "add_flow_nodes",
    "cancel_flow_state",
    "finish_flow",
    "flow_node_spec_payload",
    "pause_flow",
    "reopen_flow",
    "retry_flow_node",
    "revise_flow_node",
    "skip_flow_node",
]
