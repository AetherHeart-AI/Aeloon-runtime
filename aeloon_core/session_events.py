"""Immutable parent-linked events for durable Master conversation heads.

The first event kind deliberately stays at the already-proven Master turn commit
boundary.  Its payload contains only a reference to ``flow_turn_commits``; the
full committed snapshot remains stored exactly once in that authority table.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

SESSION_EVENT_SCHEMA_VERSION = 1
TURN_COMMITTED_EVENT = "turn.committed"

_IMMUTABLE_UPDATE_TRIGGER = "session_events_reject_update"
_IMMUTABLE_DELETE_TRIGGER = "session_events_reject_delete"


@dataclass(frozen=True)
class SessionEvent:
    """One immutable event in a conversation branch ancestry."""

    sequence: int
    event_id: str
    session_id: str
    parent_event_id: str | None
    kind: str
    schema_version: int
    turn_id: str | None
    payload: dict[str, Any]
    payload_digest: str
    turn_commit_sequence: int | None
    created_at: str


@dataclass(frozen=True)
class SessionHead:
    """Mutable pointer selecting the visible head for one Master session."""

    session_id: str
    head_event_id: str
    forked_from_session_id: str | None
    forked_from_event_id: str | None
    created_at: str
    updated_at: str

    @property
    def conversation_only_fork(self) -> bool:
        """Whether this branch shares conversation ancestry only."""

        return self.forked_from_event_id is not None


def create_session_event_schema(connection: sqlite3.Connection) -> None:
    """Create the additive durable event spine and immutable-row guards."""

    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS session_events (
          event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id TEXT NOT NULL UNIQUE,
          session_id TEXT NOT NULL,
          parent_event_id TEXT REFERENCES session_events(event_id) ON DELETE RESTRICT,
          kind TEXT NOT NULL,
          schema_version INTEGER NOT NULL,
          turn_id TEXT,
          payload_json TEXT NOT NULL,
          payload_digest TEXT NOT NULL,
          turn_commit_sequence INTEGER UNIQUE
            REFERENCES flow_turn_commits(commit_sequence) ON DELETE RESTRICT,
          created_at TEXT NOT NULL,
          UNIQUE(session_id, turn_id, kind),
          CHECK(
            kind != '{TURN_COMMITTED_EVENT}'
            OR (turn_id IS NOT NULL AND turn_commit_sequence IS NOT NULL)
          )
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS session_events_parent_idx "
        "ON session_events(parent_event_id, event_sequence)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS session_events_session_idx "
        "ON session_events(session_id, event_sequence)"
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS session_heads (
          session_id TEXT PRIMARY KEY,
          head_event_id TEXT NOT NULL
            REFERENCES session_events(event_id) ON DELETE RESTRICT,
          forked_from_session_id TEXT,
          forked_from_event_id TEXT
            REFERENCES session_events(event_id) ON DELETE RESTRICT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          CHECK(
            (forked_from_session_id IS NULL AND forked_from_event_id IS NULL)
            OR
            (forked_from_session_id IS NOT NULL AND forked_from_event_id IS NOT NULL)
          )
        )"""
    )
    connection.execute(
        f"""CREATE TRIGGER IF NOT EXISTS {_IMMUTABLE_UPDATE_TRIGGER}
        BEFORE UPDATE ON session_events
        BEGIN
          SELECT RAISE(ABORT, 'session event rows are immutable');
        END"""
    )
    connection.execute(
        f"""CREATE TRIGGER IF NOT EXISTS {_IMMUTABLE_DELETE_TRIGGER}
        BEFORE DELETE ON session_events
        BEGIN
          SELECT RAISE(ABORT, 'session event rows are immutable');
        END"""
    )


def backfill_turn_commit_events(connection: sqlite3.Connection) -> None:
    """Build a linear event ancestry for every pre-event Master commit."""

    rows = connection.execute(
        "SELECT * FROM flow_turn_commits ORDER BY commit_sequence"
    ).fetchall()
    for row in rows:
        append_turn_committed_event(connection, row)


def append_turn_committed_event(
    connection: sqlite3.Connection,
    commit_row: sqlite3.Row,
) -> SessionEvent:
    """Append one committed-turn event and atomically advance its session head."""

    commit_sequence = int(commit_row["commit_sequence"])
    existing = connection.execute(
        "SELECT * FROM session_events WHERE turn_commit_sequence = ?",
        (commit_sequence,),
    ).fetchone()
    if existing is not None:
        event = session_event_from_row(existing)
        _validate_turn_event(event, commit_row)
        return event

    session_id = str(commit_row["base_session_id"])
    turn_id = str(commit_row["turn_id"])
    payload = {"commit_sequence": commit_sequence}
    payload_json = _canonical_json(payload)
    payload_digest = hashlib.sha256(payload_json.encode()).hexdigest()
    head_row = connection.execute(
        "SELECT head_event_id FROM session_heads WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    parent_event_id = str(head_row["head_event_id"]) if head_row is not None else None
    event_id = _turn_event_id(commit_row, parent_event_id=parent_event_id)
    created_at = str(commit_row["created_at"])

    connection.execute(
        "INSERT INTO session_events("
        "event_id, session_id, parent_event_id, kind, schema_version, turn_id, "
        "payload_json, payload_digest, turn_commit_sequence, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            session_id,
            parent_event_id,
            TURN_COMMITTED_EVENT,
            SESSION_EVENT_SCHEMA_VERSION,
            turn_id,
            payload_json,
            payload_digest,
            commit_sequence,
            created_at,
        ),
    )
    connection.execute(
        "INSERT INTO session_heads("
        "session_id, head_event_id, forked_from_session_id, forked_from_event_id, "
        "created_at, updated_at"
        ") VALUES (?, ?, NULL, NULL, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET "
        "head_event_id = excluded.head_event_id, updated_at = excluded.updated_at",
        (session_id, event_id, created_at, created_at),
    )
    row = connection.execute(
        "SELECT * FROM session_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    assert row is not None
    return session_event_from_row(row)


def require_turn_committed_event(
    connection: sqlite3.Connection,
    commit_row: sqlite3.Row,
) -> SessionEvent:
    """Return the one event already paired with a replayed turn commit."""

    commit_sequence = int(commit_row["commit_sequence"])
    row = connection.execute(
        "SELECT * FROM session_events WHERE turn_commit_sequence = ?",
        (commit_sequence,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Master turn commit {commit_sequence} has no durable session event"
        )
    event = session_event_from_row(row)
    _validate_turn_event(event, commit_row)
    return event


def require_turn_commit_coverage(
    connection: sqlite3.Connection,
    session_id: str | None = None,
) -> None:
    """Fail closed when any authoritative commit lacks its paired event."""

    where = "WHERE commits.base_session_id = ? AND events.event_sequence IS NULL"
    parameters: tuple[str, ...] = (session_id,) if session_id is not None else ()
    if session_id is None:
        where = "WHERE events.event_sequence IS NULL"
    row = connection.execute(
        "SELECT commits.commit_sequence FROM flow_turn_commits AS commits "
        "LEFT JOIN session_events AS events "
        "ON events.turn_commit_sequence = commits.commit_sequence "
        f"{where} ORDER BY commits.commit_sequence LIMIT 1",
        parameters,
    ).fetchone()
    if row is not None:
        raise RuntimeError(
            f"Master turn commit {int(row['commit_sequence'])} has no durable session event"
        )


def get_session_event(
    connection: sqlite3.Connection,
    event_id: str,
) -> SessionEvent | None:
    row = connection.execute(
        "SELECT * FROM session_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    return session_event_from_row(row) if row is not None else None


def get_session_head(
    connection: sqlite3.Connection,
    session_id: str,
) -> SessionHead | None:
    row = connection.execute(
        "SELECT * FROM session_heads WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return session_head_from_row(row) if row is not None else None


def list_event_ancestry(
    connection: sqlite3.Connection,
    event_id: str,
) -> list[SessionEvent]:
    """Return one event ancestry in root-to-head order."""

    rows = connection.execute(
        """WITH RECURSIVE ancestry AS (
          SELECT * FROM session_events WHERE event_id = ?
          UNION ALL
          SELECT parent.*
          FROM session_events AS parent
          JOIN ancestry AS child ON parent.event_id = child.parent_event_id
        )
        SELECT * FROM ancestry ORDER BY event_sequence""",
        (event_id,),
    ).fetchall()
    return [session_event_from_row(row) for row in rows]


def fork_conversation_only_head(
    connection: sqlite3.Connection,
    *,
    source_session_id: str,
    fork_session_id: str,
    created_at: str,
) -> SessionHead:
    """Create an O(1) conversation-only branch by sharing the source head.

    This deliberately copies no Flow, WorkerSession, lease, or turn-commit
    ownership.  Callers must provide a pristine target Master session id.
    """

    source = get_session_head(connection, source_session_id)
    if source is None:
        raise KeyError(f"session has no durable event head: {source_session_id}")
    if get_session_head(connection, fork_session_id) is not None:
        raise ValueError("fork target already has a durable event head")
    connection.execute(
        "INSERT INTO session_heads("
        "session_id, head_event_id, forked_from_session_id, forked_from_event_id, "
        "created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        (
            fork_session_id,
            source.head_event_id,
            source_session_id,
            source.head_event_id,
            created_at,
            created_at,
        ),
    )
    forked = get_session_head(connection, fork_session_id)
    assert forked is not None
    return forked


def validate_session_event_schema(
    connection: sqlite3.Connection,
    *,
    store_version: int,
) -> None:
    """Fail startup when the durable event spine is incomplete or mutable."""

    required = {
        "session_events": {
            "event_sequence",
            "event_id",
            "session_id",
            "parent_event_id",
            "kind",
            "schema_version",
            "turn_id",
            "payload_json",
            "payload_digest",
            "turn_commit_sequence",
            "created_at",
        },
        "session_heads": {
            "session_id",
            "head_event_id",
            "forked_from_session_id",
            "forked_from_event_id",
            "created_at",
            "updated_at",
        },
    }
    table_info: dict[str, list[sqlite3.Row]] = {}
    for table, columns in required.items():
        info = connection.execute(f"PRAGMA table_info({table})").fetchall()
        table_info[table] = info
        actual = {str(row["name"]) for row in info}
        missing = columns - actual
        if missing:
            raise RuntimeError(
                f"Flow store v{store_version} is invalid: "
                f"{table} missing {sorted(missing)}"
            )

    required_primary_keys = {
        "session_events": ("event_sequence",),
        "session_heads": ("session_id",),
    }
    for table, expected in required_primary_keys.items():
        actual = tuple(
            str(row["name"])
            for row in sorted(table_info[table], key=lambda row: int(row["pk"]))
            if int(row["pk"]) > 0
        )
        if actual != expected:
            raise RuntimeError(
                f"Flow store v{store_version} is invalid: {table} primary key "
                f"is {actual!r}, expected {expected!r}"
            )

    required_unique_indexes = {
        "session_events": {
            ("event_id",),
            ("turn_commit_sequence",),
            ("session_id", "turn_id", "kind"),
        },
    }
    for table, expected in required_unique_indexes.items():
        actual = {
            tuple(
                str(column["name"])
                for column in connection.execute(
                    f"PRAGMA index_info({index['name']})"
                ).fetchall()
            )
            for index in connection.execute(f"PRAGMA index_list({table})").fetchall()
            if bool(index["unique"])
        }
        missing = expected - actual
        if missing:
            raise RuntimeError(
                f"Flow store v{store_version} is invalid: {table} missing unique "
                f"indexes {sorted(missing)}"
            )

    required_foreign_keys = {
        "session_events": {
            ("parent_event_id", "session_events", "event_id"),
            ("turn_commit_sequence", "flow_turn_commits", "commit_sequence"),
        },
        "session_heads": {
            ("head_event_id", "session_events", "event_id"),
            ("forked_from_event_id", "session_events", "event_id"),
        },
    }
    for table, expected in required_foreign_keys.items():
        actual = {
            (str(row["from"]), str(row["table"]), str(row["to"]))
            for row in connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        }
        missing = expected - actual
        if missing:
            raise RuntimeError(
                f"Flow store v{store_version} is invalid: {table} missing foreign "
                f"keys {sorted(missing)}"
            )

    trigger_names = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND tbl_name = 'session_events'"
        ).fetchall()
    }
    missing_triggers = {
        _IMMUTABLE_UPDATE_TRIGGER,
        _IMMUTABLE_DELETE_TRIGGER,
    } - trigger_names
    if missing_triggers:
        raise RuntimeError(
            f"Flow store v{store_version} is invalid: session_events missing immutable "
            f"triggers {sorted(missing_triggers)}"
        )

    require_turn_commit_coverage(connection)


def session_event_from_row(row: sqlite3.Row) -> SessionEvent:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise RuntimeError("session event payload must be a JSON object")
    return SessionEvent(
        sequence=int(row["event_sequence"]),
        event_id=str(row["event_id"]),
        session_id=str(row["session_id"]),
        parent_event_id=(
            str(row["parent_event_id"]) if row["parent_event_id"] is not None else None
        ),
        kind=str(row["kind"]),
        schema_version=int(row["schema_version"]),
        turn_id=str(row["turn_id"]) if row["turn_id"] is not None else None,
        payload=payload,
        payload_digest=str(row["payload_digest"]),
        turn_commit_sequence=(
            int(row["turn_commit_sequence"])
            if row["turn_commit_sequence"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
    )


def session_head_from_row(row: sqlite3.Row) -> SessionHead:
    return SessionHead(
        session_id=str(row["session_id"]),
        head_event_id=str(row["head_event_id"]),
        forked_from_session_id=(
            str(row["forked_from_session_id"])
            if row["forked_from_session_id"] is not None
            else None
        ),
        forked_from_event_id=(
            str(row["forked_from_event_id"])
            if row["forked_from_event_id"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _turn_event_id(
    commit_row: sqlite3.Row,
    *,
    parent_event_id: str | None,
) -> str:
    identity = {
        "kind": TURN_COMMITTED_EVENT,
        "session_id": str(commit_row["base_session_id"]),
        "turn_id": str(commit_row["turn_id"]),
        "commit_sequence": int(commit_row["commit_sequence"]),
        "commit_digest": str(commit_row["payload_digest"]),
        "parent_event_id": parent_event_id,
    }
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _validate_turn_event(event: SessionEvent, commit_row: sqlite3.Row) -> None:
    expected_sequence = int(commit_row["commit_sequence"])
    expected_payload = {"commit_sequence": expected_sequence}
    if (
        event.kind != TURN_COMMITTED_EVENT
        or event.schema_version != SESSION_EVENT_SCHEMA_VERSION
        or event.session_id != str(commit_row["base_session_id"])
        or event.turn_id != str(commit_row["turn_id"])
        or event.turn_commit_sequence != expected_sequence
        or event.payload != expected_payload
        or event.event_id
        != _turn_event_id(commit_row, parent_event_id=event.parent_event_id)
        or event.created_at != str(commit_row["created_at"])
    ):
        raise RuntimeError("durable turn event differs from its Master turn commit")
    expected_digest = hashlib.sha256(_canonical_json(expected_payload).encode()).hexdigest()
    if event.payload_digest != expected_digest:
        raise RuntimeError("durable turn event payload digest is invalid")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "SESSION_EVENT_SCHEMA_VERSION",
    "TURN_COMMITTED_EVENT",
    "SessionEvent",
    "SessionHead",
]
