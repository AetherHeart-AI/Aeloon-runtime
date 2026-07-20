from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.flows import FLOW_SCHEMA_VERSION, FlowStore
from aeloon_core.session_events import TURN_COMMITTED_EVENT


def _turn_payload(session_id: str, turn_id: str, answer: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": session_id,
        "turn_id": turn_id,
        "user_prompt": f"prompt for {answer}",
        "final_content": answer,
        "tools_used": ["finish_turn"],
        "messages": [{"role": "assistant", "content": answer}],
        "blocks": [{"type": "text", "content": answer}],
        "usage": {"total_tokens": 1},
        "transitions": [],
        "status": "completed",
    }


def _commit_turn(
    store: FlowStore,
    session_id: str,
    turn_id: str,
    answer: str,
):
    store.begin_turn(session_id, turn_id)
    try:
        return store.commit_turn_response(
            session_id,
            turn_id,
            _turn_payload(session_id, turn_id, answer),
        )
    finally:
        store.end_turn(session_id, turn_id)


def test_committed_turn_events_are_parent_linked_snapshot_references(
    tmp_path: Path,
) -> None:
    store = FlowStore(tmp_path)
    first_commit, first_created = _commit_turn(store, "master", "turn-1", "first")
    second_commit, second_created = _commit_turn(store, "master", "turn-2", "second")

    assert first_created is True
    assert second_created is True
    head = store.get_session_head("master")
    assert head is not None
    ancestry = store.session_event_ancestry("master")
    assert [event.kind for event in ancestry] == [
        TURN_COMMITTED_EVENT,
        TURN_COMMITTED_EVENT,
    ]
    assert [event.turn_id for event in ancestry] == ["turn-1", "turn-2"]
    assert ancestry[0].parent_event_id is None
    assert ancestry[1].parent_event_id == ancestry[0].event_id
    assert head.head_event_id == ancestry[1].event_id
    assert store.get_session_event(head.head_event_id) == ancestry[1]
    assert ancestry[0].payload == {"commit_sequence": first_commit.commit_sequence}
    assert ancestry[1].payload == {"commit_sequence": second_commit.commit_sequence}
    assert "messages" not in ancestry[1].payload

    materialized = store.materialize_session_head_commit("master")
    assert materialized is not None
    assert materialized.commit_sequence == second_commit.commit_sequence
    assert materialized.payload == _turn_payload("master", "turn-2", "second")

    with sqlite3.connect(store.path) as connection:
        raw_payload = connection.execute(
            "SELECT payload_json FROM session_events WHERE event_id = ?",
            (head.head_event_id,),
        ).fetchone()[0]
    assert raw_payload == f'{{"commit_sequence":{second_commit.commit_sequence}}}'


def test_exact_turn_replay_reuses_the_same_event(tmp_path: Path) -> None:
    store = FlowStore(tmp_path)
    payload = _turn_payload("master", "turn-1", "answer")
    commit, created = _commit_turn(store, "master", "turn-1", "answer")
    first_event = store.session_event_ancestry("master")[0]
    _commit_turn(store, "master", "turn-2", "later answer")
    latest_head = store.get_session_head("master")
    assert latest_head is not None

    replayed, replay_created = store.commit_turn_response("master", "turn-1", payload)

    assert created is True
    assert replay_created is False
    assert replayed == commit
    assert store.get_session_event(first_event.event_id) == first_event
    assert store.get_session_head("master") == latest_head
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] == 2


def test_event_append_failure_rolls_back_terminal_commit_and_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FlowStore(tmp_path)
    store.begin_turn("master", "turn-1")

    def fail_event_append(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("event append failed")

    monkeypatch.setattr(
        "aeloon_core.flows.session_events.append_turn_committed_event",
        fail_event_append,
    )
    with pytest.raises(RuntimeError, match="event append failed"):
        store.commit_turn_response(
            "master",
            "turn-1",
            _turn_payload("master", "turn-1", "answer"),
        )

    assert store.get_turn_commit("master", "turn-1") is None
    assert store.get_session_head("master") is None
    with sqlite3.connect(store.path) as connection:
        state = connection.execute(
            "SELECT sealed, active_turn_id FROM flow_session_state "
            "WHERE base_session_id = 'master'"
        ).fetchone()
        assert state == (0, "turn-1")
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] == 0
    store.end_turn("master", "turn-1")


def test_uncovered_legacy_commit_fails_closed_without_regressing_head(
    tmp_path: Path,
) -> None:
    store = FlowStore(tmp_path)
    _commit_turn(store, "master", "turn-1", "first")
    original_head = store.get_session_head("master")
    assert original_head is not None
    legacy_payload = _turn_payload("master", "legacy-turn", "legacy")
    payload_json = json.dumps(
        legacy_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO flow_turn_commits(base_session_id, turn_id, payload_json, "
            "payload_digest, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                "master",
                "legacy-turn",
                payload_json,
                hashlib.sha256(payload_json.encode()).hexdigest(),
                "2026-01-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(RuntimeError, match="has no durable session event"):
        store.commit_turn_response("master", "legacy-turn", legacy_payload)
    assert store.get_session_head("master") == original_head
    with pytest.raises(RuntimeError, match="has no durable session event"):
        FlowStore(tmp_path)


def test_v4_migration_backfills_existing_turn_commits(tmp_path: Path) -> None:
    original = FlowStore(tmp_path)
    first, _ = _commit_turn(original, "master", "turn-1", "first")
    second, _ = _commit_turn(original, "master", "turn-2", "second")
    with sqlite3.connect(original.path) as connection:
        connection.execute("DROP TABLE session_heads")
        connection.execute("DROP TRIGGER session_events_reject_update")
        connection.execute("DROP TRIGGER session_events_reject_delete")
        connection.execute("DROP TABLE session_events")
        connection.execute("PRAGMA user_version=4")

    migrated = FlowStore(tmp_path)

    with sqlite3.connect(migrated.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == FLOW_SCHEMA_VERSION
    ancestry = migrated.session_event_ancestry("master")
    assert [event.turn_commit_sequence for event in ancestry] == [
        first.commit_sequence,
        second.commit_sequence,
    ]
    assert ancestry[1].parent_event_id == ancestry[0].event_id
    materialized = migrated.materialize_session_head_commit("master")
    assert materialized is not None
    assert materialized.payload["final_content"] == "second"


def test_conversation_only_fork_shares_head_without_copying_events(
    tmp_path: Path,
) -> None:
    store = FlowStore(tmp_path)
    _commit_turn(store, "source", "source-1", "first")
    source_commit, _ = _commit_turn(store, "source", "source-2", "second")
    source_head = store.get_session_head("source")
    assert source_head is not None

    fork_head = store._fork_conversation_only_session_head("source", "fork")

    assert fork_head.head_event_id == source_head.head_event_id
    assert fork_head.forked_from_session_id == "source"
    assert fork_head.forked_from_event_id == source_head.head_event_id
    assert fork_head.conversation_only_fork is True
    assert store.session_event_ancestry("fork") == store.session_event_ancestry("source")
    fork_snapshot = store.materialize_session_head_commit("fork")
    assert fork_snapshot is not None
    assert fork_snapshot.commit_sequence == source_commit.commit_sequence
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM flow_turn_commits WHERE base_session_id = 'fork'"
            ).fetchone()[0]
            == 0
        )

    fork_commit, _ = _commit_turn(store, "fork", "fork-1", "fork answer")

    assert store.get_session_head("source") == source_head
    fork_ancestry = store.session_event_ancestry("fork")
    assert [event.turn_id for event in fork_ancestry] == [
        "source-1",
        "source-2",
        "fork-1",
    ]
    assert fork_ancestry[-1].session_id == "fork"
    assert fork_ancestry[-1].parent_event_id == source_head.head_event_id
    assert fork_ancestry[-1].turn_commit_sequence == fork_commit.commit_sequence
    assert store.materialize_session_head_commit("fork") == fork_commit


def test_session_event_rows_reject_updates_and_deletes(tmp_path: Path) -> None:
    store = FlowStore(tmp_path)
    _commit_turn(store, "master", "turn-1", "answer")
    head = store.get_session_head("master")
    assert head is not None

    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE session_events SET kind = 'changed' WHERE event_id = ?",
                (head.head_event_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM session_events WHERE event_id = ?",
                (head.head_event_id,),
            )
