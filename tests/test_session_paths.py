from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeloon_core.conversation import SessionStore
from tests.message_helpers import checkpoint as message_checkpoint


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(data_dir=tmp_path / "data")


def _turn_record(
    session_id: str,
    turn_id: str,
    prompt: str,
    created_at: str,
) -> dict[str, object]:
    return {
        "type": "turn",
        "session_id": session_id,
        "turn_id": turn_id,
        "created_at": created_at,
        "user_prompt": prompt,
        "final_content": f"answer for {session_id}",
        "tools_used": [],
        "messages": [{"role": "assistant", "content": f"message for {session_id}"}],
        "blocks": [],
        "usage": {},
    }


def _append_once(store: SessionStore, session_id: str, turn_id: str) -> bool:
    return store.append_turn_once(
        session_id=session_id,
        user_prompt=f"prompt for {session_id}",
        final_content=f"answer for {session_id}",
        tools_used=[],
        messages=message_checkpoint(f"message for {session_id}")["messages"],
        turn_id=turn_id,
    )


def test_unsafe_session_ids_use_distinct_contained_paths(tmp_path: Path) -> None:
    store = _store(tmp_path)

    unsafe = "team/a"
    formerly_colliding = "teama"
    traversal = "../../outside"

    assert store.session_path(unsafe) != store.session_path(formerly_colliding)
    assert store.session_path(traversal).parent == store.sessions_dir
    assert "/" not in store.session_path(unsafe).name

    assert _append_once(store, unsafe, "same-turn") is True
    assert _append_once(store, formerly_colliding, "same-turn") is True

    assert [record["session_id"] for record in store.history(unsafe)] == [unsafe]
    assert store.history(unsafe)[0]["status"] == "completed"
    assert [record["session_id"] for record in store.history(formerly_colliding)] == [
        formerly_colliding
    ]


def test_existing_safe_names_stay_stable_and_empty_has_own_namespace(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.session_path("master-1_a").name == "master-1_a.jsonl"
    assert store.session_path("") != store.session_path("default")
    assert store.session_path("").name.startswith("~")

    assert _append_once(store, "", "empty-turn") is True
    assert _append_once(store, "default", "default-turn") is True

    assert [record["turn_id"] for record in store.history("")] == ["empty-turn"]
    assert [record["turn_id"] for record in store.history("default")] == ["default-turn"]
    assert {summary.session_id for summary in store.list_sessions()} == {"", "default"}


def test_case_variants_are_isolated_on_case_insensitive_filesystems(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.session_path("TeamA") != store.session_path("teama")

    assert _append_once(store, "TeamA", "upper-turn") is True
    assert _append_once(store, "teama", "lower-turn") is True

    assert [record["turn_id"] for record in store.history("TeamA")] == ["upper-turn"]
    assert [record["turn_id"] for record in store.history("teama")] == ["lower-turn"]


def test_completed_idempotency_accepts_records_written_before_status_field(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    session_id = "legacy-status"
    turn_id = "same-turn"
    checkpoint = message_checkpoint("message for legacy-status")
    messages = checkpoint["messages"]
    record = {
        **checkpoint,
        "type": "turn",
        "session_id": session_id,
        "turn_id": turn_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "user_prompt": f"prompt for {session_id}",
        "final_content": f"answer for {session_id}",
        "tools_used": [],
        "blocks": [],
        "usage": {},
        "duration_ms": None,
    }
    store.session_path(session_id).write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    assert store.append_turn_once(
        session_id=session_id,
        user_prompt=f"prompt for {session_id}",
        final_content=f"answer for {session_id}",
        tools_used=[],
        messages=messages,
        turn_id=turn_id,
    ) is False


def test_pydantic_ai_v2_history_is_migrated_without_rewriting_source(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    session_id = "previous-kernel"
    record = {
        "type": "turn",
        "schema_version": 2,
        "message_format": "pydantic-ai-v1",
        "session_id": session_id,
        "turn_id": "old-turn",
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": "inspect",
                        "timestamp": "2026-01-01T00:00:00Z",
                    }
                ],
            },
            {
                "kind": "response",
                "model_name": "deepseek-v4-flash",
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": "read_file",
                        "tool_call_id": "call-1",
                        "args": {"path": "README.md"},
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_name": "read_file",
                        "tool_call_id": "call-1",
                        "content": "contents",
                        "outcome": "success",
                    }
                ],
            },
        ],
    }
    path = store.session_path(session_id)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    migrated = store.load_messages(session_id)

    assert [message["role"] for message in migrated] == [
        "user",
        "assistant",
        "toolResult",
    ]
    assert migrated[1]["content"][0]["name"] == "read_file"
    assert migrated[2]["toolCallId"] == "call-1"
    assert json.loads(path.read_text(encoding="utf-8"))["message_format"] == (
        "pydantic-ai-v1"
    )


def test_legacy_colliding_file_is_filtered_and_migrated_by_record_owner(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    legacy = store.sessions_dir / "teama.jsonl"
    records = [
        _turn_record("team/a", "unsafe-old", "unsafe old", "2026-01-01T00:00:00+00:00"),
        _turn_record("teama", "safe-old", "safe old", "2026-01-02T00:00:00+00:00"),
    ]
    legacy.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert [record["turn_id"] for record in store.history("team/a")] == ["unsafe-old"]
    assert [record["turn_id"] for record in store.history("teama")] == ["safe-old"]

    assert _append_once(store, "team/a", "unsafe-new") is True

    canonical = store.session_path("team/a")
    assert canonical != legacy
    assert {record["session_id"] for record in store._read_jsonl(canonical)} == {"team/a"}
    assert [record["turn_id"] for record in store.history("team/a")] == [
        "unsafe-old",
        "unsafe-new",
    ]
    assert [record["turn_id"] for record in store.history("teama")] == ["safe-old"]
    assert {summary.session_id: summary.turns for summary in store.list_sessions()} == {
        "team/a": 2,
        "teama": 1,
    }


def test_session_id_validation_rejects_unrepresentable_or_oversized_ids(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="valid Unicode"):
        store.session_path("\ud800")
    with pytest.raises(ValueError, match="too long"):
        store.session_path("/" * 200)
