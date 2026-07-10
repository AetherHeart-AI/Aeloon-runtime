from __future__ import annotations

from aeloon_core.session import SessionStore


def test_transition_trace_is_separate_from_turn_history(tmp_path) -> None:
    store = SessionStore(data_dir=tmp_path / "data", workspace=tmp_path)

    store.append_transition(
        session_id="session-1",
        turn_id="turn-1",
        transition={"sequence": 1, "node": "worker"},
    )
    store.append_turn(
        session_id="session-1",
        user_prompt="hello",
        final_content="done",
        tools_used=[],
        messages=[{"role": "assistant", "content": "done"}],
        usage={"total": {"total_tokens": 3}},
    )

    assert len(store.history("session-1")) == 1
    assert store.history("session-1")[0]["usage"]["total"]["total_tokens"] == 3
    assert store.transition_history("session-1") == [
        {
            "type": "transition",
            "schema_version": 1,
            "session_id": "session-1",
            "turn_id": "turn-1",
            "sequence": 1,
            "node": "worker",
        }
    ]

    summary = store.list_sessions()[0]
    assert summary.turns == 1
    assert summary.title == "hello"


def test_transition_trace_uses_sanitized_session_path(tmp_path) -> None:
    store = SessionStore(data_dir=tmp_path / "data", workspace=tmp_path)

    store.append_transition(
        session_id="../unsafe",
        turn_id="turn-1",
        transition={"sequence": 1},
    )

    assert store.trace_path("../unsafe").parent == tmp_path / "data" / "traces"
    assert store.transition_history("../unsafe")[0]["session_id"] == "../unsafe"
