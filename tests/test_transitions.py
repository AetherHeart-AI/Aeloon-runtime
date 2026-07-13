from __future__ import annotations

from aeloon_core.loop_guard import GuardAction, GuardEvent, GuardResolution
from aeloon_core.state import AgentNode
from aeloon_core.transitions import (
    NodeKind,
    TokenLedger,
    TransitionRecord,
    TransitionRecorder,
)


def test_token_ledger_aggregates_usage_by_node_kind() -> None:
    ledger = TokenLedger()

    normalized = ledger.record(
        NodeKind.DOMAIN,
        {"prompt_tokens": 10, "completion_tokens": 4},
    )
    ledger.add(NodeKind.HARNESS, {"total_tokens": 3})

    assert normalized["total_tokens"] == 14
    assert ledger.total_tokens == 17
    assert ledger.totals == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 17,
    }
    assert ledger.for_kind(NodeKind.DOMAIN)["total_tokens"] == 14
    assert ledger.for_kind(NodeKind.HARNESS)["total_tokens"] == 3
    assert ledger.to_dict()["by_node_kind"] == {
        "domain": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
        "harness": {"total_tokens": 3},
    }
    assert ledger.to_dict()["by_component"] == {
        "domain": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
        "harness": {"total_tokens": 3},
    }
    assert ledger.is_conserved()


def test_transition_recorder_sequences_serializes_and_persists() -> None:
    persisted = []
    recorder = TransitionRecorder(
        session_id="session-1",
        turn_id="turn-1",
        persist=persisted.append,
    )

    first = recorder.record(
        iteration=1,
        node=AgentNode.WORKER,
        node_kind=NodeKind.DOMAIN,
        component="domain:implementer",
        profile={"profile_id": "coding-team", "artifact_id": "artifact-1"},
        before_digest="before",
        after_digest="after",
        decision=GuardResolution(
            event=GuardEvent.TOOL_ERROR,
            action=GuardAction.RETRY,
            source="guard",
            evidence={"event": "tool_error"},
        ),
        token_usage={"input_tokens": 8, "output_tokens": 2},
        wall_time_ms=12.5,
    )
    second = recorder.record(
        iteration=1,
        node=AgentNode.MASTER,
        node_kind=NodeKind.DOMAIN,
        before_digest="after",
        after_digest="done",
    )

    assert [record.sequence for record in recorder.records] == [1, 2]
    assert persisted == [first, second]
    payload = recorder.to_dicts()[0]
    assert payload["schema_version"] == 2
    assert payload["session_id"] == "session-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["node"] == "worker"
    assert payload["node_kind"] == "domain"
    assert payload["component"] == "domain:implementer"
    assert payload["profile"] == {
        "profile_id": "coding-team",
        "artifact_id": "artifact-1",
    }
    assert payload["decision"] == {
        "event": "tool_error",
        "action": "retry",
        "source": "guard",
        "usage": {},
        "evidence": {"event": "tool_error"},
    }
    assert payload["token_usage"] == {
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
    }
    assert payload["wall_time_ms"] == 12.5


def test_transition_recorder_fails_open_after_trace_io_error() -> None:
    attempts = 0

    def fail_persist(_record) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("disk full")

    recorder = TransitionRecorder(persist=fail_persist)

    first = recorder.record(
        iteration=1,
        node=AgentNode.WORKER,
        node_kind=NodeKind.DOMAIN,
        before_digest="before",
        after_digest="after",
    )
    second = recorder.record(
        iteration=2,
        node=AgentNode.WORKER,
        node_kind=NodeKind.DOMAIN,
        before_digest="after",
        after_digest="done",
    )

    assert recorder.records == [first, second]
    assert recorder.persistence_error == "disk full"
    assert attempts == 1


def test_additive_transition_fields_preserve_v1_positional_constructor_order() -> None:
    record = TransitionRecord(
        1,
        2,
        "worker",
        NodeKind.DOMAIN,
        "before",
        "after",
        "session-1",
        "turn-1",
        {"legacy": "decision"},
        {"total_tokens": 3},
        1.5,
        "2026-07-11T00:00:00+00:00",
        1,
    )

    assert record.decision == {"legacy": "decision"}
    assert record.token_usage == {"total_tokens": 3}
    assert record.component is None
    assert record.profile is None
