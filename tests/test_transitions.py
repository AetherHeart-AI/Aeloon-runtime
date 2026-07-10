from __future__ import annotations

from aeloon_core.loop_guard import LoopGuardAction, LoopGuardDecision
from aeloon_core.state import AgentNode
from aeloon_core.transitions import NodeKind, TokenLedger, TransitionRecorder


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
        before_digest="before",
        after_digest="after",
        decision=LoopGuardDecision(
            LoopGuardAction.RETURN_TO_MODEL,
            reason="recover",
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
    assert payload["schema_version"] == 1
    assert payload["session_id"] == "session-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["node"] == "worker"
    assert payload["node_kind"] == "domain"
    assert payload["decision"] == {
        "action": "return_to_model",
        "reason": "recover",
        "final_content": None,
        "prompt_message": None,
        "progress_message": None,
        "budget_grant": 0,
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
