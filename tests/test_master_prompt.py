from __future__ import annotations

from aeloon_core.master_prompt import (
    MASTER_RUNTIME_MARKER,
    MASTER_SYSTEM_MARKER,
    MASTER_USER_REQUEST_MARKER,
    apply_master_system_prompt,
    master_runtime_context,
    master_system_prompt,
)


def test_master_system_prefix_excludes_volatile_control_plane_state() -> None:
    worker_types = [{"id": "builder", "description": "Build the requested outcome."}]

    stable = master_system_prompt(worker_types=worker_types)
    first_runtime = master_runtime_context(
        workers=[{"worker_id": "worker-1", "status": "running"}],
        flows=[{"flow_id": "flow-1", "status": "open"}],
    )
    second_runtime = master_runtime_context(
        workers=[{"worker_id": "worker-1", "status": "completed"}],
        flows=[],
    )

    assert stable.startswith(MASTER_SYSTEM_MARKER)
    assert MASTER_RUNTIME_MARKER not in stable
    assert first_runtime != second_runtime
    assert first_runtime.startswith(MASTER_RUNTIME_MARKER)
    assert "worker-1" in first_runtime


def test_refreshing_master_prompt_preserves_one_stable_marked_block() -> None:
    original = apply_master_system_prompt(
        [{"role": "system", "content": "base"}],
        worker_types=[{"id": "builder"}],
    )
    refreshed = apply_master_system_prompt(
        [*original, {"role": "user", "content": "prior request"}],
        worker_types=[{"id": "builder"}],
    )

    marked = [
        message
        for message in refreshed
        if str(message.get("content") or "").startswith(MASTER_SYSTEM_MARKER)
    ]
    assert refreshed[0] == {"role": "system", "content": "base"}
    assert len(marked) == 1
    assert marked[0] == original[1]
    assert refreshed[-1] == {"role": "user", "content": "prior request"}


def test_volatile_runtime_and_authoritative_request_belong_at_user_tail() -> None:
    runtime = master_runtime_context(workers=[], flows=[])
    request = f"{runtime}{MASTER_USER_REQUEST_MARKER}implement the change"

    assert request.startswith(MASTER_RUNTIME_MARKER)
    assert request.endswith("implement the change")
    assert request.rsplit(MASTER_USER_REQUEST_MARKER, 1)[-1] == "implement the change"


def test_legacy_master_prompt_keywords_remain_compatible() -> None:
    prompt = master_system_prompt(
        worker_types=[{"id": "builder"}],
        workers=[{"worker_id": "worker-1"}],
        flows=[{"flow_id": "flow-1"}],
    )

    assert prompt.startswith(MASTER_SYSTEM_MARKER)
    assert MASTER_RUNTIME_MARKER in prompt
    assert "worker-1" in prompt
    assert "flow-1" in prompt


def test_master_prompt_contains_decision_handbook_and_cost_rubrics() -> None:
    prompt = master_system_prompt(
        worker_types=[
            {"id": "builder"},
            {"id": "researcher"},
            {"id": "reviewer"},
        ]
    )

    assert "DECISION HANDBOOK" in prompt
    assert "One node is one coherent deliverable" in prompt
    assert "Fan-out width" in prompt
    assert "Every frontier costs at least one additional model round trip" in prompt
    assert "Objective writing rubric" in prompt
    assert "User asks to fix one config parsing defect" in prompt
    assert "Allow at most two review-driven revisions" in prompt
    assert "advance_mode=auto" in prompt
    assert "run_workflow" in prompt
    assert "self-contained work that can finish in this turn" in prompt
    assert "Use a durable Flow when work must survive process or turn boundaries" in prompt
