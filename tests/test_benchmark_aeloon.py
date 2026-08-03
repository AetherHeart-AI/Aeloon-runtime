from __future__ import annotations

import json
from pathlib import Path

import benchmarks.harness.base as harness_base
from benchmarks.harness.aeloon import AeloonHarness
from benchmarks.harness.base import HarnessRequest, ProcessOutcome


def test_aeloon_adapter_reads_new_result_without_expert_or_transition_fields() -> None:
    harness = object.__new__(AeloonHarness)
    payload = {
        "status": "completed",
        "session_id": "session",
        "duration_ms": 12,
        "final_content": "done",
        "tools_used": ["read", "edit"],
        "usage": {"input": 10, "output": 2, "totalTokens": 12},
        "model": "deepseek-v4-flash",
    }

    result = harness.interpret(
        ProcessOutcome(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
            duration_ms=12,
        )
    )

    assert result == {
        **payload,
        "payload_error": None,
    }
    assert "transitions" not in result
    assert "experts" not in result


def test_aeloon_adapter_run_accepts_new_model_field(tmp_path: Path, monkeypatch) -> None:
    harness = object.__new__(AeloonHarness)
    harness.executable = "/fake/python"
    harness.version = "aeloon@test"
    harness.model = "deepseek-v4-flash"
    harness.project_root = tmp_path
    outcome = ProcessOutcome(
        returncode=0,
        stdout=json.dumps(
            {
                "status": "completed",
                "final_content": "done",
                "model": "deepseek-v4-flash",
            }
        ),
        stderr="",
        duration_ms=1,
    )
    monkeypatch.setattr(harness_base, "run_process", lambda *_args, **_kwargs: outcome)

    result = harness.run(
        HarnessRequest(
            prompt="task",
            workspace=tmp_path,
            session_dir=tmp_path / "sessions",
            project_root=tmp_path,
        )
    )

    assert result.model == "deepseek-v4-flash"
    assert "turn_id" not in result.to_record()
    assert "transitions" not in result.to_record()
