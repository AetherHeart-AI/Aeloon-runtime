from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aeloon_core.__main__ import (
    _resolve_run_prompt,
    _run_prompt,
    build_parser,
)
from aeloon_core.orchestrator import TurnResult


def test_run_prompt_file_is_an_exclusive_prompt_source(tmp_path: Path) -> None:
    prompt_path = tmp_path / "task.txt"
    prompt_path.write_text("Refactor the parser.\n", encoding="utf-8")

    args = build_parser().parse_args(
        ["run", "--prompt-file", str(prompt_path), "--output", "json"]
    )

    assert _resolve_run_prompt(args) == "Refactor the parser.\n"
    assert args.output == "json"

    conflicting = build_parser().parse_args(
        ["run", "positional", "--prompt-file", str(prompt_path)]
    )
    with pytest.raises(SystemExit, match="exactly one prompt source"):
        _resolve_run_prompt(conflicting)


def test_run_reads_prompt_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    args = build_parser().parse_args(["run", "--stdin"])
    monkeypatch.setattr("sys.stdin", io.StringIO("Inspect this workspace."))

    assert _resolve_run_prompt(args) == "Inspect this workspace."


async def test_run_json_emits_one_machine_readable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_path = tmp_path / "task.txt"
    prompt_path.write_text("Make the change.", encoding="utf-8")

    class FakeSessions:
        def new_session(self) -> str:
            return "session-1"

    class FakeOrchestrator:
        def __init__(self, config: object) -> None:
            self.config = config
            self.sessions = FakeSessions()

        async def run_turn(
            self,
            prompt: str,
            *,
            session_id: str,
            on_progress: object,
        ) -> TurnResult:
            assert prompt == "Make the change."
            await on_progress.on_final("Finished.")
            return TurnResult(
                session_id=session_id,
                turn_id=on_progress.turn_id,
                status="completed",
                final_content="Finished.",
                tools_used=["workflow_execute"],
                messages=[],
                blocks=[],
                usage={"input_tokens": 12, "output_tokens": 3},
                duration_ms=on_progress.duration_ms,
                transitions=[{"kind": "terminal"}],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "aeloon_core.__main__.AeloonCoreOrchestrator",
        FakeOrchestrator,
    )
    args = build_parser().parse_args(
        [
            "run",
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "sessions"),
            "--config",
            str(tmp_path / "config.json"),
            "--prompt-file",
            str(prompt_path),
            "--output",
            "json",
        ]
    )

    await _run_prompt(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": 1,
        "session_id": "session-1",
        "turn_id": payload["turn_id"],
        "status": "completed",
        "final_content": "Finished.",
        "tools_used": ["workflow_execute"],
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "duration_ms": payload["duration_ms"],
        "transitions": [{"kind": "terminal"}],
        "workspace": str(tmp_path),
        "models": {
            "default": "anthropic/claude-sonnet-4-6",
            "master": "anthropic/claude-sonnet-4-6",
            "workers": {},
        },
    }
    assert payload["duration_ms"] is not None


async def test_run_rejects_missing_workspace_before_model_start(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        ["run", "Inspect", "--workspace", str(tmp_path / "missing")]
    )

    with pytest.raises(SystemExit, match="Workspace does not exist"):
        await _run_prompt(args)
