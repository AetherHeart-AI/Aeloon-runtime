from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeloon_core.__main__ import build_parser, main


def source(*, revision: int, role_prompt: str) -> str:
    return f"""---
schema_version: 1
id: cli-team
revision: {revision}
description: CLI test profile
default_agent: operator
max_handoffs: 2
agents:
  - id: operator
    description: Operate the test
    tools: []
---

## Shared
Stay in scope.

## Master
Always select operator.

## Agent: operator
{role_prompt}
"""


def parse_output(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def run_profile(
    args: list[str],
    *,
    data_dir: Path,
    workspace: Path,
) -> None:
    main(
        [
            "profile",
            *args,
            "--data-dir",
            str(data_dir),
            "--workspace",
            str(workspace),
        ]
    )


def test_profile_parser_exposes_complete_lifecycle() -> None:
    parser = build_parser()

    for command in (
        ["profile", "validate", "PROFILE.md"],
        ["profile", "compile", "PROFILE.md", "--compiler", "deterministic"],
        ["profile", "inspect", "a" * 64],
        ["profile", "approve", "a" * 64],
        ["profile", "activate", "a" * 64],
        ["profile", "status"],
        ["profile", "rollback", "a" * 64],
    ):
        assert parser.parse_args(command).command == "profile"


def test_profile_validate_reports_canonical_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "PROFILE.md"
    profile_path.write_text(source(revision=1, role_prompt="Complete safely."))

    main(["profile", "validate", str(profile_path)])

    result = parse_output(capsys)
    assert result["valid"] is True
    assert result["profile_id"] == "cli-team"
    assert result["agents"] == ["operator"]
    assert len(result["canonical_profile_hash"]) == 64


def test_profile_cli_compiles_approves_activates_and_rolls_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_path = workspace / "first.md"
    second_path = workspace / "second.md"
    first_path.write_text(source(revision=1, role_prompt="Complete version one."))
    second_path.write_text(source(revision=2, role_prompt="Complete version two."))

    run_profile(
        ["compile", str(first_path), "--compiler", "deterministic"],
        data_dir=data_dir,
        workspace=workspace,
    )
    first = parse_output(capsys)
    assert first["state"] == "validated"

    run_profile(
        ["approve", first["artifact_id"]],
        data_dir=data_dir,
        workspace=workspace,
    )
    assert parse_output(capsys)["state"] == "approved"
    run_profile(
        ["activate", first["artifact_id"]],
        data_dir=data_dir,
        workspace=workspace,
    )
    assert parse_output(capsys)["artifact_id"] == first["artifact_id"]

    run_profile(
        ["compile", str(second_path), "--compiler", "deterministic"],
        data_dir=data_dir,
        workspace=workspace,
    )
    second = parse_output(capsys)
    run_profile(
        ["approve", second["artifact_id"]],
        data_dir=data_dir,
        workspace=workspace,
    )
    parse_output(capsys)
    run_profile(
        ["activate", second["artifact_id"]],
        data_dir=data_dir,
        workspace=workspace,
    )
    assert parse_output(capsys)["artifact_id"] == second["artifact_id"]

    run_profile(
        ["rollback", first["artifact_id"]],
        data_dir=data_dir,
        workspace=workspace,
    )
    rolled_back = parse_output(capsys)
    assert rolled_back["rollback"] is True
    assert rolled_back["artifact_id"] == first["artifact_id"]
    assert rolled_back["rolled_back_from"] == second["artifact_id"]

    run_profile(
        ["status", "cli-team"],
        data_dir=data_dir,
        workspace=workspace,
    )
    status = parse_output(capsys)
    assert status["active"] is True
    assert status["artifact_id"] == first["artifact_id"]
    assert status["generation"] == 3


def test_profile_validate_fails_closed_on_duplicate_yaml_key(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "PROFILE.md"
    profile_path.write_text(source(revision=1, role_prompt="Complete safely.").replace(
        "revision: 1",
        "revision: 1\nrevision: 2",
    ))

    with pytest.raises(SystemExit, match="duplicate key"):
        main(["profile", "validate", str(profile_path)])
