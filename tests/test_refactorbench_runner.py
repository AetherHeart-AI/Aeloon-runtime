from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import benchmarks.run_refactorbench as runner
from benchmarks.run_refactorbench import (
    ProcessOutcome,
    WorkspaceCache,
    _build_harness_invocation,
    _capture_patch,
    _interpret_harness_output,
    _parse_agent_payload,
    _run_official_test,
    _selected_harnesses,
    build_parser,
    load_cases,
    run_benchmark,
)


def _make_refactorbench(root: Path) -> Path:
    repository = "demo_refactor"
    prompt = root / "problems" / "base_problems" / repository / "change-task.txt"
    test = root / "tests" / repository / "change-test.py"
    source = root / "repositories" / repository / "module.py"
    mapping = root / "scripts" / "base_mapping.py"
    for path in (prompt, test, source, mapping):
        path.parent.mkdir(parents=True, exist_ok=True)

    prompt.write_text("Change VALUE from zero to one.\n", encoding="utf-8")
    source.write_text("VALUE = 0\n", encoding="utf-8")
    test.write_text(
        "from pathlib import Path\n"
        "source = (Path.cwd() / '..' / 'module.py').resolve().read_text()\n"
        "raise SystemExit(0 if 'VALUE = 1' in source else 1)\n",
        encoding="utf-8",
    )
    mapping.write_text(
        "file_mapping = {\n"
        "    '../tests/demo_refactor/change-test.py': "
        "'../problems/base_problems/demo_refactor/change-task.txt',\n"
        "}\n",
        encoding="utf-8",
    )
    return root


def test_load_cases_uses_official_literal_mapping(tmp_path: Path) -> None:
    root = _make_refactorbench(tmp_path / "RefactorBench")

    cases = load_cases(root, "base")

    assert len(cases) == 1
    assert cases[0].instance_id == "demo_refactor/change"
    assert cases[0].repository == "demo_refactor"
    assert cases[0].prompt_path.read_text(encoding="utf-8").startswith("Change VALUE")


def test_workspace_cache_resets_cases_and_captures_new_files(tmp_path: Path) -> None:
    root = _make_refactorbench(tmp_path / "RefactorBench")
    case = load_cases(root, "base")[0]
    cache = WorkspaceCache(tmp_path / "cache", refactorbench_root=root)
    workspace, baseline = cache.prepare(case)

    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "new_module.py").write_text("NEW = True\n", encoding="utf-8")
    patch, changed_files, error = _capture_patch(workspace, baseline)

    assert error is None
    assert changed_files == ["module.py", "new_module.py"]
    assert "VALUE = 1" in patch
    assert "NEW = True" in patch

    same_workspace, same_baseline = cache.prepare(case)
    assert same_workspace == workspace
    assert same_baseline == baseline
    assert (workspace / "module.py").read_text(encoding="utf-8") == "VALUE = 0\n"
    assert not (workspace / "new_module.py").exists()


def test_official_test_runs_outside_agent_workspace_view(tmp_path: Path) -> None:
    root = _make_refactorbench(tmp_path / "RefactorBench")
    case = load_cases(root, "base")[0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    outcome = _run_official_test(
        test_path=case.test_path,
        workspace=workspace,
        timeout=5,
    )

    assert outcome.returncode == 0
    assert not outcome.timed_out
    assert list(workspace.glob(".aeloon-refactorbench-verify-*")) == []


def test_list_mode_has_no_workspace_or_result_side_effects(tmp_path: Path) -> None:
    root = _make_refactorbench(tmp_path / "RefactorBench")
    cache = tmp_path / "cache"
    results = tmp_path / "results.jsonl"
    args = build_parser().parse_args(
        [
            "--refactorbench-root",
            str(root),
            "--cache-dir",
            str(cache),
            "--results",
            str(results),
            "--list",
        ]
    )

    summary = run_benchmark(args)

    assert args.model == "deepseek-v4-flash"
    assert [case["instance_id"] for case in summary["cases"]] == [
        "demo_refactor/change"
    ]
    assert not cache.exists()
    assert not results.exists()


def test_agent_json_is_parsed_before_diagnostic_output_is_bounded() -> None:
    final_content = "x" * 30_000
    outcome = ProcessOutcome(
        returncode=0,
        stdout='{"status":"completed","final_content":"' + final_content + '"}',
        stderr="",
        duration_ms=1,
    )

    payload, error = _parse_agent_payload(outcome)

    assert error is None
    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["final_content"] == final_content


def test_harness_selection_and_invocations_use_noninteractive_modes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    data_dir = tmp_path / "sessions"

    assert _selected_harnesses(None) == ["aeloon"]
    assert _selected_harnesses(["codex", "pi", "codex"]) == ["codex", "pi"]
    assert _selected_harnesses(["all"]) == ["aeloon", "pi", "codex", "claude"]

    aeloon = _build_harness_invocation(
        "aeloon",
        executable=sys.executable,
        workspace=workspace,
        prompt="do the task",
        data_dir=data_dir,
        model="studio/coder",
        config_path=tmp_path / "config.json",
    )
    assert "--stdin" in aeloon.command
    assert aeloon.input_text == "do the task"
    assert aeloon.command[aeloon.command.index("--model") + 1] == "studio/coder"
    assert aeloon.command[aeloon.command.index("--config") + 1] == str(
        (tmp_path / "config.json").resolve()
    )

    pi = _build_harness_invocation(
        "pi",
        executable="/fake/pi",
        workspace=workspace,
        prompt="do the task",
        data_dir=data_dir,
        model="studio/coder",
        config_path=None,
    )
    assert pi.command[:4] == ["/fake/pi", "--print", "--mode", "json"]
    assert "--no-session" in pi.command

    codex = _build_harness_invocation(
        "codex",
        executable="/fake/codex",
        workspace=workspace,
        prompt="do the task",
        data_dir=data_dir,
        model="studio/coder",
        config_path=None,
    )
    assert codex.command[:2] == ["/fake/codex", "exec"]
    assert "--ephemeral" in codex.command
    assert codex.input_text == "do the task"

    claude = _build_harness_invocation(
        "claude",
        executable="/fake/claude",
        workspace=workspace,
        prompt="do the task",
        data_dir=data_dir,
        model="studio/coder",
        config_path=None,
    )
    assert "--output-format" in claude.command
    assert "--no-session-persistence" in claude.command


def test_external_harness_json_is_normalized() -> None:
    pi_output = "\n".join(
        [
            '{"type":"session","id":"pi-session"}',
            '{"type":"message_end","message":{"role":"assistant",'
            '"content":[{"type":"text","text":"done"}],'
            '"usage":{"input":12,"output":3,"cost":{"total":0.01}}}}',
        ]
    )
    pi = _interpret_harness_output(
        "pi",
        ProcessOutcome(0, pi_output, "", 10),
    )
    assert pi["status"] == "completed"
    assert pi["final_content"] == "done"
    assert pi["usage"]["input_tokens"] == 12
    assert pi["cost_usd"] == 0.01

    codex_output = "\n".join(
        [
            '{"type":"thread.started","thread_id":"codex-session"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}',
        ]
    )
    codex = _interpret_harness_output(
        "codex",
        ProcessOutcome(0, codex_output, "", 10),
    )
    assert codex["status"] == "completed"
    assert codex["session_id"] == "codex-session"

    claude = _interpret_harness_output(
        "claude",
        ProcessOutcome(
            0,
            '{"type":"result","subtype":"success","is_error":false,'
            '"result":"done","session_id":"claude-session","usage":{"input_tokens":12}}',
            "",
            10,
        ),
    )
    assert claude["status"] == "completed"
    assert claude["final_content"] == "done"


def test_unified_runner_archives_external_harness_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    refactorbench_root = _make_refactorbench(tmp_path / "RefactorBench")
    archive = tmp_path / "archive"
    original_run_process = runner._run_process

    def fake_run_process(
        command: list[str],
        *,
        cwd: Path,
        timeout: float,
        input_text: str | None = None,
    ) -> ProcessOutcome:
        if command[0] == "/fake/codex":
            assert input_text == "Change VALUE from zero to one.\n"
            (cwd / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            return ProcessOutcome(
                0,
                '{"type":"thread.started","thread_id":"fake"}\n'
                '{"type":"item.completed","item":{"type":"agent_message",'
                '"text":"changed"}}\n'
                '{"type":"turn.completed","usage":{"input_tokens":10,'
                '"output_tokens":2}}\n',
                "diagnostic\n",
                25,
            )
        return original_run_process(
            command,
            cwd=cwd,
            timeout=timeout,
            input_text=input_text,
        )

    monkeypatch.setattr(
        runner,
        "_resolve_harness_executable",
        lambda harness: f"/fake/{harness}",
    )
    monkeypatch.setattr(runner, "_harness_version", lambda harness, executable: "1.0")
    monkeypatch.setattr(runner, "_run_process", fake_run_process)
    args = build_parser().parse_args(
        [
            "--refactorbench-root",
            str(refactorbench_root),
            "--harness",
            "codex",
            "--output-dir",
            str(archive),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    summary = run_benchmark(args)

    assert summary["passed"] == 1
    assert summary["harnesses"]["codex"]["input_tokens"] == 10
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    records = [
        json.loads(line)
        for line in (archive / "codex" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["evaluation"]["passed"]
    assert records[0]["patch_path"].startswith("codex/patches/")
    assert (archive / records[0]["agent"]["stdout_path"]).is_file()
    assert (archive / "summary.json").is_file()

    created_at = manifest["created_at"]
    resume_args = build_parser().parse_args(
        [
            "--refactorbench-root",
            str(refactorbench_root),
            "--harness",
            "codex",
            "--output-dir",
            str(archive),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--resume",
        ]
    )
    resumed = run_benchmark(resume_args)
    resumed_manifest = json.loads(
        (archive / "manifest.json").read_text(encoding="utf-8")
    )
    assert resumed["executed_cases"] == 0
    assert resumed["skipped_completed"] == 1
    assert resumed_manifest["created_at"] == created_at


def test_missing_cli_does_not_claim_output_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    refactorbench_root = _make_refactorbench(tmp_path / "RefactorBench")
    archive = tmp_path / "archive"

    def missing_cli(harness: str) -> str:
        raise RuntimeError(f"missing {harness}")

    monkeypatch.setattr(runner, "_resolve_harness_executable", missing_cli)
    args = build_parser().parse_args(
        [
            "--refactorbench-root",
            str(refactorbench_root),
            "--harness",
            "codex",
            "--output-dir",
            str(archive),
        ]
    )

    with pytest.raises(RuntimeError, match="missing codex"):
        run_benchmark(args)

    assert not archive.exists()
