from __future__ import annotations

from pathlib import Path

from benchmarks.run_refactorbench import (
    ProcessOutcome,
    WorkspaceCache,
    _capture_patch,
    _parse_agent_payload,
    _run_official_test,
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
