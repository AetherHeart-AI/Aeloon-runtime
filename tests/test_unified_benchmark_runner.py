from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.run_bench as unified
from benchmarks import progress as benchmark_progress
from benchmarks.adapters.livecodebench import LiveCodeBenchAdapter
from benchmarks.adapters.refactorbench import RefactorBenchAdapter
from benchmarks.harness.base import (
    HarnessInvocation,
    HarnessResult,
    ProcessOutcome,
)
from benchmarks.livecodebench.runner import LiveCodeBenchCase
from benchmarks.progress import ProgressBar, configure_progress, info


class FakeHarness:
    name = "fake"
    version = "fake@1"

    def run(self, request) -> HarnessResult:
        (request.workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        invocation = HarnessInvocation(
            command=["fake", "<prompt>"],
            cwd=request.workspace,
            prompt_argument=True,
        )
        return HarnessResult(
            harness=self.name,
            version=self.version,
            invocation=invocation,
            process=ProcessOutcome(
                returncode=0,
                stdout='{"status":"completed"}',
                stderr="",
                duration_ms=1,
            ),
            status="completed",
            final_content="done",
        )


class ConcurrentFakeHarness(FakeHarness):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def run(self, request) -> HarnessResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().run(request)
        finally:
            with self._lock:
                self.active -= 1


def _make_refactorbench(root: Path) -> None:
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


def test_cli_exposes_harness_benchmark_workers_and_resume() -> None:
    parser = unified.build_parser()
    args = parser.parse_args(
        [
            "--harness",
            "aeloon",
            "pi",
            "--benchmark",
            "refactorbench",
        ]
    )

    assert args.harness == [["aeloon", "pi"]]
    assert args.benchmark == "refactorbench"
    assert args.workers == 1
    assert args.resume is None
    assert {action.dest for action in parser._actions if action.dest != "help"} == {
        "harness",
        "benchmark",
        "workers",
        "resume",
    }

    resumed = parser.parse_args(
        [
            "--harness",
            "aeloon",
            "--benchmark",
            "livecodebench",
            "--resume",
            "20260729T100902Z-d558b57d",
        ]
    )

    assert resumed.resume == "20260729T100902Z-d558b57d"


def test_benchmark_python_packages_are_not_gitignored() -> None:
    root = Path(__file__).resolve().parents[1]
    ignored_paths = {
        line.strip().casefold()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert "benchmarks/refactorbench/" not in ignored_paths
    assert "benchmarks/livecodebench/" not in ignored_paths


def test_progress_is_written_to_stderr_without_polluting_stdout(capsys) -> None:
    configure_progress()

    info("Preparing %s", "livecodebench")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INFO Preparing livecodebench" in captured.err


def test_progress_bar_renders_and_coexists_with_logs(monkeypatch) -> None:
    class TerminalStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = TerminalStream()
    monkeypatch.setattr(benchmark_progress.sys, "stderr", stream)
    configure_progress()

    with ProgressBar("refactorbench", total=2) as progress:
        progress.set_detail("running case-1")
        progress.advance(detail="case-1 PASS")
        info("Evaluator finished")
        progress.advance(detail="case-2 PASS")

    rendered = stream.getvalue()
    assert "refactorbench" in rendered
    assert "2/2" in rendered
    assert "100%" in rendered
    assert "case-2 PASS" in rendered
    assert "INFO Evaluator finished" in rendered


def test_progress_bar_uses_line_snapshots_when_stderr_is_redirected(
    monkeypatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(benchmark_progress.sys, "stderr", stream)

    with ProgressBar("livecodebench", total=2) as progress:
        progress.advance(detail="case-1")
        progress.advance(detail="case-2")

    rendered = stream.getvalue()
    assert "\r" not in rendered
    assert [line.split()[2] for line in rendered.splitlines()] == [
        "0/2",
        "1/2",
        "2/2",
    ]


def test_unified_runner_prepares_before_execute(monkeypatch, tmp_path: Path) -> None:
    events: list[object] = []
    harnesses = [object(), object()]

    class FakeAdapter:
        run = SimpleNamespace(
            source_dir=tmp_path / "source",
            output_dir=tmp_path / "results",
        )

        def prepare(self) -> None:
            events.append("prepare")

        def execute(self, selected) -> dict[str, object]:
            events.append(("execute", selected))
            return {"ok": True}

    monkeypatch.setattr(unified, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        unified,
        "get_harnesses",
        lambda names, **kwargs: events.append(("harnesses", names)) or harnesses,
    )
    monkeypatch.setattr(
        unified,
        "get_adapter",
        lambda name, **kwargs: events.append(
            (
                "adapter",
                name,
                kwargs["workers"],
                kwargs["resume_run_id"],
            )
        )
        or FakeAdapter(),
    )
    args = unified.build_parser().parse_args(
        [
            "--harness",
            "aeloon",
            "--harness",
            "codex",
            "--benchmark",
            "livecodebench",
            "--workers",
            "3",
        ]
    )

    assert unified.run(args) == {"ok": True}
    assert events == [
        ("harnesses", ["aeloon", "codex"]),
        ("adapter", "livecodebench", 3, None),
        "prepare",
        ("execute", harnesses),
    ]


def test_refactorbench_adapter_runs_through_shared_harness(tmp_path: Path) -> None:
    adapter = RefactorBenchAdapter(project_root=tmp_path)
    _make_refactorbench(adapter.run.source_dir)

    summary = adapter.execute([FakeHarness()])  # type: ignore[list-item]

    assert summary["passed"] == 1
    assert summary["harnesses"]["fake"]["passed"] == 1
    records = [
        json.loads(line)
        for line in (adapter.run.output_dir / "fake" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["evaluation"]["passed"] is True
    assert records[0]["agent"]["harness"] == "fake"
    assert (adapter.run.output_dir / records[0]["patch_path"]).is_file()


def test_refactorbench_parallelizes_isolated_repositories(tmp_path: Path) -> None:
    adapter = RefactorBenchAdapter(project_root=tmp_path, workers=2)
    root = adapter.run.source_dir
    _make_refactorbench(root)

    repository = "other_refactor"
    prompt = root / "problems" / "base_problems" / repository / "change-task.txt"
    test = root / "tests" / repository / "change-test.py"
    source = root / "repositories" / repository / "module.py"
    for path in (prompt, test, source):
        path.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text("Change VALUE from zero to one.\n", encoding="utf-8")
    source.write_text("VALUE = 0\n", encoding="utf-8")
    test.write_text(
        "from pathlib import Path\n"
        "source = (Path.cwd() / '..' / 'module.py').resolve().read_text()\n"
        "raise SystemExit(0 if 'VALUE = 1' in source else 1)\n",
        encoding="utf-8",
    )
    (root / "scripts" / "base_mapping.py").write_text(
        "file_mapping = {\n"
        "    '../tests/demo_refactor/change-test.py': "
        "'../problems/base_problems/demo_refactor/change-task.txt',\n"
        "    '../tests/other_refactor/change-test.py': "
        "'../problems/base_problems/other_refactor/change-task.txt',\n"
        "}\n",
        encoding="utf-8",
    )
    harness = ConcurrentFakeHarness()

    summary = adapter.execute([harness])  # type: ignore[list-item]

    assert summary["workers"] == 2
    assert summary["passed"] == 2
    assert harness.max_active == 2
    records = [
        json.loads(line)
        for line in (adapter.run.output_dir / "fake" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {record["instance_id"] for record in records} == {
        "demo_refactor/change",
        "other_refactor/change",
    }


def test_livecodebench_parallelizes_case_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = LiveCodeBenchAdapter(project_root=tmp_path, workers=2)
    cases = [
        LiveCodeBenchCase(
            instance_id=f"case-{index}",
            question_title="Print one",
            question_content="Print 1.",
            starter_code="",
            platform="codeforces",
            contest_id="contest",
            contest_date="2025-04-01T00:00:00",
            difficulty="easy",
        )
        for index in range(2)
    ]

    def fake_evaluate(*, generations):
        return {
            instance_id: {
                "passed": True,
                "oracle_passed": True,
                "metadata": {},
            }
            for instance_id in generations
        }

    monkeypatch.setattr(adapter, "evaluate", fake_evaluate)
    harness = ConcurrentFakeHarness()

    summary = adapter._run_harness(harness, cases)  # type: ignore[arg-type]

    assert summary["code_generation_passed"] == 2
    assert summary["self_repair_passed"] == 2
    assert harness.max_active == 2


def test_livecodebench_evaluate_accepts_complete_official_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = LiveCodeBenchAdapter(project_root=tmp_path)
    result = HarnessResult(
        harness="fake",
        version="fake@1",
        invocation=HarnessInvocation(command=["fake"], cwd=tmp_path),
        process=ProcessOutcome(
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=1,
        ),
        status="completed",
        final_content="print(1)",
    )

    def fake_run_adapter(**kwargs):
        payload = json.loads(kwargs["input_path"].read_text(encoding="utf-8"))
        assert payload == [{"instance_id": "case-1", "code": "print(1)"}]
        return {
            "results": [
                {
                    "instance_id": "case-1",
                    "passed": True,
                    "metadata": {"runtime": 0.1},
                }
            ]
        }

    monkeypatch.setattr(
        "benchmarks.adapters.livecodebench.official._run_adapter",
        fake_run_adapter,
    )

    evaluations = adapter.evaluate(
        generations={"case-1": ("print(1)", result)}
    )

    assert evaluations == {
        "case-1": {
            "passed": True,
            "oracle_passed": True,
            "metadata": {"runtime": 0.1},
        }
    }


def test_livecodebench_reuses_generation_checkpoints_after_evaluator_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = LiveCodeBenchAdapter(project_root=tmp_path)
    cases = [
        LiveCodeBenchCase(
            instance_id=f"case-{index}",
            question_title="Print one",
            question_content="Print 1.",
            starter_code="",
            platform="codeforces",
            contest_id="contest",
            contest_date="2025-04-01T00:00:00",
            difficulty="easy",
        )
        for index in range(2)
    ]

    class CountingHarness(FakeHarness):
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request) -> HarnessResult:
            self.calls += 1
            return super().run(request)

    first_harness = CountingHarness()

    def fail_evaluation(*, generations):
        assert set(generations) == {"case-0", "case-1"}
        raise RuntimeError("simulated evaluator failure")

    monkeypatch.setattr(adapter, "evaluate", fail_evaluation)

    with pytest.raises(RuntimeError, match="simulated evaluator failure"):
        adapter._run_harness(first_harness, cases)  # type: ignore[arg-type]

    assert first_harness.calls == 2
    assert len(list(adapter.run.output_dir.glob("fake/checkpoints/code-generation/*.json"))) == 2

    second_harness = CountingHarness()

    def pass_evaluation(*, generations):
        return {
            instance_id: {
                "passed": True,
                "oracle_passed": True,
                "metadata": {},
            }
            for instance_id in generations
        }

    monkeypatch.setattr(adapter, "evaluate", pass_evaluation)
    summary = adapter._run_harness(second_harness, cases)  # type: ignore[arg-type]

    assert second_harness.calls == 0
    assert summary["code_generation_passed"] == 2
    assert summary["self_repair_passed"] == 2
    results_path = adapter.run.output_dir / "fake" / "results.jsonl"
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 4

    monkeypatch.setattr(
        adapter,
        "evaluate",
        lambda **kwargs: pytest.fail("idempotent resume must not rerun evaluation"),
    )
    adapter._run_harness(second_harness, cases)  # type: ignore[arg-type]

    assert second_harness.calls == 0
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 4


def test_livecodebench_resume_imports_recovery_and_continues_self_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initial = LiveCodeBenchAdapter(project_root=tmp_path)
    cases = [
        LiveCodeBenchCase(
            instance_id=f"case-{index}",
            question_title="Print one",
            question_content="Print 1.",
            starter_code="",
            platform="codeforces",
            contest_id="contest",
            contest_date="2025-04-01T00:00:00",
            difficulty="easy",
        )
        for index in range(2)
    ]

    class CountingHarness(FakeHarness):
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request) -> HarnessResult:
            self.calls += 1
            return super().run(request)

    harness = CountingHarness()
    recovered_records = []
    for index, case in enumerate(cases):
        result = FakeHarness().run(SimpleNamespace(workspace=tmp_path))
        recovered_records.append(
            initial._record(
                harness=harness,  # type: ignore[arg-type]
                case=case,
                scenario="code-generation",
                prompt=f"prompt-{index}",
                code=f"print({index})",
                result=result,
                evaluation={
                    "passed": index == 0,
                    "oracle_passed": index == 0,
                    "metadata": {} if index == 0 else {"error_code": -2},
                },
            )
        )

    initial.write_json(
        initial.run.output_dir / "manifest.json",
        {
            **initial.manifest([harness], status="interrupted"),  # type: ignore[list-item]
            "created_at": "2026-07-29T10:09:09+00:00",
            "finished_at": "2026-07-29T11:30:26+00:00",
            "release_version": initial.release_version,
            "scenarios": list(initial.scenarios),
            "cases": [case.instance_id for case in cases],
        },
    )
    initial.write_json(
        initial.run.output_dir / "fake" / "recovered-code-generation.json",
        {"records": recovered_records},
    )

    resumed = LiveCodeBenchAdapter(
        project_root=tmp_path,
        resume_run_id=initial.run.run_id,
    )
    monkeypatch.setattr(resumed, "load_cases", lambda: cases)

    def pass_repair(*, generations):
        assert set(generations) == {"case-1"}
        return {
            "case-1": {
                "passed": True,
                "oracle_passed": True,
                "metadata": {},
            }
        }

    monkeypatch.setattr(resumed, "evaluate", pass_repair)
    summary = resumed.execute([harness])  # type: ignore[list-item]

    assert harness.calls == 1
    assert summary["harnesses"]["fake"]["code_generation_passed"] == 1
    assert summary["harnesses"]["fake"]["self_repair_passed"] == 2
    assert summary["harnesses"]["fake"]["repaired"] == 1
    results_path = resumed.run.output_dir / "fake" / "results.jsonl"
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 4

    monkeypatch.setattr(
        resumed,
        "evaluate",
        lambda **kwargs: pytest.fail("completed records must be reused"),
    )
    resumed.execute([harness])  # type: ignore[list-item]

    assert harness.calls == 1
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 4
    manifest = json.loads(
        (resumed.run.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert len(manifest["resume_events"]) == 2


def test_livecodebench_run_lock_rejects_concurrent_execution(
    tmp_path: Path,
) -> None:
    adapter = LiveCodeBenchAdapter(project_root=tmp_path)

    with adapter._run_lock():
        with pytest.raises(RuntimeError, match="already active"):
            with adapter._run_lock():
                pytest.fail("a concurrent run must not acquire the lock")
