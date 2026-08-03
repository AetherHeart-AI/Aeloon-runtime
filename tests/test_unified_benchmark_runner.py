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
from benchmarks.harness import HARNESS_NAMES
from benchmarks.harness.aeloon import AeloonHarness
from benchmarks.harness.base import (
    Harness,
    HarnessInvocation,
    HarnessRequest,
    HarnessResult,
    ProcessOutcome,
)
from benchmarks.harness.claude import ClaudeHarness
from benchmarks.harness.codex import CodexHarness
from benchmarks.harness.hermes import HermesHarness
from benchmarks.harness.openclaw import OpenClawHarness
from benchmarks.harness.pi import PiHarness
from benchmarks.livecodebench.runner import LiveCodeBenchCase
from benchmarks.progress import ProgressBar, configure_progress, info


class FakeHarness:
    name = "fake"
    version = "fake@1"
    model = "fake-model"

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


def test_cli_exposes_harness_benchmark_model_workers_and_limit() -> None:
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
    assert args.model == "deepseek-v4-flash"
    assert args.config is None
    assert args.workers == 1
    assert args.limit is None
    assert {action.dest for action in parser._actions if action.dest != "help"} == {
        "harness",
        "benchmark",
        "model",
        "config",
        "workers",
        "limit",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--harness",
                "aeloon",
                "--benchmark",
                "refactorbench",
                "--model",
                " ",
            ]
        )


@pytest.mark.parametrize(
    "harness_type, executable",
    [
        (AeloonHarness, "/fake/python"),
        (PiHarness, "/fake/pi"),
        (CodexHarness, "/fake/codex"),
        (ClaudeHarness, "/fake/claude"),
        (OpenClawHarness, "/fake/openclaw"),
        (HermesHarness, "/fake/hermes"),
    ],
)
def test_harness_invocations_forward_selected_model(
    tmp_path: Path,
    harness_type: type[Harness],
    executable: str,
) -> None:
    harness = object.__new__(harness_type)
    harness.executable = executable
    harness.model = "deepseek-v4-pro"
    request = HarnessRequest(
        prompt="solve it",
        workspace=tmp_path,
        session_dir=tmp_path / "sessions",
        project_root=tmp_path,
    )

    command = harness.build_invocation(request).command

    model_option = command.index("--model")
    assert command[model_option + 1] == "deepseek-v4-pro"
    if harness_type is AeloonHarness:
        assert command[command.index("--session-dir") + 1] == str(
            request.session_dir
        )
        assert "--data-dir" not in command


def test_aeloon_harness_forwards_explicit_config_path(tmp_path: Path) -> None:
    harness = object.__new__(AeloonHarness)
    harness.executable = "/fake/python"
    harness.model = "coder"
    config_path = tmp_path / "config.json"

    command = harness.build_invocation(
        HarnessRequest(
            prompt="solve it",
            workspace=tmp_path,
            session_dir=tmp_path / "sessions",
            project_root=tmp_path,
            config_path=config_path,
        )
    ).command

    config_option = command.index("--config")
    assert command[config_option + 1] == str(config_path.resolve())


def test_harness_registry_includes_openclaw_and_hermes() -> None:
    assert HARNESS_NAMES == (
        "aeloon",
        "pi",
        "codex",
        "claude",
        "openclaw",
        "hermes",
    )


def test_openclaw_uses_headless_json_exec_and_normalizes_result(tmp_path: Path) -> None:
    harness = object.__new__(OpenClawHarness)
    harness.executable = "/fake/openclaw"
    harness.model = "openai/gpt-5.6-sol"
    request = HarnessRequest(
        prompt="solve it",
        workspace=tmp_path,
        session_dir=tmp_path / "sessions",
        project_root=tmp_path,
    )

    invocation = harness.build_invocation(request)
    result = harness.interpret(
        ProcessOutcome(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "status": "ok",
                    "final": "done",
                    "usage": {"input": 120, "output": 8, "total": 128},
                    "costUsd": 0.0021,
                    "toolSummary": {"calls": 2, "tools": ["read", "write"]},
                    "model": "gpt-5.6-sol",
                    "provider": "openai",
                    "sessionId": "session-1",
                }
            ),
            stderr="",
            duration_ms=1,
        )
    )

    assert invocation.command[:3] == ["/fake/openclaw", "agent", "exec"]
    assert invocation.input_text == "solve it"
    assert invocation.prompt_argument is False
    assert invocation.command[invocation.command.index("--message-file") + 1] == "-"
    assert invocation.command[invocation.command.index("--cwd") + 1] == str(tmp_path)
    assert "--json" in invocation.command
    assert result == {
        "status": "completed",
        "session_id": "session-1",
        "duration_ms": None,
        "final_content": "done",
        "tools_used": ["read", "write"],
        "usage": {"input": 120, "output": 8, "total": 128},
        "models": {"model": "gpt-5.6-sol", "provider": "openai"},
        "cost_usd": 0.0021,
        "payload_error": None,
    }


def test_openclaw_reports_structured_agent_errors(tmp_path: Path) -> None:
    harness = object.__new__(OpenClawHarness)

    result = harness.interpret(
        ProcessOutcome(
            returncode=1,
            stdout=json.dumps(
                {
                    "ok": False,
                    "status": "error",
                    "final": None,
                    "error": {"kind": "model_error", "message": "provider failed"},
                }
            ),
            stderr="",
            duration_ms=1,
        )
    )

    assert result["status"] == "agent_error"
    assert result["payload_error"] == "model_error: provider failed"


def test_hermes_uses_scripted_oneshot_mode(tmp_path: Path) -> None:
    harness = object.__new__(HermesHarness)
    harness.executable = "/fake/hermes"
    harness.model = "deepseek/deepseek-v4"
    request = HarnessRequest(
        prompt="solve it",
        workspace=tmp_path,
        session_dir=tmp_path / "sessions",
        project_root=tmp_path,
    )

    invocation = harness.build_invocation(request)
    result = harness.interpret(
        ProcessOutcome(
            returncode=0,
            stdout="done\n",
            stderr="",
            duration_ms=1,
        )
    )

    assert invocation.command == [
        "/fake/hermes",
        "--yolo",
        "--model",
        "deepseek/deepseek-v4",
        "-z",
        "solve it",
    ]
    assert invocation.prompt_argument is True
    assert result == {
        "status": "completed",
        "final_content": "done",
        "payload_error": None,
    }


def test_pi_uses_json_mode_and_normalizes_token_usage(tmp_path: Path) -> None:
    harness = object.__new__(PiHarness)
    harness.executable = "/fake/pi"
    harness.model = "deepseek-v4-flash"
    request = HarnessRequest(
        prompt="solve it",
        workspace=tmp_path,
        session_dir=tmp_path / "sessions",
        project_root=tmp_path,
    )

    invocation = harness.build_invocation(request)
    result = harness.interpret(
        ProcessOutcome(
            returncode=0,
            stdout="\n".join(
                [
                    '{"type":"session","id":"pi-session"}',
                    '{"type":"message_end","message":{"role":"assistant",'
                    '"content":[{"type":"text","text":"done"}],'
                    '"provider":"deepseek","model":"deepseek-v4-flash",'
                    '"usage":{"input":12,"output":3,"cost":{"total":0.01}},'
                    '"stopReason":"stop"}}',
                    '{"type":"agent_end","messages":[]}',
                ]
            ),
            stderr="",
            duration_ms=1,
        )
    )

    assert invocation.command[:4] == ["/fake/pi", "--print", "--mode", "json"]
    assert result == {
        "status": "completed",
        "session_id": "pi-session",
        "final_content": "done",
        "usage": {
            "input": 12,
            "output": 3,
            "input_tokens": 12,
            "output_tokens": 3,
            "cost": {"total": 0.01},
        },
        "cost_usd": 0.01,
        "models": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "payload_error": None,
    }


def test_harness_run_bounds_raw_process_output(tmp_path: Path, monkeypatch) -> None:
    class CompactingHarness(Harness):
        name = "compacting"

        def resolve_executable(self) -> str:
            return "/fake/compacting"

        def build_invocation(self, request: HarnessRequest) -> HarnessInvocation:
            return HarnessInvocation(command=[self.executable], cwd=request.workspace)

        def interpret(self, outcome: ProcessOutcome) -> dict[str, object]:
            return {"status": "completed", "final_content": "done"}

    harness = object.__new__(CompactingHarness)
    harness.executable = "/fake/compacting"
    harness.version = "fake@1"
    raw_output = "x" * 100_000
    monkeypatch.setattr(
        "benchmarks.harness.base.run_process",
        lambda *args, **kwargs: ProcessOutcome(0, raw_output, "", 1),
    )

    result = harness.run(
        HarnessRequest(
            prompt="solve it",
            workspace=tmp_path,
            session_dir=tmp_path / "sessions",
            project_root=tmp_path,
        )
    )

    assert result.final_content == "done"
    assert len(result.process.stdout) < 21_000
    assert result.process.stdout.endswith("x" * 20_000)


def test_benchmark_python_packages_are_not_gitignored() -> None:
    root = Path(__file__).resolve().parents[1]
    ignored_paths = {
        line.strip().casefold()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert "benchmarks/refactorbench/" not in ignored_paths
    assert "benchmarks/livecodebench/" not in ignored_paths
    assert "benchmarks/repoqa/" not in ignored_paths


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
        lambda names, **kwargs: events.append(("harnesses", names, kwargs["model"])) or harnesses,
    )
    monkeypatch.setattr(
        unified,
        "get_adapter",
        lambda name, **kwargs: events.append(
            ("adapter", name, kwargs["workers"], kwargs["config_path"])
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
            "--model",
            "deepseek-v4-pro",
            "--config",
            str(tmp_path / "config.json"),
        ]
    )

    assert unified.run(args) == {"ok": True}
    assert events == [
        ("harnesses", ["aeloon", "codex"], "deepseek-v4-pro"),
        ("adapter", "livecodebench", 3, tmp_path / "config.json"),
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


def test_benchmark_adapter_forwards_aeloon_config_to_harness(tmp_path: Path) -> None:
    config_path = tmp_path / "aeloon.json"
    adapter = RefactorBenchAdapter(project_root=tmp_path, config_path=config_path)
    _make_refactorbench(adapter.run.source_dir)
    requests: list[HarnessRequest] = []

    class CapturingHarness(FakeHarness):
        def run(self, request: HarnessRequest) -> HarnessResult:
            requests.append(request)
            return super().run(request)

    adapter.execute([CapturingHarness()])  # type: ignore[list-item]

    assert requests[0].config_path == config_path.resolve()


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
    caplog,
) -> None:
    caplog.set_level("INFO", logger="aeloon.benchmarks")
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
    assert "[livecodebench/fake] code-generation score: 2/2 (100.00%)" in caplog.text
    assert "[livecodebench/fake] self-repair score: 2/2 (100.00%)" in caplog.text


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

    evaluations = adapter.evaluate(generations={"case-1": ("print(1)", result)})

    assert evaluations == {
        "case-1": {
            "passed": True,
            "oracle_passed": True,
            "metadata": {"runtime": 0.1},
        }
    }


def test_livecodebench_evaluator_failure_leaves_no_checkpoint(
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
    assert not (adapter.run.output_dir / "fake" / "checkpoints").exists()
