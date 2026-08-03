from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import benchmarks.run_livecodebench as runner
from benchmarks.run_livecodebench import (
    AgentGeneration,
    LiveCodeBenchCase,
    ProcessOutcome,
    _extract_python_code,
    build_parser,
    run_benchmark,
)


def _case(instance_id: str = "v6_case") -> LiveCodeBenchCase:
    return LiveCodeBenchCase(
        instance_id=instance_id,
        question_title="Add two numbers",
        question_content="Read two integers and print their sum.",
        starter_code="",
        platform="codeforces",
        contest_id="contest-6",
        contest_date="2025-04-01T00:00:00",
        difficulty="easy",
    )


def _generation(prompt: str, code: str) -> AgentGeneration:
    final_content = f"```python\n{code}\n```"
    payload = {
        "status": "completed",
        "session_id": "session",
        "final_content": final_content,
    }
    return AgentGeneration(
        prompt=prompt,
        process=ProcessOutcome(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
            duration_ms=10,
        ),
        payload=payload,
        payload_error=None,
        code=code,
        extraction_error=None,
    )


def test_default_selects_new_v6_slice_and_both_scenarios(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--livecodebench-root", str(tmp_path / "LiveCodeBench")]
    )

    assert args.release_version == "v6"
    assert args.all is False
    assert args.scenario is None
    assert args.model == "deepseek-v4-flash"


def test_agent_invocation_forwards_model_and_config(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_process(command, *, cwd, timeout, input_text=None):
        captured["command"] = command
        captured["input_text"] = input_text
        return ProcessOutcome(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "final_content": "```python\nprint(1)\n```",
                }
            ),
            stderr="",
            duration_ms=1,
        )

    monkeypatch.setattr(runner, "_run_process", fake_run_process)
    config_path = tmp_path / "config.json"
    args = build_parser().parse_args(
        [
            "--livecodebench-root",
            str(tmp_path / "LiveCodeBench"),
            "--model",
            "studio/coder",
            "--config",
            str(config_path),
            "--workspace-dir",
            str(tmp_path / "workspaces"),
            "--results",
            str(tmp_path / "results.jsonl"),
        ]
    )

    generation = runner._invoke_agent(
        "solve it",
        case=_case(),
        scenario="code-generation",
        args=args,
        release_version="v6",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--model") + 1] == "studio/coder"
    assert command[command.index("--config") + 1] == str(config_path.resolve())
    assert captured["input_text"] == "solve it"
    assert generation.code == "print(1)"


def test_default_adapter_python_comes_from_current_worktree(
    tmp_path: Path, monkeypatch
) -> None:
    worktree = tmp_path / "worktree"
    executable = worktree / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", worktree)

    assert runner._resolve_livecodebench_python(None) == executable


def test_all_switch_selects_cumulative_release_v6(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, str] = {}

    def fake_load_cases(
        root: Path,
        python: Path,
        release_version: str,
        **kwargs,
    ):
        seen["release_version"] = release_version
        return [_case()]

    monkeypatch.setattr(runner, "load_cases", fake_load_cases)
    args = build_parser().parse_args(
        [
            "--livecodebench-root",
            str(tmp_path / "LiveCodeBench"),
            "--livecodebench-python",
            sys.executable,
            "--all",
            "--list",
        ]
    )

    summary = run_benchmark(args)

    assert seen["release_version"] == "release_v6"
    assert summary["release_version"] == "release_v6"


def test_list_mode_has_no_workspace_or_result_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "load_cases", lambda *args, **kwargs: [_case()])
    workspace = tmp_path / "workspaces"
    results = tmp_path / "results.jsonl"
    args = build_parser().parse_args(
        [
            "--livecodebench-root",
            str(tmp_path / "LiveCodeBench"),
            "--livecodebench-python",
            sys.executable,
            "--workspace-dir",
            str(workspace),
            "--results",
            str(results),
            "--list",
        ]
    )

    summary = run_benchmark(args)

    assert summary["scenarios"] == ["code-generation", "self-repair"]
    assert [case["instance_id"] for case in summary["cases"]] == ["v6_case"]
    assert not workspace.exists()
    assert not results.exists()


def test_repeated_scenario_is_deduplicated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "load_cases", lambda *args, **kwargs: [_case()])
    args = build_parser().parse_args(
        [
            "--livecodebench-root",
            str(tmp_path / "LiveCodeBench"),
            "--livecodebench-python",
            sys.executable,
            "--scenario",
            "code-generation",
            "--scenario",
            "code-generation",
            "--list",
        ]
    )

    summary = run_benchmark(args)

    assert summary["scenarios"] == ["code-generation"]


def test_code_generation_and_self_repair_share_official_feedback(
    tmp_path: Path, monkeypatch
) -> None:
    case = _case()
    monkeypatch.setattr(runner, "load_cases", lambda *args, **kwargs: [case])
    prompts: list[tuple[str, str]] = []

    def fake_invoke(
        prompt: str,
        *,
        case: LiveCodeBenchCase,
        scenario: str,
        args,
        release_version: str,
    ) -> AgentGeneration:
        prompts.append((scenario, prompt))
        code = (
            "print(0)"
            if scenario == "code-generation"
            else "a,b=map(int,input().split());print(a+b)"
        )
        return _generation(prompt, code)

    evaluation_calls: list[list[str]] = []

    def fake_evaluate(generations, **kwargs):
        codes = [generation.code for generation in generations.values()]
        evaluation_calls.append(codes)
        if codes == ["print(0)"]:
            return {
                case.instance_id: {
                    "passed": False,
                    "oracle_passed": False,
                    "metadata": {
                        "error_code": -2,
                        "inputs": "2 3",
                        "output": "0",
                        "expected": "5",
                    },
                }
            }
        return {
            case.instance_id: {
                "passed": True,
                "oracle_passed": True,
                "metadata": {"execution time": 0.001},
            }
        }

    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    monkeypatch.setattr(runner, "_evaluate_generations", fake_evaluate)
    results = tmp_path / "livecodebench.jsonl"
    args = build_parser().parse_args(
        [
            "--livecodebench-root",
            str(tmp_path / "LiveCodeBench"),
            "--livecodebench-python",
            sys.executable,
            "--workspace-dir",
            str(tmp_path / "workspaces"),
            "--results",
            str(results),
        ]
    )

    summary = run_benchmark(args)
    records = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["scenario"] for record in records] == [
        "code-generation",
        "self-repair",
    ]
    assert records[0]["evaluation"]["passed"] is False
    assert records[1]["baseline"]["evaluation"]["oracle_passed"] is False
    assert records[1]["evaluation"]["passed"] is True
    assert records[1]["repaired"] is True
    assert evaluation_calls == [
        ["print(0)"],
        ["a,b=map(int,input().split());print(a+b)"],
    ]
    assert prompts[0][0] == "code-generation"
    assert prompts[1][0] == "self-repair"
    assert "Generated output: 0" in prompts[1][1]
    assert "Expected: 5" in prompts[1][1]
    assert summary["scenarios"]["self-repair"]["repaired"] == 1


def test_resume_can_restore_code_generation_from_self_repair_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    case = _case()
    monkeypatch.setattr(runner, "load_cases", lambda *args, **kwargs: [case])
    invocation_count = 0

    def fake_invoke(prompt: str, *, scenario: str, **kwargs) -> AgentGeneration:
        nonlocal invocation_count
        invocation_count += 1
        code = "print(0)" if scenario == "code-generation" else "print(1)"
        return _generation(prompt, code)

    def fake_evaluate(generations, **kwargs):
        generation = next(iter(generations.values()))
        passed = generation.code == "print(1)"
        return {
            case.instance_id: {
                "passed": passed,
                "oracle_passed": passed,
                "metadata": (
                    {"execution time": 0.001}
                    if passed
                    else {
                        "error_code": -2,
                        "inputs": "",
                        "output": "0",
                        "expected": "1",
                    }
                ),
            }
        }

    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    monkeypatch.setattr(runner, "_evaluate_generations", fake_evaluate)
    results = tmp_path / "livecodebench.jsonl"
    common = [
        "--livecodebench-root",
        str(tmp_path / "LiveCodeBench"),
        "--livecodebench-python",
        sys.executable,
        "--results",
        str(results),
    ]

    run_benchmark(
        build_parser().parse_args([*common, "--scenario", "self-repair"])
    )
    summary = run_benchmark(build_parser().parse_args([*common, "--resume"]))
    records = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
    ]

    assert invocation_count == 2
    assert [record["scenario"] for record in records] == [
        "self-repair",
        "code-generation",
    ]
    assert records[1]["prompt"] is not None
    assert records[1]["generation"]["code"] == "print(0)"
    assert summary["executed_records"] == 1
    assert summary["skipped_completed"] == 1


def test_self_repair_skips_model_when_generation_already_passes(
    tmp_path: Path, monkeypatch
) -> None:
    case = _case()
    monkeypatch.setattr(runner, "load_cases", lambda *args, **kwargs: [case])
    invocation_count = 0

    def fake_invoke(prompt: str, **kwargs) -> AgentGeneration:
        nonlocal invocation_count
        invocation_count += 1
        return _generation(prompt, "a,b=map(int,input().split());print(a+b)")

    def fake_evaluate(generations, **kwargs):
        return {
            case.instance_id: {
                "passed": True,
                "oracle_passed": True,
                "metadata": {"execution time": 0.001},
            }
        }

    monkeypatch.setattr(runner, "_invoke_agent", fake_invoke)
    monkeypatch.setattr(runner, "_evaluate_generations", fake_evaluate)
    results = tmp_path / "livecodebench.jsonl"
    args = build_parser().parse_args(
        [
            "--livecodebench-root",
            str(tmp_path / "LiveCodeBench"),
            "--livecodebench-python",
            sys.executable,
            "--results",
            str(results),
        ]
    )

    run_benchmark(args)
    records = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
    ]

    assert invocation_count == 1
    assert records[1]["scenario"] == "self-repair"
    assert records[1]["repair_attempted"] is False
    assert records[1]["agent"] is None
    assert records[1]["generation"]["code"] == records[0]["generation"]["code"]


def test_extract_python_code_prefers_last_fenced_block() -> None:
    code, error = _extract_python_code(
        "Example:\n```python\nprint('old')\n```\nFixed:\n```python\nprint('new')\n```"
    )

    assert code == "print('new')"
    assert error is None


def test_adapter_uses_official_checkout_for_listing_and_evaluation(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "LiveCodeBench"
    package = checkout / "lcb_runner"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "benchmarks.py").write_text(
        "from datetime import datetime\n"
        "from enum import Enum\n"
        "class Value(Enum):\n"
        "    ITEM = 'easy'\n"
        "class Problem:\n"
        "    question_id = 'official-v6'\n"
        "    question_title = 'Official case'\n"
        "    question_content = 'Print 1.'\n"
        "    starter_code = ''\n"
        "    platform = Value.ITEM\n"
        "    contest_id = 'six'\n"
        "    contest_date = datetime(2025, 4, 1)\n"
        "    difficulty = Value.ITEM\n"
        "    def get_evaluation_sample(self):\n"
        "        return {'input_output': '{}'}\n"
        "def load_code_generation_dataset(release_version):\n"
        "    assert release_version == 'v6'\n"
        "    print('Loaded one problem')\n"
        "    return [Problem()]\n",
        encoding="utf-8",
    )
    (package / "evaluation.py").write_text(
        "import json\n"
        "def codegen_metrics(samples, generations, **kwargs):\n"
        "    passed = generations[0][0] == 'print(1)'\n"
        "    return [{}, {0: [[passed]]}, [[json.dumps({'error_code': -2})]]]\n"
        "def extract_instance_results(results):\n"
        "    return [[all(value > 0 for value in results[0][0])]]\n",
        encoding="utf-8",
    )
    adapter = Path(runner.__file__).with_name("livecodebench_adapter.py")
    list_process = subprocess.run(
        [
            sys.executable,
            str(adapter),
            "--livecodebench-root",
            str(checkout),
            "--release-version",
            "v6",
            "list",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert list_process.returncode == 0
    assert json.loads(list_process.stdout)["cases"][0]["instance_id"] == "official-v6"
    assert "Loaded one problem" in list_process.stderr

    evaluation_input = tmp_path / "evaluation.json"
    evaluation_input.write_text(
        json.dumps([{"instance_id": "official-v6", "code": "print(1)"}]),
        encoding="utf-8",
    )
    evaluate_process = subprocess.run(
        [
            sys.executable,
            str(adapter),
            "--livecodebench-root",
            str(checkout),
            "--release-version",
            "v6",
            "evaluate",
            "--input",
            str(evaluation_input),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert evaluate_process.returncode == 0
    evaluation = json.loads(evaluate_process.stdout)["results"][0]
    assert evaluation["instance_id"] == "official-v6"
    assert evaluation["passed"] is True
