"""Legacy-compatible LiveCodeBench runner used by the unified adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = Path(__file__).with_name("livecodebench_adapter.py")
DEFAULT_SCENARIOS = ("code-generation", "self-repair")
RELEASE_V6_NEW = "v6"
RELEASE_V6_ALL = "release_v6"
MAX_CAPTURE_CHARS = 20_000
DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class LiveCodeBenchCase:
    """Prompt metadata for one official LiveCodeBench problem."""

    instance_id: str
    question_title: str
    question_content: str
    starter_code: str
    platform: str
    contest_id: str
    contest_date: str
    difficulty: str


@dataclass(frozen=True)
class ProcessOutcome:
    """Bounded result of one child process."""

    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


@dataclass(frozen=True)
class AgentGeneration:
    """One Aeloon invocation and the Python code extracted from it."""

    prompt: str
    process: ProcessOutcome
    payload: dict[str, Any] | None
    payload_error: str | None
    code: str
    extraction_error: str | None


def load_cases(
    livecodebench_root: Path,
    livecodebench_python: Path,
    release_version: str,
    *,
    adapter_timeout: float = 7200.0,
) -> list[LiveCodeBenchCase]:
    """Load public problem metadata through the official environment."""

    payload = _run_adapter(
        livecodebench_root=livecodebench_root,
        livecodebench_python=livecodebench_python,
        release_version=release_version,
        command="list",
        process_timeout=adapter_timeout,
    )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise RuntimeError("LiveCodeBench adapter list response has no cases.")

    cases: list[LiveCodeBenchCase] = []
    seen: set[str] = set()
    required = {
        "instance_id",
        "question_title",
        "question_content",
        "starter_code",
        "platform",
        "contest_id",
        "contest_date",
        "difficulty",
    }
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict) or not required <= item.keys():
            raise RuntimeError(f"Invalid LiveCodeBench case at index {index}.")
        values = {key: item[key] for key in required}
        if not all(isinstance(value, str) for value in values.values()):
            raise RuntimeError(f"LiveCodeBench case {index} contains non-string fields.")
        instance_id = values["instance_id"]
        if instance_id in seen:
            raise RuntimeError(f"Duplicate LiveCodeBench instance id: {instance_id}")
        seen.add(instance_id)
        cases.append(LiveCodeBenchCase(**values))
    return sorted(cases, key=lambda case: case.instance_id)


def select_cases(
    cases: list[LiveCodeBenchCase],
    *,
    instance_ids: list[str],
    limit: int | None,
) -> list[LiveCodeBenchCase]:
    selected = [
        case for case in cases if not instance_ids or case.instance_id in instance_ids
    ]
    if instance_ids:
        missing = sorted(set(instance_ids) - {case.instance_id for case in selected})
        if missing:
            raise RuntimeError(f"Unknown LiveCodeBench cases: {', '.join(missing)}")
    return selected[:limit] if limit is not None else selected


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected v6 cases and append durable records to a JSONL ledger."""

    benchmark_root = args.livecodebench_root.expanduser().resolve()
    benchmark_python = _resolve_livecodebench_python(args.livecodebench_python)
    release_version = RELEASE_V6_ALL if args.all else args.release_version
    scenarios = tuple(dict.fromkeys(args.scenario or DEFAULT_SCENARIOS))
    cases = select_cases(
        load_cases(
            benchmark_root,
            benchmark_python,
            release_version,
            adapter_timeout=args.adapter_timeout,
        ),
        instance_ids=args.case,
        limit=args.limit,
    )
    if not cases:
        raise RuntimeError("No LiveCodeBench cases matched the filters.")

    if args.list:
        return {
            "schema_version": 1,
            "benchmark": "livecodebench",
            "release_version": release_version,
            "scenarios": list(scenarios),
            "cases": [_case_metadata(case) for case in cases],
        }

    results_path = args.results.expanduser().resolve()
    completed = _prepare_results_file(
        results_path,
        resume=args.resume,
        overwrite=args.overwrite,
        release_version=release_version,
    )
    selected_keys = {
        (scenario, case.instance_id) for scenario in scenarios for case in cases
    }
    previously_completed = selected_keys & completed.keys()

    baseline_by_id = _load_reusable_baselines(completed, cases)
    baseline_needed = [
        case
        for case in cases
        if case.instance_id not in baseline_by_id
        and any(
            (scenario, case.instance_id) not in completed for scenario in scenarios
        )
    ]
    new_baselines = _generate_baselines(
        baseline_needed,
        args=args,
        release_version=release_version,
    )
    if new_baselines:
        evaluations = _evaluate_generations(
            new_baselines,
            livecodebench_root=benchmark_root,
            livecodebench_python=benchmark_python,
            release_version=release_version,
            results_path=results_path,
            num_processes=args.num_process_evaluate,
            timeout=args.test_timeout,
            adapter_timeout=args.adapter_timeout,
        )
        for case in baseline_needed:
            generation = new_baselines[case.instance_id]
            baseline_by_id[case.instance_id] = _baseline_data(
                generation, evaluations[case.instance_id]
            )

    executed_records: list[dict[str, Any]] = []
    if "code-generation" in scenarios:
        for case in cases:
            key = ("code-generation", case.instance_id)
            if key in completed:
                continue
            baseline = baseline_by_id[case.instance_id]
            record = _code_generation_record(
                case,
                release_version=release_version,
                config_path=args.config,
                baseline=baseline,
            )
            _append_jsonl(results_path, record)
            completed[key] = record
            executed_records.append(record)

    if "self-repair" in scenarios:
        pending_repair = [
            case
            for case in cases
            if ("self-repair", case.instance_id) not in completed
        ]
        repair_generations = _generate_repairs(
            pending_repair,
            baseline_by_id=baseline_by_id,
            args=args,
            release_version=release_version,
        )
        attempted = {
            instance_id: generation
            for instance_id, generation in repair_generations.items()
            if generation is not None
        }
        repair_evaluations = (
            _evaluate_generations(
                attempted,
                livecodebench_root=benchmark_root,
                livecodebench_python=benchmark_python,
                release_version=release_version,
                results_path=results_path,
                num_processes=args.num_process_evaluate,
                timeout=args.test_timeout,
                adapter_timeout=args.adapter_timeout,
            )
            if attempted
            else {}
        )
        for case in pending_repair:
            baseline = baseline_by_id[case.instance_id]
            generation = repair_generations[case.instance_id]
            evaluation = (
                baseline["evaluation"]
                if generation is None
                else repair_evaluations[case.instance_id]
            )
            record = _self_repair_record(
                case,
                release_version=release_version,
                config_path=args.config,
                baseline=baseline,
                generation=generation,
                evaluation=evaluation,
            )
            _append_jsonl(results_path, record)
            completed[("self-repair", case.instance_id)] = record
            executed_records.append(record)

    selected_records = [
        completed[(scenario, case.instance_id)]
        for scenario in scenarios
        for case in cases
        if (scenario, case.instance_id) in completed
    ]
    return _summary(
        cases=cases,
        scenarios=scenarios,
        release_version=release_version,
        selected_records=selected_records,
        executed_records=executed_records,
        skipped_completed=len(previously_completed),
        results_path=results_path,
    )


def _generate_baselines(
    cases: list[LiveCodeBenchCase],
    *,
    args: argparse.Namespace,
    release_version: str,
) -> dict[str, AgentGeneration]:
    generations: dict[str, AgentGeneration] = {}
    for index, case in enumerate(cases, start=1):
        print(
            f"[code-generation {index}/{len(cases)}] {case.instance_id}",
            file=sys.stderr,
            flush=True,
        )
        prompt = _code_generation_prompt(case)
        generations[case.instance_id] = _invoke_agent(
            prompt,
            case=case,
            scenario="code-generation",
            args=args,
            release_version=release_version,
        )
    return generations


def _generate_repairs(
    cases: list[LiveCodeBenchCase],
    *,
    baseline_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    release_version: str,
) -> dict[str, AgentGeneration | None]:
    generations: dict[str, AgentGeneration | None] = {}
    attempted_cases = [
        case
        for case in cases
        if not bool(baseline_by_id[case.instance_id]["evaluation"]["oracle_passed"])
    ]
    attempted_index = 0
    for case in cases:
        baseline = baseline_by_id[case.instance_id]
        if baseline["evaluation"]["oracle_passed"]:
            generations[case.instance_id] = None
            continue
        attempted_index += 1
        print(
            f"[self-repair {attempted_index}/{len(attempted_cases)}] {case.instance_id}",
            file=sys.stderr,
            flush=True,
        )
        prompt = _self_repair_prompt(
            case,
            code=str(baseline["code"]),
            metadata=baseline["evaluation"]["metadata"],
        )
        generations[case.instance_id] = _invoke_agent(
            prompt,
            case=case,
            scenario="self-repair",
            args=args,
            release_version=release_version,
        )
    return generations


def _invoke_agent(
    prompt: str,
    *,
    case: LiveCodeBenchCase,
    scenario: str,
    args: argparse.Namespace,
    release_version: str,
) -> AgentGeneration:
    workspace_root = args.workspace_dir.expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    session_root = (
        args.results.expanduser().resolve().parent
        / f"{args.results.stem}.sessions"
        / release_version
        / scenario
        / _safe_id(case.instance_id)
    )
    command = [
        sys.executable,
        "-m",
        "aeloon_core",
        "run",
        "--stdin",
        "--output",
        "json",
        "--data-dir",
        str(session_root),
        "--model",
        args.model,
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config.expanduser().resolve())])

    with tempfile.TemporaryDirectory(
        prefix=f".{scenario}-",
        dir=workspace_root,
    ) as temporary_workspace:
        command.extend(["--workspace", temporary_workspace])
        process = _run_process(
            command,
            cwd=PROJECT_ROOT,
            timeout=args.agent_timeout,
            input_text=prompt,
        )

    payload, payload_error = _parse_agent_payload(process)
    final_content = (payload or {}).get("final_content")
    code, extraction_error = _extract_python_code(final_content)
    return AgentGeneration(
        prompt=prompt,
        process=process,
        payload=payload,
        payload_error=payload_error,
        code=code,
        extraction_error=extraction_error,
    )


def _evaluate_generations(
    generations: dict[str, AgentGeneration],
    *,
    livecodebench_root: Path,
    livecodebench_python: Path,
    release_version: str,
    results_path: Path,
    num_processes: int,
    timeout: int,
    adapter_timeout: float,
) -> dict[str, dict[str, Any]]:
    if not generations:
        return {}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"instance_id": instance_id, "code": generation.code}
        for instance_id, generation in generations.items()
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=".livecodebench-eval-",
        dir=results_path.parent,
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        input_path = Path(stream.name)
    try:
        response = _run_adapter(
            livecodebench_root=livecodebench_root,
            livecodebench_python=livecodebench_python,
            release_version=release_version,
            command="evaluate",
            input_path=input_path,
            num_processes=num_processes,
            timeout=timeout,
            process_timeout=adapter_timeout,
        )
    finally:
        input_path.unlink(missing_ok=True)

    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("LiveCodeBench adapter evaluate response has no results.")
    evaluations: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("instance_id"), str)
            or not isinstance(item.get("passed"), bool)
            or not isinstance(item.get("metadata"), dict)
        ):
            raise RuntimeError("LiveCodeBench adapter returned an invalid evaluation.")
        instance_id = item["instance_id"]
        generation = generations.get(instance_id)
        if generation is None:
            raise RuntimeError(f"LiveCodeBench adapter returned unexpected case: {instance_id}")
        oracle_passed = item["passed"]
        evaluations[instance_id] = {
            "passed": _agent_status(generation) == "completed" and oracle_passed,
            "oracle_passed": oracle_passed,
            "metadata": item["metadata"],
        }
    missing = sorted(set(generations) - evaluations.keys())
    if missing:
        raise RuntimeError(
            f"LiveCodeBench adapter omitted evaluations: {', '.join(missing)}"
        )
    return evaluations


def _run_adapter(
    *,
    livecodebench_root: Path,
    livecodebench_python: Path,
    release_version: str,
    command: str,
    input_path: Path | None = None,
    num_processes: int = 4,
    timeout: int = 6,
    process_timeout: float = 7200.0,
) -> dict[str, Any]:
    adapter_command = [
        str(livecodebench_python),
        str(ADAPTER_PATH),
        "--livecodebench-root",
        str(livecodebench_root),
        "--release-version",
        release_version,
        command,
    ]
    if command == "evaluate":
        if input_path is None:
            raise RuntimeError("Adapter evaluation requires an input file.")
        adapter_command.extend(
            [
                "--input",
                str(input_path),
                "--num-processes",
                str(num_processes),
                "--timeout",
                str(timeout),
            ]
        )
    outcome = _run_process(
        adapter_command,
        cwd=livecodebench_root,
        timeout=process_timeout,
    )
    if outcome.timed_out:
        raise RuntimeError("LiveCodeBench adapter timed out.")
    if outcome.returncode != 0:
        detail = outcome.stderr.strip() or outcome.stdout.strip()
        raise RuntimeError(f"LiveCodeBench adapter failed: {_bounded(detail)}")
    try:
        payload = json.loads(outcome.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LiveCodeBench adapter returned invalid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise RuntimeError("LiveCodeBench adapter response must be a JSON object.")
    return payload


def _code_generation_prompt(case: LiveCodeBenchCase) -> str:
    prompt = (
        "You are an expert Python programmer. You will be given a problem "
        "specification and must generate a correct Python program that passes all "
        "tests. Return the complete solution in exactly one fenced Python code block.\n\n"
        f"### Question:\n{case.question_content}\n\n"
    )
    if case.starter_code:
        prompt += (
            "### Starter code:\n"
            "Use this starter code to write the solution.\n"
            f"```python\n{case.starter_code}\n```\n\n"
        )
    else:
        prompt += (
            "Read input from stdin, solve the problem, and write the answer to stdout. "
            "Do not hard-code the sample inputs.\n\n"
        )
    return prompt + "### Answer:\n"


def _self_repair_prompt(
    case: LiveCodeBenchCase,
    *,
    code: str,
    metadata: dict[str, Any],
) -> str:
    feedback = _official_feedback(metadata)
    return (
        "You are an expert Python programmer repairing an incorrect solution. "
        "Briefly identify the defect, then return the complete fixed program in "
        "exactly one fenced Python code block.\n\n"
        f"### Question:\n{case.question_content}\n\n"
        f"### Incorrect solution:\n```python\n{code}\n```\n\n"
        f"### Test feedback:\n{feedback}\n\n"
        "The fixed program must satisfy the original input/output contract.\n\n"
        "### Answer:\n"
    )


def _official_feedback(metadata: dict[str, Any]) -> str:
    error_code = metadata.get("error_code")
    if error_code == -1:
        return (
            "The code failed to complete within the global time limit.\n"
            f"{metadata.get('error', '')}"
        ).strip()
    if error_code == -2:
        return (
            "The code produced a wrong answer.\n"
            f"Input: {metadata.get('inputs', '')}\n"
            f"Generated output: {metadata.get('output', '')}\n"
            f"Expected: {metadata.get('expected', '')}"
        )
    if error_code == -3:
        return (
            "The code exceeded the time limit.\n"
            f"{metadata.get('error', '')}\n"
            f"Input: {metadata.get('inputs', '')}\n"
            f"Expected: {metadata.get('expected', '')}"
        )
    if error_code == -4:
        return (
            "The code raised a runtime error.\n"
            f"Input: {metadata.get('inputs', '')}\n"
            f"Expected: {metadata.get('expected', '')}\n"
            f"{metadata.get('error', '')}"
        ).strip()
    return (
        f"{metadata.get('error_message', 'The code failed the official tests.')} "
        f"{metadata.get('error', '')}"
    ).strip()


def _extract_python_code(value: Any) -> tuple[str, str | None]:
    if not isinstance(value, str) or not value.strip():
        return "", "agent final_content was empty"
    matches = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        code = matches[-1].strip()
        return (code, None) if code else ("", "last fenced code block was empty")

    stripped = value.strip()
    if stripped.lower().startswith("python\n"):
        stripped = stripped.split("\n", 1)[1].strip()
    return stripped, "agent final_content had no fenced Python code block"


def _baseline_data(
    generation: AgentGeneration,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": generation.prompt,
        "code": generation.code,
        "extraction_error": generation.extraction_error,
        "agent": _agent_record(generation),
        "evaluation": evaluation,
    }


def _load_reusable_baselines(
    completed: dict[tuple[str, str], dict[str, Any]],
    cases: list[LiveCodeBenchCase],
) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for case in cases:
        codegen = completed.get(("code-generation", case.instance_id))
        if codegen is not None:
            baseline = _baseline_from_code_generation_record(codegen)
            if baseline is not None:
                baselines[case.instance_id] = baseline
                continue
        repair = completed.get(("self-repair", case.instance_id))
        if repair is not None and isinstance(repair.get("baseline"), dict):
            baseline = repair["baseline"]
            if _valid_baseline(baseline):
                baselines[case.instance_id] = baseline
    return baselines


def _baseline_from_code_generation_record(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    generation = record.get("generation")
    evaluation = record.get("evaluation")
    agent = record.get("agent")
    if not isinstance(generation, dict):
        return None
    baseline = {
        "prompt": record.get("prompt"),
        "code": generation.get("code"),
        "extraction_error": generation.get("extraction_error"),
        "agent": agent,
        "evaluation": evaluation,
    }
    return baseline if _valid_baseline(baseline) else None


def _valid_baseline(value: dict[str, Any]) -> bool:
    evaluation = value.get("evaluation")
    return (
        isinstance(value.get("code"), str)
        and isinstance(value.get("agent"), dict)
        and isinstance(evaluation, dict)
        and isinstance(evaluation.get("oracle_passed"), bool)
        and isinstance(evaluation.get("metadata"), dict)
    )


def _code_generation_record(
    case: LiveCodeBenchCase,
    *,
    release_version: str,
    config_path: Path | None,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_record_header(
            case,
            scenario="code-generation",
            release_version=release_version,
            config_path=config_path,
        ),
        "prompt": baseline.get("prompt"),
        "agent": baseline["agent"],
        "generation": {
            "code": baseline["code"],
            "extraction_error": baseline["extraction_error"],
        },
        "evaluation": baseline["evaluation"],
        "false_completed": (
            baseline["agent"].get("status") == "completed"
            and not baseline["evaluation"]["oracle_passed"]
        ),
    }


def _self_repair_record(
    case: LiveCodeBenchCase,
    *,
    release_version: str,
    config_path: Path | None,
    baseline: dict[str, Any],
    generation: AgentGeneration | None,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    repair_attempted = generation is not None
    agent = _agent_record(generation) if generation is not None else None
    code = generation.code if generation is not None else baseline["code"]
    extraction_error = (
        generation.extraction_error
        if generation is not None
        else baseline["extraction_error"]
    )
    baseline_snapshot = {
        "prompt": baseline.get("prompt"),
        "code": baseline["code"],
        "extraction_error": baseline["extraction_error"],
        "agent": baseline["agent"],
        "evaluation": baseline["evaluation"],
    }
    return {
        **_record_header(
            case,
            scenario="self-repair",
            release_version=release_version,
            config_path=config_path,
        ),
        "prompt": generation.prompt if generation is not None else None,
        "repair_attempted": repair_attempted,
        "agent": agent,
        "baseline": baseline_snapshot,
        "generation": {
            "code": code,
            "extraction_error": extraction_error,
        },
        "evaluation": evaluation,
        "repaired": (
            not baseline["evaluation"]["oracle_passed"]
            and evaluation["oracle_passed"]
        ),
        "false_completed": (
            agent is not None
            and agent.get("status") == "completed"
            and not evaluation["oracle_passed"]
        ),
    }


def _record_header(
    case: LiveCodeBenchCase,
    *,
    scenario: str,
    release_version: str,
    config_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "benchmark": "livecodebench",
        "release_version": release_version,
        "scenario": scenario,
        **_case_metadata(case),
        "aeloon_core_commit": _git_revision(PROJECT_ROOT),
        "config_path": (
            str(config_path.expanduser().resolve()) if config_path is not None else None
        ),
    }


def _case_metadata(case: LiveCodeBenchCase) -> dict[str, str]:
    return {
        "instance_id": case.instance_id,
        "question_title": case.question_title,
        "question_content": case.question_content,
        "starter_code": case.starter_code,
        "platform": case.platform,
        "contest_id": case.contest_id,
        "contest_date": case.contest_date,
        "difficulty": case.difficulty,
    }


def _agent_record(generation: AgentGeneration) -> dict[str, Any]:
    payload = generation.payload or {}
    return {
        "status": _agent_status(generation),
        "returncode": generation.process.returncode,
        "timed_out": generation.process.timed_out,
        "wall_time_ms": generation.process.duration_ms,
        "session_id": payload.get("session_id"),
        "duration_ms": payload.get("duration_ms"),
        "final_content": payload.get("final_content"),
        "tools_used": payload.get("tools_used", []),
        "usage": payload.get("usage", {}),
        "model": payload.get("model"),
        "stdout": _bounded(generation.process.stdout) if generation.payload_error else None,
        "stderr": _bounded(generation.process.stderr),
        "payload_error": generation.payload_error,
    }


def _agent_status(generation: AgentGeneration) -> str:
    if generation.process.timed_out:
        return "timeout"
    if generation.process.returncode != 0:
        return "process_error"
    if generation.payload is None:
        return "invalid_output"
    return str(generation.payload.get("status") or "unknown")


def _summary(
    *,
    cases: list[LiveCodeBenchCase],
    scenarios: tuple[str, ...],
    release_version: str,
    selected_records: list[dict[str, Any]],
    executed_records: list[dict[str, Any]],
    skipped_completed: int,
    results_path: Path,
) -> dict[str, Any]:
    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        records = [
            record for record in selected_records if record.get("scenario") == scenario
        ]
        passed = sum(
            bool(record.get("evaluation", {}).get("passed")) for record in records
        )
        scenario_summary: dict[str, Any] = {
            "recorded_cases": len(records),
            "passed": passed,
            "pass_rate": passed / len(records) if records else None,
        }
        if scenario == "self-repair":
            scenario_summary["repaired"] = sum(
                bool(record.get("repaired")) for record in records
            )
            scenario_summary["repair_attempts"] = sum(
                bool(record.get("repair_attempted")) for record in records
            )
        scenario_summaries[scenario] = scenario_summary
    return {
        "schema_version": 1,
        "benchmark": "livecodebench",
        "release_version": release_version,
        "selected_cases": len(cases),
        "selected_scenarios": list(scenarios),
        "skipped_completed": skipped_completed,
        "executed_records": len(executed_records),
        "scenarios": scenario_summaries,
        "results": str(results_path),
    }


def _prepare_results_file(
    path: Path,
    *,
    resume: bool,
    overwrite: bool,
    release_version: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {}
    if overwrite:
        path.write_text("", encoding="utf-8")
        return {}
    if not resume:
        raise RuntimeError(
            f"Results file already exists: {path}. Use --resume or --overwrite."
        )

    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from None
        if (
            isinstance(record, dict)
            and record.get("benchmark") == "livecodebench"
            and record.get("release_version") == release_version
            and record.get("scenario") in DEFAULT_SCENARIOS
            and isinstance(record.get("instance_id"), str)
        ):
            completed[(record["scenario"], record["instance_id"])] = record
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
) -> ProcessOutcome:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return ProcessOutcome(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessOutcome(
            returncode=None,
            stdout=_process_text(exc.stdout),
            stderr=_process_text(exc.stderr),
            duration_ms=round((time.monotonic() - started) * 1000),
            timed_out=True,
        )


def _parse_agent_payload(
    outcome: ProcessOutcome,
) -> tuple[dict[str, Any] | None, str | None]:
    if outcome.timed_out:
        return None, "agent process timed out"
    if outcome.returncode != 0:
        return None, f"agent process exited with code {outcome.returncode}"
    try:
        payload = json.loads(outcome.stdout)
    except json.JSONDecodeError as exc:
        return None, f"agent stdout was not JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "agent JSON payload was not an object"
    return payload, None


def _resolve_livecodebench_python(configured: Path | None) -> Path:
    if configured is not None:
        executable = configured.expanduser().resolve()
    else:
        executable = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not executable.is_file():
            executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise RuntimeError(f"LiveCodeBench Python executable does not exist: {executable}")
    return executable


def _git_revision(workspace: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _safe_id(instance_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id).strip("._")
    return sanitized or "case"


def _bounded(value: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"[... {omitted} earlier characters omitted ...]\n{value[-limit:]}"


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LiveCodeBench v6 code-generation and self-repair tests through "
            "Aeloon Core."
        )
    )
    parser.add_argument(
        "--livecodebench-root",
        type=Path,
        required=True,
        help="Official LiveCodeBench/LiveCodeBench checkout.",
    )
    parser.add_argument(
        "--livecodebench-python",
        type=Path,
        default=None,
        help="Python with LiveCodeBench dependencies (default: worktree .venv).",
    )
    release = parser.add_mutually_exclusive_group()
    release.add_argument(
        "--release-version",
        choices=(RELEASE_V6_NEW, RELEASE_V6_ALL),
        default=RELEASE_V6_NEW,
        help="Official dataset slice (default: v6, only newly added v6 cases).",
    )
    release.add_argument(
        "--all",
        action="store_true",
        help="Run cumulative release_v6 instead of only the newly added v6 slice.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=DEFAULT_SCENARIOS,
        default=None,
        help="Scenario to run; repeat as needed (default: both).",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this exact question id; repeat to select more.",
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected cases without creating workspaces or running agents.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Aeloon model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Aeloon Core config JSON path.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=PROJECT_ROOT / ".benchmark-workspaces" / "livecodebench",
        help="Parent for disposable per-call agent workspaces.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results" / "livecodebench-v6.jsonl",
        help="Append-only result ledger.",
    )
    parser.add_argument("--agent-timeout", type=_positive_timeout, default=900.0)
    parser.add_argument(
        "--adapter-timeout",
        type=_positive_timeout,
        default=7200.0,
        help="Total timeout for official dataset loading or an evaluation batch.",
    )
    parser.add_argument("--test-timeout", type=_positive_int, default=6)
    parser.add_argument("--num-process-evaluate", type=_positive_int, default=4)
    result_mode = parser.add_mutually_exclusive_group()
    result_mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenario/case records already present in the result ledger.",
    )
    result_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate an existing result ledger before running.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = run_benchmark(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
