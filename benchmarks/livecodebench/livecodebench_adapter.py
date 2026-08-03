"""Bridge benchmark adapters to an official LiveCodeBench checkout.

This module is intentionally executed with the Python environment belonging to
the LiveCodeBench checkout.  Keeping the adapter in a separate process avoids
adding LiveCodeBench's heavyweight inference and evaluation dependencies to
Aeloon Core itself.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any


def _activate_checkout(root: Path) -> Path:
    checkout = root.expanduser().resolve()
    package = checkout / "lcb_runner"
    if not package.is_dir():
        raise RuntimeError(f"Missing official LiveCodeBench package: {package}")
    sys.path.insert(0, str(checkout))
    return checkout


def _load_problems(root: Path, release_version: str) -> list[Any]:
    _activate_checkout(root)
    try:
        from lcb_runner.benchmarks import load_code_generation_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Could not import LiveCodeBench with the selected Python environment. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    # The official loader prints its case count. Keep stdout machine-readable.
    with contextlib.redirect_stdout(sys.stderr):
        problems = load_code_generation_dataset(release_version)
    return sorted(problems, key=lambda problem: str(problem.question_id))


def list_cases(root: Path, release_version: str) -> dict[str, Any]:
    problems = _load_problems(root, release_version)
    return {
        "schema_version": 1,
        "release_version": release_version,
        "cases": [
            {
                "instance_id": str(problem.question_id),
                "question_title": str(problem.question_title),
                "question_content": str(problem.question_content),
                "starter_code": str(problem.starter_code or ""),
                "platform": _enum_value(problem.platform),
                "contest_id": str(problem.contest_id),
                "contest_date": problem.contest_date.isoformat(),
                "difficulty": _enum_value(problem.difficulty),
            }
            for problem in problems
        ],
    }


def evaluate_codes(
    root: Path,
    release_version: str,
    input_path: Path,
    *,
    num_processes: int,
    timeout: int,
) -> dict[str, Any]:
    problems = _load_problems(root, release_version)
    problem_by_id = {str(problem.question_id): problem for problem in problems}
    requested = _read_evaluation_input(input_path)

    missing = sorted(
        {
            item["instance_id"]
            for item in requested
            if item["instance_id"] not in problem_by_id
        }
    )
    if missing:
        raise RuntimeError(f"Unknown LiveCodeBench cases: {', '.join(missing)}")

    selected = [problem_by_id[item["instance_id"]] for item in requested]
    samples = [problem.get_evaluation_sample() for problem in selected]
    generations = [[item["code"]] for item in requested]

    try:
        from lcb_runner.evaluation import codegen_metrics, extract_instance_results
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the official LiveCodeBench evaluator."
        ) from exc

    # The official evaluator reports progress and aggregate metrics to stdout.
    # The adapter reserves stdout for its JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        metrics = codegen_metrics(
            samples,
            generations,
            num_process_evaluate=num_processes,
            timeout=timeout,
        )
        graded = extract_instance_results(metrics[1])

    results = []
    for index, item in enumerate(requested):
        metadata = _decode_metadata(metrics[2][index][0])
        results.append(
            {
                "instance_id": item["instance_id"],
                "passed": bool(graded[index][0]),
                "metadata": metadata,
            }
        )
    return {
        "schema_version": 1,
        "release_version": release_version,
        "results": results,
    }


def _read_evaluation_input(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid evaluation input {path}: {exc}") from None
    if not isinstance(payload, list):
        raise RuntimeError("Evaluation input must be a JSON list.")

    requested: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"Evaluation input item {index} must be an object.")
        instance_id = item.get("instance_id")
        code = item.get("code")
        if not isinstance(instance_id, str) or not isinstance(code, str):
            raise RuntimeError(
                f"Evaluation input item {index} requires string instance_id and code."
            )
        if instance_id in seen:
            raise RuntimeError(f"Duplicate evaluation instance id: {instance_id}")
        seen.add(instance_id)
        requested.append({"instance_id": instance_id, "code": code})
    return requested


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"error": value, "error_code": -5, "error_message": "InvalidMetadata"}
    if isinstance(value, dict):
        return value
    return {
        "error": repr(value),
        "error_code": -5,
        "error_message": "InvalidMetadata",
    }


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an official LiveCodeBench checkout to list or evaluate cases."
    )
    parser.add_argument(
        "--livecodebench-root",
        type=Path,
        required=True,
        help="Official LiveCodeBench/LiveCodeBench checkout.",
    )
    parser.add_argument(
        "--release-version",
        choices=("v6", "release_v6"),
        default="v6",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--num-processes", type=_positive_int, default=4)
    evaluate.add_argument("--timeout", type=_positive_int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            payload = list_cases(args.livecodebench_root, args.release_version)
        else:
            payload = evaluate_codes(
                args.livecodebench_root,
                args.release_version,
                args.input,
                num_processes=args.num_processes,
                timeout=args.timeout,
            )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
