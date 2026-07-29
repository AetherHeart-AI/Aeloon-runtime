"""Unified benchmark entry point."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from benchmarks.adapters import BENCHMARK_NAMES, get_adapter
from benchmarks.harness import HARNESS_NAMES, get_harnesses
from benchmarks.progress import configure_progress, info

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Prepare and run one official benchmark through one or more coding harnesses.")
    )
    parser.add_argument(
        "--harness",
        action="append",
        nargs="+",
        choices=(*HARNESS_NAMES, "all"),
        required=True,
        help=(
            "Harness name(s). Repeat the option or provide several values; use "
            "'all' for every registered harness."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_NAMES,
        required=True,
        help="Official benchmark adapter to prepare and run.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Maximum concurrent benchmark cases (default: 1).",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help=(
            "Resume an existing benchmark run by id. Completed records and "
            "generation checkpoints are reused."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    harness_names = [name for group in args.harness for name in group]
    started = time.monotonic()
    info(
        "Selected benchmark=%s harnesses=%s workers=%d",
        args.benchmark,
        ", ".join(harness_names),
        args.workers,
    )
    harnesses = get_harnesses(harness_names, project_root=PROJECT_ROOT)
    adapter = get_adapter(
        args.benchmark,
        project_root=PROJECT_ROOT,
        workers=args.workers,
        resume_run_id=args.resume,
    )
    info("Preparing benchmark environment...")
    adapter.prepare()
    info(
        "Preparation completed; source=%s",
        adapter.run.source_dir,
    )
    info("Starting benchmark execution; results=%s", adapter.run.output_dir)
    summary = adapter.execute(harnesses)
    info(
        "Benchmark completed in %.1fs; results=%s",
        time.monotonic() - started,
        adapter.run.output_dir,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_progress()
    try:
        summary = run(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
