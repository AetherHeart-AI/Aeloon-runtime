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
DEFAULT_MODEL = "deepseek-v4-flash"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _model_name(value: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("model name must not be empty")
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
        "--model",
        type=_model_name,
        default=DEFAULT_MODEL,
        help=f"Model name passed to every selected harness (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Aeloon Runtime config JSON path (used by the aeloon harness).",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Maximum concurrent benchmark cases (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Run only the first N deterministically selected cases.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    harness_names = [name for group in args.harness for name in group]
    started = time.monotonic()
    info(
        "Selected benchmark=%s harnesses=%s model=%s workers=%d limit=%s",
        args.benchmark,
        ", ".join(harness_names),
        args.model,
        args.workers,
        args.limit if args.limit is not None else "all",
    )
    harnesses = get_harnesses(
        harness_names,
        project_root=PROJECT_ROOT,
        model=args.model,
    )
    adapter = get_adapter(
        args.benchmark,
        project_root=PROJECT_ROOT,
        limit=args.limit,
        workers=args.workers,
        config_path=args.config,
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
