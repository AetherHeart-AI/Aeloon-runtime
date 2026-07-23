"""Compare before/after result ledgers for the Master–Worker evaluation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

REQUIRED_FIELDS = (
    "acceptance_passed",
    "master_request_count",
    "end_to_end_duration_ms",
    "input_tokens",
    "output_tokens",
    "revision_count",
)


def compare_ledgers(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Validate two ledgers and return stable aggregate deltas."""

    before_rows = _indexed_rows(before)
    after_rows = _indexed_rows(after)
    if set(before_rows) != set(after_rows):
        raise ValueError("before and after ledgers must contain the same case_id values")
    ordered_ids = sorted(before_rows)
    before_summary = summarize([before_rows[case_id] for case_id in ordered_ids])
    after_summary = summarize([after_rows[case_id] for case_id in ordered_ids])
    return {
        "case_count": len(ordered_ids),
        "before": before_summary,
        "after": after_summary,
        "delta": {
            key: after_summary[key] - before_summary[key]
            for key in before_summary
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate one complete evaluation run."""

    if not rows:
        raise ValueError("an evaluation ledger requires at least one result")
    for row in rows:
        _validate_row(row)
    durations = sorted(float(row["end_to_end_duration_ms"]) for row in rows)
    return {
        "acceptance_pass_rate": mean(
            1.0 if row["acceptance_passed"] else 0.0 for row in rows
        ),
        "master_request_count_mean": mean(
            float(row["master_request_count"]) for row in rows
        ),
        "end_to_end_duration_p50_ms": _percentile(durations, 0.50),
        "end_to_end_duration_p95_ms": _percentile(durations, 0.95),
        "input_tokens_mean": mean(float(row["input_tokens"]) for row in rows),
        "output_tokens_mean": mean(float(row["output_tokens"]) for row in rows),
        "revision_rate": mean(
            1.0 if row["revision_count"] > 0 else 0.0 for row in rows
        ),
    }


def _indexed_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ledger must contain a nonempty results array")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each result must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each result requires a nonempty case_id")
        if case_id in indexed:
            raise ValueError(f"duplicate case_id: {case_id}")
        _validate_row(row)
        indexed[case_id] = row
    return indexed


def _validate_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("acceptance_passed"), bool):
        raise ValueError("acceptance_passed must be boolean")
    for field in REQUIRED_FIELDS[1:]:
        value = row.get(field)
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a nonnegative number")


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    print(json.dumps(compare_ledgers(before, after), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
