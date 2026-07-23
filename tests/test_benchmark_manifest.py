from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.compare_master_worker_eval import compare_ledgers


def test_master_worker_evaluation_manifest_covers_control_plane_risks() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "benchmarks" / "master_worker_eval_cases.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == 1
    assert {
        "acceptance_passed",
        "master_request_count",
        "end_to_end_duration_ms",
        "input_tokens",
        "output_tokens",
        "revision_count",
    } <= set(payload["metrics"])
    cases = payload["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert {
        "single_node",
        "parallel_dag",
        "research_dynamic_graph",
        "review_revision",
        "budget_partial",
        "request_master",
    } <= {case["category"] for case in cases}


def test_evaluation_comparator_reports_quality_latency_and_cost_deltas() -> None:
    before = {
        "results": [
            {
                "case_id": "one",
                "acceptance_passed": False,
                "master_request_count": 4,
                "end_to_end_duration_ms": 100,
                "input_tokens": 1_000,
                "output_tokens": 200,
                "revision_count": 1,
            },
            {
                "case_id": "two",
                "acceptance_passed": True,
                "master_request_count": 6,
                "end_to_end_duration_ms": 300,
                "input_tokens": 2_000,
                "output_tokens": 400,
                "revision_count": 0,
            },
        ]
    }
    after = {
        "results": [
            {
                "case_id": "one",
                "acceptance_passed": True,
                "master_request_count": 2,
                "end_to_end_duration_ms": 80,
                "input_tokens": 800,
                "output_tokens": 150,
                "revision_count": 0,
            },
            {
                "case_id": "two",
                "acceptance_passed": True,
                "master_request_count": 4,
                "end_to_end_duration_ms": 160,
                "input_tokens": 1_600,
                "output_tokens": 300,
                "revision_count": 0,
            },
        ]
    }

    compared = compare_ledgers(before, after)

    assert compared["delta"]["acceptance_pass_rate"] == pytest.approx(0.5)
    assert compared["delta"]["master_request_count_mean"] == pytest.approx(-2)
    assert compared["delta"]["end_to_end_duration_p50_ms"] == pytest.approx(-80)
    assert compared["delta"]["input_tokens_mean"] == pytest.approx(-300)
