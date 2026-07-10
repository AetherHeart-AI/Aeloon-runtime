from __future__ import annotations

import pytest

from aeloon_core.ablation import run_ablation, summarize_ablation
from aeloon_core.fault_injection import build_fault_scenario, scenario_names


def test_every_fault_scenario_builds_fresh_offline_fixtures() -> None:
    first = build_fault_scenario("fail_n_then_succeed")
    second = build_fault_scenario("fail_n_then_succeed")

    assert len(scenario_names()) == 5
    assert first.provider is not second.provider
    assert first.tools is not second.tools
    assert first.provider.api_key is None


def test_unknown_fault_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fault scenario"):
        build_fault_scenario("unknown")


@pytest.mark.asyncio
async def test_ablation_runs_all_groups_and_exposes_guard_delta() -> None:
    rows = await run_ablation()

    assert len(rows) == len(scenario_names()) * 4
    by_key = {(row["scenario"], row["group"]): row for row in rows}
    assert by_key[("duplicate_loop", "A1")]["success"] is False
    assert by_key[("duplicate_loop", "A2")]["success"] is True
    assert by_key[("duplicate_loop", "A2")]["tokens"]["harness"] > 0
    assert by_key[("duplicate_loop", "A1")]["unproductive_rounds"] == 2
    assert all(
        row["unproductive_rounds"] == 0 for row in rows if row["group"] == "A0"
    )
    assert by_key[("fail_n_then_succeed", "A0")]["success"] is False
    assert by_key[("fail_n_then_succeed", "A1")]["success"] is True
    assert (
        by_key[("fail_n_then_succeed", "A3")]["tokens"]["context_estimated_saved"]
        > by_key[("fail_n_then_succeed", "A2")]["tokens"]["context_estimated_saved"]
    )
    assert all(row["tokens"]["total"] >= 0 for row in rows)

    summary = summarize_ablation(rows)
    assert summary["A0"]["recovery_rate"] == 0.0
    assert summary["A1"]["recovery_rate"] == 0.6
    assert summary["A2"]["recovery_rate"] == 1.0
    assert summary["A3"]["recovery_rate"] == 1.0
