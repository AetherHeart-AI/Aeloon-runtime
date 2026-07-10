"""Deterministic A0-A3 UASM fault-injection ablation runner."""

from __future__ import annotations

from typing import Any

from aeloon_core.fault_injection import build_fault_scenario, scenario_names
from aeloon_core.kernel import run_agent_kernel
from aeloon_core.state_machine import run_agent_loop
from aeloon_core.transitions import NodeKind

GROUPS: dict[str, dict[str, bool]] = {
    "A0": {
        "uasm": True,
        "rule_engine_enabled": False,
        "temporary_guard_enabled": False,
        "minimal_context_enabled": False,
    },
    "A1": {
        "uasm": False,
        "rule_engine_enabled": True,
        "temporary_guard_enabled": False,
        "minimal_context_enabled": False,
    },
    "A2": {
        "uasm": True,
        "rule_engine_enabled": True,
        "temporary_guard_enabled": True,
        "minimal_context_enabled": False,
    },
    "A3": {
        "uasm": True,
        "rule_engine_enabled": True,
        "temporary_guard_enabled": True,
        "minimal_context_enabled": True,
    },
}


async def run_ablation() -> list[dict[str, Any]]:
    """Run every scenario/group pair with fresh deterministic fixtures."""

    rows: list[dict[str, Any]] = []
    for scenario_name in scenario_names():
        for group, settings in GROUPS.items():
            scenario = build_fault_scenario(scenario_name)
            messages = _benchmark_messages(scenario_name)
            if settings["uasm"]:
                state = await run_agent_loop(
                    provider=scenario.provider,
                    model="offline-script",
                    tools=scenario.tools,
                    messages=messages,
                    max_iterations=8,
                    max_auto_continue_iterations=0,
                    max_finalization_iterations=1,
                    rule_engine_enabled=settings["rule_engine_enabled"],
                    temporary_guard_enabled=settings["temporary_guard_enabled"],
                    minimal_context_enabled=settings["minimal_context_enabled"],
                    guard_decision_mode="binary",
                    experiment_labels={"group": group, "scenario": scenario_name},
                )
                final_content = state.metadata.final_content
                iterations = state.metadata.iteration
                transitions = len(state.transitions)
                domain_tokens = state.token_ledger.for_kind(NodeKind.DOMAIN).get(
                    "total_tokens", 0
                )
                harness_tokens = state.token_ledger.for_kind(NodeKind.HARNESS).get(
                    "total_tokens", 0
                )
                context_usage = state.token_ledger.for_kind(NodeKind.CONTEXT_PROCESSING)
                context_tokens = context_usage.get("total_tokens", 0)
                context_estimated_input = context_usage.get(
                    "estimated_input_tokens_after", 0
                )
                context_estimated_saved = context_usage.get(
                    "estimated_input_tokens_saved", 0
                )
                unproductive_rounds = state.guard_state.unproductive_tool_rounds
            else:
                final_content, _tools_used, _messages = await run_agent_kernel(
                    provider=scenario.provider,
                    model="offline-script",
                    tools=scenario.tools,
                    messages=messages,
                    max_iterations=8,
                    max_auto_continue_iterations=0,
                    max_finalization_iterations=1,
                )
                iterations = scenario.provider.domain_calls
                transitions = 0
                domain_tokens = scenario.provider.domain_usage.get("total_tokens", 0)
                harness_tokens = scenario.provider.harness_usage.get("total_tokens", 0)
                context_tokens = 0
                context_estimated_input = 0
                context_estimated_saved = 0
                unproductive_rounds = None

            success = final_content == scenario.success_text
            rows.append(
                {
                    "scenario": scenario_name,
                    "group": group,
                    "success": success,
                    "recovered_after_first_tool_error": (
                        scenario.has_tool_error and success
                    ),
                    "iterations": iterations,
                    "unproductive_rounds": unproductive_rounds,
                    "transitions": transitions,
                    "tokens": {
                        "domain": domain_tokens,
                        "harness": harness_tokens,
                        "context_processing": context_tokens,
                        "context_estimated_input": context_estimated_input,
                        "context_estimated_saved": context_estimated_saved,
                        "total": domain_tokens + harness_tokens + context_tokens,
                    },
                }
            )
    return rows


def summarize_ablation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate deterministic success and recovery rates by ablation group."""

    summary: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        eligible = sum(
            row["recovered_after_first_tool_error"] is not None for row in group_rows
        )
        recovered = sum(bool(row["recovered_after_first_tool_error"]) for row in group_rows)
        successes = sum(bool(row["success"]) for row in group_rows)
        summary[group] = {
            "scenarios": len(group_rows),
            "successes": successes,
            "success_rate": successes / len(group_rows) if group_rows else 0.0,
            "recovery_eligible": eligible,
            "recovered": recovered,
            "recovery_rate": recovered / eligible if eligible else None,
        }
    return summary


def _benchmark_messages(scenario_name: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "offline benchmark"},
        {"role": "user", "content": "old context " + ("alpha " * 400)},
        {"role": "assistant", "content": "old answer " + ("beta " * 100)},
        {"role": "user", "content": "recent context " + ("gamma " * 100)},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": f"run {scenario_name}"},
    ]
