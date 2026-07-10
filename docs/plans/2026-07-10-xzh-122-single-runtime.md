# XZH-122: collapse to one state-machine runtime

## Goal

Make the explicit state machine the only production agent loop. Preserve the
current dual-runtime implementation as a Git baseline instead of carrying the
legacy loop, compatibility adapters, and runtime selector in the MVP codebase.

## Baseline preservation

Before deleting legacy code:

1. Commit the current dual-runtime implementation on
   `xz89166/xzh-122-mvp-v10`.
2. Create local branch `xzh-122-a1-baseline` at that commit.
3. Remain on `xz89166/xzh-122-mvp-v10` for the single-runtime refactor.
4. Exclude this plan file from the baseline commit so the baseline contains only
   the already-verified implementation it describes.

The baseline branch replaces the runtime `enabled` switch for historical A1
comparisons. No push is part of this task.

## Runtime simplification

- `AeloonCoreOrchestrator` always calls `run_agent_loop`.
- Remove `uasm.enabled` and its CLI setter.
- Remove legacy/UASM branching, legacy usage fallbacks, and the `legacy` result
  status.
- Keep one configuration namespace for state-machine policies:
  `rule_engine_enabled`, `temporary_guard_enabled`,
  `minimal_context_enabled`, transition tracing, guard mode, and context bounds.
- The default runtime is the full state-machine path (A3 policy set).

## File boundaries

Keep:

- `state_machine.py`
- `agents.py`
- `state.py`
- `temporary_guard.py`
- `minimal_context.py`
- `transitions.py`

Extract the message, streaming, provider-input, and tool-batch helpers currently
living beside the legacy loop into `runtime_support.py`, then delete:

- `aeloon_core/kernel.py`
- `tests/test_kernel.py`

Remove the tuple-shaped `run_uasm_kernel` compatibility wrapper. Tests and
callers use the typed `LightweightState` result from `run_agent_loop`.

## A0-A3 experiments

All four groups use `run_agent_loop`:

| Group | Rule engine | TemporaryGuard | Minimal context |
| --- | --- | --- | --- |
| A0 | off | off | off |
| A1 | on | off | off |
| A2 | on | on | off |
| A3 | on | on | on |

The hard `max_iterations` cap remains active as a safety boundary in every
group. With the rule engine disabled, reaching it stops the run without
auto-continuation, recovery prompts, or finalization passes.

The offline benchmark continues to report success, recovery rate, iterations,
transition counts, billed domain/harness/context tokens, and estimated context
savings. Git branch `xzh-122-a1-baseline` remains the code-level comparison for
the former loop.

## Documentation and compatibility

- Update README to describe a single default runtime rather than opt-in UASM.
- Document the baseline branch and the A0-A3 state-machine policy switches.
- Preserve additive session/turn/trace schemas and terminal event contracts.
- Preserve deterministic rule, guard, context, finalization, and provider-error
  behavior already covered by state-machine tests.
- Remove tests whose only purpose is legacy-loop equivalence.

## Verification

- `git branch --list xzh-122-a1-baseline` points at the baseline commit.
- No production or test references remain for `run_agent_kernel`,
  `run_uasm_kernel`, `uasm.enabled`, or `status == "legacy"`.
- `aeloon_core/kernel.py` and `tests/test_kernel.py` are absent.
- A0-A3 benchmark uses only `run_agent_loop` and still produces 20 scenario rows
  plus deterministic recovery-rate summaries.
- Run `uv run pytest -q`, `uv run ruff check .`, `uv build`, benchmark assertions,
  and `git diff --check`.
- Browser testing remains N/A unless a browser-facing surface appears in the
  diff.
