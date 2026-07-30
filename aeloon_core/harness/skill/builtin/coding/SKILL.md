---
name: coding
description: Implement a bounded repository change through planning, building, independent review, and one corrective pass.
kind: expert
runner: builtin.coding
capabilities:
  - filesystem
  - filesystem_read
  - shell
  - repo_context
  - planning
model_tier: strong
concurrency_mode: exclusive
max_calls_per_turn: 4
---
# Coding expert

Use this expert for a self-contained change in the current workspace.

The runner performs a bounded pipeline:

1. a read-only planner identifies scope, constraints, and verification;
2. a builder edits the workspace and runs appropriate checks;
3. an independent reviewer returns structured actionable findings;
4. if findings remain, one fixer pass and one re-review are allowed.

There is no unbounded repair loop. Remaining findings after re-review produce a partial result.
