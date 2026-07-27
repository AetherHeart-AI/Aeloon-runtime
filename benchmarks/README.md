# RefactorBench

This directory contains a lightweight runner for the official
[microsoft/RefactorBench](https://github.com/microsoft/RefactorBench). It uses
the benchmark's published prompt-to-test mappings and AST tests; it does not
copy those tests into the agent workspace.

## Run

Clone or download RefactorBench separately, then list the official base cases:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --list
```

Run a small smoke sample:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --instruction-set base \
  --limit 3 \
  --results benchmarks/results/refactorbench-base.jsonl
```

Filter to one repository or exact case:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --repository fastapi_refactor \
  --case fastapi_refactor/get-auth-scheme-param \
  --results benchmarks/results/refactorbench-fastapi.jsonl
```

Use `--resume` to skip cases already recorded in the JSONL ledger. Use
`--overwrite` only when the existing ledger should be truncated.

The first case for each of the nine source repositories creates a local git
snapshot under `.benchmark-workspaces/refactorbench`. Later cases reuse that
snapshot and reset it to the recorded baseline, avoiding a full repository copy
per task. Each record contains the agent result, official test verdict, changed
files, token usage, timings, and a path to the saved patch.

A case counts as passed only when `aeloon-core run` completes successfully and
the mapped official AST test exits with code zero. The ledger records the AST
verdict separately so timeouts/process failures remain diagnosable.

## Isolation

RefactorBench's official Docker setup provides stronger isolation. This runner
executes the agent and AST test directly on the host to keep feedback fast.
Use it only with trusted benchmark content and model-generated code in a
disposable workspace.
