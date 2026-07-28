# Benchmarks

## LiveCodeBench v6

`run_livecodebench.py` runs the official LiveCodeBench code-generation and
self-repair scenarios through Aeloon Core. By default it selects `v6`, the
problems added in the sixth release, and runs both scenarios once per problem.
Use `--all` to select cumulative `release_v6` instead.

Clone the official runner, then install Aeloon's development dependencies into
the current worktree environment:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
uv sync
```

List the newly added v6 cases without running agents:

```bash
uv run python benchmarks/run_livecodebench.py \
  --livecodebench-root /path/to/LiveCodeBench \
  --list
```

Run a smoke sample with the default code-generation plus self-repair flow:

```bash
uv run python benchmarks/run_livecodebench.py \
  --livecodebench-root /path/to/LiveCodeBench \
  --limit 3 \
  --results benchmarks/results/livecodebench-v6-smoke.jsonl
```

The runner automatically uses this worktree's `.venv/bin/python` for dataset
loading and official evaluation. It adds the LiveCodeBench checkout to that
process's import path, so the checkout does not need a separate installation.
To override the environment, pass
`--livecodebench-python /path/to/python`.

Select one scenario, one exact problem, or the cumulative v6 release:

```bash
uv run python benchmarks/run_livecodebench.py \
  --livecodebench-root /path/to/LiveCodeBench \
  --scenario code-generation \
  --case abc123 \
  --all
```

Self repair first runs (or resumes) code generation and its official tests. A
failed solution is given the official error feedback and repaired once; an
already-correct solution is reused without another model call. The JSONL ledger
stores separate `code-generation` and `self-repair` records. Use `--resume` to
skip completed scenario/problem pairs and `--overwrite` to replace an existing
ledger. `--test-timeout` controls each official test and `--adapter-timeout`
controls total dataset-loading or batch-evaluation time.

LiveCodeBench executes model-generated Python during evaluation. Its reliability
guard is not a security sandbox, so run it only in a disposable environment.

## RefactorBench

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
