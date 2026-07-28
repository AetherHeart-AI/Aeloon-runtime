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

`run_refactorbench.py` is the single entry point for running Aeloon Core and
baseline coding harnesses against the official
[microsoft/RefactorBench](https://github.com/microsoft/RefactorBench). It uses
the published prompt-to-test mappings and AST tests; those tests are never
copied into the agent workspace.

## Prerequisites

Clone or download RefactorBench separately. To run baselines, install and
authenticate the corresponding `pi`, `codex`, and `claude` CLIs first. The
runner deliberately uses each CLI's existing local model and authentication
configuration.

List the selected official cases without creating any files:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --list
```

## Run

With no `--harness`, the runner uses Aeloon Core:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --instruction-set base \
  --limit 3
```

Run every supported harness over the same cases:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --harness all \
  --limit 3
```

Or select baselines explicitly:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --harness pi \
  --harness codex \
  --harness claude \
  --output-dir benchmarks/results/my-baseline-run
```

The adapters intentionally stay thin:

| Harness | Non-interactive invocation |
|---|---|
| `aeloon` | `python -m aeloon_core run --stdin --output json` |
| `pi` | `pi --print --mode json --no-session` |
| `codex` | `codex exec --sandbox workspace-write --ephemeral --json -` |
| `claude` | `claude --print --output-format json --no-session-persistence --dangerously-skip-permissions` |

`--config` applies only to Aeloon Core. The other harnesses use their normal
local CLI configuration.

Filter to one repository or exact case with repeatable `--repository` and
`--case` options:

```bash
uv run python benchmarks/run_refactorbench.py \
  --refactorbench-root /path/to/RefactorBench \
  --harness aeloon \
  --harness codex \
  --repository fastapi_refactor \
  --case fastapi_refactor/get-auth-scheme-param
```

## Result archive

By default, every invocation creates a new timestamped directory under
`benchmarks/results/refactorbench/`. Use `--output-dir` for a stable location:

```text
my-baseline-run/
├── manifest.json
├── summary.json
├── aeloon/
│   ├── results.jsonl
│   ├── logs/
│   ├── patches/
│   └── sessions/
├── codex/
│   ├── results.jsonl
│   ├── logs/
│   ├── patches/
│   └── sessions/
└── ...
```

The manifest records the selected cases, source revision, CLI paths, and CLI
versions. Each append-only JSONL record contains the normalized agent result,
official test verdict, changed files, timing, token usage when exposed by the
CLI, and relative paths to the full stdout, stderr, and patch artifacts.
`summary.json` makes pass rates and basic cost/timing comparisons available
without re-reading every ledger.

Use `--resume` with the same `--output-dir`, harness order, and case selection
to skip completed harness/case pairs. Use `--overwrite` only to replace managed
ledgers and artifacts in a compatible archive. The legacy `--results
path.jsonl` option remains available for single-harness callers, but new
automation should use `--output-dir`.

## Workspace reuse and isolation

The first case for each source repository creates a local git snapshot under
`.benchmark-workspaces/refactorbench`. Every harness/case pair resets that
workspace to the recorded baseline, so all harnesses see identical source while
avoiding a full copy for every task.

A case passes only when the harness process completes successfully and the
mapped official AST test exits with code zero. The ledger records process and
oracle failures separately.

RefactorBench's official Docker setup provides stronger isolation. This runner
executes agents and AST tests directly on the host to keep comparison runs
simple and fast. In particular, Claude Code runs with permission prompts
disabled. Use only trusted benchmark content in the disposable runner-owned
workspaces.
