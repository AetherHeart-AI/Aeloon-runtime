# Benchmarks

The benchmark runner has one public command:

```bash
uv run python run_bench.py \
  --harness aeloon \
  --model deepseek-v4-flash \
  --benchmark refactorbench
```

Run the same benchmark through several harnesses by listing them together or
repeating the option:

```bash
uv run python run_bench.py \
  --harness aeloon pi codex \
  --workers 4 \
  --benchmark refactorbench
```

Supported benchmarks are `refactorbench`, `livecodebench`, and `repoqa`.
Supported harnesses are `aeloon`, `pi`, `codex`, `claude`, `openclaw`, and
`hermes`; `--harness all` selects all of them.

`--model MODEL` passes the same explicit model selection to every harness and
defaults to `deepseek-v4-flash`. This keeps comparisons on the same model
instead of inheriting each CLI's configured default.

`--workers N` enables opt-in case concurrency and defaults to `1`. RefactorBench
assigns each source repository to one writable lane, so cases that share a
repository never mutate the same workspace concurrently. LiveCodeBench runs
independent generation or repair cases concurrently and keeps official
evaluation batched. RepoQA assigns each repository to one lane and reuses
separately reset workspaces for each harness. Start with `--workers 2` or
`--workers 4`; higher values may hit model-provider rate limits.

Use `--limit N` for a deterministic smoke run. RepoQA interleaves languages and
repositories before applying the limit, so a small run does not collapse to a
single language:

```bash
uv run python run_bench.py \
  --harness aeloon \
  --workers 4 \
  --limit 60 \
  --benchmark repoqa
```

Each invocation creates a new run. If it is interrupted, rerun the command to
start from the beginning; the unified runner does not keep generation
checkpoints or support `--resume`.

## Automatic preparation

No separate setup command is required. Before a run, the selected adapter:

1. clones the official benchmark into
   `.benchmark-workspaces/sources/<benchmark>` when it is not cached;
2. installs the dependencies needed by the official evaluator;
3. loads the official cases and runs them through every selected harness;
4. writes a manifest, summary, JSONL ledgers, process logs, and patches under
   `benchmarks/results/<benchmark>/<run-id>/`.

Preparation, dataset loading, harness execution, evaluation, and result paths
are reported on stderr. Case execution includes a progress bar with completed
and total counts, percentage, elapsed time, ETA, and the current case. The bar
updates in place in an interactive terminal and falls back to one line per
completed case when stderr is redirected (for example, in CI). The final
machine-readable summary remains the only content written to stdout.

LiveCodeBench uses a dedicated environment under
`.benchmark-workspaces/environments/livecodebench`. The adapter installs only
the public dataset and evaluation dependencies; model inference is supplied by
the selected harness, so LiveCodeBench's GPU inference stack is unnecessary.

RepoQA uses the pinned official `2024-06-23` dataset. Preparation downloads and
verifies the compressed release once; task execution and grading are then
local. This integration is an agentized Search Needle Function task: the prompt
contains only the repository, language, and natural-language function
description. The harness must search the materialized repository and finish
with:

```text
REPOQA_RESULT: {"path":"relative/path.ext","symbol":"function_name"}
```

The grader uses exact path-and-symbol matching. Repository changes are recorded
as collateral damage and make the task fail. Unlike upstream RepoQA's
long-context generation setup, this integration does not install model
backends, tokenizers, Tree-sitter, or BLEU dependencies. The benchmark itself
does not require network access after preparation; network policy for a remote
model or harness remains the responsibility of that harness.

The `pi`, `codex`, `claude`, `openclaw`, and `hermes` commands must already be
installed and authenticated. Aeloon uses the current project environment and
configuration.

## Architecture

```text
benchmarks/
├── adapters/
│   ├── base.py
│   ├── refactorbench.py
│   ├── livecodebench.py
│   └── repoqa.py
├── harness/
│   ├── base.py
│   ├── aeloon.py
│   ├── pi.py
│   ├── codex.py
│   ├── claude.py
│   ├── openclaw.py
│   └── hermes.py
├── refactorbench/
├── livecodebench/
├── repoqa/
└── run_bench.py
run_bench.py
```

`BenchmarkAdapter` owns source acquisition, dependency preparation, official
evaluation, and durable result writing. `Harness` owns non-interactive CLI
invocation and normalizes status, output, token usage, and diagnostics.

The old `benchmarks/run_refactorbench.py` and
`benchmarks/run_livecodebench.py` modules remain as compatibility imports for
existing automation. New integrations should use the two base classes and
register themselves in the corresponding registry.

## Safety

Both benchmarks execute model-generated code or modifications. Run them in a
disposable environment. The benchmark cache and result directories are ignored
by Git.
