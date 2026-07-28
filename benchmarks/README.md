# Benchmarks

The benchmark runner has one public command:

```bash
uv run python run_bench.py \
  --harness aeloon \
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

Supported benchmarks are `refactorbench` and `livecodebench`. Supported
harnesses are `aeloon`, `pi`, `codex`, and `claude`; `--harness all` selects all
of them.

`--workers N` enables opt-in case concurrency and defaults to `1`. RefactorBench
assigns each source repository to one writable lane, so cases that share a
repository never mutate the same workspace concurrently. LiveCodeBench runs
independent generation or repair cases concurrently and keeps official
evaluation batched. Start with `--workers 2` or `--workers 4`; higher values may
hit model-provider rate limits.

## Automatic preparation

No separate setup command is required. Before a run, the selected adapter:

1. clones the official benchmark into
   `.benchmark-workspaces/sources/<benchmark>` when it is not cached;
2. installs the dependencies needed by the official evaluator;
3. loads the official cases and runs them through every selected harness;
4. writes a manifest, summary, JSONL ledgers, process logs, and patches under
   `benchmarks/results/<benchmark>/<run-id>/`.

Preparation, dataset loading, harness execution, evaluation, and result paths
are reported as `INFO` progress on stderr. The final machine-readable summary
remains the only content written to stdout.

LiveCodeBench uses a dedicated environment under
`.benchmark-workspaces/environments/livecodebench`. The adapter installs only
the public dataset and evaluation dependencies; model inference is supplied by
the selected harness, so LiveCodeBench's GPU inference stack is unnecessary.

The `pi`, `codex`, and `claude` commands must already be installed and
authenticated. Aeloon uses the current project environment and configuration.

## Architecture

```text
benchmarks/
├── adapters/
│   ├── base.py
│   ├── refactorbench.py
│   └── livecodebench.py
├── harness/
│   ├── base.py
│   ├── aeloon.py
│   ├── pi.py
│   ├── codex.py
│   └── claude.py
├── refactorbench/
├── livecodebench/
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
