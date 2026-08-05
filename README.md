# Aeloon Core

Aeloon Core combines a stateless Python agent-run engine with a stateful application runtime. It
provides a CLI and Python API for tool-driven coding tasks, resumable sessions, configurable model
providers, retries, and automatic context compaction.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- an OpenAI-compatible local API or an Aeloon Cloud account

## Quick start

Install the project and connect a local endpoint:

```bash
uv sync

uv run aeloon local add studio \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-coder \
  --no-api-key

uv run aeloon "Inspect this repository and explain its entry points"
```

Omit `--no-api-key` when the endpoint requires a key. The CLI reads the key from a hidden prompt
and stores it in the mode-`0600` config file.

To use Aeloon Cloud instead:

```bash
uv run aeloon login
uv run aeloon models
uv run aeloon "Fix the failing tests"
```

## Common workflows

The task is the default command:

```bash
# Start a saved task in the current workspace
uv run aeloon "fix the failing tests"

# Continue the newest task for this workspace
uv run aeloon resume "continue with the implementation"

# Read a task from a file or standard input
uv run aeloon --file task.md
printf 'review this change' | uv run aeloon

# Select a workspace or model for one run
uv run aeloon -C ../project -m studio/qwen3-coder "review the repository"

# Run without saving, return JSON, or show tool activity
uv run aeloon --ephemeral "answer without saving a session"
uv run aeloon --json "return one machine-readable result"
uv run aeloon -v "show concise tool activity"
uv run aeloon -vv "also show lifecycle events"
```

Useful management commands:

```bash
uv run aeloon local list
uv run aeloon models
uv run aeloon models use studio/qwen3-coder
uv run aeloon history
uv run aeloon doctor
uv run aeloon whoami
uv run aeloon logout
```

Fresh installations have no pinned default model. Aeloon uses the first available model until
`models use` pins a default. Models have stable `provider/model` IDs; provider-local names are
resolved in catalog order, so use the full ID when names overlap. `-m MODEL` overrides the
selection for one run without changing the saved default.

Shell completion is available without an additional runtime dependency:

```bash
uv run aeloon completion zsh > ~/.zfunc/_aeloon
uv run aeloon completion bash > ~/.local/share/bash-completion/completions/aeloon
uv run aeloon completion fish > ~/.config/fish/completions/aeloon.fish
```

## Project resources

Global resources live in `~/.aeloon-core`; workspace resources live in
`<workspace>/.aeloon-core`:

```text
.aeloon-core/
├── SYSTEM.md
├── APPEND_SYSTEM.md
├── skills/<name>/SKILL.md
└── prompts/<name>.md
```

`SYSTEM.md` replaces the generated base prompt. `APPEND_SYSTEM.md`, project instructions, skills,
and the working directory are appended in a deterministic order. Workspace resources override
same-named global resources and are reloaded at every turn boundary.

## Python API

```python
from aeloon_core.core import (
    DeepSeekProvider,
    RunRequest,
    UserMessage,
    create_all_tools,
    get_deepseek_model,
    run_agent,
)

workspace = "/path/to/repository"
provider = DeepSeekProvider(api_key="...")
tools = tuple(create_all_tools(workspace).values())

try:
    result = await run_agent(RunRequest(
        run_id="example-run",
        messages=(),
        input=(UserMessage("Implement the requested change"),),
        system_prompt="You are a coding agent.",
        tools=tools,
        active_tool_names=("read", "bash", "edit", "write"),
        provider=provider,
        model=get_deepseek_model("deepseek/deepseek-v4-flash"),
    ))
    print(result.final_message.text)
finally:
    await provider.close()
```

`run_agent()` retains no state after it returns. Stateful applications should use the runtime API:

```python
from aeloon_core.bootstrap import create_runtime_service

runtime = create_runtime_service()
session = await runtime.create_session(workspace="/path/to/repository")
```

Runtime owns sessions, context construction, persistence, provider selection, and operation
scheduling. Bridge clients access the same runtime through the stable Bridge v2 protocol.

## Security

The built-in `read`, `bash`, `edit`, and `write` tools are active by default; `grep`, `find`, and
`ls` are available when enabled. These tools are not a sandbox: they accept absolute paths,
inherit the process environment, and can execute arbitrary shell commands. Run Aeloon only where
the model is allowed to access the filesystem, credentials, and processes.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv build
git diff --check
```

The default test suite is offline and uses local fixtures. Optional live tests require credentials
in an explicit test config and are not part of default CI.

## Documentation

- [Architecture](docs/architecture.md)
- [Migration guide](MIGRATION.md)
- [Changelog](CHANGELOG.md)
- [Benchmark guide](benchmarks/README.md)
