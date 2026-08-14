# Aeloon Core

Aeloon Core combines a stateless Python agent-run engine with a stateful application runtime. It
provides a CLI and Python API for tool-driven coding tasks, resumable sessions, configurable model
providers, retries, and automatic context compaction.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- a Custom OpenAI-compatible API or an Aeloon Cloud account

## Quick start

Install the project and connect a local endpoint:

```bash
uv sync

uv run aeloon-core provider add studio \
  --endpoint http://127.0.0.1:8000

uv run aeloon-core "Inspect this repository and explain its entry points"
```

Pass `--api-key` when the endpoint requires one. The key is stored in the mode-`0600` config file.
Custom Providers try the supplied URL and its `/v1` form, discover the model catalog, and make one
small best-effort image request for each model whose image capability is not present in metadata.
Capability probe failures leave that model available as text-only and do not block setup.

To use Aeloon Cloud instead:

```bash
uv run aeloon-core login
uv run aeloon-core models
uv run aeloon-core "Fix the failing tests"
```

## Common workflows

The task is the default command:

```bash
# Start a saved task in the current workspace
uv run aeloon-core "fix the failing tests"

# Continue the newest task for this workspace
uv run aeloon-core resume "continue with the implementation"

# Read a task from a file or standard input
uv run aeloon-core --file task.md
printf 'review this change' | uv run aeloon-core

# Select a workspace or model for one run
uv run aeloon-core -C ../project -m studio/qwen3-coder "review the repository"

# Run without saving, return JSON, or show tool activity
uv run aeloon-core --ephemeral "answer without saving a session"
uv run aeloon-core --json "return one machine-readable result"
uv run aeloon-core -v "show concise tool activity"
uv run aeloon-core -vv "also show lifecycle events"
```

Useful management commands:

```bash
uv run aeloon-core provider list
uv run aeloon-core models
uv run aeloon-core models use studio/qwen3-coder
uv run aeloon-core history
uv run aeloon-core doctor
uv run aeloon-core whoami
uv run aeloon-core logout
```

Fresh installations have no pinned default model. Aeloon uses the first available model until
`models use` pins a default. Models have stable `provider/model` IDs; provider-local names are
resolved in catalog order, so use the full ID when names overlap. `-m MODEL` overrides the
selection for one run without changing the saved default.

Shell completion is available without an additional runtime dependency:

```bash
uv run aeloon-core completion zsh > ~/.zfunc/_aeloon-core
uv run aeloon-core completion bash > ~/.local/share/bash-completion/completions/aeloon-core
uv run aeloon-core completion fish > ~/.config/fish/completions/aeloon-core.fish
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

Runtime bundles one Python-driven Office skill: `aeloon-office-lite` for fast, simple reading,
creation, validation, and visual rendering of local PDF, DOCX, PPTX, and XLSX files. On first
startup after installation, the missing built-in skill is copied from package resources into
`<data_dir>/skills` (normally
`~/.aeloon-core/skills`). Existing same-named files and directories are preserved, so local
customizations are never overwritten or removed.

The main package contains only lightweight Python document libraries. Text-bearing files are read
directly; scanned PDFs are rendered into page images for a vision-capable model instead of
downloading an OCR runtime. LibreOffice remains optional and is used only to render DOCX/PPTX/XLSX
files for visual QA. A missing renderer is reported explicitly and never treated as completed
visual verification. Python installation hints default to the Tsinghua PyPI mirror for Chinese
users.

Skill discovery is progressive: Runtime reads only each `SKILL.md` frontmatter for the catalog and
system-prompt index. The full instructions are read only when the model chooses an enabled skill or
when a prompt starts with an explicit command such as `/review inspect this patch`. Workbench clients
select enabled skills through `settings.update.resources.enabled_skill_ids`; they send
the user's text as a normal prompt, and runtime resolves the slash command.

`catalog.get.skills` includes each skill's command, source, location, selection and invocation
status, plus `content_loading: "on_demand"`. `settings.get.resources.enabled_skill_ids` is the
effective selection. A null persisted selection enables every discovered skill; an explicit list
is subtractive.

Runtime also injects the intrinsic `present_files` delivery tool. Skills use it after verifying
final office, PDF, image, Markdown, or HTML files. Runtime validates the paths, persists their
display metadata outside model context, and exposes them to Workbench clients as structured
artifacts; stateless core tools and message types remain format-agnostic.

## Python API

```python
from aeloon_core.core import RunRequest, UserMessage, run_agent
from aeloon_core.runtime.providers import DeepSeekProvider, get_deepseek_model
from aeloon_core.tool import BuiltinToolSet

workspace = "/path/to/repository"
provider = DeepSeekProvider(api_key="...")
tools = BuiltinToolSet(workspace)

try:
    result = await run_agent(RunRequest(
        run_id="example-run",
        messages=(),
        input=(UserMessage("Implement the requested change"),),
        system_prompt="You are a coding agent.",
        tools=tools.tools,
        active_tool_names=("read", "bash", "edit", "write"),
        inference=provider,
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
scheduling. The Electron workbench accesses the runtime through the private `aeloon-rpc-v2`
Unix-socket adapter.

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
uv run python -m aeloon_core.rpc.manifest --check
uv build
git diff --check
```

The default test suite is offline and uses local fixtures. Optional live tests require credentials
in an explicit test config and are not part of default CI.

The checked-in `aeloon-rpc-v2.manifest.json` is generated from Core's typed RPC registry. Run
`uv run python -m aeloon_core.rpc.manifest --output aeloon_core/rpc/aeloon-rpc-v2.manifest.json`
after changing a public method, result, or event payload; CI rejects stale output.

### Desktop distribution

Aeloon Core is distributed as part of the Aeloon desktop installer. The UI release workflow locks
this repository to an exact commit, builds a wheel, and installs the wheel plus its frozen
production dependencies into the desktop application's bundled Python runtime. Core does not
publish a separate PyInstaller executable or desktop archive.

The desktop application starts Core with `python -m aeloon_core`, and the Agent, built-in Skills,
and terminal use that same bundled interpreter. Office Lite uses those bundled lightweight Python
libraries directly and does not create a second Python or OCR environment.

For local package validation:

```bash
uv build --wheel --out-dir dist
uv venv wheel-smoke
uv pip install --python wheel-smoke/bin/python dist/aeloon_core-*.whl
wheel-smoke/bin/python -m aeloon_core --version
```

## Documentation

- [Architecture](docs/architecture.md)
- [Changelog](CHANGELOG.md)
- [Benchmark guide](benchmarks/README.md)
