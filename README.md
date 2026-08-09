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

uv run aeloon provider add studio \
  --endpoint http://127.0.0.1:8000

uv run aeloon "Inspect this repository and explain its entry points"
```

Pass `--api-key` when the endpoint requires one. The key is stored in the mode-`0600` config file.
Custom Providers try the supplied URL and its `/v1` form, discover the model catalog, and make one
small best-effort image request for each model whose image capability is not present in metadata.
Capability probe failures leave that model available as text-only and do not block setup.

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
uv run aeloon provider list
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

Runtime bundles exactly three Python-driven Office skills: `document-reader` for local document
ingestion and PDF layout inspection, `word-docx` for editable Word documents, and
`powerpoint-pptx` for editable presentations. On first startup after installation, missing built-in
skills are copied from package resources into `<data_dir>/skills` (normally
`~/.aeloon-core/skills`). Existing same-named files and directories are preserved, so local
customizations and retired user-owned skill copies are never overwritten or removed.

The main package contains the digital-document engines. Docling with RapidOCR runs only from its
locked, separately prepared `uv` environment and model cache; the `document-reader preflight`
action reports whether that cache is ready, while `prepare-ocr` installs and warms it for later
offline use.
LibreOffice remains optional and is used to render DOCX/PPTX files for visual QA. A missing renderer
is reported explicitly and never treated as completed visual verification.

Skill discovery is progressive: Runtime reads only each `SKILL.md` frontmatter for the catalog and
system-prompt index. The full instructions are read only when the model chooses an enabled skill or
when a prompt starts with an explicit command such as `/review inspect this patch`. Workbench clients
select enabled skills through `settings.update.resources.enabled_skill_ids`; they continue to send
the user's text as a normal prompt, and runtime resolves the slash command.

`catalog.get.skills` includes each skill's command, source, location, selection and invocation
status, plus `content_loading: "on_demand"`. `settings.get.resources.enabled_skill_ids` is the
persisted selection. On existing configurations, all discovered skills remain selected until a
client saves an explicit list.

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
scheduling. The Electron workbench accesses the runtime through the private, incompatible
`aeloon-rpc-v1` Unix-socket adapter. When Core starts with a Browser Runtime socket, all 22
Browser Use tools are part of every agent tool catalog and execute through
`browser-runtime-v1`.

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

### Desktop distribution

Aeloon Core is distributed as part of the Aeloon desktop installer. The UI release workflow locks
this repository to an exact commit, builds a wheel, and installs the wheel plus its frozen
production dependencies into the desktop application's bundled Python runtime. Core does not
publish a separate PyInstaller executable or desktop archive.

The desktop application starts Core with `python -m aeloon_core`, and the Agent, built-in Skills
and terminal use that same bundled interpreter. The optional Docling/RapidOCR environment remains
isolated because of its size and native dependency constraints, but it is created by the desktop
runtime's Python and uv rather than downloading another Python distribution.

For local package validation:

```bash
uv build --wheel --out-dir dist
uv venv wheel-smoke
uv pip install --python wheel-smoke/bin/python dist/aeloon_core-*.whl
wheel-smoke/bin/python -m aeloon_core --version
```

## Documentation

- [Architecture](docs/architecture.md)
- [Migration guide](MIGRATION.md)
- [Changelog](CHANGELOG.md)
- [Benchmark guide](benchmarks/README.md)
