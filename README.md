# Aeloon Runtime

Aeloon Runtime combines the stateless Python agent engine with a stateful application runtime. It
provides a standalone CLI and Python API for tool-driven coding tasks, resumable sessions,
configurable model providers, retries, and automatic context compaction.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- a Custom OpenAI-compatible API or an Aeloon Cloud account

## Quick start

Install the project and connect a local endpoint:

```bash
uv sync

uv run aeloon-runtime provider add studio \
  --endpoint http://127.0.0.1:8000

uv run aeloon-runtime "Inspect this repository and explain its entry points"
```

Pass `--api-key` when the endpoint requires one. The key is stored in the mode-`0600` config file.
Custom Providers try the supplied URL and its `/v1` form, discover the model catalog, and make one
small best-effort image request for each model whose image capability is not present in metadata.
Capability probe failures leave that model available as text-only and do not block setup.

To use Aeloon Cloud instead:

```bash
uv run aeloon-runtime login
uv run aeloon-runtime models
uv run aeloon-runtime "Fix the failing tests"
```

## Common workflows

The task is the default command:

```bash
# Start a saved task in the current workspace
uv run aeloon-runtime "fix the failing tests"

# Continue the newest task for this workspace
uv run aeloon-runtime resume "continue with the implementation"

# Read a task from a file or standard input
uv run aeloon-runtime --file task.md
printf 'review this change' | uv run aeloon-runtime

# Select a workspace or model for one run
uv run aeloon-runtime -C ../project -m studio/qwen3-coder "review the repository"

# Run without saving, return JSON, or show tool activity
uv run aeloon-runtime --ephemeral "answer without saving a session"
uv run aeloon-runtime --json "return one machine-readable result"
uv run aeloon-runtime -v "show concise tool activity"
uv run aeloon-runtime -vv "also show lifecycle events"
```

Useful management commands:

```bash
uv run aeloon-runtime provider list
uv run aeloon-runtime models
uv run aeloon-runtime models use studio/qwen3-coder
uv run aeloon-runtime history
uv run aeloon-runtime doctor
uv run aeloon-runtime whoami
uv run aeloon-runtime logout
```

The desktop Runtime can also run independently over a private Unix socket. It keeps serving after
the UI exits; use `system.shutdown` (or the uninstall flow) for an explicit stop. Boundary traces
are disabled by default and require an explicit local directory:

```bash
uv run aeloon-runtime serve --unix /tmp/aeloon-runtime.sock \
  --data-dir ~/.aeloon-runtime --workspace-root "$PWD" \
  --record-trace ~/.aeloon-runtime/traces
```

Fresh installations have no pinned default model. Aeloon uses the first available model until
`models use` pins a default. Models have stable `provider/model` IDs; provider-local names are
resolved in catalog order, so use the full ID when names overlap. `-m MODEL` overrides the
selection for one run without changing the saved default.

Shell completion is available without an additional runtime dependency:

```bash
uv run aeloon-runtime completion zsh > ~/.zfunc/_aeloon-runtime
uv run aeloon-runtime completion bash > ~/.local/share/bash-completion/completions/aeloon-runtime
uv run aeloon-runtime completion fish > ~/.config/fish/completions/aeloon-runtime.fish
```

## Project resources

Global resources live in `~/.aeloon-runtime`; workspace resources live in
`<workspace>/.aeloon-runtime`:

```text
.aeloon-runtime/
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
`~/.aeloon-runtime/skills`). Existing same-named files and directories are preserved, so local
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
from aeloon_runtime.core import RunRequest, UserMessage, run_agent
from aeloon_runtime.runtime.providers import DeepSeekProvider, get_deepseek_model
from aeloon_runtime.tool import BuiltinToolSet

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
from aeloon_runtime.bootstrap import create_runtime_service

runtime = create_runtime_service()
session = await runtime.create_session(workspace="/path/to/repository")
```

Runtime owns sessions, context construction, persistence, provider selection, workspace operations,
and operation scheduling. The Electron desktop client connects through the `aeloon-rpc` v3 Unix
socket gateway; the Runtime remains alive when the client exits.

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
uv run python tools/gen_v3_manifest.py --check
uv run python tools/gen_rpc_docs.py
uv build
git diff --check
```

The default test suite is offline and uses local fixtures. Optional live tests require credentials
in an explicit test config and are not part of default CI.

The checked-in `aeloon-rpc-v3.manifest.json` is generated from `docs/rpc-v3.json`; run
`uv run python tools/gen_v3_manifest.py` and `uv run python tools/gen_rpc_docs.py` after changing a
public method, result, or event payload. CI rejects stale generated output.

### Desktop distribution

Aeloon Runtime is distributed independently as the `aeloon-runtime` wheel and as macOS ARM64 and
Linux ARM64 bundles. The desktop lock pins the Runtime archive URL and SHA-256; the UI does not
checkout this repository at build time.

The desktop application starts Runtime on a stable Unix socket. The Agent, built-in Skills, Git/fs,
and terminal operations all execute in that Runtime process. Office Lite uses the bundled
lightweight Python libraries directly and does not create a second Python or OCR environment.

For local package validation:

```bash
uv build --wheel --out-dir dist
uv venv wheel-smoke
uv pip install --python wheel-smoke/bin/python dist/aeloon_runtime-*.whl
wheel-smoke/bin/aeloon-runtime --version
```

## Documentation

- [Architecture](docs/architecture.md)
- [Changelog](CHANGELOG.md)
- [Benchmark guide](benchmarks/README.md)
