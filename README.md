# Aeloon Core

Aeloon Core is a stateful coding-agent harness implemented entirely in Python. Its behavior is
modeled after Pi Coding Agent 0.83.0: natural tool-loop completion, Pi-style system prompts and
resources, steer/follow-up queues, append-only session trees, retries, and semantic compaction.

It does not depend on `pi-ai`, `pi-agent-core`, Node, or Bun at runtime, during development, or in
CI. Expert routing, Expert implementations, MCP, and the former Web UI are intentionally outside
this version and will be redesigned separately.

## Quick start

```bash
uv sync

# Connect an OpenAI-compatible local API. The key is read from a hidden prompt;
# use --no-api-key for Ollama or another unauthenticated endpoint.
uv run aeloon local add studio \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-coder \
  --no-api-key
uv run aeloon "Inspect this repository and explain its entry points"
uv run aeloon --file task.md --json
printf 'Fix the failing tests' | uv run aeloon
```

Fresh installations have no pinned default model and never read `DEEPSEEK_API_KEY`. After you
connect a local API or sign in to Aeloon Cloud, runs automatically use the first model shown by
`aeloon models`. Use `aeloon models use MODEL` only when you want to pin a different default.
Every model has a stable `provider/model` ID, so local API and cloud models can appear in one
catalog without collisions. Existing explicitly configured DeepSeek providers and sessions remain
compatible, but DeepSeek is not selected or shown as an available CLI model without a stored key.
Thinking defaults to `off`.

Aeloon Cloud is an optional account-backed provider. Core owns the account session and refresh
credential; UI clients receive only public account status and provider-qualified model IDs such as
`aeloon-cloud/reasoner`. The unified Provider registry also accepts user-added OpenAI-compatible
local APIs. Sign-in is exposed through the Core CLI and Bridge v2, and cloud models are added to
the dynamic catalog only while the account is authenticated.

## Python API

```python
from aeloon_core.harness import (
    AgentHarness,
    DeepSeekProvider,
    JsonlSessionRepository,
    ResourceLoader,
    get_deepseek_model,
)

workspace = "/path/to/repository"
repository = JsonlSessionRepository("~/.aeloon-core")
session = await repository.create(cwd=workspace)
provider = DeepSeekProvider(api_key="...")
harness = AgentHarness(
    provider=provider,
    model=get_deepseek_model("deepseek/deepseek-v4-flash"),
    cwd=workspace,
    session=session,
    resource_loader=ResourceLoader(cwd=workspace),
)

try:
    response = await harness.prompt("Implement the requested change")
    print(response.text)
finally:
    await harness.close()
```

`aeloon_core.harness` exports provider-neutral messages and content blocks, `AgentTool`,
`ToolResult`, models and stream options, typed harness events, resources, sessions, and the
deterministic `ScriptedProvider` used by tests and offline integrations.

`AgentHarness` owns a strict `idle / turn / compaction / branch_summary / retry` state machine.
Normal prompts are rejected with `HarnessError(code="busy")` while it is active. During a turn,
`steer()` injects input after the current tool batch, `follow_up()` continues after natural
completion, and `next_turn()` queues input for the next explicit prompt. Queue modes support
`one-at-a-time` and `all`.

## Tool loop and tools

An assistant response with no tool calls completes naturally. Tool results—including errors—are
returned to the model and the loop continues. A `length` stop containing tool calls skips the
entire batch because its arguments may be truncated. Calls run in parallel by default; mutations
to the same file are serialized and tool results remain in call order.

The built-in tools are:

- active by default: `read`, `bash`, `edit`, `write`;
- available but inactive by default: `grep`, `find`, `ls`.

`read` supports text continuation and image attachments. `bash` streams updates, has no default
timeout, and stores complete output in a temporary file when its visible tail is truncated.
`edit` performs one or more unique, non-overlapping exact replacements against the original file,
preserving UTF-8 BOM and line-ending style. `write` creates parents and overwrites the target.

> **Security boundary:** these tools are deliberately not a sandbox. Like Pi, they accept absolute
> paths, inherit the complete environment, and allow arbitrary shell commands. Run Aeloon only in
> an environment whose filesystem, credentials, and processes the model is allowed to access.

## System prompt and resources

Resources use `~/.aeloon-core` globally and `<workspace>/.aeloon-core` for the project:

```text
.aeloon-core/
├── SYSTEM.md
├── APPEND_SYSTEM.md
├── skills/<name>/SKILL.md
└── prompts/<name>.md
```

`SYSTEM.md` replaces the generated base prompt. `APPEND_SYSTEM.md`, project instructions, Skills
XML, and the current working directory are still appended in Pi order. Project resources override
same-named global resources. Resources are reloaded at every turn boundary.

Project context is loaded from a global `AGENTS.md`/`CLAUDE.md`, then from the workspace's ancestor
chain from outermost to innermost. In each directory, only the first matching filename is used.

## Sessions and compaction

New sessions are stored under `<data-dir>/harness-sessions/`; the legacy `sessions/` directory is
neither changed nor read. Each session is a version-3 JSONL file with an immutable header and
append-only tree entries for messages, model/thinking/tool changes, compactions, branch summaries,
custom messages, labels, session info, leaf navigation, and Bridge v2 `run_start`/`run_end`
boundaries. Run boundaries are ignored by model context but allow stable turn reconstruction and
daemon-crash interruption detection. Compaction retains the `run_start` paired with kept messages.

Every `message_end` is persisted and fsynced immediately. Restarting therefore restores the last
save point instead of a cumulative turn snapshot. `navigate_tree()` can return to any entry and can
optionally summarize the abandoned branch.

Automatic semantic compaction is enabled by default with:

```json
{"reserve_tokens": 16384, "keep_recent_tokens": 20000}
```

The cut point is turn-safe. Summaries preserve goals, decisions, progress, exact paths and errors,
include file read/write summaries, and merge an earlier compaction summary. The current model is
used for both compaction and branch summaries.

## CLI

The task is the default command. A normal workflow only needs the task itself and, when needed,
`resume`:

```bash
# Start a saved task in the current workspace
uv run aeloon "fix the failing tests"

# Continue the newest task for this workspace; no session id is needed
uv run aeloon resume "continue with the implementation"

# Read a task from a file or pipe
uv run aeloon --file task.md
printf 'review this change' | uv run aeloon

# Common task options
uv run aeloon -C ../project -m studio/qwen3-coder "review the repository"
uv run aeloon --ephemeral "answer without saving a session"
uv run aeloon --json "return one machine-readable result"
uv run aeloon -v "show concise tool activity"
uv run aeloon -vv "also show lifecycle events"
```

Local API, cloud account, model selection, and recovery commands are intentionally explicit:

```bash
# Local OpenAI-compatible API; omit --model to discover GET /models
uv run aeloon local add ollama \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3-coder \
  --no-api-key
uv run aeloon local list

# Or connect Aeloon Cloud
uv run aeloon login
uv run aeloon whoami
uv run aeloon logout

# With no pinned default, the first model in this list is used automatically
uv run aeloon models
# Optionally pin a local or cloud model as the default
uv run aeloon models use ollama/qwen3-coder
uv run aeloon models use aeloon-cloud/reasoner

uv run aeloon history
uv run aeloon doctor
```

`history` and `models` render compact tables for people and accept `--json` for automation.
`doctor` checks the config, workspace, effective model, credentials, and optional Bridge daemon,
then prints a concrete recovery command for each problem. `local add` reads an optional API key
from a hidden prompt and stores it in the mode-0600 config; `login` stores cloud credentials in the
account vault. Neither command accepts a secret as a command-line argument.
`-m PROVIDER/MODEL` overrides either the automatic or pinned default for one run without changing
the saved selection.

Shell completion scripts can be generated without an additional runtime dependency:

```bash
uv run aeloon completion zsh > ~/.zfunc/_aeloon
uv run aeloon completion bash > ~/.local/share/bash-completion/completions/aeloon
uv run aeloon completion fish > ~/.config/fish/completions/aeloon.fish
```

Advanced configuration and compatibility provider commands remain available:

```bash
uv run aeloon config show
uv run aeloon config set model studio/qwen3-coder
uv run aeloon provider list

# Bridge administration is grouped under the advanced system surface
uv run aeloon system bridge status
uv run aeloon system bridge stop
```

The former command surface remains as a compatibility layer for existing scripts:

```bash
# Explicit new session, existing session, or ephemeral run
uv run aeloon-core run "task"
uv run aeloon-core run "continue" --session <id>
uv run aeloon-core run "one shot" --no-session

# Output modes
uv run aeloon-core run "task" --output text
uv run aeloon-core run "task" --output json
uv run aeloon-core run "task" --output stream-json

# Show tool execution details on stderr (default is quiet)
uv run aeloon-core run "task" --verbose

# Session inspection
uv run aeloon-core session list
uv run aeloon-core session show <id>

# Configuration
uv run aeloon-core config path
uv run aeloon-core config init
uv run aeloon-core config show
uv run aeloon-core config set model deepseek/deepseek-v4-pro
uv run aeloon-core config set max-retries 5

# Aeloon Cloud account (the password is read from a hidden terminal prompt)
uv run aeloon-core cloud login <username>
uv run aeloon-core cloud status
uv run aeloon-core cloud logout

# Unified Provider commands (`cloud ...` is a compatibility alias)
uv run aeloon-core provider login <username>
uv run aeloon-core provider list

# Original Bridge v2 path
uv run aeloon-core bridge ensure --output json
uv run aeloon-core bridge status --output json
uv run aeloon-core bridge schema
uv run aeloon-core bridge stop
```

`stream-json` writes one typed harness event per line followed by a `result` object. `json` writes
only the result object to stdout. In an interactive terminal, normal text mode keeps a single
status line updated on stderr and renders the final Markdown response on stdout. Redirected output
remains stable and quiet. Pass `--quiet` to suppress interactive status, `-v` for concise
run/read/write/search summaries, or `-vv` for additional lifecycle events.

## Bridge v2

`CoreService` is the application boundary between the private harness/session/config layers and
external clients. The Bridge transport is JSON-RPC 2.0 over NDJSON on a Unix domain socket; it
exposes stable session, operation, catalog and revisioned settings DTOs rather than Python types,
raw configuration, system prompts or provider payloads.

`bridge ensure` is concurrency-safe and starts a detached daemon only when needed. The runtime
directory is mode `0700`, the socket and daemon metadata are `0600`, and conflicting config,
data-directory or socket parameters never kill an existing daemon. `--socket`, `--config` and
`--data-dir` are supported by the daemon-management commands.

The `cloud login`, `cloud status`, and `cloud logout` commands use the same daemon account methods
as UI clients, keeping the in-memory account state, credential vault, events, and dynamic model
catalog synchronized. `cloud login` never accepts a password argument; it reads the password with
terminal echo disabled. All three commands support `--output json`, `--config`, `--data-dir`, and
`--socket`.

`provider login/status/logout` expose the same cloud account through the unified Provider surface.
`local add` (and its compatibility alias `provider add`) registers an OpenAI-compatible endpoint
and prompts for its API key without placing the key in shell history; pass `--no-api-key` for
endpoints such as an unauthenticated Ollama server. It discovers models from `GET /models` when
`--model` is omitted; repeat `--model` to register them explicitly. Core stores the key only in the
mode-`0600` config, returns redacted Provider DTOs, and strips the provider prefix before sending
the model key to the upstream API.

Bridge clients use `provider.list`, `provider.local.add`, `provider.local.remove`, and
`provider.cloud.login/status/logout`. `catalog.get` returns both `providers` and the merged `models`
array; every model in that array uses the same `provider/model` namespace.

An explicit `system.shutdown` first publishes a `system.shutdown` event with
`intentional: true`. `status` and `stop` are idempotent: a missing daemon, refused connection or
stale socket returns `status: stopped`; `stop --output json` is available for automation.

The daemon serializes operations within a session and limits cross-session concurrency. It retains
5,000 ordered public events for cursor replay. Client disconnects do not cancel work; clients
reconcile with `session.get` when the daemon instance changes or replay has a gap. Attachments are
validated against roots declared during handshake and copied into Core-owned per-session storage
before an operation is queued.

Bridge v2 is intentionally incompatible with Bridge v1. Expert, MCP and the former capability mode
are not part of this protocol.

## Development

The default test suite is offline and uses only Python behavior specifications—no Pi install,
generated Pi fixtures, or network access.

```bash
uv run pytest -q
uv run ruff check .
uv build
git diff --check
```

Provider credentials used for optional live tests must be written to an explicit test config; Core
does not import API keys from the process environment. Live tests are not part of default CI.
