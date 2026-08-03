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
export DEEPSEEK_API_KEY="your API key"

uv run aeloon-core run "Inspect this repository and explain its entry points"
uv run aeloon-core run --prompt-file task.md --output json
printf 'Fix the failing tests' | uv run aeloon-core run --stdin --output stream-json
```

The default model is `deepseek-v4-flash`; `deepseek-v4-pro` is also built in. Both use Pi 0.83.0's
1M context-window and 384K output-ceiling metadata. Thinking defaults to `off`.

Aeloon Cloud is an optional account-backed provider. Core owns the account session and refresh
credential; UI clients receive only public account status and provider-qualified model IDs such as
`aeloon-cloud/reasoner`. Sign-in is exposed through Bridge v2, and cloud models are added to the
dynamic catalog only while the account is authenticated.

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
    model=get_deepseek_model("deepseek-v4-flash"),
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

```bash
# New session, existing session, or ephemeral run
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
uv run aeloon-core config set model deepseek-v4-pro
uv run aeloon-core config set max-retries 5

# User-level Bridge v2 daemon
uv run aeloon-core bridge ensure --output json
uv run aeloon-core bridge status --output json
uv run aeloon-core bridge schema
uv run aeloon-core bridge stop
```

`stream-json` writes one typed harness event per line followed by a `result` object. `json` writes
only the result object to stdout. Text mode runs quietly by default: only the final response is
buffered and rendered as Markdown when stdout is an interactive terminal (redirected text output
remains plain text). Pass `--verbose` to also write concise run/read/write/search summaries to
stderr as the tools execute.

## Bridge v2

`CoreService` is the application boundary between the private harness/session/config layers and
external clients. The Bridge transport is JSON-RPC 2.0 over NDJSON on a Unix domain socket; it
exposes stable session, operation, catalog and revisioned settings DTOs rather than Python types,
raw configuration, system prompts or provider payloads.

`bridge ensure` is concurrency-safe and starts a detached daemon only when needed. The runtime
directory is mode `0700`, the socket and daemon metadata are `0600`, and conflicting config,
data-directory or socket parameters never kill an existing daemon. `--socket`, `--config` and
`--data-dir` are supported by the daemon-management commands.

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

An optional live DeepSeek smoke test should be run only when a real `DEEPSEEK_API_KEY` is supplied;
it is not part of default CI.
