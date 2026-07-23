# Aeloon Core

Aeloon Core is a small dynamic agent workflow runtime.

> **Master 写 Flow、看结果、动态改图；Worker 自己找路、交付节点结果。**

The Master owns the conversation and authors a durable dynamic Flow. It decides
dependencies, parallel frontiers, review-driven revisions, and termination. A Worker
receives one outcome-oriented node objective, chooses its own tools and Skills,
completes the work, and returns a bounded report. Selecting a Worker type is only an
executor-binding detail inside that larger Flow.

Master and Worker both use PydanticAI Core for the model–tool loop. Aeloon remains
the control plane for Flow scheduling, WorkerSession identity, Namespace/Skill
capabilities, permissions, cancellation, persistence, and commit ordering.

## Quick start

```bash
uv sync
cd aeloon_core/tui && bun install && cd ../..

export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"
export ANTHROPIC_API_KEY="你的 API Key"
export ANTHROPIC_MODEL="k3[1m]"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=1048576
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576

uv run aeloon-core
```

The model gateway uses PydanticAI's official Anthropic integration. With the Kimi
base URL above, the Anthropic SDK sends requests to
`https://api.kimi.com/coding/v1/messages`; tool definitions and history use
Claude `tool_use` / `tool_result` content blocks. Aeloon accepts Claude Code's
`k3[1m]` environment value and automatically sends Kimi's API model ID `k3`.

### Volcano Engine Ark Agent Plan

The Volcano Engine provider uses Ark Agent Plan's OpenAI-compatible Responses API,
which is the protocol recommended by the official OpenCode integration guide:

```bash
export AELOON_CORE_PROVIDER="volcengine"   # sets agents.defaults.provider
export ARK_API_KEY="你的火山方舟 API Key"
export ARK_MODEL="ark-code-latest"

uv run aeloon-core
```

The provider defaults to the Agent Plan endpoint
`https://ark.cn-beijing.volces.com/api/plan/v3`. Do not replace it with the regular
pay-as-you-go `/api/v3` endpoint when you intend to consume an Agent Plan subscription.
Set `ARK_BASE_URL` only when you intentionally need a different Ark endpoint.

Run one non-interactive turn with:

```bash
uv run aeloon-core run "Inspect the repository and explain its entry points"
```

Useful TUI commands include:

```text
/worker-types
/workers
/worker <label>
/spawn <worker-type> <objective>
/resume-worker <response…>
/cancel [run]
/sessions
/new
/quit
```

Independent Workers run concurrently. Runs in one WorkerSession retain a single
private context lineage and are continued explicitly.

The TUI renders streaming Master responses as concealed Markdown. Concurrent Workers
remain visible as live inline blocks in the Master transcript; use the right-hand
catalog or `Tab` to expand a Worker's full phase, todo, timeline, and result detail.
Set `AELOON_CORE_TUI_ASCII=1` when the terminal needs ASCII-only status glyphs.

## Master and Worker boundary

| Actor | Tools |
|---|---|
| Master | `list`, `read`, `glob`, `grep`; create/list/inspect/extend/advance/revise/pause/complete Flow; low-level Worker lifecycle escape hatches |
| Worker | `list`, `read`, `write`, `str_replace`, `glob`, `grep`, `exec`, `webfetch`, `websearch`, `todowrite`, optional `skill` |
| Worker terminal | `complete_work`, `request_master` |

The Master handles tiny observations itself. It should not create a Worker just to
perform an `ls -la`-sized action, and it should not send command lists or prescribed
steps. A delegated request has one field: `objective`.

### Internal module boundaries

The control plane separates pure state contracts from persistence and execution:

| Concern | Module |
|---|---|
| Flow DAG models and state transitions | `aeloon_core.flow_state` |
| Flow runtime projections and WorkerRun bindings | `aeloon_core.flow_runtime` |
| Flow SQLite authority and turn commits | `aeloon_core.flows` |
| Flow orchestration and frontier execution | `aeloon_core.flow_control` |
| Worker lifecycle models and bounded envelopes | `aeloon_core.worker_state` |
| Worker SQLite authority and execution fencing | `aeloon_core.worker_sessions` |
| Live Worker task scheduling and heartbeats | `aeloon_core.worker_manager` |

`aeloon_core.flows` and `aeloon_core.worker_sessions` continue to re-export their
historical domain types so existing integrations can migrate without a flag day.
New control-plane code should import domain contracts from `flow_state` or
`worker_state` and persistence stores from `flows` or `worker_sessions`.

## Dynamic Flows

Multi-stage work is represented as a first-class, appendable DAG rather than an
implicit sequence of Worker calls. For example:

```text
plan
  ├─ build_1 ─┐
  ├─ build_2 ─┼─ review
  └─ build_3 ─┘
```

The Master creates semantic nodes with `objective`, `worker_type_id`, and
`depends_on`. A node may also set `worker_session_policy` to `auto` (the default) or
`fresh`, plus up to four explicit `context_refs`. Flows default to
`advance_mode: "checkpointed"`, where each `advance_flow` launches and joins one ready
frontier before returning control to Master. When every downstream objective is already
complete and its inputs live in the shared workspace, Master may explicitly choose
`advance_mode: "auto"` and a limit of at most four frontiers. Auto mode continues only
after an entirely completed frontier and stops after executing a reviewer frontier, at
the configured limit, or on any partial, failed, cancelled, waiting, timeout, launch
error, active Run, round limit, or lack of ready work. It never revises, retries, skips,
or completes a Flow on Master's behalf.

After any checkpoint, Master can evaluate results and dynamically choose one of:

- `add_flow_nodes` to expand or replan the graph;
- `revise_flow_node` to create a new generation and rerun only affected descendants;
- `retry_flow_node` for a technical/non-successful Run outcome; a reusable `partial`
  checkpoint requires a Master-authored `budget_increase`;
- `resume_flow_node` for an exact `waiting_for_context` continuation, optionally with
  a Master-authored `budget_increase`;
- `complete_flow` for an explicit completed, partial, or blocked outcome.

WorkerSession selection follows a durable, inspectable policy:

| Situation | WorkerSession action |
|---|---|
| New independent node or branch | Create a new WorkerSession |
| Revision of the same node | Reuse its healthy WorkerSession by default |
| Ordinary retry of the same node | Reuse its healthy WorkerSession by default |
| `waiting_for_context` | Always resume the exact WorkerSession and source Run |
| Worker missing, outcome unknown, or context polluted | Create a new WorkerSession |
| Reviewer requiring an independent audit | Set `worker_session_policy: "fresh"` |

`fresh` applies to new non-resume executions; it never breaks the exact-continuation
invariant for `waiting_for_context`. Lost or unknown Worker state automatically falls
back to a new WorkerSession. When Master judges a context polluted, it passes
`fresh_worker=true` and a concrete `fresh_reason` to `revise_flow_node` or
`retry_flow_node`. Flow inspection exposes the requested policy, the resolved
`new`/`reuse`/`resume` action, its reason, and the pending/last Run budget.

Dependencies are scheduling edges, not implicit data pipes. Upstream reports remain
untrusted task data and are never silently appended to a downstream authoritative
objective. When build objectives depend on a planner's conclusions, Master advances
the planner first and then dynamically authors the build nodes from its own synthesis.
Static downstream nodes are appropriate when their inputs already live as durable
shared-workspace artifacts or their objectives are known in advance.

For a fresh follow-up Worker that needs evidence from related work, Master opts in
with `context_refs`. A `flow_node` reference may target an ancestor in the same Flow;
a `worker_run` reference may target a settled Run owned by the same Master session,
including a prior Flow. Each reference records a relation and selects bounded
`objective`, `summary`, `artifacts`, `evidence`, or `unresolved` sections. The packet is
sent separately as untrusted reference material: it neither changes the authoritative
objective nor creates WorkerSession lineage. Flow inspection exposes these durable
associations and each WorkerRun's resolved reference ids.

Only `completed` and explicitly `skipped` dependencies unlock the default join.
`partial`, `failed`, `cancelled`, `waiting_for_context`, `queued`, and `running` are
never mistaken for success. A partial node cannot be converted to `skipped`: Master
must increase its request/output/token/time/tool target, revise it, or finish the Flow
as partial. A node may opt into `all_terminal` when its purpose is to diagnose failed
branches.

Review approval is not inferred from words in a free-form Worker report. Master reads
the report and explicitly completes or revises the Flow. Revision increments the
target generation, marks only transitive descendants stale, and preserves unaffected
parallel branches. Stable node/generation/attempt idempotency keys make frontier
recovery safe after an interrupted dispatch.

An open or cancelling Flow prevents Master from ending the turn with bare text.
`complete_flow`
persists one Flow's explicit outcome without ending the turn; after every open Flow
has been completed, paused, blocked, or cancelled, the terminal
`finish_turn(final_content)` tool produces the user response. If user input is needed,
Master pauses a quiescent Flow, asks the question, and resumes the same persisted Flow
on a later turn. `max_rounds` provides an independent liveness bound for dynamic
revision loops.

`cancel_flow` first enters a durable `cancelling` state. It becomes cleanly `cancelled`
only after every Worker has settled without an uncertain in-flight tool outcome. If an
owner disappears during tool execution, the Flow becomes `blocked` instead and exposes
the unknown outcome for inspection. Tool boundaries re-check Run authority, so a stale
owner is fenced from issuing new mutations after cancellation wins.

Worker types are soft responsibilities. `explorer`, `builder`, `researcher`, and `reviewer`
receive the same domain capability set; their definitions only change the
responsibility prompt. Workers never receive Worker scheduling tools or any nested
agent capability. The Master never receives mutation, shell, web, or Skill tools.

## Worker definitions

Built-ins live in `aeloon_core/builtin_workers`. A project can add or override a type
with `.aeloon-core/workers/*.md`:

```markdown
---
id: reviewer
description: Independently inspect changes and return evidence-backed risks
---
Review the requested outcome in the shared workspace. Verify findings and report
only actionable issues with concrete evidence.
```

Frontmatter is strict: only `id` and `description` are accepted, duplicate keys and
YAML aliases are rejected, and the Markdown body must be non-empty. Definitions are
discovered once at process startup. A project definition overrides a built-in with
the same `id`.

Creating a WorkerSession persists the complete immutable snapshot:

```text
WorkerSnapshot(id, description, prompt, source, digest)
```

Changing a file affects new WorkerSessions after restart. Existing sessions continue
with their stored snapshot and digest.

## Completion and continuation

A WorkerRun must end with exactly one typed PydanticAI output:

```text
complete_work(summary, artifacts=[], evidence=[])
request_master(summary, question, artifacts=[], evidence=[])
```

A terminal call mixed with any other tool call—or multiple terminal calls in one
response—is rejected before any tool executes. Plain Worker text cannot complete a
Run; PydanticAI requests a corrected typed output within the request budget.

Evidence is structured and host-validated:

```json
{
  "kind": "file | test | typecheck | lint | runtime | source",
  "locator": "file:line, test node, URL, or runtime object",
  "claim": "what the evidence proves",
  "status": "passed | failed | observed | not_applicable",
  "method": "actual command or observation method",
  "finding_id": "optional reviewer finding ID"
}
```

A passed executable check must match a command actually run in the current WorkerRun
with exit code zero, and file locators must resolve to valid workspace lines. A builder
that used mutation tools cannot complete without test, typecheck, and lint accounting
plus at least one successful executable check. A reviewer must run a relevant test or
runtime reproduction, and every reported finding ID must link to evidence. Invalid
terminal output receives `ModelRetry`; the Run is not marked completed. Historical
string evidence remains readable as `legacy` evidence but cannot be emitted by new Runs.

Run states are:

```text
queued → running → completed | partial | waiting_for_context | failed | cancelled
```

`waiting_for_context` is settled, so `await_workers` returns immediately. The Master
answers with `resume_worker(run_id, response, idempotency_key)`. Resume creates the
next Run in the same WorkerSession, references the exact `source_run_id`, and restores
that Run's checkpoint. The waiting Run is never reopened.

Checkpoint, structured question, result, and waiting status are committed in one
SQLite transaction. Reuse and resume inherit the prior permission domain; the host
rejects permission expansion and idempotency conflicts. Workers cannot extend their
own limits. Master may provide a `budget_increase` with strictly higher target limits;
the resolved grant is durably attached to the next exact-checkpoint Run.

WorkerRuns use the configured request limit and have no cumulative token or tool-call
cap by default. The model context
window is a separate per-request concern. A new Worker starts from the minimal dispatch
envelope—its objective, permission domain, and budget—rather than a copy of the Master
transcript. PydanticAI `ProcessHistory` applies Aeloon's client-side compaction policy
when history approaches the configured context window. A finite internal grant remains a hard Run
bound when an embedding host explicitly supplies one, and the wall-clock timeout plus
`cancel_worker` remain the liveness controls.

Cancellation of queued work settles immediately. For a detached running Worker, the
control service first records a durable cancellation request; the owning runner cancels
its model/tool coroutine and only then acknowledges the Run as `cancelled`. A Flow Run
is reserved first, durably bound, and only then marked activated, allowing another runner
to recover it without ever claiming an unbound reservation. Every executing tool holds a
durable in-flight marker, and each Run claim records a unique process-owner epoch backed by
an exclusive local file lock. While the owner is healthy or performing controlled teardown,
the marker prevents lease expiry from exposing a terminal Flow result before the tool and
its cleanup finish. A stale in-flight
marker is recovered only after the kernel releases that exact owner's lock on process exit;
heartbeat delay or an event-loop stall alone cannot clear it. Continuous runners keep polling
for unrelated queued work while existing Workers are active. If an owner dies with a tool in
flight, the Run is `failed` with `tool_outcome=unknown` even when cancellation was already
requested: clearing the control-plane marker cannot prove that shell descendants or remote
side effects stopped or rolled back. A cancelling Flow containing such a Run becomes `blocked`
and instructs the Master to inspect side effects; only cancellation with no uncertain in-flight
tool work becomes cleanly `cancelled`.

## Runtime limits and prompt caching

The loop deterministically detects an identical successful read-only tool action and
observation repeated across four distinct model steps. It keeps at most 20 complete
exchanges, compares canonical digests rather than raw large results, and never treats
mutating, failed, incomplete, or same-batch calls as stuck. Before another identical
call executes, the runtime returns `ModelRetry` and requires a different strategy.
`UsageLimits.request_limit` remains the final liveness boundary. Raising it requires a
Master-authored continuation; there is no model Guard, automatic budget continuation,
or tool-error reviewer.

Each model request sets `max_tokens` from `agents.defaults.max_output_tokens` (default
`32768`). Without an explicit value, Anthropic-compatible providers silently use `4096`,
which thinking models often exhaust before emitting tools or text. Finite Worker cumulative
token grants are checked before every request. The runtime counts the next input when
supported, otherwise uses a conservative estimate, and lowers that request's `max_tokens`
so an over-budget response cannot reach tool execution.

Anthropic prompt caching is enabled by default. The Master instruction block and Worker
type metadata form a stable system prefix; volatile Worker/Flow state is appended at the
user tail. Anthropic-compatible gateways that clearly reject `cache_control` are retried
once without it and remembered for later requests; unrelated provider errors are not
masked by this compatibility fallback.

## Model routing and telemetry

Model selection is process-scoped and role-aware. The default pair lives under
`agents.defaults.provider` + `agents.defaults.model` (there is no global
`providers.active` switch). On the official Anthropic endpoint with that default
provider, Master, explorer, and researcher use `claude-haiku-4-5-20251001`; builder,
reviewer, and unknown custom Worker types use the default pair. Compatible
gateways and Volcengine safely use the default pair unless a role is explicitly
overridden via `agents.routing` (bare model name or `provider/model`). When an
override cannot be used, the router falls back to the config default pair.
Provider/model bundles are reused while each bundle keeps its own prompt
cache state. Constructor-level `model` and `model_settings` injection remains an all-role
override for tests and embeddings.

Each Worker result records its resolved model, role, request and tool counts, tokens,
duration, budget utilization, partial reason, and budget-increase count. Flow state
records frontier widths, auto-advanced frontier count, revision count, and stop reason.
The global `max_iterations` default remains 25; optional per-role request budgets allow
telemetry-driven tuning without changing the default policy.

## Skills

Skills belong only to Workers. Each WorkerRun receives the current process's Skill
summary and the `skill` tool when Skills are enabled. Loaded Skill content remains
complete during the current Run; later Runs can load it again from the refreshed
catalog instead of permanently injecting every Skill into the system prompt.

Project-native locations are:

```text
.aeloon-core/skill/<name>/SKILL.md
.aeloon-core/skills/<name>/SKILL.md
.opencode/skill/<name>/SKILL.md
.opencode/skills/<name>/SKILL.md
```

Compatible `.claude/skills` and `.agents/skills` locations and their global variants
are enabled by default. Configure discovery under `skills`:

```json
{
  "skills": {
    "enabled": true,
    "external": true,
    "claude_code": true,
    "paths": ["./team-skills"]
  }
}
```

Host-discovered Worker definitions and Skill contents are trusted workflow
configuration. Workspace files, tool output, web content, and data referenced by a
Skill remain untrusted task data.

## Persistence and PydanticAI migration

Master turns remain JSONL projections. Flow state, idempotent graph decisions, turn
leases, and terminal-response commits use the independent `flow-control.sqlite3` store
with current schema version 5. Each terminal commit now atomically appends an immutable,
parent-linked `turn.committed` event and advances a durable session-head pointer before
JSONL projection or UI delivery. Event payloads reference the existing full commit
snapshot instead of duplicating it. A crash can therefore recover and project the result
idempotently without rerunning the model. Master resume resolves the current head snapshot
through indexed lookups before falling back to the compatibility JSONL projection, while
the shared head gives internal conversation-only forks O(1) data creation and establishes
the boundary for future incremental events. Flow and Worker ownership are deliberately not
inherited by such a fork.
Worker control state uses `worker-control.sqlite3` at schema version 5; private
Worker transcripts are stored below `worker-sessions/` and are never exposed to Master.

New Master turns, Flow commits, Worker transcripts, and checkpoints use
`schema_version: 2`, `message_format: "pydantic-ai-v1"`, and PydanticAI's
`ModelMessagesTypeAdapter`. Legacy JSONL, Flow payloads, and checkpoints remain listable
and auditable but cannot be resumed; execution raises `LegacySessionError` and requires a
new session. Migration never deletes or overwrites those records.

## Config

Configuration defaults to `~/.aeloon-core/config.json`:

```bash
uv run aeloon-core config init \
  --api-key sk-... \
  --base-url https://api.kimi.com/coding/ \
  --model 'k3[1m]'

uv run aeloon-core config show
uv run aeloon-core config set max-iterations 25
uv run aeloon-core config set master-model claude-haiku-4-5-20251001
uv run aeloon-core config set reviewer-model claude-sonnet-4-6
uv run aeloon-core config set explorer-max-iterations 12
uv run aeloon-core config set context-compaction-enabled true
uv run aeloon-core config set prompt-caching true
uv run aeloon-core config set runtime-stuck-detection-threshold 4
```

Alternatively, initialize a Volcano Engine Agent Plan configuration:

```bash
uv run aeloon-core config init \
  --provider volcengine \
  --api-key ark-... \
  --model ark-code-latest
```

When loading a v1 config, v2 ignores the removed `base_profile_id`, `profile_id`,
and `max_handoffs` settings. The next `config set` or `config init --force` writes
the file back without them. Legacy `providers.active` migrates to
`agents.defaults.provider`. Legacy `uasm` trace/stuck settings migrate to `runtime`;
removed `tool_error_guard_threshold`, `budget_auto_continues`, and per-round
minimal-context settings are ignored. Unrelated unknown agent-default settings remain
validation errors.

Equivalent default provider/model, routing, and role-budget settings can be written
directly:

```json
{
  "agents": {
    "defaults": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6"
    },
    "routing": {
      "master": "anthropic/claude-haiku-4-5-20251001",
      "workers": {
        "explorer": "fast-model",
        "reviewer": "volcengine/ark-code-latest"
      }
    },
    "budgets": {
      "master": 20,
      "workers": {"explorer": 12, "reviewer": 30}
    }
  }
}
```

Common environment overrides are:

- `AELOON_CORE_PROVIDER` (`anthropic` or `volcengine`; sets `agents.defaults.provider`)
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`
- `ARK_API_KEY`
- `ARK_BASE_URL`
- `ARK_MODEL`
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW`
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS`
- `AELOON_CORE_WORKSPACE`
- `AELOON_CORE_DATA_DIR`
- `AELOON_CORE_SKILLS_ENABLED`
- `AELOON_CORE_SKILL_PATHS`

## Atomic file tools

`write` atomically creates or overwrites a complete UTF-8 file. Use `str_replace`
for exact edits; an empty `old_str` creates a missing file, while non-empty matches
must be unique unless `replace_all=true`. Both advertise model-aware per-call
character guidance (soft preference only), reject symlink/protected paths, recheck
the target before commit, and use an atomic same-directory replacement. The only
hard size ceiling is the 16 MiB UTF-8 file limit.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv build
(cd aeloon_core/tui && bun run check)
git diff --check
```

The representative control-plane evaluation manifest is
`benchmarks/master_worker_eval_cases.json`. It covers single-node delivery, parallel
DAGs, research-gated graph construction, review-driven revision, budget continuation,
and `request_master`, with stable quality and latency metrics for before/after runs.
After recording one result ledger from each version, compare them with:

```bash
uv run python -m benchmarks.compare_master_worker_eval before.json after.json
```
