# Aeloon Core

Aeloon Core is a small dynamic agent workflow runtime.

> **Master 写 Flow、看结果、动态改图；Worker 自己找路、交付节点结果。**

The Master owns the conversation and authors a durable dynamic Flow. It decides
dependencies, parallel frontiers, review-driven revisions, and termination. A Worker
receives one outcome-oriented node objective, chooses its own tools and Skills,
completes the work, and returns a bounded report. Selecting a Worker type is only an
executor-binding detail inside that larger Flow.

There is one agent loop for both actors. Its nodes are always
`router → model → tool/guard → done`; responsibility and tool configuration vary,
not the execution engine.

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

The model gateway now uses Anthropic Messages format end to end. With the Kimi
base URL above, the SDK sends requests to
`https://api.kimi.com/coding/v1/messages`; tool definitions and history use
Claude `tool_use` / `tool_result` content blocks. Aeloon accepts Claude Code's
`k3[1m]` environment value and automatically sends Kimi's API model ID `k3`.

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

## Master and Worker boundary

| Actor | Tools |
|---|---|
| Master | `list`, `read`, `glob`, `grep`; create/list/inspect/extend/advance/revise/pause/complete Flow; low-level Worker lifecycle escape hatches |
| Worker | `list`, `read`, `write`, `str_replace`, `glob`, `grep`, `exec`, `webfetch`, `websearch`, `todowrite`, optional `skill` |
| Worker terminal | `complete_work`, `request_master` |

The Master handles tiny observations itself. It should not create a Worker just to
perform an `ls -la`-sized action, and it should not send command lists or prescribed
steps. A delegated request has one field: `objective`.

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
`fresh`. Each `advance_flow` executes exactly one ready frontier: it launches all
independent nodes as distinct WorkerSessions, joins their Runs, synchronizes bounded
results, and returns control to Master. It deliberately does not start the next
frontier in the same call, so Master can evaluate results and dynamically choose one
of:

- `add_flow_nodes` to expand or replan the graph;
- `revise_flow_node` to create a new generation and rerun only affected descendants;
- `retry_flow_node` for a technical/non-successful Run outcome;
- `resume_flow_node` for an exact `waiting_for_context` continuation;
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
`new`/`reuse`/`resume` action, and its reason.

Dependencies are scheduling edges, not implicit data pipes. Upstream reports remain
untrusted task data and are never silently appended to a downstream authoritative
objective. When build objectives depend on a planner's conclusions, Master advances
the planner first and then dynamically authors the build nodes from its own synthesis.
Static downstream nodes are appropriate when their inputs already live as durable
shared-workspace artifacts or their objectives are known in advance.

Only `completed` and explicitly `skipped` dependencies unlock the default join.
`partial`, `failed`, `cancelled`, `waiting_for_context`, `queued`, and `running` are
never mistaken for success. A node may opt into `all_terminal` when its purpose is to
diagnose failed branches.

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

A WorkerRun must end with exactly one terminal tool call:

```text
complete_work(summary, artifacts=[], evidence=[])
request_master(summary, question)
```

A terminal call mixed with any other tool call—or multiple terminal calls in one
response—is rejected before any tool executes. Plain Worker text cannot complete a
Run; it enters the normal correction and Guard path.

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
rejects permission expansion and idempotency conflicts. Each continuation receives
the current Run budget instead of inheriting an obsolete cap.

WorkerRuns have no cumulative token or tool-call cap by default. The model context
window is a separate per-request concern: Workers use the same automatic context
compaction as Master, followed by the bounded minimal-context view. A finite internal
grant remains a hard Run bound when an embedding host explicitly supplies one, and
the wall-clock timeout plus `cancel_worker` remain the liveness controls.

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

## Persistence and v2 migration

Master turns are JSONL records. Flow state, idempotent graph decisions, turn leases, and
terminal-response commits use the independent `flow-control.sqlite3` store with current
schema version 4. A terminal response is committed there before JSONL projection or UI
delivery; a crash can therefore recover and project it idempotently without rerunning the
model.
Worker control state uses `worker-control.sqlite3` at schema version 5; private
Worker transcripts are stored below `worker-sessions/` and are never exposed to Master.

The architecture-v2 upgrade is intentionally destructive only for legacy Worker schema
v1. On first startup it locks the database, drops v1 Worker UI/checkpoint/run/session
tables, creates the current schema, and deletes old `worker-sessions/` transcripts.
Subsequent v2-to-v5 migrations preserve Worker data while adding activation, execution,
and process-owner fences. A migrated v4 in-flight marker without an owner epoch fails
closed and requires the old runner to finish or operator intervention; ownership is never
fabricated from elapsed time. Existing Master sessions and ordinary transition traces
remain. Old external definition/artifact directories are not read and are not deleted.

Stop any old detached runner before upgrading. A migration or transcript-cleanup
failure aborts startup instead of running against a half-migrated store.

## Config

Configuration defaults to `~/.aeloon-core/config.json`:

```bash
uv run aeloon-core config init \
  --api-key sk-... \
  --base-url https://api.kimi.com/coding/ \
  --model 'k3[1m]'

uv run aeloon-core config show
uv run aeloon-core config set max-iterations 25
uv run aeloon-core config set context-compaction-enabled true
```

When loading a v1 config, v2 ignores the removed `base_profile_id`, `profile_id`,
and `max_handoffs` settings. The next `config set` or `config init --force` writes
the file back without them; unrelated unknown agent settings remain validation
errors.

Common environment overrides are:

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW`
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS`
- `AELOON_CORE_WORKSPACE`
- `AELOON_CORE_DATA_DIR`
- `AELOON_CORE_SKILLS_ENABLED`
- `AELOON_CORE_SKILL_PATHS`

## Atomic file tools

`write` creates a new UTF-8 file or appends with the exact byte
`expected_offset`; it never silently overwrites an existing file. `str_replace`
requires an exact unique match unless `replace_all=true`. Both enforce the model-aware
per-call character limit, reject symlink/protected paths, recheck the file baseline,
and commit through an atomic same-directory replacement.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv build
(cd aeloon_core/tui && bun run check)
git diff --check
```
