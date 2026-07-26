# Aeloon Core

Aeloon Core is a conversation-scoped master/agent runtime built on Pydantic AI
Harness.

> **Master 负责当前对话；Harness agents 在当前 turn 内完成全部工作。**

Phase 1 intentionally has one execution model:

- the Master owns the user conversation and final answer;
- Pydantic AI Harness `DynamicWorkflow` provides fan-out, chaining, voting, and
  synthesis;
- every child agent is isolated and must finish before the current turn returns;
- child-agent context, checkpoints, leases, and workflow state are not persisted.

There is no custom durable Flow DAG, WorkerSession store, detached runner, resume
protocol, or duplicate filesystem/shell implementation. Harness supplies child-agent
filesystem access, shell execution, repository context, planning, usage forwarding,
agent-call limits, and sliding-window history compaction.

## Quick start

```bash
uv sync
bun --version  # requires Bun >= 1.3

export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"
export ANTHROPIC_API_KEY="你的 API Key"
export ANTHROPIC_MODEL="k3[1m]"

uv run aeloon-core
```

The default command starts the local Web UI on `127.0.0.1:7331` and opens it in
your browser. The bootstrap URL contains a one-time token; after the first load,
the browser receives a process-local HttpOnly cookie.

Useful variants:

```bash
uv run aeloon-core serve --no-open
uv run aeloon-core serve --port 0
uv run aeloon-core run "Inspect the repository and explain its entry points"
```

The Web UI has three pieces: the Master conversation, live agents for the current
turn, and diagnostic logs. Live agent events are display-only and disappear when
the next turn starts. There are no Flow recovery or Worker resume controls.

## Execution model

The Master handles small observations with read-only `list`, `read`, `glob`, and
`grep` tools. For a self-contained delegated outcome, it calls Harness
`run_workflow` with a small sandboxed Python program:

```python
import asyncio

implementation, review = await asyncio.gather(
    builder(task="Implement the scoped change and verify it"),
    reviewer(task="Independently inspect the requested behavior and report risks"),
)
{"implementation": implementation, "review": review}
```

Each named call:

- starts an isolated Pydantic AI agent;
- receives the same host-owned dependency context but no Master transcript;
- can use Harness `FileSystem`, `Shell`, `RepoContext`, and `Planning`;
- cannot recursively delegate;
- returns a typed `WorkerReport`;
- contributes usage to the parent run.

The workflow program may also chain results when the next objective genuinely
depends on an earlier report. Reports and workspace content remain untrusted task
data, not higher-priority instructions.

If work needs user input, the Master asks the user directly. Phase 1 does not leave
a child agent waiting across turns.

## Worker definitions

Built-in responsibilities live in `aeloon_core/builtin_workers`. Projects may add
or override them with `.aeloon-core/workers/*.md`:

```markdown
---
id: reviewer
description: Independently inspect changes and return evidence-backed risks
---
Review the requested outcome. Verify findings and return only actionable issues.
```

Definitions are discovered once at process startup. Frontmatter accepts only `id`
and `description`; the Markdown body is the responsibility prompt.

Child agents return:

```text
WorkerReport(
  summary,
  artifacts=[],
  evidence=[{kind, locator, claim, status, method?, finding_id?}],
  unresolved=[],
)
```

## Persistence boundary

Only completed Master conversation turns are stored as JSONL under the configured
`data_dir`. That history lets a user continue the conversation.

Child-agent prompts, messages, plans, and tool state live only in memory for the
current turn. A process interruption ends them; the next turn starts fresh agents.
This is deliberate in Phase 1.

## Runtime structure

| Concern | Module |
|---|---|
| Master turn and conversation persistence | `aeloon_core.orchestrator`, `aeloon_core.session` |
| Harness capabilities and ephemeral agents | `aeloon_core.harness_runtime` |
| Worker prompt catalog and typed reports | `aeloon_core.workers` |
| Provider/model routing | `aeloon_core.model_router`, `aeloon_core.pydantic_model` |
| Pydantic AI policy and event adapter | `aeloon_core.pydantic_runtime` |
| Web transport and live projections | `aeloon_core.web_bridge`, `aeloon_core.turn_events` |

## Configuration

Configuration is JSON. The relevant agent section is:

```json
{
  "agents": {
    "defaults": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "max_iterations": 25,
      "max_output_tokens": 32768,
      "context_window_tokens": 128000,
      "context_compaction": {
        "enabled": true,
        "trigger_ratio": 0.9,
        "preserve_recent_tokens": null
      }
    },
    "routing": {
      "master": null,
      "workers": {
        "reviewer": "volcengine/ark-code-latest"
      }
    },
    "harness": {
      "max_agent_calls": 16,
      "sub_agent_request_limit": 25,
      "workflow_cpu_seconds": 10.0
    }
  }
}
```

Use `uv run aeloon-core config show`, `config init`, and `config set` to inspect
or change it.

## Development

```bash
uv run pytest -q
uv run ruff check .
(cd aeloon_core/web && bun test)
uv build
git diff --check
```
