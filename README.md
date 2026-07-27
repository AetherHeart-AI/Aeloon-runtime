# Aeloon Core

Aeloon Core is a conversation-scoped master/agent runtime built on Pydantic AI
Harness.

> **Master 负责当前对话；Harness agents 在当前 turn 内完成全部工作。**

The runtime has two in-turn execution paths:

- the Master owns the user conversation and final answer;
- trusted Python `WorkflowTemplate` classes provide a validated fast path for
  common fixed patterns;
- Pydantic AI Harness `DynamicWorkflow` remains the fallback for unmatched or
  genuinely dynamic work;
- every child agent is isolated and must finish before the current turn returns;
- child-agent context, checkpoints, leases, and workflow state are not persisted.

There is no durable Flow DAG, WorkerSession store, detached runner, resume protocol,
or result cache. Harness continues to supply child-agent filesystem access, shell
execution, repository context, planning, and sliding-window history compaction.

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
uv run aeloon-core run \
  --workspace /path/to/repository \
  --prompt-file /path/to/task.txt \
  --output json
```

`run` accepts exactly one of positional prompt text, `--prompt-file`, or
`--stdin`. JSON output suppresses progress rendering so automation receives one
machine-readable result on stdout. See [benchmarks/README.md](benchmarks/README.md)
for the official RefactorBench runner.

The Web UI has three pieces: the Master conversation, live agents for the current
turn, and diagnostic logs. Live agent events are display-only and disappear when
the next turn starts. There are no Flow recovery or Worker resume controls.

## Execution model

The Master handles small observations with read-only `list`, `read`, `glob`, and
`grep` tools. Before each Master request, the Host performs a deterministic local
search over the Workflow Template catalog. A clear match is executed through
`workflow_execute`, so the Master supplies validated inputs instead of generating
orchestration code:

```json
{
  "template_id": "implement-review",
  "inputs": {
    "objective": "Implement the scoped change",
    "acceptance": "Affected tests pass"
  },
  "tuning": {
    "review_focus": "Correctness and data integrity"
  }
}
```

Built-in templates cover single-role delegation, parallel read-only investigation,
implementation plus independent review, and a bounded review/fix/re-review loop.
Run-scoped tuning is validated and never persisted.

When no fixed template is compatible, the Master calls Harness `run_workflow` with
a sandboxed Python program:

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
- returns the Role's typed structured output (built-ins use `WorkerReport`);
- contributes usage to the parent run.

The workflow program may also chain results when the next objective genuinely
depends on an earlier report. Reports and workspace content remain untrusted task
data, not higher-priority instructions.

If work needs user input, the Master asks the user directly. Phase 1 does not leave
a child agent waiting across turns.

## Python Roles and Workflow Templates

Project-facing definitions use the stable bases in
`aeloon_core.harness.agent` and `aeloon_core.harness.workflow`. Roles configure
prompts, output types, model tiers, capabilities, and concurrency. The Harness
still owns agent construction, budgets, lifecycle events, and execution:

```python
from aeloon_core.harness.agent import Role

class ProjectReviewer(Role):
    id = "reviewer"
    description = "Review project changes"
    system_prompt = "Return only verified, actionable findings."
    model_tier = "strong"
    capabilities = ("filesystem", "shell", "repo_context", "planning")
    concurrency_mode = "parallel_safe"
```

`parallel_safe` is a trusted declaration: use it only for read-only work or work
whose mutations cannot overlap. Use `exclusive` for general workspace mutation.

Workflow Templates compile Pydantic inputs into finite validated plans:

```python
from pydantic import BaseModel
from aeloon_core.harness.workflow import WorkflowNode, WorkflowPlan, WorkflowTemplate

class Inputs(BaseModel):
    task: str

class ProjectWorkflow(WorkflowTemplate):
    id = "project-review"
    description = "Run the project review role"
    tags = ("review",)
    when_to_use = "Use for project review requests."
    avoid_when = "Avoid when implementation is also required."
    input_model = Inputs

    def build(self, inputs, tuning):
        return WorkflowPlan(nodes=(
            WorkflowNode(id="review", role_id="reviewer", objective=inputs.task),
        ))
```

Projects export trusted definitions from `.aeloon-core/catalog.py`:

```python
ROLES = (ProjectReviewer,)
WORKFLOWS = (ProjectWorkflow,)
```

Definitions are loaded once at process startup; restart after changes. Project IDs
override built-ins.

## Module boundaries

The Harness is one system organized by cohesive feature packages. Each feature
keeps its base contracts, presets, and runtime implementation together:

```text
aeloon_core/
├── harness/
│   ├── agent/              # Role base, presets, prompts, and agent factory
│   ├── model/              # model bindings and routing
│   ├── provider/           # provider construction and shared transport policy
│   ├── tool/               # tool base, registry, filesystem, and search tools
│   ├── workflow/           # template base, presets, runner, and Master tools
│   ├── execution/          # run engine, events, traces, and stuck detection
│   └── catalog.py          # built-in and project definition discovery
├── conversation/           # message serialization and session persistence
├── web/                    # bridge, event projection, launcher, and UI assets
├── orchestrator.py         # application composition root
└── config.py               # configuration schema and persistence
```

Project definitions subclass only the `agent` and `workflow` base contracts.
They describe behavior but cannot own budgets, execution, lifecycle, or model
construction. `orchestrator.py` remains the composition root and imports explicit
feature APIs instead of relying on a flat compatibility layer.

## Persistence boundary

Only completed Master conversation turns are stored as JSONL under the configured
`data_dir`. That history lets a user continue the conversation.

Child-agent prompts, messages, plans, and tool state live only in memory for the
current turn. A process interruption ends them; the next turn starts fresh agents.
This is deliberate in Phase 1.

## Runtime structure

| Concern | Module |
|---|---|
| Master turn and conversation persistence | `aeloon_core.orchestrator`, `aeloon_core.conversation` |
| Role contracts, presets, prompts, and construction | `aeloon_core.harness.agent` |
| Workflow contracts, presets, execution, and tools | `aeloon_core.harness.workflow` |
| Tool contracts and built-in tools | `aeloon_core.harness.tool` |
| Provider construction and model routing | `aeloon_core.harness.provider`, `aeloon_core.harness.model` |
| Pydantic AI execution, events, and runtime safeguards | `aeloon_core.harness.execution` |
| Web transport and live projections | `aeloon_core.web` |

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
      "max_worker_continuations": 4,
      "workflow_cpu_seconds": 10.0
    },
    "templates": {
      "enabled": true,
      "max_concurrency": 4,
      "max_nodes": 16,
      "max_upstream_chars": 32000,
      "presearch_limit": 3
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
