# Aeloon Core

Aeloon Core is a conversation-scoped agent runtime built on Pydantic AI Harness.
It has two explicit capability-policy modes:

- `normal`: Master can use every discovered plain Skill, every configured MCP
  server, and the full workspace capability surface;
- `expert`: Master and predefined ExpertSkills receive explicit Skill, MCP, and
  host-capability scopes.

> **Normal 放开 Master；Expert 模式收紧 Master 与 ExpertSkill 的能力边界。**

The Master owns the user conversation, final answer, filesystem mutation, shell
execution, repository context, and planning. It can also call enabled
`ExpertSkill`s:

```text
Master = Ultra Worker
           + ExpertSkill calls
           ├── research: plan → 2–4 explorers → primary docs → reduce
           └── coding:   plan → build → review → one fix → one re-review
```

Expert work is ephemeral. It must finish inside the current turn and has no
checkpoint, resume, detached runner, background task, or cross-turn state. Core
does not provide a generic DAG interface.

## Quick start

```bash
uv sync
bun --version  # requires Bun >= 1.3

export DEEPSEEK_API_KEY="your API key"
export EXA_API_KEY="your Exa API key"  # only needed by builtin:research

uv run aeloon-core
```

The default command starts the local Web UI on `127.0.0.1:7331`. Other useful
forms:

```bash
uv run aeloon-core serve --no-open
uv run aeloon-core run "Inspect the repository and explain its entry points"
uv run aeloon-core run --workspace /path/to/repo --prompt-file task.txt --output json
```

## Modes, Skills, and isolation

A Skill is passive, lazily loaded instructions and resources. An ExpertSkill is
also a Skill, but adds a registered runner, capability declaration, dependencies,
and execution policy.

Discovery is intentionally narrow:

1. package built-ins;
2. `<workspace>/.aeloon-core/skills`;
3. roots explicitly listed in config.

Aeloon does not implicitly scan `~/.codex`, `~/.claude`, or `~/.agents`. Canonical
IDs are `<root-id>:<name>`, so equal names in different roots do not override each
other.

`normal` is the default. Master sees all plain Skills found in the discovery
roots, all MCP servers in the explicit MCP config, and all standard workspace
capabilities. In `expert` mode, Master sees only:

- plain Skills in `skills.master_allowlist`;
- MCP servers in `mcp.master_allowlist`;
- host capability groups in `tools.master_capabilities`;
- ExpertSkills in `experts.enabled`.

In both modes Master can use:

- `skill_search`
- `skill_load`
- `skill_read`
- `expert_run`

An expert sees only itself, its declared plain-Skill dependencies, declared MCP
servers, and declared host capabilities. It never receives `expert_run`, and
expert nesting is rejected by the registry.
Standard passive metadata such as `license`, `compatibility`, `metadata`, and
`allowed-tools` is accepted, but it never grants runtime capabilities.

This isolates discovery, prompt context, and Skill tools. It is not an operating
system confidentiality boundary against a generally authorized shell.

## `SKILL.md`

Plain Skill:

```markdown
---
name: project-conventions
description: Project-specific implementation and verification conventions.
---
# Project conventions

Read references/checks.md before changing production code.
```

ExpertSkill:

```markdown
---
name: ppt-builder
description: Build and verify a presentation.
kind: expert
runner: project.ppt
dependencies:
  - workspace:slide-style
capabilities:
  - filesystem
  - shell
  - repo_context
  - planning
mcp-servers:
  - github
model_tier: strong
concurrency_mode: exclusive
max_calls_per_turn: 2
---
# PPT builder

Create the deck, render it, inspect the result, and report artifact paths.
```

Manifests reference a registered runner ID; they cannot import arbitrary Python.
Trusted project runner extensions live in `.aeloon-core/catalog.py`:

```python
from my_project.experts import PptExpertRunner

EXPERT_RUNNERS = {
    "project.ppt": PptExpertRunner(),
}
```

The old `ROLES` and `WORKFLOWS` entries are rejected with a migration error.
Custom one-agent experts can use `runner: builtin.prompt`. A trusted compiled
LangGraph can be registered directly or wrapped with `LangGraphExpertRunner`;
install the optional adapter dependency with:

```bash
uv sync --extra langgraph
```

LangGraph remains an implementation detail of that expert. Core only calls its
runner and normalizes the final `ExpertResult`.

## MCP

MCP servers use the standard `mcpServers` JSON shape:

```json
{
  "mcpServers": {
    "github": {"url": "http://127.0.0.1:8000/mcp"},
    "local": {
      "command": "python",
      "args": ["server.py"]
    }
  }
}
```

Set `mcp.config_path` to this file. Normal mode exposes every server and every
tool returned by those servers to Master. Expert mode applies the configured
server scopes. MCP and Skill discovery never generates ExpertSkills; executable
experts must already have a manifest and registered runner.

## Built-in experts

`builtin:research` runs a bounded research pipeline:

1. a planner emits two to four independent assignments;
2. explorers fan out with the Harness Exa tools;
3. a docs stage verifies key claims against official or primary sources;
4. a reducer produces direct URLs, uncertainty, and unresolved points.

If Exa is unavailable or `EXA_API_KEY` is missing, the expert returns `blocked`;
the Master turn does not crash.

`builtin:coding` runs:

1. a read-only plan;
2. a workspace build and verification;
3. an independent read-only review;
4. at most one fix and one re-review.

Remaining findings produce `partial`; there is no unbounded repair loop.

## Configuration

```json
{
  "mode": "expert",
  "agents": {
    "defaults": {
      "provider": "deepseek",
      "model": "deepseek-v4-flash",
      "max_iterations": null,
      "max_output_tokens": 32768,
      "runtime": {
        "max_retries": 3
      }
    },
    "routing": {
      "master": null,
      "experts": {
        "builtin:research/reduce": "deepseek/deepseek-v4-pro"
      }
    }
  },
  "skills": {
    "roots": [
      {"id": "team", "path": "/opt/team-skills"}
    ],
    "master_allowlist": []
  },
  "mcp": {
    "config_path": ".aeloon-core/mcp.json",
    "master_allowlist": ["github"]
  },
  "tools": {
    "master_capabilities": [
      "filesystem",
      "shell",
      "repo_context",
      "planning"
    ]
  },
  "experts": {
    "enabled": ["builtin:research", "builtin:coding"],
    "max_calls_per_turn": 8,
    "max_concurrency": 4,
    "stage_request_limit": 25,
    "timeout_seconds": 1800,
    "max_upstream_chars": 32000,
    "web_backend": "exa"
  }
}
```

Master follows Pi's natural tool loop by default: it continues until it returns
plain text, is cancelled, or another runtime guard stops it. Set
`agents.defaults.max_iterations` to an integer for an optional hard model-request
limit. A bounded run reserves its final request for a tool-free progress
checkpoint instead of failing immediately after the last tool call. Expert
stages remain bounded by `experts.stage_request_limit`.

```bash
uv run aeloon-core config set max-iterations unlimited
uv run aeloon-core config set max-iterations 100
```

An exact `<expert-id>/<stage-id>` route wins over the expert route, then the
default model is used.

Use `uv run aeloon-core config show`, `config init`, and `config set`. The planned
command that scans arbitrary installed Skill collections and asks a model to
generate disabled ExpertSkill drafts is intentionally not part of this MVP.

```bash
uv run aeloon-core config set mode normal
uv run aeloon-core config set mode expert
```

## Runtime boundaries

```text
aeloon_core/
├── harness/
│   ├── skill/          # manifest parsing, discovery, scopes, lazy tools
│   ├── expert/         # contracts, registry, runtime, runners, adapters
│   ├── agent/          # Master prompt
│   ├── capabilities.py # Ultra and Expert Harness capabilities
│   ├── execution/      # Pydantic AI run engine and tracing
│   ├── model/          # Master/expert-stage routing
│   ├── mcp/            # configured MCP loading and per-agent scopes
│   ├── provider/       # provider construction
│   └── tool/           # host tool contracts and observation tools
├── conversation/       # persisted Master turns only
├── web/                # transport and live lifecycle projection
├── orchestrator.py     # composition root
└── config.py
```

Only completed Master turns are persisted. Expert prompts, reports, stage state,
and tools live only in memory for the current turn.

## Development

```bash
uv run pytest -q
uv run ruff check .
(cd aeloon_core/web && bun test)
uv build
git diff --check
```
