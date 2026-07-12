# Aeloon Core

A minimal, independent Aeloon agent-loop playground. It provides an explicit
state-machine runtime, an OpenAI-compatible provider, a small set of local tools,
JSONL session persistence, and a terminal debugging CLI.

## Quick Start

```bash
uv sync
export AELOON_CORE_API_KEY="..."
export AELOON_CORE_API_BASE="https://api.openai.com/v1"
export AELOON_CORE_MODEL="gpt-4.1-mini"
uv run python -m aeloon_core "List the current directory and read README.md"
```

Run the opencode-style terminal CLI:

```bash
uv run aeloon-core
# or explicitly:
uv run aeloon-core chat
```

Inside the CLI, type prompts directly. The default terminal view is deliberately
quiet: it keeps the workspace, model, session id, streamed assistant output,
tool and sub-agent lifecycles, flow-changing Guard decisions, errors, and compact
token statistics. Raw reasoning, routine status updates, profile artifact
metadata, turn UUID separators, and gateway logs are hidden by default. A Guard
decision is rendered as a concise line such as
`⚠ Guard [rule_engine] · 重试 · malformed tool call`.

Useful interactive commands:

```text
/help
/sessions
/resume <session-id>
/new
/logs debug
/logs off
/quit
```

You can also run one rich-rendered turn and exit:

```bash
uv run aeloon-core chat "List the current directory"
uv run aeloon-core tui "Read README.md"
```

Enable gateway diagnostics explicitly when needed:

```bash
# Compact gateway logs at the default INFO level
uv run aeloon-core chat --show-gateway-logs

# Setting a level or requesting detail also enables gateway logs
uv run aeloon-core tui --gateway-log-level DEBUG --gateway-log-detail "Read README.md"

# The compatibility flag always wins when options are composed by scripts
uv run aeloon-core chat --gateway-log-level DEBUG --hide-gateway-logs
```

Runtime commands use the directory where you invoke `aeloon-core` as the
workspace. To target a different folder for one command, pass `--workspace`.

## Config

By default Aeloon Core persists config at `~/.aeloon-core/config.json`. You can
create it from the CLI:

```bash
uv run aeloon-core config init \
  --api-key sk-... \
  --api-base https://api.openai.com/v1 \
  --model gpt-4.1-mini
```

Inspect or update it later:

```bash
uv run aeloon-core config path
uv run aeloon-core config show
uv run aeloon-core config set model gpt-4.1-mini
uv run aeloon-core config set max-auto-continue-iterations 25
uv run aeloon-core config set max-finalization-iterations 2
uv run aeloon-core config set context-compaction-enabled true
uv run aeloon-core config set context-compaction-trigger-ratio 0.9
uv run aeloon-core config set uasm-guard-decision-mode full
```

You can override the path with `AELOON_CORE_CONFIG` or `--config`.
Environment variables override file values:

- `AELOON_CORE_API_KEY`
- `AELOON_CORE_API_BASE`
- `AELOON_CORE_MODEL`
- `AELOON_CORE_DATA_DIR`
- `AELOON_CORE_PROFILE_ID`

Minimal file example:

```json
{
  "providers": {
    "custom": {
      "api_key": "sk-...",
      "api_base": "https://api.openai.com/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "gpt-4.1-mini"
    }
  }
}
```

Normal model calls do not set `max_tokens`; the provider controls output length.

Context compaction is enabled by default. Before every agent-loop model call, Aeloon
estimates the complete model-visible request, including tool definitions. At 90% of
the model context window, it summarizes older turns into a synthetic system checkpoint
and keeps the recent tail intact. Context windows come from LiteLLM's public model
table, falling back to `agents.defaults.context_window_tokens`. Tunables live under
`agents.defaults.context_compaction`.

## Unified Agentic State Machine

The Unified Agentic State Machine (UASM) is the only agent-loop runtime. Rule
handling, TemporaryGuard, and bounded minimal context are always enabled:

```bash
uv run aeloon-core config set uasm-transition-trace-enabled true
```

UASM makes the `MasterAgent -> WorkerAgent/ToolAgent` route explicit. Canonical
conversation history lives in `LightweightState`; forward minimal context is a
per-call view and does not replace persisted messages. Deterministic loop rules
handle known failures. Only a repeated ambiguous failure that would otherwise
stop the loop is escalated to a stateless `TemporaryGuard`, which sees a bounded
evidence object rather than the transcript.

Completed UASM turns persist transition records separately at
`~/.aeloon-core/traces/<session-id>.jsonl`. Each record includes state digests,
the node and decision, wall time, and token usage. Turn records aggregate tokens
by `domain`, `harness`, and `context_processing` without mixing transition rows
into session history. The additive `by_component` view distinguishes
`profile_master`, `domain:<role>`, `tool`, `control`, `temporary_guard`, and
`minimal_context`; both views conserve the same aggregate counters.

## Agent Profiles (v1.5)

Profiles provide explicitly declared agent teams while keeping the same
`run_agent_loop`. Aeloon ships with `coding` as the zero-config default: a
`planner`, `implementer`, and independent `reviewer`. On the first turn, the
host deterministically compiles the package-owned source, records a system
approval and activation audit, then pins the immutable artifact for the turn.
It also copies the source to `.aeloon-core/profiles/coding/PROFILE.md` without
overwriting an existing workspace file.

Disable profiles explicitly with `uv run aeloon-core config set profile-id none`.
That preserves the v1.0 deterministic-master path: text completes the turn and
neither compiler nor profile-master calls occur.

A profile lives at `.aeloon-core/profiles/<profile-id>/PROFILE.md` and declares
roles and requested tools in strict YAML, followed by shared, master, and role
instructions in Markdown:

```markdown
---
schema_version: 1
id: coding-team
revision: 1
description: Coding and review team
default_agent: implementer
max_handoffs: 8
agents:
  - id: planner
    description: Analyze requirements
    tools: [read, glob, grep]
  - id: implementer
    description: Implement and verify changes
    tools: [read, write, edit, exec]
---

## Shared
Keep changes scoped and verified.

## Master
Select the role that owns the next step.

## Agent: planner
Inspect the repository and produce an implementation approach.

## Agent: implementer
Implement, verify, and report the result.
```

Custom profiles are built and activated explicitly:

```bash
PROFILE=.aeloon-core/profiles/coding-team/PROFILE.md

uv run aeloon-core profile validate "$PROFILE"
uv run aeloon-core profile compile "$PROFILE" --compiler deterministic
uv run aeloon-core profile inspect <artifact-id>
uv run aeloon-core profile approve <artifact-id> --approved-by operator
uv run aeloon-core profile activate <artifact-id>
uv run aeloon-core config set profile-id coding-team
uv run aeloon-core profile status coding-team
```

The deterministic compiler is the reference backend. The optional `llm`
backend is explicit and offline from turns:

```bash
uv run aeloon-core profile compile "$PROFILE" --compiler llm --model gpt-4.1-mini
```

It has no tools, uses temperature zero, receives one repair attempt, and may
rewrite prompts only. It cannot alter role ids, descriptions, tools, the
default role, or handoff budget. Keep it experimental unless a golden-corpus
evaluation shows measurable gains over the deterministic artifact.

Compiled Python is an inert review format: only one constant-only
`CompiledProfile` class is allowed, values are decoded with `ast.literal_eval`,
and generated source is never imported, executed, or passed to `compile`.
Artifacts must move through `validated -> approved -> active`; activation is an
audited, cross-process-serialized commit whose active pointer is published last.
During Profile turns, filesystem tools cannot access the operator data directory,
and exec is sandboxed away from it (or disabled when that isolation is
unavailable). A turn pins the active artifact once, so activation during a turn
affects only the next turn.

At runtime, the profile master can select only a declared role. Roles see the
intersection of their requested tools and the host registry plus two internal
control operations:

- `handoff_agent(summary, recommended_agent?)`
- `complete_task(final_content)`

Control calls must be the response's only tool call. External tools are hidden
by role and checked again immediately before execution. Tool results always
return to the calling role; only an accepted handoff invokes the profile master
again. A second protocol violation stops with visible output. Finalization,
TemporaryGuard, and provider-failure termination remain host-controlled.

Rollback selects a prior approved compatible artifact for future turns; it
cannot undo tool side effects from completed turns:

```bash
uv run aeloon-core profile rollback <prior-artifact-id>
uv run aeloon-core config set profile-id none  # restore the v1.0 path
```

See [UASM profile operations](docs/uasm-profiles.md) for the artifact layout,
failure handling, compatibility rules, and operational checklist.

## Core Tools

The runtime registers these tools:

- `exec`
- `read`
- `write`
- `edit`
- `glob`
- `grep`
- `skill` when skills are enabled
- `webfetch`
- `websearch`
- `todowrite`

File writes follow an OpenCode-style safety pattern:

- Use `read` with `offset`/`limit` to inspect files in chunks.
- Use `edit` for existing files whenever possible.
- `write` refuses to overwrite an existing file unless `overwrite=true`.
- Large `write` calls require an `end_marker` appended to the end of `content`;
  the marker is stripped before the file is saved. If the marker is missing,
  Aeloon treats the write as possibly truncated and refuses to touch the file.

## Skills

Aeloon Core discovers OpenCode-style `SKILL.md` files at startup. The model sees
only names and descriptions in system context, then loads the full instructions
on demand with the `skill` tool.

Standard locations:

- Project native: `.aeloon-core/skill/<name>/SKILL.md` and
  `.aeloon-core/skills/<name>/SKILL.md`
- Project OpenCode-compatible: `.opencode/skill/<name>/SKILL.md` and
  `.opencode/skills/<name>/SKILL.md`
- Project Claude-compatible: `.claude/skills/<name>/SKILL.md`
- Project agent-compatible: `.agents/skills/<name>/SKILL.md`
- Global native: `~/.aeloon-core/skill/<name>/SKILL.md` and
  `~/.aeloon-core/skills/<name>/SKILL.md`
- Global OpenCode-compatible: `~/.config/opencode/skill/<name>/SKILL.md` and
  `~/.config/opencode/skills/<name>/SKILL.md`
- Global Claude-compatible: `~/.claude/skills/<name>/SKILL.md`
- Global agent-compatible: `~/.agents/skills/<name>/SKILL.md`

For project-local external and config directories, Aeloon walks upward from the
workspace to the git worktree root. Later discoveries override earlier duplicate
skill names, so project-native skills can override global or compatibility
skills.

Minimal `SKILL.md`:

```markdown
---
name: git-release
description: Prepare consistent releases and changelogs.
---

## Workflow

Draft release notes, check the versioning scheme, and produce the release
command.
```

Aeloon only reads simple scalar `name` and `description` fields from the
frontmatter; other fields are ignored.

Additional settings live under `skills`:

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

Environment overrides:

- `AELOON_CORE_SKILLS_ENABLED`
- `AELOON_CORE_DISABLE_EXTERNAL_SKILLS`
- `AELOON_CORE_DISABLE_CLAUDE_CODE_SKILLS`
- `AELOON_CORE_SKILL_PATHS` using the OS path separator
