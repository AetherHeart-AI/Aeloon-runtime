# Aeloon Core

A minimal, independent Aeloon agent-loop playground. It keeps the reusable LLM
and tool iteration kernel, an OpenAI-compatible provider, a small set of local
tools, JSONL session persistence, and a terminal debugging CLI.

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

Inside the CLI, type prompts directly. The terminal view keeps the core turn
information visible: workspace, model, session id, streamed assistant output,
tool calls, tool results, token usage, and compact gateway logs. Useful commands:

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
uv run aeloon-core tui --gateway-log-level DEBUG "Read README.md"
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
uv run aeloon-core config set max-tokens auto
uv run aeloon-core config set max-auto-continue-iterations 25
uv run aeloon-core config set max-finalization-iterations 2
```

You can override the path with `AELOON_CORE_CONFIG` or `--config`.
Environment variables override file values:

- `AELOON_CORE_API_KEY`
- `AELOON_CORE_API_BASE`
- `AELOON_CORE_MODEL`
- `AELOON_CORE_MAX_TOKENS` (`auto` uses model-aware defaults)
- `AELOON_CORE_DATA_DIR`

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
      "model": "gpt-4.1-mini",
      "max_tokens": null
    }
  }
}
```

`max_tokens: null` means auto. Aeloon resolves the output budget from public
model metadata before each call: LiteLLM's `model_prices_and_context_window.json`
table, and 4,096 only when metadata is unavailable.

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
