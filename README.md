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
model metadata before each call: OpenRouter's `/api/v1/models` table for
OpenRouter routes, LiteLLM's `model_prices_and_context_window.json` table for
other OpenAI-compatible routes, and 4,096 only when metadata is unavailable.

## Core Tools

The runtime registers exactly these tools:

- `exec`
- `read`
- `write`
- `edit`
- `glob`
- `grep`
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
