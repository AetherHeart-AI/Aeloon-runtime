# Aeloon Core

A minimal, independent Aeloon agent-loop playground. It keeps the reusable LLM
and tool iteration kernel, an OpenAI-compatible provider, a small set of local
tools, JSONL session persistence, and a WebSocket driven debugging UI.

## Quick Start

```bash
uv sync
export AELOON_CORE_API_KEY="..."
export AELOON_CORE_API_BASE="https://api.openai.com/v1"
export AELOON_CORE_MODEL="gpt-4.1-mini"
uv run python -m aeloon_core "List the current directory and read README.md"
```

Run the local Web UI:

```bash
cd web
npm install
npm run build
cd ..
uv run aeloon-core webui
```

Then open `http://127.0.0.1:8765`.

You can point the Web UI at another workspace without changing directories:

```bash
uv run aeloon-core webui --workspace /path/to/project --port 8766
```

## Config

By default Aeloon Core persists config at `~/.aeloon-core/config.json`. You can
create it from the CLI:

```bash
uv run aeloon-core config init \
  --workspace /path/to/project \
  --api-key sk-... \
  --api-base https://api.openai.com/v1 \
  --model gpt-4.1-mini
```

Inspect or update it later:

```bash
uv run aeloon-core config path
uv run aeloon-core config show
uv run aeloon-core config set workspace /path/to/another-project
uv run aeloon-core config set model gpt-4.1-mini
```

You can override the path with `AELOON_CORE_CONFIG` or `--config`.
Environment variables override file values:

- `AELOON_CORE_API_KEY`
- `AELOON_CORE_API_BASE`
- `AELOON_CORE_MODEL`
- `AELOON_CORE_WORKSPACE`
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
      "model": "gpt-4.1-mini"
    }
  },
  "workspace": "/Users/zhangxin/Desktop/aeloon-core"
}
```

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
