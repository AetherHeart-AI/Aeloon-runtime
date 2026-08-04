# Migration Guide

This guide collects compatibility details that are useful when upgrading existing scripts,
configuration, sessions, or Bridge clients. New installations can follow the
[README](README.md) directly.

## CLI commands

The `aeloon` command is the primary interface. Existing automation can continue to use the
low-level `aeloon-core` command surface:

```bash
# Explicit saved, resumed, or ephemeral runs
uv run aeloon-core run "task"
uv run aeloon-core run "continue" --session <id>
uv run aeloon-core run "one shot" --no-session

# Machine-readable output
uv run aeloon-core run "task" --output json
uv run aeloon-core run "task" --output stream-json

# Session and configuration inspection
uv run aeloon-core session list
uv run aeloon-core session show <id>
uv run aeloon-core config path
uv run aeloon-core config show
```

`stream-json` emits typed harness events followed by a result object. `json` emits only the result
object. Interactive text output uses stderr for status and stdout for the final response, so
redirected output remains stable.

## Provider and model configuration

Fresh installations do not pin a default model or import provider keys from the process
environment. After a local endpoint is added or a cloud account is connected, Aeloon selects the
first available model. Pin a model with:

```bash
uv run aeloon models use PROVIDER/MODEL
```

Provider-local names are resolved in catalog order; use the full `provider/model` ID to
disambiguate overlapping names. Existing explicitly configured DeepSeek providers and sessions
remain supported, but the provider appears in the CLI catalog only when its credential is stored.

The following compatibility commands remain available:

```bash
uv run aeloon config show
uv run aeloon config set model studio/qwen3-coder
uv run aeloon provider list

uv run aeloon-core provider login <username>
uv run aeloon-core provider list
uv run aeloon-core cloud login <username>
uv run aeloon-core cloud status
uv run aeloon-core cloud logout
```

`provider add` remains an alias for `local add`. Cloud credentials are stored in the account
vault, and local API keys are stored in the mode-`0600` config. Neither login flow accepts secrets
as command-line arguments.

## Session storage

New sessions are stored under `<data-dir>/harness-sessions/`. The older `sessions/` directory is
not read or modified automatically, so copy or convert historical data explicitly if it must be
retained.

Current sessions use append-only version-3 JSONL records. Each completed message is persisted and
fsynced immediately. Restarting restores the last saved entry rather than reconstructing a
cumulative turn snapshot.

## Bridge clients

Bridge v2 uses JSON-RPC 2.0 over NDJSON on a Unix domain socket. Clients should consume the stable
session, operation, catalog, and revisioned settings DTOs instead of internal Python objects or raw
provider payloads.

Administration commands are:

```bash
uv run aeloon system bridge status
uv run aeloon system bridge stop

uv run aeloon-core bridge ensure --output json
uv run aeloon-core bridge status --output json
uv run aeloon-core bridge schema
uv run aeloon-core bridge stop
```

Older Bridge clients are not protocol-compatible with v2 and must update their method names,
handshake handling, and DTO parsing together. `status` and `stop` are idempotent when the daemon or
socket is absent.
