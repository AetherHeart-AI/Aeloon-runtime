# Runtime v3 release gate

The base release is frozen only after these commands and platform jobs pass:

```bash
uv run python tools/gen_v3_manifest.py --check
uv run python tools/check_rpc_compat.py OLD.json aeloon_core/rpc/aeloon-rpc-v3.manifest.json
uv run python tools/docker_smoke.py
uv run pytest -q
uv run ruff check aeloon_core tools tests
```

The Docker smoke builds the image and starts it with mounted workspace, data,
and Docker-managed `/run/aeloon` socket volumes. It performs v3 handshake,
health, and controlled shutdown from inside the container; a Docker Desktop
host bind mount is intentionally not used for the socket because virtiofs can
reject `chmod(0600)` on socket inodes.

Trace capture is opt-in and must be explicitly requested for a replay fixture:

```bash
uv run aeloon-runtime serve --unix /tmp/aeloon-runtime.sock \
  --data-dir ~/.aeloon-runtime --record-trace ~/.aeloon-runtime/traces
```

The raw JSONL and `blobs/` directory are mode `0600`/`0700` local artifacts. Run the deterministic
sanitizer and manually review the result before adding a trace to the repository:

```bash
uv run python tools/sanitize_trace.py RAW.jsonl REVIEWED.jsonl
```

Every Runtime data directory also keeps a bounded `runtime.log` plus one rotated
`runtime.log.1`; both are private `0600` lifecycle diagnostics and are not replay
fixtures.

The protocol contract is 66 methods, 31 events, 16 error codes, a 40 MiB
frame, 25 MiB ordinary-file and 10 MiB image limits. Migration jobs must cover
empty data, WAL databases, malformed input, interrupted worktree moves, dirty
and missing worktrees, idempotent reruns and rollback. Platform CI additionally
builds the macOS ARM64 and Linux ARM64 Runtime archives, computes their SHA-256
with `tools/build_runtime_bundle.py`, updates the UI lock through its
`scripts/update-runtime-lock.ts` workflow, and runs the UI `bun run check` plus
Playwright/Electron packaging checks.

## Handoff evidence

The six base-release handoff gates map to these checked-in tests and commands:

1. `tests/test_runtime_lifecycle.py` starts the real `aeloon-runtime serve`
   command, closes and reconnects a client, verifies the PID and restored
   projection, then performs an explicit shutdown.
2. `bun run architecture:check` enforces the renderer/Electron boundary and
   rejects the removed `src/server`, Workbench process, Bun sidecar and old
   Core client paths.
3. `tests/test_migration.py` covers empty data, WAL, attachments, worktrees,
   dirty/missing inputs, interrupted journals, repeatability and rollback.
4. `tests/replay-scenario.test.ts` and `scripts/replay-scenario.ts` provide
   protocol-neutral legacy/v3 adapters plus file-tree, Git, turn and PTY
   final-state oracles.
5. `uv run python tools/docker_smoke.py`, the wheel build, and UI packaging
   contract tests cover reproducible local packaging. Native ARM64 archives
   and a non-zero Runtime lock digest remain release-CI gates until the
   `runtime-v0.1.0` assets are published.
6. `tools/check_rpc_compat.py` and the handshake range tests enforce additive
   minor changes and major-version breaks; the generated protocol package is
   checked against the single `docs/rpc-v3.json` source.
