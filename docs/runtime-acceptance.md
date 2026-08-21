# Runtime v4 release gate

The coordinated release is frozen only after these checks pass:

```bash
uv run python tools/gen_v4_manifest.py --check
uv run python tools/gen_rpc_docs.py --check
uv run python tools/docker_smoke.py
uv run pytest -q
uv run ruff check aeloon_runtime tools tests
```

The Runtime accepts only the exact `4.0.0` handshake. WSS pairing uses a
one-shot `devices.claim` request followed by a `device_token` handshake;
host-lifecycle methods remain Unix/CLI-only. Workspace authorization is
root-based and `project.add` accepts only `{root_id, relative_path}`.

The protocol contract is 73 methods, 31 events, 17 error codes, a 40 MiB
frame, 25 MiB ordinary-file and 10 MiB image limits. Runtime data is not
migrated: old state blocks startup until the operator runs the explicit,
force-confirmed v4 reset command.

<!-- r3-acceptance:start -->
R3 remote numbers are written by `bun run test:r3:remote` after a clean-tree
benchmark whose WSS client runs on the operator machine through an SSH tunnel,
plus the four fault-proxy resilience scenarios against that same forwarded Runtime.
<!-- r3-acceptance:end -->

## Handoff evidence

1. `tests/test_runtime_lifecycle.py` starts the real Runtime, reconnects a
   client, verifies persistence, and performs an explicit Unix shutdown.
2. `tests/test_runtime_server.py` covers exact handshake identity/host
   metadata, workspace roots, path boundaries, WSS lifecycle restrictions,
   and Runtime-owned project/thread behavior.
3. `tests/test_pairing.py` and `tests/test_pairing_gateway.py` cover strict
   pairing URLs, one-shot claims, v4 device storage, and token revocation.
4. UI `bun run check` covers protocol generation, architecture boundaries,
   TypeScript, unit/Electron tests, Playwright, Unix replay, WSS replay
   equivalence, and the production build.
5. The wheel and bundle workflows verify the v4 manifest, generated docs,
   Runtime package, and platform archives.
