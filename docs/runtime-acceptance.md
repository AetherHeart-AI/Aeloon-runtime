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
## R3 remote acceptance

Date: 2026-08-21
Runtime commit: 924bf94092a730058525e141379bed4afe5ab4c0
UI commit: 4c43b898603adf5eeca3c50cb61a9ef05ae1d100
Path: ssh-tunnel (WSS client on the operator machine through SSH local-forward)
Runtime host: autodl-container-e5354dbac4-b13fcaaf / linux / x86_64
Client host: zhangxins-MacBook-Air.local
Protocol: 4.0.0
Gate: passed

| Probe | warmup / n | p50 | p95 | budget | extra |
| --- | --- | --- | --- | --- | --- |
| Raw tunnel upload | 1 / 3 | 7080.5 ms | 7123.6 ms | baseline | 8 MiB, 1.12 MiB/s |
| Raw tunnel download | 1 / 3 | 1581.0 ms | 1694.1 ms | baseline | 8 MiB, 4.72 MiB/s |
| PTY echo | 5 / 50 | 44.7 ms | 85.0 ms | ≤ 250 ms | pass |
| thread.get | 3 / 30 | 198.9 ms | 854.5 ms | ≤ 1000.0 ms | max(1 s, raw-download projection × 1.25); encoded 1228360 B / 5 MiB; pass |
| 25 MiB attachment | 1 / 10 | 30032.5 ms | 30831.5 ms | ≤ 37102.3 ms | max(8 s, raw-link projection × 1.25); base64 + unique payloads + SHA-256; pass |
| First available | 5 / 30 | 361.0 ms | 403.5 ms | ≤ 2000 ms | TLS + handshake + subscribe + capabilities + snapshot; pass |

Event throughput 1 KiB: 1000 events/s delivered=true, receive p50=988.9 / p95=989.6, highest stable=2000 events/s.

Latency budgets are hard floors over SSH-tunneled WSS. Bulk response and attachment times are bounded against independent raw-tunnel download/upload baselines with 25% protocol headroom, and every stored attachment is SHA-256 verified. Private benchmark methods stay off the RPC manifest.
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
