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

## R4 release acceptance

Date: 2026-08-22
Runtime commit: 64699536a7d248ff0811705a38eaee2fb4dbda72
Tag: runtime-v0.1.0
UI commit: 04f2e680712ed4ee004a7d166dd9dbcd0c6c00b6
Gate: **partial — see "Not verified" below**

R4 set out to turn work that had only ever run on one developer machine into a
release that CI has checked and that something can actually install. The release
pipeline fixes were the means; being consumable was the goal.

### Published

`runtime-v0.1.0` carries five assets:

| Asset | Size |
| --- | --- |
| aeloon-runtime-darwin-aarch64.tar.zst | 73,225,848 |
| aeloon-runtime-linux-aarch64.tar.zst | 85,583,747 |
| aeloon-runtime-linux-x86_64.tar.zst | 97,135,508 |
| aeloon_runtime-0.1.0-py3-none-any.whl | 250,244 |
| aeloon-protocol-4.0.0.tgz | 4,891 |

`linux-x86_64` is new in this release. Only aarch64 archives existed before,
which a Runtime meant for LAN and cloud hosts cannot be deployed from.

### Verified

- Both repositories' CI passes and the work is merged to `main`. Neither had
  ever run CI before this phase.
- The release workflow completes in one run: the tag, `pyproject` and
  `RUNTIME_VERSION` are checked against each other before anything is built,
  all three platform archives are produced, and every asset uploads.
- With no `AELOON_UI_REPOSITORY_TOKEN` configured, the UI notification step
  reports success rather than failing a release whose assets are already
  published.
- The published `linux-x86_64` archive hashes to what
  `runtime-bundle.lock.json` pins, byte for byte, and the pin was computed from
  the downloaded bytes rather than a build log.
- That archive contains genuine x86-64 ELF binaries for Python, uv and ripgrep;
  its manifest declares `linux-x86_64`, names this commit, and its recorded
  component digests match the files beside them.
- A clean clone of the client with no local state installs, builds, validates
  the lock under `--release`, typechecks and passes its unit tests.
- The client suite runs against a real Runtime in CI, including the replay
  corpus over both transports with byte-identical transcripts. Before this
  phase that equivalence had only ever been demonstrated on one machine.

### Not verified

Two exit criteria are open. Nothing below has been demonstrated, and the
release should not be described as consumable until they are.

1. **The published x86_64 archive has never been executed.** Its integrity and
   contents are checked, but no process has been started from it. That needs an
   x86_64 Linux host; the machine used for R3 stopped accepting SSH
   (`kex_exchange_identification`, before key exchange) and was not replaced.
2. **No client has connected to a Runtime started from a release artifact.** The
   clean-clone client was built but never pointed at one, because (1) is open.

Everything to date shows the release can be built and pinned. Whether it can be
installed and used is the question these two answer, and it is the one that
matters — a pipeline that produces artifacts nobody has started is the ordinary
way this goes wrong.

### Known gap

The remote-pairing e2e specs skip in CI. Electron reports
`isEncryptionAvailable=false backend=basic_text` on a stock runner, and the
client refuses remote profiles rather than keep a device token in plaintext, so
those specs would be asserting against a flow the product declines to offer.
Two attempts to give the runner a working keyring did not change the reported
backend. They still run on a developer machine with a real keychain.
