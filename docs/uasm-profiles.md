# UASM v1.5 Profile Operations

Profiles separate build-time compilation from runtime execution. `PROFILE.md` is
strictly parsed, compiled to a literal-only `CompiledProfile`, validated as AST
data, and stored as an immutable content-addressed artifact. Runtime loads only
an approved, compatible artifact and pins its id and generation for the turn.

## Bundled profiles

`coding` is the default profile. It declares a read-only planner, a full coding
implementer, and an independent reviewer. `research` is an optional built-in
with a research lead, parallel source scouts, and an independent fact checker;
its delegated branches expose only read-only web tools. Select it with
`uv run aeloon-core config set profile-id research`.

Before the first turn using either built-in, the host installs the package-owned
source through the normal deterministic compile, approval, audit, and activation
chain. Repeated startup is idempotent, and an operator-approved active artifact
is never overwritten.

When a built-in is first installed, its source is copied into that bootstrap
workspace at `.aeloon-core/profiles/<profile-id>/PROFILE.md` when absent so users
and agents can inspect and adapt it. The copy is best-effort and is not required
to load the artifact from another workspace. The packaged resource remains the
built-in source of truth. Workspace edits are never auto-approved; activate them
through the CLI. Set `profile-id` to `none` to use the no-profile runtime.

## CLI

```bash
uv run aeloon-core profile validate PROFILE.md
uv run aeloon-core profile compile PROFILE.md --compiler deterministic
uv run aeloon-core profile inspect ARTIFACT_ID
uv run aeloon-core profile approve ARTIFACT_ID --approved-by operator
uv run aeloon-core profile activate ARTIFACT_ID
uv run aeloon-core profile status PROFILE_ID
uv run aeloon-core profile rollback ARTIFACT_ID
```

The deterministic compiler is the reference backend. The optional LLM backend
receives no tools, runs at temperature zero, may rewrite prompts only, and gets
one structured repair attempt. Compilation is outside the turn critical path.

## Artifact safety

Artifacts live below `<data-dir>/profile-artifacts/<artifact-id>/` and contain
the source snapshot, generated class, manifest, validation report, and semantic
diff. State is derived: invalid candidates are `quarantined`, valid artifacts
are `validated`, an approval record makes them `approved`, and an active pointer
makes them `active`.

Activation validates the manifest, approval, ABI, and tool fingerprints while a
cross-process lock is held. It writes a content-addressed audit record and fsyncs
it before atomically replacing the active pointer. The pointer is the sole commit
point; readers reject mismatched pointer, audit, approval, or artifact hashes.
Rollback selects a previously approved compatible artifact for future turns and
does not undo tool side effects.

## Runtime protocol

The master always gives precedence to terminal state, guard decisions,
finalization, pending control, pending external tools, and tool-return affinity.
Profile roles receive only declared tools intersected with the host registry;
`ScopedToolRegistry` enforces the same boundary before execution.

`handoff_agent`, `delegate_tasks`, and `complete_task` are internal control
calls. A control call must be the sole tool call in the response. Mixed batches
have no external side effects. The first protocol error receives a correction;
the second terminates with visible output. Handoffs are counted against the
profile and host caps.

`delegate_tasks` runs two to four isolated branches concurrently and joins their
bounded reports back into the calling role. Every delegated role is preflighted
before any branch starts and may expose only tools whose concurrency mode is
`read_only`. Branches have their own state, Guard, context, iteration budget,
messages, and tool scope; they share neither the parent runtime's mutable state
nor one another's transcript. Branch streaming is suppressed, tool-call ids are
prefixed for stable TUI pairing, ordinary branch failures are isolated, and
parent cancellation cancels all outstanding branches. Each branch also has a
five-minute work deadline in addition to its iteration limits, and lifecycle callbacks
have a separate five-second deadline. Reports are marked as
untrusted task data and restored to input order at join time. The host permits
at most two delegation rounds per turn, and a delegated branch cannot invoke
profile control operations recursively.

The fork/join operation is profile control protocol version 2. Artifacts built
against version 1 remain immutable but are incompatible with the new host until
their source is recompiled, approved, and activated.

Prompts and bounded handoff context are forward-only. Canonical history stores
normal assistant/tool pairs, while profile provenance and component usage remain
additive trace data. Agents cannot approve or activate their own profiles.
