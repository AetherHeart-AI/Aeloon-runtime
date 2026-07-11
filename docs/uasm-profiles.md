# UASM v1.5 Profile Operations

Profiles separate build-time compilation from runtime execution. `PROFILE.md` is
strictly parsed, compiled to a literal-only `CompiledProfile`, validated as AST
data, and stored as an immutable content-addressed artifact. Runtime loads only
an approved, compatible artifact and pins its id and generation for the turn.

## Bundled coding profile

`coding` is the default profile. It declares a read-only planner, a full coding
implementer, and an independent reviewer. Before the first default-profile turn,
the host installs the package-owned source through the normal deterministic
compile, approval, audit, and activation chain. Repeated startup is idempotent,
and an operator-approved active `coding` artifact is never overwritten.

The bundled source is copied to
`.aeloon-core/profiles/coding/PROFILE.md` when absent so users and agents can
inspect and adapt it. Workspace edits are never auto-approved; activate them
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

`handoff_agent` and `complete_task` are internal control calls. A control call
must be the sole tool call in the response. Mixed batches have no external side
effects. The first protocol error receives a correction; the second terminates
with visible output. Handoffs are counted against the profile and host caps.

Prompts and bounded handoff context are forward-only. Canonical history stores
normal assistant/tool pairs, while profile provenance and component usage remain
additive trace data. Agents cannot approve or activate their own profiles.
