# UASM v1.5: Profile compiler and explicit multi-agent collaboration

## Goal

Add profile-defined agent teams to the single UASM runtime while preserving the
v1.0 path when profiles are explicitly disabled. The trusted bundled `coding`
Profile is the zero-config default; custom Profiles remain explicitly compiled
and approved outside the turn-critical path. Runtime code consumes immutable,
validated artifacts and treats role selection, handoff, and completion as
explicit protocol operations.

The implementation has two trust boundaries:

- **Build Plane:** `PROFILE.md` parsing, strict validation, compilation,
  literal-only Python artifact validation, approval, activation, rollback, and
  an independent compilation ledger.
- **Runtime Plane:** immutable artifact loading, profile master routing,
  role-scoped worker execution, internal control operations, external tools,
  TemporaryGuard, and additive trace/provenance accounting.

Dynamic role or tool discovery, semantic tool grouping, generated executable
code, automatic self-activation, and crash-resumable mid-turn checkpoints are
not part of v1.5.

## Compatibility invariant

`run_agent_loop(..., profile=None)` remains the v1.0 loop:

- deterministic master;
- existing text completion semantics;
- no profile master or compiler provider calls;
- unchanged canonical message sequence, final hooks, and status behavior;
- profile-only state is omitted from the state digest and serialized provenance.

Profile behavior remains additive, and setting `profile_id=None` retains the
no-Profile contract. The bundled coding Profile is selected by default. There
remains exactly one `run_agent_loop` implementation.

## Delivery sequence

1. Implement the profile source model, parser, artifact contract, deterministic
   compiler, store, lifecycle, and CLI without connecting it to runtime.
2. Add scoped tool enforcement, profile provenance, component usage, and the
   internal control protocol.
3. Connect a single-role artifact and prove explicit completion plus no-profile
   compatibility.
4. Add multiple independent role agents, handoff, tool-return affinity, and a
   deterministic profile-master fixture.
5. Add the provider-backed JSON-only profile master while keeping all safety
   precedence deterministic.
6. Add the experimental LLM compiler, one repair attempt, cache behavior, and
   activation audit.
7. Add evaluation fixtures, operational documentation, and full verification.

## Profile source contract

Profiles live at `.aeloon-core/profiles/<profile-id>/PROFILE.md`. The file has a
YAML frontmatter document and these Markdown sections:

```markdown
---
schema_version: 1
id: coding-team
revision: 1
description: Coding and review team
default_agent: implementer
max_handoffs: 8
agents:
  - id: planner
    description: Analyze requirements and produce an implementation approach
    tools: [read, glob, grep]
  - id: implementer
    description: Implement and verify changes
    tools: [read, write, edit, glob, grep, exec]
---

## Shared
Shared constraints.

## Master
Routing criteria.

## Agent: planner
Planner instructions.

## Agent: implementer
Implementer instructions.
```

Validation rules:

- Pydantic models reject unknown fields.
- A safe YAML loader rejects duplicate mapping keys and custom tags.
- Profile and role identifiers match `[a-z][a-z0-9_-]{0,63}`.
- Reserved runtime/control node names cannot be role identifiers.
- A profile declares 1-16 unique roles and every role has exactly one matching
  `## Agent: <id>` section; undeclared role sections are rejected.
- `default_agent` names a declared role.
- Tools are explicit names and duplicate requested tools are rejected.
- The compiler cannot add, remove, or rename roles, expand a tool list, or alter
  a handoff budget.
- `revision` is display metadata. Artifact identity comes from canonical content
  and compatibility inputs.

## Compiled artifact contract

Both compiler backends produce the same constant-only source form:

```python
class CompiledProfile:
    profile_schema_version = 1
    compiled_api_version = 1
    profile_id = "coding-team"
    revision = 1
    description = "Coding and review team"
    default_agent_id = "implementer"
    max_handoffs = 8
    master_prompt = "..."
    shared_prompt = "..."
    agents = (...)
```

The AST validator permits only an optional module docstring followed by one
plain `CompiledProfile` class. The class has no bases, keywords, decorators, or
metaclass. Its body contains exactly the allowlisted literal assignments.
Imports, methods, calls, attributes, comprehensions, lambdas, control flow,
annotations, and other top-level statements are rejected. Values are decoded
only with `ast.literal_eval` into frozen `RuntimeProfileSpec` values. Generated
source is never imported, executed, or passed to `compile`.

Backends:

- `deterministic` is the reference implementation. It combines Shared, Master,
  and role sections with stable templates and deterministic serialization.
- `llm` receives profile content as untrusted data and may only rewrite routing,
  shared, and role prompts. It has no tools, uses deterministic generation, and
  receives one repair attempt after a structured validation failure. The
  compiler verifies role, tool, and budget invariants against the source model.
  If evaluation does not show measurable value, this backend remains
  experimental.

## Artifact storage and lifecycle

Artifacts are content-addressed directories under
`<data_dir>/profile-artifacts/<artifact-id>/` containing:

- the exact source snapshot;
- generated `compiled_profile.py`;
- a canonical JSON manifest;
- validation report;
- semantic diff between source declarations and runtime spec;
- lifecycle metadata.

The manifest/cache identity binds canonical profile and grammar versions;
compiler backend/version, prompt/model/output hashes; AST, validator,
RuntimeProfileSpec, UASM, and Control protocol versions; requested tool schema
fingerprints; and validation result. Compilation telemetry records provider
tokens, duration, repair count, and cache hits in a separate compile ledger and
never contributes to turn usage.

Lifecycle states are `candidate -> validated -> approved -> active`, with
invalid candidates quarantined. Approval binds the validated artifact digest.
Activation atomically replaces the active-profile pointer and synchronously
writes an append-only activation audit; an audit failure aborts activation.
Rollback can activate only an approved, compatible prior artifact and cannot
undo prior external tool effects.

CLI surface:

- `profile validate <PROFILE.md>`
- `profile compile <PROFILE.md> --compiler deterministic|llm`
- `profile inspect <artifact-id>`
- `profile approve <artifact-id>`
- `profile activate <artifact-id>`
- `profile status`
- `profile rollback <artifact-id>`

## Runtime assembly

The public loop gains one optional input:

```python
run_agent_loop(..., profile: RuntimeProfileSpec | None = None)
```

Configuration adds `agents.defaults.profile_id: str | None` and
`agents.defaults.max_handoffs: int = 8`. The orchestrator resolves the selected
active artifact exactly once at turn start and pins profile id, revision,
artifact id, and activation generation in a `ProfileRef`. Activation during a
turn affects only later turns.

Profile roles do not extend the closed control enum. Each role is represented by
an independent `ProfileDomainAgent`, while `active_agent_id` records the active
role and traces use `domain:<role-id>`. Control phases remain closed:

```text
master -> worker -> control/tool/temporary_guard -> master -> ... -> done
```

The deterministic master precedence is:

1. terminal -> done;
2. pending guard -> temporary guard;
3. finalizing -> the existing text-only worker with no tools;
4. pending internal control call -> control;
5. pending external call -> tool;
6. completed tool with a resume role -> that role, with no profile-master call;
7. initial routing or accepted handoff -> profile master.

The profile master has no tools and returns strict JSON containing only
`{"agent_id": "<declared-role>"}`. Its bounded input includes the user goal,
role ids/descriptions, handoff summary, remaining budget, and a state digest.
Invalid JSON, unknown roles, or provider errors fall back first to a legal
handoff recommendation and otherwise to the declared default role. It cannot
mutate messages, budgets, pending work, permissions, or terminal state.

## Control protocol

Every profile role sees two internal operations that are not registered in the
external `ToolRegistry`:

```text
handoff_agent(summary: str, recommended_agent?: str)
complete_task(final_content: str)
```

Rules:

- A control operation must be the response's sole tool call. Mixed control and
  external calls produce no external side effect.
- Empty final content, unknown recommendations, malformed arguments, and empty
  handoff summaries are protocol errors.
- The first protocol error appends a local correction and resumes the same role.
  A second error terminates by rule with visible final text.
- An accepted handoff increments `handoff_count`. The effective cap is
  `min(profile.max_handoffs, host.max_handoffs)`. Once exhausted, the active
  role is instructed to complete; another handoff terminates visibly.
- An external tool always resumes its calling role. Only an explicit accepted
  handoff invokes the profile master again.
- Bare text does not normally complete a profile run. The first instance asks
  the role to call `complete_task`; the second terminates with that text visible
  and a protocol-violation status.
- RuleEngine finalization, TemporaryGuard, and provider-failure termination keep
  their existing semantics and do not require `complete_task`.
- Assistant control calls and paired tool results enter canonical history.
  Profile, master, and role system prompts are forward-only context and never
  enter the persisted transcript.
- Every path invokes `on_final` at most once.

## Role-scoped external tools

For each role:

```text
effective_tools = profile_requested_tools intersect host_registry
```

The model sees only these tools plus the two internal control definitions. A
`ScopedToolRegistry` independently checks the tool name immediately before
execution. Unauthorized calls return a protocol failure and never invoke a tool
handler. Profiles cannot register tools, broaden tool-specific filesystem or
network policy, access secrets, or expand host budgets.

## State, provenance, and accounting

`LightweightState` gains typed profile fields:

- `ProfileRef`;
- `active_agent_id` and `resume_agent_id`;
- `handoff_count`;
- `pending_handoff` and `pending_control_call`;
- `control_protocol_retries`.

Profile fields appear in the digest only when a profile is active. Traces retain
the existing `domain`, `harness`, and `context_processing` node kinds and add
component accounting for `profile_master`, `temporary_guard`, `domain:<role>`,
`tool`, and `control`. Transition and turn records gain optional profile
provenance so old records remain readable. Provider usage must be conserved
between total, node-kind, and component ledgers.

Terminal events expose role selection, handoff, completion, and pinned artifact
provenance. Workspace file and execution primitives provide agent-native profile
editing and validation, but agents cannot autonomously approve or activate their
own profile.

## Test matrix

Profile and compiler tests cover strict/invalid schemas, duplicate keys, unknown
fields and tags, identifiers, section mismatches, canonical hashing,
deterministic goldens, scripted LLM output, one repair, and cache hits with zero
provider calls.

AST tests reject imports, inheritance, decorators, methods, calls, attributes,
comprehensions, environment/file/network access attempts, and prove no generated
source is executed.

Artifact tests cover content/ABI/tool-schema incompatibility, approval digest
binding, atomic activation, turn pinning, rollback eligibility, and activation
audit failure.

Runtime tests cover initial routing, tool-return role affinity, A-to-B handoff,
B completion, handoff limits, invalid-master fallback, mixed-call zero-side
effects, bare-text correction/termination, finalization exemption, schema hiding,
and execution-layer denial.

Observability tests cover provenance, component/node/total conservation,
compile-turn ledger separation, and exactly one final hook. Compatibility tests
must preserve the existing suite and no-profile node/message/status behavior.

## Verification gates

- `uv run pytest -q`
- `uv run ruff check .`
- `uv build`
- deterministic compiler golden and malicious AST suites
- profile runtime/control/scoping/provenance tests
- existing runtime and safety contract assertions
- `git diff --check`
- plan-aware correctness, maintainability, testing, Python, and adversarial
  review; resolve all approved findings before completion
- browser testing is N/A unless a browser-facing surface is added
