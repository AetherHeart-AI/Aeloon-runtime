# Architecture

Aeloon is one Python distribution with inward-only dependencies:

```mermaid
flowchart LR
    Client["Bridge client"] --> Bridge["bridge v3"]
    Bridge --> Runtime["runtime"]
    Runtime --> Core["stateless core"]
    Runtime --> Tool["tool"]
    Tool --> Core
    Bootstrap["bootstrap"] --> Runtime
    Bootstrap --> Cloud["cloud account"]
```

The fixed dependency directions are `bridge → runtime → core`, `runtime → tool/core`, and
`tool → core`. Bootstrap is the composition root. Core never imports Runtime, Cloud, Bridge,
`httpx`, Pillow, or a concrete vendor implementation.

## Core: one stateless inference run

`aeloon_core.core.run_agent()` receives a complete `RunRequest` and returns a `RunResult`.
`RunRequest.inference` is an `InferencePort`; tools implement the neutral `Tool` protocol. Core
owns only invocation-local engine, controller, queues, cancellation tasks, messages, and tool-loop
state. Nothing is retained after the await completes.

Core contains model identity and general capability metadata, streaming inference contracts,
events, token estimation, and the `ContextCompactor` port. It coordinates threshold and overflow
compaction but has no Session selection, summarization prompt, transport, authentication, model
discovery, or vendor compatibility logic.

## Tool: object-oriented built-ins

`aeloon_core.tool` contains `BaseTool`, `ToolContext`, filesystem tools, `BashTool`, search tools,
and `BuiltinToolSet`. A ToolSet shares one context-scoped mutation-lock map; there is no process
global write registry. Writes and edits replace their target atomically.

Runtime's `RuntimeToolSet` explicitly adds `PresentFilesTool`. This composition is intentionally
small and does not introduce a plugin registry.

## Runtime: state and capabilities

Runtime owns Sessions, Skills, prompt templates, prompt construction, artifacts, compaction
selection and persistence, Provider configuration, and all resource lifecycles. `SessionAgent`
uses one operation-scoped `ProviderManager`, so the main run, compaction, branch summary, and
automatic title reuse one inference instance. Closing the agent closes the whole manager.

`ProviderManager` constructs Providers lazily from a fixed driver factory mapping. It resolves
qualified and unqualified model IDs, isolates transient model-discovery failures by Provider, and
closes every instantiated Provider idempotently. Catalog and settings operations use short-lived
managers and always close them in `finally` blocks.

Bootstrap also gives each Manager a lazy Cloud account gateway bound to the same configuration
snapshot. Updating settings can therefore replace the service-level account client without
mutating an operation that is already running.

Concrete implementations live in `aeloon_core.runtime.providers`: a shared
`OpenAICompatibleProvider`, DeepSeek, Ollama, Aeloon Cloud, and the testing-only
`ScriptedProvider`.

## Bridge and Cloud

Bridge v3 is a transport adapter over Runtime. It owns JSON-RPC dispatch, handshake negotiation,
wire errors, event sequencing/replay, and JSON DTOs. It does not import Core.

Cloud owns login, refresh tokens, the vault, and raw model-catalog access. It does not create Core
models or inference implementations. Bootstrap adapts `CloudAccountService` to `AccountGateway`
and injects it into Runtime's Provider manager factory.

Sessions remain append-only JSONL schema v3. Existing Session v3 files remain readable even
though newly serialized models no longer carry transport fields.
