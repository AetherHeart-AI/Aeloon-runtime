# Architecture

Aeloon Core is one Python distribution with inward-only dependencies. Electron and UI code are
not dependencies of this repository.

```mermaid
flowchart LR
    Workbench["Bun Workbench"] --> RPC["aeloon-rpc-v1 adapter"]
    RPC --> Runtime["Core runtime"]
    Runtime --> Agent["stateless agent core"]
    Runtime --> Tools["runtime tool set"]
    Tools --> Agent
```

The fixed dependency directions are `rpc → runtime → core`, `runtime → tool/core`, and
`tool → core`. Bootstrap is the composition root. Core never imports Electron, Bun, React, UI,
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

## Runtime: state and resources

Runtime owns Sessions, Skills, prompt templates, prompt construction, artifacts, compaction
selection and persistence, Provider configuration, and all resource lifecycles. `SessionAgent`
uses one operation-scoped `ProviderManager`, so the main run, compaction, branch summary, and
automatic title reuse one inference instance. Closing the agent closes the whole manager.

The built-in Office boundary is intentionally narrow. Runtime provisions exactly
`aeloon-office-lite` and dispatches its `preflight`, `read`, `write`, `render`, and `validate`
actions to one Python entry point. The main process includes only lightweight document libraries.
Scanned PDFs are rendered for the model's visual capability instead of starting an OCR runtime.
Optional LibreOffice rendering is a validation enhancement, not a condition that may be silently
assumed.

`ProviderManager` constructs Providers lazily from a fixed driver factory mapping. It resolves
qualified and unqualified model IDs, isolates transient model-discovery failures by Provider, and
closes every instantiated Provider idempotently. Catalog and settings operations use short-lived
managers and always close them in `finally` blocks.

Bootstrap also gives each Manager a lazy Cloud account gateway bound to the same configuration
snapshot. Updating settings can therefore replace the service-level account client without
mutating an operation that is already running.

Concrete implementations live in `aeloon_core.runtime.providers`: Custom OpenAI-compatible APIs,
DeepSeek, Aeloon Cloud, and the testing-only `ScriptedProvider`.

## Local RPC and Cloud

`aeloon-rpc-v1` is a small private transport adapter over Runtime. It uses length-prefixed JSON on
a restricted Unix socket and owns dispatch, cancellation, frame limits, timeouts, event delivery,
and JSON DTOs. It has no legacy negotiation, token, certificate, capability grant, or background
discovery. The adapter does not import UI or Electron code.

Cloud owns login, refresh tokens, the vault, and raw model-catalog access. It does not create Core
models or inference implementations. Bootstrap adapts `CloudAccountService` to `AccountGateway`
and injects it into Runtime's Provider manager factory.

Sessions remain append-only JSONL. Socket paths and transport state are operation-local and are
never serialized into Session data.
