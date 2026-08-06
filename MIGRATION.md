# Migrating to 0.4

Version 0.4 intentionally removes the old Provider configuration, Python imports, CLI aliases,
and Bridge v2 contract. JSONL Session schema v3 is unchanged and existing sessions remain
readable.

## Configuration

Move all Provider settings into the `providers` mapping:

```yaml
providers:
  deepseek:
    driver: deepseek
    name: DeepSeek
    enabled: true
    endpoint: https://api.deepseek.com
    api_key: null
    proxy: null
    headers: {}

  aeloon-cloud:
    driver: cloud
    name: Aeloon Cloud
    enabled: true
    endpoint: https://api.aetherheart.com
    proxy: null
    device_name: Aeloon Core
    allow_insecure_http: false
```

The IDs `deepseek` and `aeloon-cloud` are reserved. They may be edited or disabled, but cannot be
removed or assigned another driver. API keys are nullable; the old `"no-key"` sentinel is not
accepted.

Files containing any old top-level `deepseek`, `local_providers`, or `cloud` key are rejected with
`ConfigMigrationError`. Aeloon does not automatically convert or accept both formats. Rewrite the
file before starting 0.4.

## Python API

- `Provider`, `ProviderContext`, `ProviderError`, and `ProviderRuntime` become `InferencePort`,
  `InferenceContext`, `InferenceError`, and `InferenceRuntime`.
- `RunRequest.provider` becomes `RunRequest.inference`.
- `AgentTool` becomes the `Tool` protocol.
- Concrete Providers are imported from `aeloon_core.runtime.providers`.
- Built-in tools and `BuiltinToolSet` are imported from `aeloon_core.tool`.
- Skill, `LoadedSkill`, `PromptTemplate`, and prompt construction are imported from
  `aeloon_core.runtime`.
- `PresentFilesTool` replaces the old factory and is normally composed by Runtime.

There are no old-name re-exports.

## CLI

Use only the unified Provider commands:

```bash
aeloon provider list
aeloon provider add ollama --driver ollama
aeloon provider add studio --driver openai-compatible --endpoint https://api.example/v1
aeloon provider remove studio
```

`--model` may be repeated. If it is omitted, the selected driver discovers `/models` and stores a
normalized model list. Cloud account operations remain separate as `aeloon login/logout/whoami`
or the low-level `aeloon cloud ...` commands. The old `local`, `provider local`, Provider login,
and cloud Provider aliases are removed.

## Bridge

Bridge v3 replaces `provider.local.add/remove` with `provider.add/remove`, removes
`provider.cloud.*`, and keeps `cloud.account.*`. `settings.get` exposes a unified `providers`
mapping without API keys. Provider DTOs contain `id`, `name`, `driver`, `kind`, `endpoint`,
`enabled`, `authenticated`, `credential_configured`, and `model_ids`.

Clients must offer protocol version 3 during `system.handshake`. A client offering only v2
receives `protocol_incompatible`. The v3 schema is available through `aeloon bridge schema` and
at `aeloon_core/bridge/bridge-protocol-v3.json`.
