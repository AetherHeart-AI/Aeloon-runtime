# Changelog

Significant changes to Aeloon Core are recorded here. New work is added under `Unreleased`; when a
release is cut, those entries should move to a versioned section with an ISO date.

## Unreleased

## 0.0.3 - 2026-08-07

### Changed

- Unified URL-configured model endpoints under the `custom` Provider driver. Provider setup now
  discovers the usable API base and model list automatically, then best-effort probes undeclared
  image-input capability without failing setup when a capability probe is rejected or times out.
- Custom Provider discovery now reads declared image capability from nested metadata such as
  `meta.allow_image`, avoiding an image probe when the endpoint already reports the capability.

## 0.4.0 - 2026-08-06

### Changed

- Made `core` a strictly stateless, vendor-neutral inference engine and renamed its contracts to
  `InferencePort`, `InferenceContext`, `InferenceError`, and `InferenceRuntime`.
- Added the object-oriented `aeloon_core.tool` package and moved filesystem, shell, and search
  tools out of Core. Runtime explicitly composes them with `PresentFilesTool`.
- Added `runtime.providers`, including OpenAI-compatible, DeepSeek, Ollama, Cloud, testing, and
  operation-scoped `ProviderManager` implementations.
- Replaced legacy top-level Provider settings with the discriminated `Config.providers` mapping.
  Legacy `deepseek`, `local_providers`, and `cloud` keys now fail with a migration error.
- Replaced CLI `local` commands with `aeloon provider add/list/remove` and required an explicit
  `ollama` or `openai-compatible` driver when adding an endpoint.
- Upgraded Bridge to v3 with `provider.add/remove`, unified Provider/settings DTOs, and explicit
  v2 handshake rejection. Append-only JSONL Session v3 remains readable and unchanged.
- Moved Skill metadata/content, prompt construction, summarization, compaction selection, and
  typed artifacts into Runtime. Cloud now owns only account, token, vault, and raw catalog access.
- Removed old Python import re-exports and compatibility shims; package version is now `0.4.0`.
- Expanded session context statistics with context-window occupancy, token share by
  message type, and cache token/request hit rates.
- Simplified the main README around installation, common workflows, configuration, and
  development.
- Moved compatibility guidance and internal architecture details into dedicated documents.
- Reworded public project descriptions and core module documentation to focus on Aeloon behavior.
- Added runtime-owned skill selection, `/skill` prompt resolution, metadata-rich Bridge catalog
  entries, and on-demand loading of full skill instructions.
- Bundled office, presentation, document-writing, and reporting skills and provisioned missing
  presets into the Core data directory without overwriting user-owned skills.
