# Changelog

Significant changes to Aeloon Core are recorded here. New work is added under `Unreleased`; when a
release is cut, those entries should move to a versioned section with an ISO date.

## Unreleased

## 0.0.11 - 2026-08-09

### Fixed

- Made llama.cpp reasoning deterministic by requesting the structured DeepSeek reasoning format,
  forwarding per-turn thinking intent, and preserving reasoning in assistant history.
- Added streaming compatibility for `reasoning`, `thinking`, and related reasoning fields, with a
  chunk-safe fallback for legacy `<think>` content.

## 0.0.6 - 2026-08-09

### Changed

- Removed the standalone PyInstaller desktop artifacts and release workflows. Core is now built and
  tested as a wheel for inclusion in the unified Aeloon UI desktop installer.
- Required the locked OCR environment to inherit the Aeloon-managed Python runtime and disabled uv
  Python downloads, keeping heavyweight Skill isolation without introducing a second interpreter.

## 0.0.5 - 2026-08-09

### Added

- Added the Python-driven `document-reader`, `word-docx`, and `powerpoint-pptx` built-in skills.
  They provide document ingestion with evidence sidecars, editable DOCX/PPTX build and targeted
  editing, structural validation, and optional LibreOffice-backed visual verification.
- Added a locked, opt-in Docling/RapidOCR runtime with explicit preparation, cache manifests, and
  offline cache-miss behavior. External Marker and PyMuPDF4LLM engines require a recorded license
  acceptance and are never selected by default.
- Added Word tracked replacements and comments through the versioned `word-edit-spec/v1.1`, plus
  an independently gated Chinese scanned-PDF OCR quality benchmark.

### Changed

- Replaced the nine former Office skills with three focused single-entry Python skills. Existing
  copies in user data directories are preserved, but the retired built-in IDs now return migration
  guidance instead of running compatibility implementations.
- Removed the bundled Node.js/PptxGenJS, PaddlePaddle/PaddleOCR, ReportLab, and related packaging
  branches. LibreOffice and the system `uv` executable remain optional and all missing validation
  or download capabilities are reported explicitly.

### Removed

- Removed built-in PDF creation, merging, splitting, rotation, form filling, and encryption. These
  requests now require a reviewed custom PDF skill or a dedicated PDF tool; document ingestion and
  PDF rendering remain supported by `document-reader`.

## 0.0.4 - 2026-08-07

### Added

- Bundled local-first office execution skills for MarkItDown document extraction, PDF inspection,
  PaddleOCR PP-StructureV3 parsing, editable PPTX generation, and Chinese DOCX formatting. The
  office router now delegates digital, scanned, presentation, and document-writing workflows to
  these skills.
- Official packages now include the Python, Node.js, PptxGenJS, PDF, DOCX, PaddlePaddle, and
  PaddleOCR execution dependencies required by the built-in office skills. OCR model weights are
  cached separately after first use, and LibreOffice remains an optional visual-QA dependency.
- Extended the desktop child-process startup window for large single-file packages so cold extraction on
  slower disks does not fail after five seconds.

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
- Unified the former desktop transport Provider/settings DTOs and made unsupported protocol
  versions fail explicitly. Append-only JSONL Session data remained readable at that release.
- Moved Skill metadata/content, prompt construction, summarization, compaction selection, and
  typed artifacts into Runtime. Cloud now owns only account, token, vault, and raw catalog access.
- Removed old Python import re-exports and compatibility shims; package version is now `0.4.0`.
- Expanded session context statistics with context-window occupancy, token share by
  message type, and cache token/request hit rates.
- Simplified the main README around installation, common workflows, configuration, and
  development.
- Moved compatibility guidance and internal architecture details into dedicated documents.
- Reworded public project descriptions and core module documentation to focus on Aeloon behavior.
- Added runtime-owned skill selection, `/skill` prompt resolution, metadata-rich client catalog
  entries, and on-demand loading of full skill instructions.
- Bundled office, presentation, document-writing, and reporting skills and provisioned missing
  presets into the Core data directory without overwriting user-owned skills.
