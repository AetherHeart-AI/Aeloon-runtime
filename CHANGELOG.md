# Changelog

Significant changes to Aeloon Core are recorded here. New work is added under `Unreleased`; when a
release is cut, those entries should move to a versioned section with an ISO date.

## Unreleased

### Changed

- Split the package into stateless `core`, stateful `runtime`, wire-facing `bridge`, and optional
  `cloud` modules while retaining one Python distribution.
- Replaced `AgentHarness` with `core.run_agent()` and `CoreService` with
  `runtime.RuntimeService`; removed the old Python modules without compatibility shims.
- Preserved the CLI surface, Bridge v2 wire contract, and append-only JSONL Session v3 format.
- Expanded session context statistics with context-window occupancy, token share by
  message type, and cache token/request hit rates.
- Simplified the main README around installation, common workflows, configuration, and
  development.
- Moved compatibility guidance and internal architecture details into dedicated documents.
- Reworded public project descriptions and core module documentation to focus on Aeloon behavior.
