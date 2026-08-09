# Extraction sidecars

Every `ingest` run writes these files to the output directory:

- `source.md`: extracted Markdown, or an explicit extraction-failure notice.
- `source.manifest.json`: stable outcome summary and safe reading order.
- `source.evidence.json`: engine attempts, PDF audit, runtime state, metrics, and license acceptances.

Both JSON files use versioned identifiers. Consumers must reject an unknown major version and may ignore new fields within version 1.

## Manifest (`document-reader-manifest/v1`)

Key fields are `status`, `risk`, `source`, `engine`, `outputs`, `reading_order`, and `error`. `status` is one of:

- `good`: direct extraction passed the quality gate.
- `salvaged`: OCR or a fallback engine produced usable text; verify visually.
- `failed_for_agent`: do not use `source.md` as extracted content.

`risk.level` is `low`, `medium`, or `high`; `risk.reasons` explains the decision. `error.code` is stable enough for automation, while `error.message` is diagnostic text.

## Evidence (`document-reader-evidence/v1`)

Read `attempts` in order. Each entry records an engine, outcome, and metrics or error. `pdf_audit` contains page coverage, suspicious-character, table, and multi-column signals. `runtime` contains only local environment/cache facts. `license_acceptances` persists explicit external-engine choices in this directory.
