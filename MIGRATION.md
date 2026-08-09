# Breaking runtime rewrite

The Electron runtime rewrite is intentionally incompatible with prior desktop transports and
data. There is no automatic protocol, Session, or configuration migration path.

The supported desktop integration is now:

- `aeloon rpc serve --socket <path> --browser-runtime-socket <path>` for the Bun Workbench;
- `aeloon-rpc-v1` for Workbench-to-Core requests and events;
- `browser-runtime-v1` for Core Browser Use execution;
- one UI thread UUID used directly as the Core Session and Browser scope ID.

Start with the new Electron application data directory. Old desktop data is neither read nor
deleted. Python API consumers can continue to compose `RuntimeService` directly, but removed CLI
commands and transports have no compatibility aliases.

## Built-in Office skill replacement

The built-in Office catalog now contains only `document-reader`, `word-docx`, and
`powerpoint-pptx`. The former IDs are retired as follows:

| Retired ID | Replacement |
| --- | --- |
| `office`, `markitdown`, `pdf`, `paddleocr-doc-parsing` | `document-reader` |
| `document-writing`, `reports`, `document-format-skills` | `word-docx` |
| `ppt`, `pptx-generator` | `powerpoint-pptx` |

Runtime does not delete or overwrite any same-named directories under `~/.aeloon-core/skills`.
Those user-owned copies can coexist with the new built-in IDs. Calling a retired built-in ID no
longer runs its former implementation and instead returns the replacement guidance above.

`document-reader --offline` prohibits downloads and network-dependent environment changes. It can
use Docling/RapidOCR only after `prepare-ocr` has produced a complete cache manifest. On an offline
cache miss, a digital document may fall back to MarkItDown; a scanned document produces
`failed_for_agent` with `offline_cache_miss` evidence rather than an empty success.

The built-in PDF skill no longer creates, merges, splits, rotates, fills forms, encrypts, or
otherwise writes PDFs. Install a reviewed custom PDF skill or use a dedicated PDF application for
those operations. Aeloon will not silently substitute an unreviewed tool. PDF reading, OCR,
extraction, page rendering, and visual inspection remain available through `document-reader`.

Node.js, PptxGenJS, PaddlePaddle, PaddleOCR, ReportLab, and `nodejs-wheel` are no longer package
dependencies. Docling and RapidOCR live in a separate locked `uv` environment; LibreOffice remains
an optional renderer. Missing `uv`, models, cache entries, or LibreOffice are always surfaced in
preflight or validation output.
