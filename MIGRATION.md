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

The built-in Office catalog now contains only `aeloon-office-lite`. It replaces the previous
document-reading, Word, and PowerPoint skills with one fast path for simple PDF, DOCX, PPTX, and
XLSX reads and writes. The former IDs are retired as follows:

| Retired ID | Replacement |
| --- | --- |
| `document-reader`, `word-docx`, `powerpoint-pptx` | `aeloon-office-lite` |
| `office`, `markitdown`, `pdf`, `paddleocr-doc-parsing` | `aeloon-office-lite` |
| `document-writing`, `reports`, `document-format-skills` | `aeloon-office-lite` |
| `ppt`, `pptx-generator` | `aeloon-office-lite` |

Runtime does not delete or overwrite any same-named directories under `~/.aeloon-core/skills`.
Those user-owned copies can coexist with the new built-in IDs. Calling a retired built-in ID no
longer runs its former implementation and instead returns the replacement guidance above.

`aeloon-office-lite` does not download OCR models. It extracts normal text directly and renders
scanned PDFs into PNG pages for the model's visual capability. It creates simple PDFs, DOCX files,
editable 16:9 PPTX decks, and XLSX workbooks from one compact JSON schema. Complex templates,
macros, revisions, animations, formula calculation, and pixel-exact layout are intentionally out
of scope.

The implementation uses Python only: pypdf/pypdfium2, python-docx, python-pptx, openpyxl, and
ReportLab. LibreOffice is an optional Office-to-PDF renderer. When a Python dependency must be
installed, prompts and project locking default to the Tsinghua PyPI mirror.
