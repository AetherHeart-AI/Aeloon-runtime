---
name: document-reader
description: Read and quality-check trusted local PDF, DOCX, PPTX, XLSX/XLS, CSV, HTML, Markdown, text, and scanned image files into Markdown with evidence sidecars. Use for document extraction, OCR, table or multi-column PDF intake, local Office-file reading, and PDF page rendering; do not use for creating, merging, splitting, rotating, filling, or encrypting PDFs.
---

# Document Reader

Use the bundled Python entry point through `aeloon system skill document-reader ACTION`.

## Workflow

1. Run `preflight --json` before OCR-sensitive work. It reports the packaged readers, `uv`, LibreOffice, the isolated Docling/RapidOCR environment, and model-cache integrity.
2. For a scan, run `prepare-ocr` before an offline job. This is the only explicit setup action: it creates a locked uv environment and downloads Docling plus Chinese-capable RapidOCR ONNX artifacts without uploading the source document.
3. Run `ingest INPUT --output-dir DIR --engine auto`. Add `--offline` when network access is forbidden. Read `source.manifest.json` first, then follow its `reading_order`.
4. Treat `failed_for_agent` as a failed extraction even though diagnostic sidecars exist. Report the reason; never infer missing text or claim OCR succeeded.
5. Use `render-pdf INPUT --output-dir DIR` when page images are needed to verify extraction. Do not present those PNGs unless requested.

`auto` audits PDFs with pypdf/pdfplumber, uses MarkItDown for normal text-bearing documents, and selects Docling with RapidOCR ONNX for scans or structurally complex PDFs. In offline mode, Docling runs only when the locked environment and models are complete; a scan with a cache miss fails with `offline_cache_miss`.

Only accept local paths. For `.doc`, `.ppt`, `.wps`, `.dps`, or `.et`, ask the user to use Office/WPS **Save As** to create `.docx`, `.pptx`, or `.xlsx`. Refuse URI and remote inputs.

## External engines

Marker and PyMuPDF4LLM are never installed or selected automatically. Invoke one only after the user accepts its terms with `--engine ENGINE --accept-external-license ENGINE`. The acceptance, detected version, and license class are written to `source.evidence.json` and reused when continuing in the same output directory.

Read [references/schemas.md](references/schemas.md) when consuming sidecars, [references/ocr-runtime.md](references/ocr-runtime.md) when preparing or diagnosing OCR, and [references/licenses.md](references/licenses.md) before proposing an external engine.

## Delivery

Keep source files unchanged. Treat Markdown, JSON evidence, model caches, and rendered pages as working artifacts unless the user asks for them. Deliver requested final files once with `present_files`.

This skill does not write PDFs. For create, merge, split, rotate, form-fill, or encryption requests, state that the built-in capability was retired and recommend a reviewed custom PDF skill or a dedicated local PDF tool.
