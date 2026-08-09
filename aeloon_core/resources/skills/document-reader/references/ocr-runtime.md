# OCR runtime

OCR is isolated from the main application. `runtime/uv.lock` freezes Docling, RapidOCR, ONNXRuntime, openpyxl, xlrd, and their transitive dependencies. `prepare-ocr` sets `UV_PROJECT_ENVIRONMENT` to the selected cache rather than writing into the bundled skill.

The default cache is `${XDG_CACHE_HOME:-~/.cache}/aeloon/document-reader`; override it with `--cache-dir`. It contains:

- `venv/`: locked Python environment.
- `uv/`: uv package cache.
- `models/`: prefetched Docling layout, table, and RapidOCR ONNX artifacts.
- `runtime-manifest.json`: package versions, paths, file count, size/path fingerprint, and integrity state.

The RapidOCR configuration uses ONNXRuntime and Chinese PP-OCR detection/recognition settings. Documents remain local; package and model hosts receive no document bytes.

`--offline` sets library offline flags and blocks Python socket creation in the OCR worker. It never runs `uv sync` or a model downloader. If the runtime manifest or declared paths fail integrity checks, text-bearing files may still use MarkItDown, but scans return `failed_for_agent` with `offline_cache_miss`.
