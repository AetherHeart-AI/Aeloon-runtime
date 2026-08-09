#!/usr/bin/env python3
"""Read trusted local documents into Markdown with quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROJECT = SKILL_ROOT / "runtime"
RUNTIME_WORKER = RUNTIME_PROJECT / "ocr_runtime.py"
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".markdown",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
NATIVE_SUFFIXES = {".doc", ".ppt", ".wps", ".dps", ".et"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXTERNAL_LICENSES = {
    "marker": "apache-2.0-source_with_separate_model-weight-terms",
    "pymupdf4llm": "agpl-3.0-or-commercial",
}
MANIFEST_SCHEMA = "document-reader-manifest/v1"
EVIDENCE_SCHEMA = "document-reader-evidence/v1"
RUNTIME_SCHEMA = "document-reader-ocr-runtime/v1"


class ReaderError(RuntimeError):
    code = "reader_error"


class InputError(ReaderError):
    code = "invalid_input"


class OfflineCacheMiss(ReaderError):
    code = "offline_cache_miss"


class LicenseAcceptanceRequired(ReaderError):
    code = "external_license_not_accepted"


class ExtractionError(ReaderError):
    code = "extraction_failed"


@dataclass(frozen=True)
class ExtractionBundle:
    markdown: Path
    manifest: Path
    evidence: Path
    status: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "aeloon" / "document-reader").resolve(strict=False)


def resolve_cache_dir(value: str | None) -> Path:
    return Path(value).expanduser().resolve(strict=False) if value else default_cache_dir()


def local_input(value: str, *, suffixes: set[str] | None = None) -> Path:
    lowered = value.strip().lower()
    uri_scheme = re.match(r"^[a-z][a-z0-9+.-]*:", lowered)
    windows_drive = re.match(r"^[a-z]:[\\/]", lowered)
    if uri_scheme and not windows_drive:
        raise InputError("URI and remote inputs are not allowed; provide a trusted local path")
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise InputError(f"input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in NATIVE_SUFFIXES:
        raise InputError(
            f"{suffix} is unsupported; use Office/WPS Save As to create DOCX, PPTX, or XLSX"
        )
    allowed = SUPPORTED_SUFFIXES if suffixes is None else suffixes
    if suffix not in allowed:
        names = ", ".join(sorted(allowed))
        raise InputError(f"unsupported file type {suffix or '(none)'}; expected one of: {names}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def deny_network(enabled: bool):
    """Block Python socket connections for an offline in-process conversion."""
    if not enabled:
        yield
        return
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for offline document reading")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


def package_state(distribution: str, module: str | None = None) -> dict[str, Any]:
    module_name = module or distribution.replace("-", "_")
    available = importlib.util.find_spec(module_name) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"available": available, "version": version}


def runtime_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "runtime-manifest.json"


def venv_python(cache_dir: Path) -> Path:
    if os.name == "nt":
        return cache_dir / "venv" / "Scripts" / "python.exe"
    return cache_dir / "venv" / "bin" / "python"


def venv_command(cache_dir: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    directory = "Scripts" if os.name == "nt" else "bin"
    return cache_dir / "venv" / directory / f"{name}{suffix}"


def model_fingerprint(model_dir: Path) -> tuple[int, int, str]:
    entries: list[tuple[str, int]] = []
    if model_dir.is_dir():
        for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
            entries.append((path.relative_to(model_dir).as_posix(), path.stat().st_size))
    encoded = json.dumps(entries, separators=(",", ":")).encode()
    return len(entries), sum(size for _, size in entries), hashlib.sha256(encoded).hexdigest()


def inspect_ocr_runtime(cache_dir: Path) -> dict[str, Any]:
    path = runtime_manifest_path(cache_dir)
    state: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "manifest": str(path),
        "environment_ready": venv_python(cache_dir).is_file(),
        "models_ready": False,
        "complete": False,
    }
    if not path.is_file():
        state["reason"] = "runtime_manifest_missing"
        return state
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        state["reason"] = f"runtime_manifest_invalid: {exc}"
        return state
    state["versions"] = manifest.get("versions", {})
    models = cache_dir / "models"
    count, size, fingerprint = model_fingerprint(models)
    recorded = manifest.get("models", {})
    models_ready = bool(
        count > 0
        and count == recorded.get("file_count")
        and size == recorded.get("total_bytes")
        and fingerprint == recorded.get("fingerprint")
    )
    state["models_ready"] = models_ready
    state["model_files"] = count
    state["complete"] = bool(
        state["environment_ready"]
        and models_ready
        and manifest.get("schema") == RUNTIME_SCHEMA
        and manifest.get("integrity") == "complete"
    )
    if not state["complete"]:
        state["reason"] = "runtime_or_model_integrity_mismatch"
    return state


def collect_preflight(cache_dir: Path) -> dict[str, Any]:
    uv_path = shutil.which("uv")
    libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
    return {
        "schema": "document-reader-preflight/v1",
        "timestamp": utc_now(),
        "cache_dir": str(cache_dir),
        "components": {
            "markitdown": package_state("markitdown"),
            "pypdf": package_state("pypdf"),
            "pdfplumber": package_state("pdfplumber"),
            "pypdfium2": package_state("pypdfium2"),
            "uv": {"available": bool(uv_path), "path": uv_path},
            "libreoffice": {"available": bool(libreoffice), "path": libreoffice},
        },
        "ocr": inspect_ocr_runtime(cache_dir),
    }


def prepare_ocr(cache_dir: Path) -> dict[str, Any]:
    uv = shutil.which("uv")
    if not uv:
        raise ReaderError("uv is required for OCR setup; install uv and rerun prepare-ocr")
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(cache_dir / "venv"),
            "UV_CACHE_DIR": str(cache_dir / "uv"),
            "UV_PYTHON_DOWNLOADS": "never",
            "HF_HOME": str(cache_dir / "huggingface"),
            "DOCLING_ARTIFACTS_PATH": str(cache_dir / "models"),
        }
    )
    bundled_python = os.environ.get("AELOON_RUNTIME_PYTHON", "").strip()
    if bundled_python:
        if not Path(bundled_python).is_file():
            raise ReaderError(f"Aeloon bundled Python is missing: {bundled_python}")
        environment["UV_PYTHON"] = bundled_python
    environment.setdefault("HF_HUB_DISABLE_XET", "1")
    environment.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    sync = subprocess.run(
        [
            uv,
            "sync",
            "--frozen",
            "--no-dev",
            "--no-python-downloads",
            "--project",
            str(RUNTIME_PROJECT),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if sync.returncode:
        raise ReaderError(f"locked OCR environment setup failed: {sync.stderr.strip()}")
    downloader = venv_command(cache_dir, "docling-tools")
    if not downloader.is_file():
        raise ReaderError("Docling model downloader is missing from the locked environment")
    models = cache_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    download_command = [
        str(downloader),
        "models",
        "download",
        "layout",
        "tableformer",
        "rapidocr",
        "--output-dir",
        str(models),
    ]
    download_errors: list[str] = []
    for attempt in range(1, 4):
        download = subprocess.run(
            download_command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if not download.returncode:
            break
        detail = download.stderr.strip() or download.stdout.strip()
        download_errors.append(f"attempt {attempt}: {detail}")
    else:
        raise ReaderError(
            "OCR model prefetch failed after 3 resumable attempts: "
            + "\n".join(download_errors)
        )
    python = venv_python(cache_dir)
    version_code = (
        "import importlib.metadata as m,json;"
        "print(json.dumps({n:m.version(n) for n in "
        "['docling','rapidocr','onnxruntime','openpyxl','xlrd']}))"
    )
    version_run = subprocess.run(
        [str(python), "-c", version_code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if version_run.returncode:
        raise ReaderError(f"OCR runtime version check failed: {version_run.stderr.strip()}")
    count, total, fingerprint = model_fingerprint(models)
    if count == 0:
        raise ReaderError("OCR model prefetch completed without model files")
    manifest = {
        "schema": RUNTIME_SCHEMA,
        "created_at": utc_now(),
        "integrity": "complete",
        "environment": str(cache_dir / "venv"),
        "lock": str(RUNTIME_PROJECT / "uv.lock"),
        "versions": json.loads(version_run.stdout),
        "ocr": {
            "engine": "rapidocr",
            "backend": "onnxruntime",
            "language": "ch",
            "model_family": "PP-OCR",
        },
        "models": {
            "path": str(models),
            "file_count": count,
            "total_bytes": total,
            "fingerprint": fingerprint,
        },
    }
    atomic_json(runtime_manifest_path(cache_dir), manifest)
    return inspect_ocr_runtime(cache_dir)


def _page_has_multiple_columns(page: Any, words: list[dict[str, Any]]) -> bool:
    width = float(getattr(page, "width", 0) or 0)
    if width <= 0 or len(words) < 30:
        return False
    left = sum(float(word.get("x0", 0)) < width * 0.38 for word in words)
    right = sum(float(word.get("x0", 0)) > width * 0.56 for word in words)
    gap = sum(width * 0.43 <= float(word.get("x0", 0)) <= width * 0.52 for word in words)
    return left >= 12 and right >= 12 and gap <= max(2, len(words) // 25)


def audit_pdf(source: Path) -> dict[str, Any]:
    try:
        import pdfplumber
        from pypdf import PdfReader
    except ImportError as exc:
        raise ReaderError("packaged pypdf/pdfplumber audit runtime is missing") from exc
    try:
        reader = PdfReader(str(source))
        if reader.is_encrypted:
            raise ExtractionError("encrypted PDFs are unsupported")
        pypdf_pages = len(reader.pages)
        page_characters: list[int] = []
        table_count = 0
        table_cells = 0
        multicolumn_pages = 0
        suspicious = 0
        total_chars = 0
        with pdfplumber.open(str(source)) as document:
            for page in document.pages:
                text = page.extract_text() or ""
                count = len(text.strip())
                page_characters.append(count)
                total_chars += len(text)
                suspicious += text.count("\ufffd") + sum(
                    unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text
                )
                tables = page.extract_tables() or []
                table_count += len(tables)
                table_cells += sum(
                    len(row) for table in tables for row in table if isinstance(row, list)
                )
                words = page.extract_words() or []
                multicolumn_pages += int(_page_has_multiple_columns(page, words))
        pages = max(pypdf_pages, len(page_characters))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"PDF audit failed: {exc}") from exc
    covered = sum(count >= 40 for count in page_characters)
    coverage = covered / pages if pages else 0.0
    average = sum(page_characters) / pages if pages else 0.0
    suspicious_ratio = suspicious / max(total_chars, 1)
    reasons: list[str] = []
    if not pages:
        reasons.append("no_pages")
    if coverage < 0.6 or average < 40:
        reasons.append("low_text_layer_coverage")
    if suspicious_ratio > 0.02:
        reasons.append("suspicious_text_encoding")
    if table_count >= 2 or table_cells >= 16:
        reasons.append("complex_tables")
    if multicolumn_pages:
        reasons.append("multi_column_layout")
    return {
        "pages": pages,
        "page_characters": page_characters,
        "text_layer_coverage": round(coverage, 4),
        "average_characters_per_page": round(average, 2),
        "suspicious_character_ratio": round(suspicious_ratio, 6),
        "tables": table_count,
        "table_cells": table_cells,
        "multi_column_pages": multicolumn_pages,
        "needs_docling": bool(reasons),
        "reasons": reasons,
    }


def convert_markitdown(source: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise ExtractionError("packaged MarkItDown runtime is missing") from exc
    converter = MarkItDown()
    result = (
        converter.convert_local(source)
        if hasattr(converter, "convert_local")
        else converter.convert(str(source))
    )
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        markdown = getattr(result, "text_content", None)
    if not isinstance(markdown, str) or not markdown.strip():
        raise ExtractionError("MarkItDown produced no text")
    return markdown


def convert_pdfplumber(source: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ExtractionError("packaged pdfplumber runtime is missing") from exc
    pages = []
    try:
        with pdfplumber.open(str(source)) as document:
            for index, page in enumerate(document.pages, 1):
                pages.append(f"## Page {index}\n\n{(page.extract_text() or '').strip()}")
    except Exception as exc:
        raise ExtractionError(f"pdfplumber extraction failed: {exc}") from exc
    markdown = "\n\n".join(pages)
    if not re.search(r"\w", markdown):
        raise ExtractionError("pdfplumber produced no text")
    return markdown


def run_docling(source: Path, cache_dir: Path, *, offline: bool) -> tuple[str, dict]:
    state = inspect_ocr_runtime(cache_dir)
    if not state["complete"]:
        if offline:
            raise OfflineCacheMiss(
                "offline OCR cache is incomplete; run prepare-ocr while online first"
            )
        state = prepare_ocr(cache_dir)
    if not state["complete"]:
        raise ExtractionError("OCR runtime failed its integrity check")
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(cache_dir / "huggingface"),
            "DOCLING_ARTIFACTS_PATH": str(cache_dir / "models"),
        }
    )
    if offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with tempfile.TemporaryDirectory(prefix="docling-", dir=cache_dir) as temporary:
        temporary_path = Path(temporary)
        markdown_path = temporary_path / "result.md"
        report_path = temporary_path / "report.json"
        command = [
            str(venv_python(cache_dir)),
            str(RUNTIME_WORKER),
            str(source),
            str(markdown_path),
            str(report_path),
            "--cache-dir",
            str(cache_dir),
        ]
        if offline:
            command.append("--offline")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ExtractionError(f"Docling/RapidOCR failed: {detail[-2000:]}")
        if not markdown_path.is_file() or not report_path.is_file():
            raise ExtractionError("Docling/RapidOCR did not produce expected output files")
        return (
            markdown_path.read_text(encoding="utf-8"),
            json.loads(report_path.read_text(encoding="utf-8")),
        )


def _external_version(engine: str) -> str | None:
    distributions = [engine]
    if engine == "marker":
        distributions = ["marker-pdf", "marker"]
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def run_marker(source: Path, *, offline: bool) -> str:
    binary = shutil.which("marker_single")
    if not binary:
        raise ExtractionError("Marker is not installed; install it in a user-managed environment")
    environment = os.environ.copy()
    if offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with tempfile.TemporaryDirectory(prefix="marker-") as temporary:
        completed = subprocess.run(
            [binary, str(source), "--output_dir", temporary],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if completed.returncode:
            raise ExtractionError(f"Marker failed: {completed.stderr.strip()[-2000:]}")
        candidates = list(Path(temporary).rglob("*.md"))
        if not candidates:
            raise ExtractionError("Marker produced no Markdown file")
        markdown = candidates[0].read_text(encoding="utf-8")
    if not markdown.strip():
        raise ExtractionError("Marker produced empty Markdown")
    return markdown


def run_pymupdf4llm(source: Path) -> str:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise ExtractionError(
            "PyMuPDF4LLM is not installed; install it in a user-managed environment"
        ) from exc
    markdown = pymupdf4llm.to_markdown(str(source))
    if not isinstance(markdown, str) or not markdown.strip():
        raise ExtractionError("PyMuPDF4LLM produced empty Markdown")
    return markdown


def load_license_acceptances(output_dir: Path) -> list[dict[str, Any]]:
    evidence = output_dir / "source.evidence.json"
    if not evidence.is_file():
        return []
    try:
        value = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    acceptances = value.get("license_acceptances", [])
    return acceptances if isinstance(acceptances, list) else []


def ensure_external_acceptance(
    engine: str, supplied: set[str], previous: list[dict[str, Any]]
) -> dict[str, Any]:
    for acceptance in previous:
        if acceptance.get("engine") == engine and acceptance.get("accepted") is True:
            return acceptance
    if engine not in supplied:
        raise LicenseAcceptanceRequired(
            f"{engine} requires explicit --accept-external-license {engine}"
        )
    return {
        "engine": engine,
        "accepted": True,
        "accepted_at": utc_now(),
        "license_class": EXTERNAL_LICENSES[engine],
        "version": _external_version(engine),
    }


def _source_record(source: Path) -> dict[str, Any]:
    return {
        "path": str(source),
        "name": source.name,
        "suffix": source.suffix.lower(),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _quality(markdown: str, *, pages: int | None = None) -> dict[str, Any]:
    meaningful = len(re.sub(r"\s+", "", markdown))
    return {
        "characters": len(markdown),
        "non_whitespace_characters": meaningful,
        "pages": pages,
        "has_content": meaningful > 0,
    }


def _write_bundle(
    output_dir: Path,
    *,
    markdown: str,
    source: dict[str, Any],
    requested_engine: str,
    selected_engine: str | None,
    status: str,
    risk_reasons: list[str],
    attempts: list[dict[str, Any]],
    pdf_audit: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
    acceptances: list[dict[str, Any]],
    error: ReaderError | None = None,
) -> ExtractionBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "source.md"
    manifest_path = output_dir / "source.manifest.json"
    evidence_path = output_dir / "source.evidence.json"
    persisted_markdown = markdown.rstrip() + "\n"
    atomic_text(markdown_path, persisted_markdown)
    risk_level = {"good": "low", "salvaged": "medium"}.get(status, "high")
    quality = _quality(persisted_markdown, pages=(pdf_audit or {}).get("pages"))
    quality["usable_for_agent"] = status != "failed_for_agent"
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "created_at": utc_now(),
        "source": source,
        "attempts": attempts,
        "pdf_audit": pdf_audit,
        "runtime": runtime,
        "quality": quality,
        "license_acceptances": acceptances,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "status": status,
        "risk": {"level": risk_level, "reasons": risk_reasons},
        "source": source,
        "engine": {"requested": requested_engine, "selected": selected_engine},
        "outputs": {
            "markdown": str(markdown_path),
            "manifest": str(manifest_path),
            "evidence": str(evidence_path),
        },
        "reading_order": ["source.md", "source.evidence.json", "source.manifest.json"],
        "error": ({"code": error.code, "message": str(error)} if error is not None else None),
    }
    atomic_json(evidence_path, evidence)
    atomic_json(manifest_path, manifest)
    return ExtractionBundle(markdown_path, manifest_path, evidence_path, status)


def ingest_document(
    input_value: str,
    output_dir: Path,
    *,
    engine: str = "auto",
    offline: bool = False,
    cache_dir: Path | None = None,
    accepted_licenses: set[str] | None = None,
) -> ExtractionBundle:
    output_dir = output_dir.expanduser().resolve(strict=False)
    cache_dir = (cache_dir or default_cache_dir()).expanduser().resolve(strict=False)
    accepted_licenses = accepted_licenses or set()
    previous_acceptances = load_license_acceptances(output_dir)
    attempts: list[dict[str, Any]] = []
    pdf_audit = None
    selected: str | None = None
    source_record: dict[str, Any] = {"path": input_value}
    current_acceptances = list(previous_acceptances)
    try:
        source = local_input(input_value)
        source_record = _source_record(source)
        for artifact in ("source.md", "source.manifest.json", "source.evidence.json"):
            if (output_dir / artifact).resolve(strict=False) == source:
                raise InputError("output sidecars would overwrite the input file")
        if source.suffix.lower() == ".pdf":
            pdf_audit = audit_pdf(source)
        if engine in EXTERNAL_LICENSES:
            acceptance = ensure_external_acceptance(engine, accepted_licenses, previous_acceptances)
            if not any(item.get("engine") == engine for item in current_acceptances):
                current_acceptances.append(acceptance)
        if engine == "auto":
            if source.suffix.lower() in IMAGE_SUFFIXES:
                selected = "docling"
            elif pdf_audit and pdf_audit["needs_docling"]:
                selected = "docling"
            else:
                selected = "markitdown"
        else:
            selected = engine
        try:
            with deny_network(offline and selected not in {"docling", "marker"}):
                if selected == "markitdown":
                    markdown = convert_markitdown(source)
                    metrics = _quality(markdown)
                elif selected == "pdfplumber":
                    if source.suffix.lower() != ".pdf":
                        raise InputError("pdfplumber accepts PDF inputs only")
                    markdown = convert_pdfplumber(source)
                    metrics = _quality(markdown, pages=(pdf_audit or {}).get("pages"))
                elif selected == "docling":
                    markdown, metrics = run_docling(source, cache_dir, offline=offline)
                elif selected == "marker":
                    if source.suffix.lower() != ".pdf":
                        raise InputError("Marker accepts PDF inputs only")
                    if offline:
                        raise InputError(
                            "offline mode cannot enforce network denial for external Marker; "
                            "use the locked Docling engine instead"
                        )
                    markdown = run_marker(source, offline=offline)
                    metrics = _quality(markdown)
                elif selected == "pymupdf4llm":
                    if source.suffix.lower() != ".pdf":
                        raise InputError("PyMuPDF4LLM accepts PDF inputs only")
                    markdown = run_pymupdf4llm(source)
                    metrics = _quality(markdown, pages=(pdf_audit or {}).get("pages"))
                else:
                    raise InputError(f"unsupported engine: {selected}")
            attempts.append({"engine": selected, "status": "success", "metrics": metrics})
        except ExtractionError as first_error:
            attempts.append({"engine": selected, "status": "failed", "error": str(first_error)})
            if engine != "auto" or selected == "docling":
                raise
            selected = "docling"
            markdown, metrics = run_docling(source, cache_dir, offline=offline)
            attempts.append({"engine": selected, "status": "success", "metrics": metrics})
        salvaged = selected in {"docling", "marker", "pymupdf4llm"} or len(attempts) > 1
        status = "salvaged" if salvaged else "good"
        reasons = list((pdf_audit or {}).get("reasons", []))
        if salvaged:
            reasons.append(f"extracted_with_{selected}")
        runtime = inspect_ocr_runtime(cache_dir) if selected == "docling" else None
        return _write_bundle(
            output_dir,
            markdown=markdown,
            source=source_record,
            requested_engine=engine,
            selected_engine=selected,
            status=status,
            risk_reasons=sorted(set(reasons)),
            attempts=attempts,
            pdf_audit=pdf_audit,
            runtime=runtime,
            acceptances=current_acceptances,
        )
    except (OSError, ReaderError, ValueError) as raw_error:
        exc = raw_error if isinstance(raw_error, ReaderError) else ReaderError(str(raw_error))
        if not attempts or attempts[-1].get("status") != "failed":
            attempts.append({"engine": selected or engine, "status": "failed", "error": str(exc)})
        markdown = f"# Extraction unavailable\n\n{exc.code}: {exc}\n"
        return _write_bundle(
            output_dir,
            markdown=markdown,
            source=source_record,
            requested_engine=engine,
            selected_engine=selected,
            status="failed_for_agent",
            risk_reasons=[exc.code],
            attempts=attempts,
            pdf_audit=pdf_audit,
            runtime=inspect_ocr_runtime(cache_dir) if selected == "docling" else None,
            acceptances=current_acceptances,
            error=exc,
        )


def render_pdf(source: Path, output_dir: Path, *, dpi: int = 144) -> list[Path]:
    if dpi < 36 or dpi > 600:
        raise InputError("DPI must be between 36 and 600")
    try:
        import pypdfium2
    except ImportError as exc:
        raise ReaderError("packaged pypdfium2 renderer is missing") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pypdfium2.PdfDocument(str(source))
    rendered: list[Path] = []
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                image = page.render(scale=dpi / 72).to_pil()
                target = output_dir / f"{source.stem}-{index + 1:04d}.png"
                image.save(target)
                rendered.append(target)
            finally:
                page.close()
    finally:
        document.close()
    if not rendered:
        raise ExtractionError("PDF renderer produced no pages")
    return rendered


def normalize_chinese(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(char for char in normalized if "\u3400" <= char <= "\u9fff")


def chinese_character_recall(expected: str, actual: str) -> float:
    expected_chars = Counter(normalize_chinese(expected))
    if not expected_chars:
        return 1.0
    actual_chars = Counter(normalize_chinese(actual))
    matched = sum((expected_chars & actual_chars).values())
    return matched / sum(expected_chars.values())


def table_cell_recall(expected_cells: list[str], actual: str) -> float:
    if not expected_cells:
        return 1.0

    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)

    haystack = normalize(actual)
    expected = [normalize(cell) for cell in expected_cells]
    return sum(bool(cell) and cell in haystack for cell in expected) / len(expected)


def evaluate_chinese_ocr_quality(
    *,
    expected_text: str,
    actual_text: str,
    expected_table_cells: list[str],
    expected_pages: int,
    successful_pages: int,
) -> dict[str, Any]:
    character_recall = chinese_character_recall(expected_text, actual_text)
    cell_recall = table_cell_recall(expected_table_cells, actual_text)
    page_coverage = successful_pages / expected_pages if expected_pages else 1.0
    return {
        "page_coverage": page_coverage,
        "chinese_character_recall": character_recall,
        "table_cell_recall": cell_recall,
        "passed": bool(page_coverage == 1.0 and character_recall >= 0.90 and cell_recall >= 0.85),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    preflight = subparsers.add_parser("preflight", help="report local reader state")
    preflight.add_argument("--cache-dir")
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument("--require-ocr", action="store_true")

    prepare = subparsers.add_parser("prepare-ocr", help="prepare locked local OCR")
    prepare.add_argument("--cache-dir")

    ingest = subparsers.add_parser("ingest", help="extract a local document")
    ingest.add_argument("input")
    ingest.add_argument("--output-dir", required=True)
    ingest.add_argument(
        "--engine",
        choices=("auto", "markitdown", "pdfplumber", "docling", "marker", "pymupdf4llm"),
        default="auto",
    )
    ingest.add_argument("--offline", action="store_true")
    ingest.add_argument("--cache-dir")
    ingest.add_argument(
        "--accept-external-license",
        action="append",
        choices=tuple(EXTERNAL_LICENSES),
        default=[],
    )

    render = subparsers.add_parser("render-pdf", help="render a local PDF to PNG")
    render.add_argument("input")
    render.add_argument("--output-dir", required=True)
    render.add_argument("--dpi", type=int, default=144)
    return parser


def _print_preflight(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    for name, state in report["components"].items():
        print(f"{name}={'available' if state['available'] else 'missing'}")
    print(f"ocr={'ready' if report['ocr']['complete'] else 'not-ready'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "preflight":
            report = collect_preflight(resolve_cache_dir(args.cache_dir))
            _print_preflight(report, as_json=args.json)
            return 0 if not args.require_ocr or report["ocr"]["complete"] else 2
        if args.action == "prepare-ocr":
            report = prepare_ocr(resolve_cache_dir(args.cache_dir))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.action == "ingest":
            bundle = ingest_document(
                args.input,
                Path(args.output_dir),
                engine=args.engine,
                offline=args.offline,
                cache_dir=resolve_cache_dir(args.cache_dir),
                accepted_licenses=set(args.accept_external_license),
            )
            print(bundle.manifest)
            return 0 if bundle.status in {"good", "salvaged"} else 2
        if args.action == "render-pdf":
            source = local_input(args.input, suffixes={".pdf"})
            for output in render_pdf(source, Path(args.output_dir), dpi=args.dpi):
                print(output)
            return 0
    except (OSError, ReaderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
