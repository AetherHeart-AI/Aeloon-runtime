from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SKILL_DIR = Path(__file__).parents[1] / "aeloon_core" / "resources" / "skills" / "document-reader"


def load_cli():
    path = SKILL_DIR / "scripts" / "cli.py"
    spec = importlib.util.spec_from_file_location("test_document_reader_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reader():
    return load_cli()


def test_skill_metadata_and_locked_runtime_are_complete() -> None:
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = skill_text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert set(metadata) == {"name", "description"}
    assert metadata["name"] == "document-reader"
    assert "present_files" in body
    interface = yaml.safe_load((SKILL_DIR / "agents" / "openai.yaml").read_text(encoding="utf-8"))[
        "interface"
    ]
    assert "$document-reader" in interface["default_prompt"]
    assert "MIT License" in (SKILL_DIR / "LICENSE.txt").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Rael" in (SKILL_DIR / "LICENSE-docling-skill.txt").read_text(
        encoding="utf-8"
    )
    runtime_lock = (SKILL_DIR / "runtime" / "uv.lock").read_text(encoding="utf-8")
    for package in ("docling", "rapidocr", "onnxruntime", "openpyxl", "xlrd"):
        assert f'name = "{package}"' in runtime_lock
    worker = (SKILL_DIR / "runtime" / "ocr_runtime.py").read_text(encoding="utf-8")
    assert 'backend="onnxruntime"' in worker
    assert 'lang=["chinese"]' in worker
    assert "rapidocr_params=" not in worker


def test_local_input_rejects_remote_and_legacy_formats(reader, tmp_path: Path) -> None:
    with pytest.raises(reader.InputError, match="URI and remote"):
        reader.local_input("https://example.com/report.pdf")
    with pytest.raises(reader.InputError, match="URI and remote"):
        reader.local_input("s3://private-bucket/report.pdf")
    legacy = tmp_path / "report.wps"
    legacy.write_bytes(b"wps")
    with pytest.raises(reader.InputError, match="Save As"):
        reader.local_input(str(legacy))
    old_word = tmp_path / "report.doc"
    old_word.write_bytes(b"doc")
    with pytest.raises(reader.InputError, match="DOCX"):
        reader.local_input(str(old_word))


def test_ingest_markitdown_writes_three_quality_sidecars(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "brief.docx"
    source.write_bytes(b"trusted-local-docx")
    monkeypatch.setattr(reader, "convert_markitdown", lambda _source: "# Brief\n\nContent")
    output = tmp_path / "result"

    bundle = reader.ingest_document(str(source), output, cache_dir=tmp_path / "cache")

    assert bundle.status == "good"
    assert {path.name for path in (bundle.markdown, bundle.manifest, bundle.evidence)} == {
        "source.md",
        "source.manifest.json",
        "source.evidence.json",
    }
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    evidence = json.loads(bundle.evidence.read_text(encoding="utf-8"))
    assert manifest["schema"] == "document-reader-manifest/v1"
    assert manifest["status"] == "good"
    assert manifest["engine"] == {"requested": "auto", "selected": "markitdown"}
    assert evidence["attempts"][0]["engine"] == "markitdown"
    assert evidence["quality"]["has_content"] is True
    assert evidence["quality"]["characters"] == len(bundle.markdown.read_text(encoding="utf-8"))


def test_offline_scan_cache_miss_is_explicit_failed_for_agent(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "scan.png"
    source.write_bytes(b"image")
    output = tmp_path / "result"
    monkeypatch.setattr(
        reader,
        "prepare_ocr",
        lambda _cache: pytest.fail("offline ingest must not prepare or download OCR"),
    )

    bundle = reader.ingest_document(
        str(source), output, offline=True, cache_dir=tmp_path / "empty-cache"
    )

    assert bundle.status == "failed_for_agent"
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    evidence = json.loads(bundle.evidence.read_text(encoding="utf-8"))
    assert manifest["error"]["code"] == "offline_cache_miss"
    assert manifest["risk"]["level"] == "high"
    assert evidence["quality"]["usable_for_agent"] is False
    assert "Extraction unavailable" in bundle.markdown.read_text(encoding="utf-8")


def test_auto_routes_low_quality_pdf_to_docling(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"pdf")
    audit = {
        "pages": 2,
        "page_characters": [0, 0],
        "text_layer_coverage": 0,
        "average_characters_per_page": 0,
        "suspicious_character_ratio": 0,
        "tables": 0,
        "table_cells": 0,
        "multi_column_pages": 0,
        "needs_docling": True,
        "reasons": ["low_text_layer_coverage"],
    }
    monkeypatch.setattr(reader, "audit_pdf", lambda _source: audit)
    monkeypatch.setattr(
        reader,
        "run_docling",
        lambda _source, _cache, *, offline: ("# OCR text", {"pages": 2}),
    )

    bundle = reader.ingest_document(str(source), tmp_path / "result", cache_dir=tmp_path / "cache")

    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert bundle.status == "salvaged"
    assert manifest["engine"]["selected"] == "docling"
    assert "low_text_layer_coverage" in manifest["risk"]["reasons"]


def test_damaged_pdf_still_writes_failure_sidecars(reader, tmp_path: Path) -> None:
    source = tmp_path / "damaged.pdf"
    source.write_bytes(b"not a PDF")

    bundle = reader.ingest_document(str(source), tmp_path / "result", cache_dir=tmp_path / "cache")

    assert bundle.status == "failed_for_agent"
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert manifest["error"]["code"] == "extraction_failed"


def test_prepare_ocr_reports_missing_uv(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(reader.shutil, "which", lambda _name: None)
    with pytest.raises(reader.ReaderError, match="uv is required"):
        reader.prepare_ocr(tmp_path / "cache")


def test_prepare_ocr_retries_resumable_model_downloads(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    downloader = cache / "venv" / "bin" / "docling-tools"
    downloader.parent.mkdir(parents=True)
    downloader.write_text("", encoding="utf-8")
    (downloader.parent / "python").write_text("", encoding="utf-8")
    bundled_python = tmp_path / "bundled" / "python3"
    bundled_python.parent.mkdir()
    bundled_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("AELOON_RUNTIME_PYTHON", str(bundled_python))
    calls: list[tuple[list[str], dict[str, str]]] = []
    download_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal download_attempts
        calls.append((command, kwargs["env"]))
        if command[0] == str(downloader):
            download_attempts += 1
            if download_attempts == 1:
                return SimpleNamespace(returncode=1, stderr="transient TLS EOF", stdout="")
            models = cache / "models"
            models.mkdir(exist_ok=True)
            (models / "model.onnx").write_bytes(b"complete-model")
            return SimpleNamespace(returncode=0, stderr="", stdout="downloaded")
        if command[0].endswith("python"):
            versions = {
                "docling": "2.118.1",
                "rapidocr": "3.9.2",
                "onnxruntime": "1.28.0",
                "openpyxl": "3.1.5",
                "xlrd": "2.0.2",
            }
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(versions))
        return SimpleNamespace(returncode=0, stderr="", stdout="synced")

    monkeypatch.setattr(reader.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(reader.subprocess, "run", fake_run)

    state = reader.prepare_ocr(cache)

    assert state["complete"] is True
    assert download_attempts == 2
    sync_command, sync_environment = calls[0]
    assert sync_command[:4] == ["/usr/bin/uv", "sync", "--frozen", "--no-dev"]
    assert "--no-python-downloads" in sync_command
    assert sync_environment["UV_PYTHON"] == str(bundled_python)
    assert sync_environment["UV_PYTHON_DOWNLOADS"] == "never"
    download_environments = [
        environment for command, environment in calls if command[0] == str(downloader)
    ]
    assert all(environment["HF_HUB_DISABLE_XET"] == "1" for environment in download_environments)
    assert all(
        environment["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"
        for environment in download_environments
    )


def test_external_engine_acceptance_is_recorded_and_reused(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf")
    audit = {
        "pages": 1,
        "page_characters": [100],
        "text_layer_coverage": 1.0,
        "average_characters_per_page": 100,
        "suspicious_character_ratio": 0,
        "tables": 0,
        "table_cells": 0,
        "multi_column_pages": 0,
        "needs_docling": False,
        "reasons": [],
    }
    monkeypatch.setattr(reader, "audit_pdf", lambda _source: audit)
    monkeypatch.setattr(reader, "run_pymupdf4llm", lambda _source: "# Extracted")
    monkeypatch.setattr(reader, "_external_version", lambda _engine: "test-version")
    output = tmp_path / "result"

    rejected = reader.ingest_document(
        str(source), output, engine="pymupdf4llm", cache_dir=tmp_path / "cache"
    )
    assert rejected.status == "failed_for_agent"
    accepted = reader.ingest_document(
        str(source),
        output,
        engine="pymupdf4llm",
        accepted_licenses={"pymupdf4llm"},
        cache_dir=tmp_path / "cache",
    )
    assert accepted.status == "salvaged"
    evidence = json.loads(accepted.evidence.read_text(encoding="utf-8"))
    assert evidence["license_acceptances"] == [
        {
            "engine": "pymupdf4llm",
            "accepted": True,
            "accepted_at": evidence["license_acceptances"][0]["accepted_at"],
            "license_class": "agpl-3.0-or-commercial",
            "version": "test-version",
        }
    ]

    reused = reader.ingest_document(
        str(source), output, engine="pymupdf4llm", cache_dir=tmp_path / "cache"
    )
    assert reused.status == "salvaged"


def test_pdf_render_uses_pdfium_and_closes_pages(
    reader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved: list[Path] = []
    pages = []

    class Image:
        def save(self, path):
            saved.append(Path(path))
            Path(path).write_bytes(b"png")

    class Page:
        closed = False

        def render(self, *, scale):
            assert scale == 2
            return SimpleNamespace(to_pil=lambda: Image())

        def close(self):
            self.closed = True

    class Document:
        closed = False

        def __init__(self, _source):
            pages.extend([Page(), Page()])

        def __len__(self):
            return len(pages)

        def __getitem__(self, index):
            return pages[index]

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=Document))
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf")

    outputs = reader.render_pdf(source, tmp_path / "pages", dpi=144)

    assert outputs == saved
    assert [item.name for item in outputs] == ["paper-0001.png", "paper-0002.png"]
    assert all(page.closed for page in pages)


def test_preflight_reports_optional_components(reader, tmp_path: Path) -> None:
    report = reader.collect_preflight(tmp_path / "cache")
    assert report["schema"] == "document-reader-preflight/v1"
    assert set(report["components"]) == {
        "markitdown",
        "pypdf",
        "pdfplumber",
        "pypdfium2",
        "uv",
        "libreoffice",
    }
    assert report["ocr"]["complete"] is False
    assert report["ocr"]["reason"] == "runtime_manifest_missing"


def test_chinese_ocr_quality_gate(reader) -> None:
    expected = "中文标题这是正文金额一百元"
    actual = "中文标题 这是正文 金额一百元 表格 甲 乙"
    passed = reader.evaluate_chinese_ocr_quality(
        expected_text=expected,
        actual_text=actual,
        expected_table_cells=["甲", "乙"],
        expected_pages=3,
        successful_pages=3,
    )
    assert passed["passed"] is True

    failed = reader.evaluate_chinese_ocr_quality(
        expected_text=expected,
        actual_text="中文",
        expected_table_cells=["甲", "乙"],
        expected_pages=3,
        successful_pages=2,
    )
    assert failed["passed"] is False
    assert failed["page_coverage"] == pytest.approx(2 / 3)


def test_cli_ingest_failure_returns_nonzero_but_keeps_sidecars(reader, tmp_path: Path) -> None:
    output = tmp_path / "result"
    code = reader.main(["ingest", "https://example.com/scan.pdf", "--output-dir", str(output)])
    assert code == 2
    assert (output / "source.md").is_file()
    manifest = json.loads((output / "source.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_for_agent"
