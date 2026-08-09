#!/usr/bin/env python3
"""Run Docling with local RapidOCR ONNX artifacts."""

from __future__ import annotations

import argparse
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def deny_network(enabled: bool):
    if not enabled:
        yield
        return
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for offline OCR")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


def configure_cache(cache_dir: Path, *, offline: bool) -> Path:
    models = (cache_dir / "models").resolve()
    os.environ["DOCLING_ARTIFACTS_PATH"] = str(models)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return models


def verify_offline_network_block() -> str:
    """Prove the worker's offline socket policy is active before conversion."""
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    except RuntimeError as exc:
        if "network access is disabled" in str(exc):
            return "blocked"
        raise
    except OSError as exc:
        raise RuntimeError("offline socket policy was not active") from exc
    raise RuntimeError("offline socket policy allowed a network connection")


def convert(source: Path, output: Path, cache_dir: Path, *, offline: bool) -> dict:
    models = configure_cache(cache_dir, offline=offline)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    ocr = RapidOcrOptions(
        backend="onnxruntime",
        lang=["chinese"],
        mode="full_page",
    )
    pipeline = PdfPipelineOptions(
        artifacts_path=models,
        do_ocr=True,
        do_table_structure=True,
        enable_remote_services=False,
        ocr_options=ocr,
    )
    formats = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline),
    }
    converter = DocumentConverter(format_options=formats)
    with deny_network(offline):
        result = converter.convert(source)
    markdown = result.document.export_to_markdown()
    if not markdown or not markdown.strip():
        raise RuntimeError("Docling produced no Markdown content")
    output.write_text(markdown, encoding="utf-8")
    pages = getattr(result.document, "pages", {})
    return {"pages": len(pages), "characters": len(markdown)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("report")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    report = Path(args.report).resolve(strict=False)
    with deny_network(args.offline):
        network_probe = verify_offline_network_block() if args.offline else "not_requested"
        metrics = convert(
            source,
            output,
            Path(args.cache_dir).expanduser().resolve(strict=False),
            offline=args.offline,
        )
    metrics["offline_network_probe"] = network_probe
    report.write_text(json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
