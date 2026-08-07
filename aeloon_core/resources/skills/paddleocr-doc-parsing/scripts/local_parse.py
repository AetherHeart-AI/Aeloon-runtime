#!/usr/bin/env python3
"""Parse one local PDF or image with the local PaddleOCR PP-StructureV3 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def local_input(value: str) -> Path:
    if value.lower().startswith(("http://", "https://", "data:", "file:")):
        raise ValueError("only local files are accepted")
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    return path


def configure_environment(cache: Path, *, offline: bool) -> None:
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


@contextmanager
def deny_network(enabled: bool) -> Iterator[None]:
    """Reject network connections while an offline OCR run is active."""
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


def import_pipeline():
    try:
        from paddleocr import PPStructureV3
    except ImportError as exc:
        raise RuntimeError(
            "local PaddleOCR is missing; install paddlepaddle and paddleocr[doc-parser]"
        ) from exc
    return PPStructureV3


def safe_asset_path(output_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise RuntimeError(f"PaddleOCR returned an absolute asset path: {relative}")
    target = (output_dir / relative).resolve(strict=False)
    try:
        target.relative_to(output_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"PaddleOCR returned an asset path outside the output directory: {relative}"
        ) from exc
    return target


def parse_document(args: argparse.Namespace) -> tuple[Path, int]:
    source = local_input(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    cache = Path(args.model_cache).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    configure_environment(cache, offline=args.offline)

    with deny_network(args.offline):
        pipeline_type = import_pipeline()
        pipeline = pipeline_type(
            device=args.device,
            use_doc_orientation_classify=args.use_doc_orientation,
            use_doc_unwarping=args.use_doc_unwarping,
            use_textline_orientation=args.use_textline_orientation,
        )
        results = list(pipeline.predict(input=str(source)))
    if not results:
        raise RuntimeError("PaddleOCR produced no pages")

    markdown_pages = []
    markdown_images = []
    for index, result in enumerate(results, 1):
        page_dir = output_dir / f"page-{index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        result.save_to_json(save_path=str(page_dir))
        result.save_to_markdown(save_path=str(page_dir))
        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, dict):
            markdown_pages.append(markdown)
            images = markdown.get("markdown_images", {})
            if isinstance(images, dict):
                markdown_images.append(images)

    if not markdown_pages:
        raise RuntimeError("PaddleOCR produced no Markdown content")
    combined = pipeline.concatenate_markdown_pages(markdown_pages)
    output_file = output_dir / f"{source.stem}.md"
    output_file.write_text(combined, encoding="utf-8")
    for images in markdown_images:
        for relative_path, image in images.items():
            asset = safe_asset_path(output_dir, str(relative_path))
            asset.parent.mkdir(parents=True, exist_ok=True)
            image.save(asset)
    manifest = {
        "source": str(source),
        "output": str(output_file),
        "pages": len(results),
        "device": args.device,
        "offline": args.offline,
        "model_cache": str(cache),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_file, len(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="local PDF or image")
    parser.add_argument("--output-dir", default="output/paddleocr")
    parser.add_argument(
        "--model-cache", default="~/.aeloon-core/models/paddleocr"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--no-doc-orientation", dest="use_doc_orientation", action="store_false"
    )
    parser.add_argument(
        "--no-doc-unwarping", dest="use_doc_unwarping", action="store_false"
    )
    parser.add_argument(
        "--no-textline-orientation",
        dest="use_textline_orientation",
        action="store_false",
    )
    parser.set_defaults(
        use_doc_orientation=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check:
            import_pipeline()
            print("local PaddleOCR runtime is available")
            return 0
        if not args.input:
            raise ValueError("input is required unless --check is used")
        output, pages = parse_document(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{output} ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
