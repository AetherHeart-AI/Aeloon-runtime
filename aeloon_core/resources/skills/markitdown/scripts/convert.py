#!/usr/bin/env python3
"""Convert one trusted local file to Markdown without remote URI dispatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _local_file(value: str) -> Path:
    if value.lower().startswith(("http://", "https://", "data:", "file:")):
        raise ValueError("remote and URI inputs are not allowed")
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    if path.suffix.lower() in {".wps", ".dps", ".et"}:
        raise ValueError("native WPS formats are unsupported; save as DOCX, PPTX, or XLSX")
    return path


def convert_file(source: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "MarkItDown is missing; install markitdown[pdf,docx,pptx,xlsx]"
        ) from exc

    converter = MarkItDown()
    if hasattr(converter, "convert_local"):
        result = converter.convert_local(source)
    else:  # MarkItDown 0.0.x compatibility
        result = converter.convert(str(source))
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        markdown = getattr(result, "text_content", None)
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError("conversion produced no text; the document may require local OCR")
    return markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="trusted local source file")
    parser.add_argument("output", help="Markdown output path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = _local_file(args.input)
        output = Path(args.output).expanduser().resolve(strict=False)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(convert_file(source), encoding="utf-8")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
