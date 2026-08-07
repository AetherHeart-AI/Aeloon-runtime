#!/usr/bin/env python3
"""Render a local PDF to PNG pages with packaged pypdfium2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def import_pdfium():
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError("bundled pypdfium2 runtime is missing; reinstall Aeloon Core") from exc
    return pypdfium2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?")
    parser.add_argument("--output-dir", default="tmp/pdf-pages")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        try:
            import_pdfium()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("pypdfium2=available")
        return 0
    if not args.input:
        print("error: input is required unless --check is used", file=sys.stderr)
        return 2
    source = Path(args.input).expanduser().resolve(strict=False)
    if not source.is_file() or source.suffix.lower() != ".pdf":
        print("error: input must be an existing local PDF", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdfium = import_pdfium()
        document = pdfium.PdfDocument(str(source))
        pages = []
        try:
            for index in range(len(document)):
                page = document[index]
                try:
                    image = page.render(scale=args.dpi / 72).to_pil()
                    output = output_dir / f"{source.stem}-{index + 1}.png"
                    image.save(output)
                    pages.append(output)
                finally:
                    page.close()
        finally:
            document.close()
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not pages:
        print("error: PDF renderer produced no pages", file=sys.stderr)
        return 2
    for page in pages:
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
