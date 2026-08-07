#!/usr/bin/env python3
"""Render a PPTX to PNG slides using LibreOffice and packaged pypdfium2."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def libreoffice() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def import_pdfium():
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError(
            "bundled pypdfium2 runtime is missing; reinstall Aeloon Core"
        ) from exc
    return pypdfium2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?")
    parser.add_argument("--output-dir", default="tmp/pptx-render")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    office = libreoffice()
    if args.check:
        print(f"libreoffice={office or 'missing'}")
        try:
            import_pdfium()
        except RuntimeError as exc:
            print(f"pypdfium2=missing ({exc})")
            return 2
        print("pypdfium2=available")
        return 0 if office else 2
    if not office:
        print("error: LibreOffice is required for PPTX rendering", file=sys.stderr)
        return 2
    if not args.input:
        print("error: input is required unless --check is used", file=sys.stderr)
        return 2
    source = Path(args.input).expanduser().resolve(strict=False)
    if not source.is_file() or source.suffix.lower() != ".pptx":
        print("error: input must be an existing local PPTX", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aeloon-pptx-render-") as temp:
        profile = Path(temp) / "libreoffice-profile"
        profile.mkdir()
        completed = subprocess.run(
            [
                office,
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp,
                str(source),
            ],
            check=False,
        )
        pdf = Path(temp) / f"{source.stem}.pdf"
        if completed.returncode or not pdf.is_file():
            print("error: LibreOffice did not produce a PDF", file=sys.stderr)
            return completed.returncode or 2
        try:
            document = import_pdfium().PdfDocument(str(pdf))
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
        print("error: no slide images were produced", file=sys.stderr)
        return 2
    for page in pages:
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
