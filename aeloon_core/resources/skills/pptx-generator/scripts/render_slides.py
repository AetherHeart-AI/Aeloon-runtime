#!/usr/bin/env python3
"""Render a PPTX to PNG slides using LibreOffice and Poppler."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def dependencies() -> tuple[str | None, str | None]:
    return shutil.which("soffice") or shutil.which("libreoffice"), shutil.which("pdftoppm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?")
    parser.add_argument("--output-dir", default="tmp/pptx-render")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    office, poppler = dependencies()
    if args.check:
        print(f"libreoffice={office or 'missing'}")
        print(f"pdftoppm={poppler or 'missing'}")
        return 0 if office and poppler else 2
    if not office or not poppler:
        print("error: LibreOffice and Poppler are required", file=sys.stderr)
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
        prefix = output_dir / source.stem
        completed = subprocess.run(
            [poppler, "-png", "-r", str(args.dpi), str(pdf), str(prefix)], check=False
        )
        if completed.returncode:
            return completed.returncode
    pages = sorted(output_dir.glob(f"{source.stem}-*.png"))
    if not pages:
        print("error: no slide images were produced", file=sys.stderr)
        return 2
    for page in pages:
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
