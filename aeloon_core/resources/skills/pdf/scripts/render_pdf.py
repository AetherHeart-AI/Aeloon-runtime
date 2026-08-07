#!/usr/bin/env python3
"""Render a local PDF to PNG pages with Poppler."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?")
    parser.add_argument("--output-dir", default="tmp/pdf-pages")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    executable = shutil.which("pdftoppm")
    if args.check:
        print(f"pdftoppm={executable or 'missing'}")
        return 0 if executable else 2
    if not args.input:
        print("error: input is required unless --check is used", file=sys.stderr)
        return 2
    source = Path(args.input).expanduser().resolve(strict=False)
    if not source.is_file() or source.suffix.lower() != ".pdf":
        print("error: input must be an existing local PDF", file=sys.stderr)
        return 2
    if executable is None:
        print("error: pdftoppm is missing; install Poppler", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / source.stem
    completed = subprocess.run(
        [executable, "-png", "-r", str(args.dpi), str(source), str(prefix)],
        check=False,
    )
    if completed.returncode:
        return completed.returncode
    pages = sorted(output_dir.glob(f"{source.stem}-*.png"))
    if not pages:
        print("error: Poppler produced no pages", file=sys.stderr)
        return 2
    for page in pages:
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
