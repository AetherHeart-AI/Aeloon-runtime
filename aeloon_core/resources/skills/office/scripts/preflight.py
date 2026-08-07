#!/usr/bin/env python3
"""Report packaged and external runtimes used by the bundled office skills."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    available: bool
    detail: str
    hint: str = ""


def module_check(module: str, *, label: str, hint: str) -> Check:
    available = importlib.util.find_spec(module) is not None
    return Check(available, f"{label} {'available' if available else 'missing'}", hint)


def executable_check(*names: str, label: str, hint: str) -> Check:
    executable = next((path for name in names if (path := shutil.which(name))), None)
    return Check(executable is not None, executable or f"{label} missing", hint)


def pptxgenjs_check(node: Check) -> Check:
    hint = "reinstall an official Aeloon release containing the bundled PPTX runtime"
    if not node.available:
        return Check(False, "PptxGenJS unavailable because bundled Node.js is missing", hint)
    package = (
        Path(__file__).resolve().parents[2]
        / "pptx-generator"
        / "runtime"
        / "node_modules"
        / "pptxgenjs"
        / "package.json"
    )
    return Check(
        package.is_file(),
        str(package) if package.is_file() else "PptxGenJS missing",
        hint,
    )


def model_cache_check() -> Check:
    cache = Path(
        os.environ.get(
            "AELOON_PADDLEOCR_MODEL_CACHE", "~/.aeloon-core/models/paddleocr"
        )
    ).expanduser()
    ready = cache.is_dir() and any(path.is_file() for path in cache.rglob("*"))
    hint = (
        "run one trusted online PP-StructureV3 task to populate the cache, or copy a "
        "prewarmed cache before using --offline"
    )
    return Check(ready, f"{cache} ({'ready' if ready else 'not ready'})", hint)


def collect_checks() -> dict[str, Check]:
    node = module_check(
        "nodejs_wheel",
        label="bundled Node.js",
        hint="reinstall Aeloon Core with its built-in Skill dependencies",
    )
    pdfplumber = importlib.util.find_spec("pdfplumber") is not None
    pypdf = importlib.util.find_spec("pypdf") is not None
    return {
        "python": Check(True, sys.version.split()[0]),
        "markitdown": module_check(
            "markitdown",
            label="MarkItDown",
            hint="reinstall Aeloon Core with its built-in Skill dependencies",
        ),
        "pdf-text": Check(
            pdfplumber or pypdf,
            "pdfplumber/pypdf available" if pdfplumber or pypdf else "PDF text library missing",
            "reinstall Aeloon Core with its built-in Skill dependencies",
        ),
        "pdf-render": module_check(
            "pypdfium2",
            label="pypdfium2",
            hint="reinstall Aeloon Core with its built-in Skill dependencies",
        ),
        "paddlepaddle": module_check(
            "paddle",
            label="PaddlePaddle",
            hint="reinstall Aeloon Core with its built-in Skill dependencies",
        ),
        "paddleocr": module_check(
            "paddleocr",
            label="PaddleOCR",
            hint="reinstall Aeloon Core with its built-in Skill dependencies",
        ),
        "ocr-model-cache": model_cache_check(),
        "node": node,
        "pptxgenjs": pptxgenjs_check(node),
        "python-docx": module_check(
            "docx",
            label="python-docx",
            hint="reinstall Aeloon Core with its built-in Skill dependencies",
        ),
        "libreoffice": executable_check(
            "soffice",
            "libreoffice",
            label="LibreOffice",
            hint="install LibreOffice for DOCX/PPTX rendering QA",
        ),
    }


COMPONENTS = {
    "markitdown": ("python", "markitdown"),
    "pdf": ("python", "pdf-text", "pdf-render"),
    "ocr": ("python", "paddlepaddle", "paddleocr", "ocr-model-cache"),
    "pptx": ("node", "pptxgenjs"),
    "pptx-render": ("node", "pptxgenjs", "libreoffice", "pdf-render"),
    "docx": ("python", "python-docx"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="append",
        choices=sorted(COMPONENTS),
        default=[],
        help="fail when this skill runtime is incomplete; may be repeated",
    )
    args = parser.parse_args(argv)
    checks = collect_checks()
    required_keys = {
        key for component in args.require for key in COMPONENTS[component]
    }

    for name, check in checks.items():
        marker = "ok" if check.available else "missing"
        print(f"[{marker}] {name}: {check.detail}")
        if not check.available and check.hint:
            print(f"  hint: {check.hint}")

    missing = sorted(key for key in required_keys if not checks[key].available)
    if missing:
        print(f"required runtime incomplete: {', '.join(missing)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
