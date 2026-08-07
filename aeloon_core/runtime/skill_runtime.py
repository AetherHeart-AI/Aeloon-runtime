"""Run trusted bundled Skill entry points with Aeloon's packaged runtimes."""

from __future__ import annotations

import os
import runpy
import sys
from importlib.resources import files
from pathlib import Path

_SCRIPT_COMMANDS = {
    ("office", "preflight"): "office/scripts/preflight.py",
    ("markitdown", "convert"): "markitdown/scripts/convert.py",
    ("pdf", "render"): "pdf/scripts/render_pdf.py",
    ("paddleocr-doc-parsing", "parse"): (
        "paddleocr-doc-parsing/scripts/local_parse.py"
    ),
    ("pptx-generator", "render"): "pptx-generator/scripts/render_slides.py",
    ("document-format-skills", "from-text"): (
        "document-format-skills/scripts/from_text.py"
    ),
    ("document-format-skills", "process"): (
        "document-format-skills/scripts/process.py"
    ),
}


def bundled_skill_root() -> Path:
    return Path(str(files("aeloon_core.resources").joinpath("skills"))).resolve()


def bundled_pptx_node_modules() -> Path:
    return bundled_skill_root() / "pptx-generator" / "runtime" / "node_modules"


def _exit_code(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    print(value, file=sys.stderr)
    return 1


def _run_python_script(relative_path: str, arguments: list[str]) -> int:
    script = bundled_skill_root() / relative_path
    if not script.is_file():
        raise RuntimeError(f"bundled Skill script is missing: {relative_path}")
    previous_argv = sys.argv
    sys.argv = [str(script), *arguments]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return _exit_code(exc.code)
    finally:
        sys.argv = previous_argv
    return 0


def _run_pptx_javascript(arguments: list[str]) -> int:
    if not arguments:
        raise ValueError("pptx-generator node requires a local JavaScript file")
    source = Path(arguments[0]).expanduser().resolve(strict=False)
    if not source.is_file() or source.suffix.lower() not in {".js", ".cjs", ".mjs"}:
        raise ValueError("pptx-generator node requires an existing local JavaScript file")
    node_modules = bundled_pptx_node_modules()
    if not (node_modules / "pptxgenjs" / "package.json").is_file():
        raise RuntimeError("bundled PptxGenJS runtime is missing")

    try:
        from nodejs_wheel import node
    except ImportError as exc:
        raise RuntimeError("bundled Node.js runtime is missing") from exc

    previous_node_path = os.environ.get("NODE_PATH")
    os.environ["NODE_PATH"] = str(node_modules)
    try:
        completed = node(
            [str(source), *arguments[1:]],
            return_completed_process=True,
        )
    finally:
        if previous_node_path is None:
            os.environ.pop("NODE_PATH", None)
        else:
            os.environ["NODE_PATH"] = previous_node_path
    return int(completed.returncode)


def run_bundled_skill(skill_id: str, action: str, arguments: list[str]) -> int:
    if (skill_id, action) == ("pptx-generator", "node"):
        return _run_pptx_javascript(arguments)
    relative_path = _SCRIPT_COMMANDS.get((skill_id, action))
    if relative_path is None:
        available = ", ".join(
            f"{item_skill}/{item_action}"
            for item_skill, item_action in sorted(
                [*_SCRIPT_COMMANDS, ("pptx-generator", "node")]
            )
        )
        raise ValueError(f"unknown bundled Skill action; expected one of: {available}")
    return _run_python_script(relative_path, arguments)


__all__ = [
    "bundled_pptx_node_modules",
    "bundled_skill_root",
    "run_bundled_skill",
]
