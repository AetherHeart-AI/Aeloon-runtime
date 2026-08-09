"""Run trusted bundled Python Skill entry points."""

from __future__ import annotations

import runpy
import sys
from importlib.resources import files
from pathlib import Path

_SKILL_ENTRIES = {
    "document-reader": "document-reader/scripts/cli.py",
    "word-docx": "word-docx/scripts/cli.py",
    "powerpoint-pptx": "powerpoint-pptx/scripts/cli.py",
}

_SKILL_ACTIONS = {
    "document-reader": ("preflight", "prepare-ocr", "ingest", "render-pdf"),
    "word-docx": ("build", "edit", "validate", "render"),
    "powerpoint-pptx": ("build", "inspect-template", "apply-template", "validate", "render"),
}

_RETIRED_SKILLS = {
    "office": "document-reader, word-docx, or powerpoint-pptx",
    "ppt": "powerpoint-pptx",
    "document-writing": "word-docx",
    "reports": "word-docx",
    "markitdown": "document-reader",
    "pdf": "document-reader",
    "paddleocr-doc-parsing": "document-reader",
    "pptx-generator": "powerpoint-pptx",
    "document-format-skills": "word-docx",
}

def bundled_skill_root() -> Path:
    return Path(str(files("aeloon_core.resources").joinpath("skills"))).resolve()


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


def run_bundled_skill(skill_id: str, action: str, arguments: list[str]) -> int:
    replacement = _RETIRED_SKILLS.get(skill_id)
    if replacement is not None:
        raise ValueError(
            f"bundled Skill '{skill_id}' has been retired; use {replacement} instead"
        )

    relative_path = _SKILL_ENTRIES.get(skill_id)
    if relative_path is None:
        available = ", ".join(sorted(_SKILL_ENTRIES))
        raise ValueError(f"unknown bundled Skill '{skill_id}'; expected one of: {available}")

    valid_actions = _SKILL_ACTIONS[skill_id]
    if action not in valid_actions:
        available = ", ".join(valid_actions)
        raise ValueError(
            f"unknown action '{action}' for bundled Skill '{skill_id}'; "
            f"expected one of: {available}"
        )
    return _run_python_script(relative_path, [action, *arguments])


__all__ = [
    "bundled_skill_root",
    "run_bundled_skill",
]
