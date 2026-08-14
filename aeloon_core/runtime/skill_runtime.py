"""Run trusted bundled Python Skill entry points."""

from __future__ import annotations

import runpy
import sys
from importlib.resources import files
from pathlib import Path

_SKILL_ENTRIES = {
    "aeloon-office-lite": "aeloon-office-lite/scripts/cli.py",
}

_SKILL_ACTIONS = {
    "aeloon-office-lite": ("preflight", "read", "write", "render", "validate"),
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
