"""Skill loading tool."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from aeloon_core.skills import SkillInfo, SkillRegistry, xml_leaf
from aeloon_core.tools.base import Tool


class SkillTool(Tool):
    """Load skill instructions on demand."""

    name = "skill"
    concurrency_mode = "read_only"
    description = "\n".join(
        [
            "Load a specialized skill when the task at hand matches one of the available "
            "skills in the system context.",
            "",
            "Use this tool to inject the skill's instructions and resources into the "
            "current conversation. The output may contain detailed workflow guidance as "
            "well as references to scripts, files, etc. in the same directory as the skill.",
            "",
            "The skill name must match one of the available skills in the system context.",
        ]
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the skill from the available skills list.",
            }
        },
        "required": ["name"],
    }
    _FILE_LIMIT = 10

    def __init__(self, *, registry: SkillRegistry) -> None:
        self.registry = registry

    async def execute(self, name: str, **kwargs: Any) -> str:
        del kwargs
        skill = self.registry.get(name)
        if skill is None:
            available = ", ".join(self.registry.available_names) or "none"
            return f"Error: Skill '{name}' not found. Available skills: {available}"
        files, total = _list_skill_files(skill.directory, self._FILE_LIMIT)
        return _format_skill_output(skill, files, total)


def _format_skill_output(skill: SkillInfo, files: list[Path], total: int) -> str:
    directory = skill.directory
    if total > len(files):
        listing_note = f"Showing {len(files)} of {total} files (sorted by path)."
    else:
        listing_note = "Files in this skill:"
    return "\n".join(
        [
            f'<skill_content name="{html.escape(skill.name, quote=True)}">',
            f"# Skill: {skill.name}",
            "",
            skill.content.strip(),
            "",
            f"Base directory for this skill: {directory}",
            "Relative paths in this skill (e.g., scripts/, reference/) are relative to "
            "this base directory.",
            listing_note,
            "",
            "<skill_files>",
            *[xml_leaf("file", str(file)) for file in files],
            "</skill_files>",
            "</skill_content>",
        ]
    )


def _list_skill_files(directory: Path, limit: int) -> tuple[list[Path], int]:
    """Return up to ``limit`` files (sorted by path) plus the total file count."""

    try:
        all_files = sorted(
            path.resolve(strict=False)
            for path in directory.rglob("*")
            if path.is_file() and path.name != "SKILL.md"
        )
    except OSError:
        return [], 0
    return all_files[:limit], len(all_files)
