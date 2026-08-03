"""Aeloon-namespaced resource discovery with Pi-compatible precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aeloon_core.harness.types import PromptTemplate, Resources, Skill

_CONTEXT_NAMES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


class ResourceLoader:
    """Load system prompts, project instructions, skills, and prompt templates."""

    def __init__(
        self,
        *,
        cwd: Path | str,
        agent_dir: Path | str | None = None,
        additional_roots: tuple[Path | str, ...] = (),
        no_skills: bool = False,
        no_prompt_templates: bool = False,
        no_context_files: bool = False,
    ) -> None:
        self.cwd = Path(cwd).expanduser().resolve(strict=False)
        self.agent_dir = Path(agent_dir or "~/.aeloon-core").expanduser().resolve(strict=False)
        self.additional_roots = tuple(
            Path(path).expanduser().resolve(strict=False) for path in additional_roots
        )
        self.no_skills = no_skills
        self.no_prompt_templates = no_prompt_templates
        self.no_context_files = no_context_files
        self._resources = Resources()

    @property
    def resources(self) -> Resources:
        return self._resources

    def reload(self) -> Resources:
        project_dir = self.cwd / ".aeloon-core"
        system_source = _first_file(project_dir / "SYSTEM.md", self.agent_dir / "SYSTEM.md")
        append_source = _first_file(
            project_dir / "APPEND_SYSTEM.md", self.agent_dir / "APPEND_SYSTEM.md"
        )
        skills = () if self.no_skills else self._load_skills(project_dir)
        prompts = () if self.no_prompt_templates else self._load_prompts(project_dir)
        context = () if self.no_context_files else self._load_context_files()
        self._resources = Resources(
            skills=skills,
            prompt_templates=prompts,
            context_files=context,
            system_prompt=_read_optional(system_source),
            append_system_prompt=(
                (_read_optional(append_source) or "",) if append_source is not None else ()
            ),
        )
        return self._resources

    def _load_context_files(self) -> tuple[tuple[str, str], ...]:
        files: list[Path] = []
        global_context = _context_file(self.agent_dir)
        if global_context is not None:
            files.append(global_context)
        ancestors: list[Path] = []
        current = self.cwd
        while True:
            context = _context_file(current)
            if context is not None:
                ancestors.insert(0, context)
            if current.parent == current:
                break
            current = current.parent
        seen: set[Path] = set()
        result: list[tuple[str, str]] = []
        for path in (*files, *ancestors):
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                result.append((str(path), path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError):
                continue
        return tuple(result)

    def _load_skills(self, project_dir: Path) -> tuple[Skill, ...]:
        roots = (self.agent_dir / "skills", project_dir / "skills", *self.additional_roots)
        selected: dict[str, Skill] = {}
        for root in roots:
            for path in _resource_files(root, "SKILL.md"):
                skill = _parse_skill(path)
                if skill is not None:
                    selected[skill.name] = skill
        return tuple(selected[name] for name in sorted(selected))

    def _load_prompts(self, project_dir: Path) -> tuple[PromptTemplate, ...]:
        roots = (self.agent_dir / "prompts", project_dir / "prompts")
        selected: dict[str, PromptTemplate] = {}
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                prompt = _parse_prompt(path)
                selected[prompt.name] = prompt
        return tuple(selected[name] for name in sorted(selected))


def _first_file(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _read_optional(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _context_file(directory: Path) -> Path | None:
    return _first_file(*(directory / name for name in _CONTEXT_NAMES))


def _resource_files(root: Path, filename: str) -> tuple[Path, ...]:
    if root.is_file() and root.name == filename:
        return (root,)
    if not root.is_dir():
        return ()
    return tuple(sorted(path for path in root.rglob(filename) if path.is_file()))


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    try:
        raw = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        raw = {}
    return (dict(raw) if isinstance(raw, dict) else {}), text[end + 5 :]


def _parse_skill(path: Path) -> Skill | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    metadata, content = _frontmatter(raw)
    name = str(metadata.get("name") or path.parent.name).strip()
    description = str(metadata.get("description") or "").strip()
    if not name or not description:
        return None
    return Skill(
        name=name,
        description=description,
        content=content.strip(),
        file_path=str(path.resolve(strict=False)),
        disable_model_invocation=bool(metadata.get("disable-model-invocation", False)),
    )


def _parse_prompt(path: Path) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    metadata, content = _frontmatter(raw)
    return PromptTemplate(
        name=str(metadata.get("name") or path.stem),
        description=(str(metadata["description"]) if metadata.get("description") else None),
        content=content.strip(),
    )


__all__ = ["ResourceLoader"]
