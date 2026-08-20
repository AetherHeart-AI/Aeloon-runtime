"""Runtime-owned resource discovery with deterministic precedence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONTEXT_NAMES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    file_path: str
    disable_model_invocation: bool = False
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    skill: Skill
    content: str

    @property
    def name(self) -> str:
        return self.skill.name

    @property
    def file_path(self) -> str:
        return self.skill.file_path


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    content: str
    description: str | None = None

    def format(self, arguments: Sequence[str] = ()) -> str:
        rendered = self.content
        for index, argument in enumerate(arguments, 1):
            rendered = rendered.replace(f"${index}", argument)
        return rendered.replace("$@", " ".join(arguments))


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    skills: tuple[Skill, ...] = ()
    prompt_templates: tuple[PromptTemplate, ...] = ()
    context_files: tuple[tuple[str, str], ...] = ()
    system_prompt: str | None = None
    append_system_prompt: tuple[str, ...] = ()


class ResourceLoader:
    """Load system prompts, project instructions, skills, and prompt templates."""

    def __init__(
        self,
        *,
        cwd: Path | str,
        agent_dir: Path | str | None = None,
        additional_roots: tuple[Path | str, ...] = (),
        enabled_skills: tuple[str, ...] | None = None,
        no_skills: bool = False,
        no_prompt_templates: bool = False,
        no_context_files: bool = False,
    ) -> None:
        self.cwd = Path(cwd).expanduser().resolve(strict=False)
        self.agent_dir = Path(agent_dir or "~/.aeloon-runtime").expanduser().resolve(strict=False)
        self.additional_roots = tuple(
            Path(path).expanduser().resolve(strict=False) for path in additional_roots
        )
        self.enabled_skills = None if enabled_skills is None else frozenset(enabled_skills)
        self.no_skills = no_skills
        self.no_prompt_templates = no_prompt_templates
        self.no_context_files = no_context_files
        self._resources = RuntimeResources()
        self._available_skills: tuple[Skill, ...] = ()

    @property
    def resources(self) -> RuntimeResources:
        return self._resources

    @property
    def available_skills(self) -> tuple[Skill, ...]:
        """All discovered skill descriptors, including disabled skills."""

        return self._available_skills

    def reload(self) -> RuntimeResources:
        project_dir = self.cwd / ".aeloon-runtime"
        system_source = _first_file(project_dir / "SYSTEM.md", self.agent_dir / "SYSTEM.md")
        append_source = _first_file(
            project_dir / "APPEND_SYSTEM.md", self.agent_dir / "APPEND_SYSTEM.md"
        )
        self._available_skills = self._load_skills(project_dir)
        skills = (
            ()
            if self.no_skills
            else tuple(
                skill
                for skill in self._available_skills
                if self.enabled_skills is None or skill.name in self.enabled_skills
            )
        )
        prompts = () if self.no_prompt_templates else self._load_prompts(project_dir)
        context = () if self.no_context_files else self._load_context_files()
        self._resources = RuntimeResources(
            skills=skills,
            prompt_templates=prompts,
            context_files=context,
            system_prompt=_read_optional(system_source),
            append_system_prompt=(
                (_read_optional(append_source) or "",) if append_source is not None else ()
            ),
        )
        return self._resources

    def load_skill(self, name: str) -> LoadedSkill:
        """Load one enabled skill body after metadata-only discovery."""

        skill = next((item for item in self._resources.skills if item.name == name), None)
        if skill is None:
            raise KeyError(name)
        try:
            raw = Path(skill.file_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Could not read skill: {name}") from exc
        metadata, content = _frontmatter(raw)
        loaded_name = str(metadata.get("name") or Path(skill.file_path).parent.name).strip()
        if loaded_name != skill.name:
            raise ValueError(f"Skill metadata changed while loading: {name}")
        return LoadedSkill(skill=skill, content=content.strip())

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
        roots = (
            (self.agent_dir / "skills", "user"),
            (project_dir / "skills", "workspace"),
            *((root, "additional") for root in self.additional_roots),
        )
        selected: dict[str, Skill] = {}
        for root, source in roots:
            for path in _resource_files(root, "SKILL.md"):
                skill = _parse_skill(path, source=source)
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


def _parse_skill(path: Path, *, source: str) -> Skill | None:
    """Read only frontmatter during discovery; the body is loaded on invocation."""

    metadata = _frontmatter_metadata(path)
    if metadata is None:
        return None
    name = str(metadata.get("name") or path.parent.name).strip()
    description = str(metadata.get("description") or "").strip()
    if not name or not description:
        return None
    return Skill(
        name=name,
        description=description,
        file_path=str(path.resolve(strict=False)),
        disable_model_invocation=bool(metadata.get("disable-model-invocation", False)),
        source=source,
    )


def _frontmatter_metadata(path: Path) -> dict[str, Any] | None:
    """Parse bounded YAML frontmatter without reading the skill instructions."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline().removeprefix("\ufeff").rstrip("\r\n")
            if first != "---":
                return {}
            lines: list[str] = []
            size = 0
            for line in handle:
                if line.rstrip("\r\n") == "---":
                    break
                size += len(line)
                if size > 64 * 1024:
                    return None
                lines.append(line)
            else:
                return {}
    except (OSError, UnicodeError):
        return None
    try:
        raw = yaml.safe_load("".join(lines)) or {}
    except yaml.YAMLError:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _parse_prompt(path: Path) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    metadata, content = _frontmatter(raw)
    return PromptTemplate(
        name=str(metadata.get("name") or path.stem),
        description=(str(metadata["description"]) if metadata.get("description") else None),
        content=content.strip(),
    )


__all__ = [
    "LoadedSkill",
    "PromptTemplate",
    "ResourceLoader",
    "RuntimeResources",
    "Skill",
]
