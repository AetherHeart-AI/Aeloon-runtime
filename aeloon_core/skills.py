"""Skill discovery and prompt guidance."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from aeloon_core.config import Config

CLAUDE_SKILL_DIR = ".claude"
AGENTS_SKILL_DIR = ".agents"
EXTERNAL_SKILL_SUBDIR = "skills"
NATIVE_SKILL_DIRS = (".opencode", ".aeloon-core")
NATIVE_SKILL_SUBDIRS = ("skill", "skills")

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(?P<data>.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)


@dataclass(frozen=True)
class SkillInfo:
    """Loaded metadata and content for one skill."""

    name: str
    description: str | None
    location: Path
    content: str

    @property
    def directory(self) -> Path:
        """Return the directory containing this skill."""

        return self.location.parent


class SkillRegistry:
    """Discover skills and provide prompt guidance plus on-demand lookup."""

    def __init__(self, skills: dict[str, SkillInfo], *, enabled: bool = True) -> None:
        self._skills = skills
        self.enabled = enabled
        self._sorted = sorted(skills.values(), key=lambda item: item.name)

    @classmethod
    def discover(cls, config: Config) -> SkillRegistry:
        """Discover configured skills using OpenCode-compatible locations."""

        if not config.skills.enabled:
            return cls({}, enabled=False)

        # ``_discover_skill_files`` yields paths in ascending precedence order, so a
        # later file with the same skill name deliberately overrides an earlier one
        # (custom paths > native roots > project external > global external).
        matches = _discover_skill_files(config)
        skills: dict[str, SkillInfo] = {}
        for match in matches:
            info = load_skill_file(match)
            if info is None:
                continue
            if info.name in skills:
                logger.warning(
                    "duplicate skill name",
                    name=info.name,
                    existing=str(skills[info.name].location),
                    duplicate=str(info.location),
                )
            skills[info.name] = info
        return cls(skills)

    def all(self) -> list[SkillInfo]:
        """Return all loaded skills sorted by name."""

        return list(self._sorted)

    def described(self) -> list[SkillInfo]:
        """Return skills with descriptions, sorted by name."""

        return [skill for skill in self._sorted if skill.description]

    def get(self, name: str) -> SkillInfo | None:
        """Return a skill by name."""

        return self._skills.get(name)

    def format_guidance(self) -> str | None:
        """Render available-skill guidance for the model."""

        if not self.enabled:
            return None

        skills = self.described()
        lines = [
            "Skills provide specialized instructions and workflows for specific tasks.",
            "Use the skill tool to load a skill when the task matches its description.",
        ]
        if not skills:
            lines.append("No skills are currently available.")
            return "\n".join(lines)

        lines.append("<available_skills>")
        for skill in skills:
            lines.extend(
                [
                    "  <skill>",
                    f"    {xml_leaf('name', skill.name)}",
                    f"    {xml_leaf('description', skill.description or '')}",
                    "  </skill>",
                ]
            )
        lines.append("</available_skills>")
        return "\n".join(lines)

    @property
    def available_names(self) -> list[str]:
        """Return available skill names sorted by name."""

        return [skill.name for skill in self._sorted]


def _discover_skill_files(config: Config) -> list[Path]:
    # Roots are scanned from lowest to highest precedence; callers rely on the
    # ordering so that later entries win when two skills share a name.
    workspace = config.workspace.resolve(strict=False)
    home = Path.home()
    state = _DiscoveryState()

    if config.skills.external:
        global_external = [AGENTS_SKILL_DIR]
        if config.skills.claude_code:
            global_external.insert(0, CLAUDE_SKILL_DIR)
        for dirname in global_external:
            _scan_external_root(state, home / dirname)

        for ancestor in _ancestors_from_worktree(workspace):
            if config.skills.claude_code:
                _scan_external_root(state, ancestor / CLAUDE_SKILL_DIR)
            _scan_external_root(state, ancestor / AGENTS_SKILL_DIR)

    for root in _native_config_roots(config, workspace, home):
        _scan_native_root(state, root)

    for item in config.skills.paths:
        root = _resolve_config_path(item, workspace, home)
        if not root.is_dir():
            logger.warning("skill path not found", path=str(root))
            continue
        _scan_any_skill_root(state, root)

    return state.matches


class _DiscoveryState:
    def __init__(self) -> None:
        self._seen: set[Path] = set()
        self.matches: list[Path] = []

    def add(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        if resolved in self._seen:
            return
        self._seen.add(resolved)
        self.matches.append(resolved)


def _scan_external_root(state: _DiscoveryState, root: Path) -> None:
    _scan_pattern(state, root / EXTERNAL_SKILL_SUBDIR, "**/SKILL.md")


def _scan_native_root(state: _DiscoveryState, root: Path) -> None:
    for subdir in NATIVE_SKILL_SUBDIRS:
        _scan_pattern(state, root / subdir, "**/SKILL.md")


def _scan_any_skill_root(state: _DiscoveryState, root: Path) -> None:
    _scan_pattern(state, root, "**/SKILL.md")


def _scan_pattern(state: _DiscoveryState, root: Path, pattern: str) -> None:
    if not root.is_dir():
        return
    try:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
    except OSError as exc:
        logger.warning("failed to scan skills", dir=str(root), error=str(exc))
        return
    for match in matches:
        state.add(match)


def _native_config_roots(config: Config, workspace: Path, home: Path) -> list[Path]:
    roots: list[Path] = [
        home / ".config" / "opencode",
        home / ".aeloon-core",
        config.data_dir,
    ]
    for ancestor in _ancestors_from_worktree(workspace):
        for dirname in NATIVE_SKILL_DIRS:
            roots.append(ancestor / dirname)
    return roots


def _ancestors_from_worktree(start: Path) -> list[Path]:
    current = start if start.is_dir() else start.parent
    current = current.resolve(strict=False)
    stop = _find_worktree_root(current)
    upward: list[Path] = []
    node = current
    while True:
        upward.append(node)
        if node == stop or node.parent == node:
            break
        node = node.parent
    return list(reversed(upward))


def _find_worktree_root(start: Path) -> Path:
    node = start
    while True:
        if (node / ".git").exists():
            return node
        if node.parent == node:
            return start
        node = node.parent


def _resolve_config_path(item: str, workspace: Path, home: Path) -> Path:
    expanded = item
    if item == "~":
        expanded = str(home)
    elif item.startswith("~/"):
        expanded = str(home / item[2:])
    path = Path(expanded)
    if not path.is_absolute():
        path = workspace / path
    return path.resolve(strict=False)


def load_skill_file(path: Path) -> SkillInfo | None:
    """Load one SKILL.md file."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("failed to read skill", path=str(path), error=str(exc))
        return None

    parsed = _parse_skill_markdown(raw)
    if not parsed:
        return None
    data, content = parsed
    name = data.get("name")
    if not name:
        return None
    description = data.get("description")
    return SkillInfo(
        name=name,
        description=description,
        location=path.resolve(strict=False),
        content=content,
    )


def _parse_skill_markdown(raw: str) -> tuple[dict[str, str], str] | None:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return None
    return _parse_frontmatter(match.group("data")), raw[match.end() :]


def _parse_frontmatter(text: str) -> dict[str, str]:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("failed to parse skill frontmatter", error=str(exc))
        return {}
    if not isinstance(loaded, dict):
        return {}
    data: dict[str, str] = {}
    for key in ("name", "description"):
        value = loaded.get(key)
        if isinstance(value, str):
            data[key] = value
        elif value is not None:
            data[key] = str(value)
    return data


def xml_leaf(tag: str, value: str) -> str:
    """Render a single XML-ish element with escaped text content."""

    return f"<{tag}>{html.escape(value)}</{tag}>"


__all__ = [
    "SkillInfo",
    "SkillRegistry",
    "load_skill_file",
    "xml_leaf",
]
