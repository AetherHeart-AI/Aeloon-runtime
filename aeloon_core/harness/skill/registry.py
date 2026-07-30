"""Isolated discovery and scope construction for Skills and ExpertSkills."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aeloon_core.config import Config
from aeloon_core.harness.skill.base import (
    MAX_SKILL_INSTRUCTION_CHARS,
    ExpertSkillSnapshot,
    SkillDefinitionError,
    SkillScope,
    SkillSnapshot,
    parse_skill_manifest,
    resolve_dependency_id,
)


class SkillRegistry:
    """Immutable process-start snapshot of explicitly configured Skill roots."""

    __slots__ = ("_skills", "_roots")

    def __init__(
        self,
        *,
        skills: dict[str, SkillSnapshot | ExpertSkillSnapshot],
        roots: dict[str, Path],
    ) -> None:
        self._skills = MappingProxyType(dict(skills))
        self._roots = MappingProxyType(dict(roots))

    @classmethod
    def discover(cls, config: Config) -> SkillRegistry:
        """Discover built-ins, workspace Skills, and only explicit extra roots."""

        builtin_root = Path(__file__).parent / "builtin"
        roots: list[tuple[str, Path, bool]] = [
            ("builtin", builtin_root.resolve(strict=False), True),
            (
                "workspace",
                (config.workspace / ".aeloon-core" / "skills").resolve(strict=False),
                False,
            ),
            *[
                (item.id, item.path.resolve(strict=False), True)
                for item in config.skills.roots
            ],
        ]
        snapshots: dict[str, SkillSnapshot | ExpertSkillSnapshot] = {}
        for root_id, root_path, required in roots:
            if not root_path.exists():
                if required:
                    raise SkillDefinitionError(
                        f"configured skill root {root_id!r} does not exist: {root_path}"
                    )
                continue
            if not root_path.is_dir():
                raise SkillDefinitionError(
                    f"configured skill root {root_id!r} is not a directory: {root_path}"
                )
            for manifest_path in sorted(root_path.rglob("SKILL.md")):
                if manifest_path.is_symlink():
                    continue
                snapshot = parse_skill_manifest(
                    manifest_path,
                    root_id=root_id,
                    root_path=root_path,
                )
                if snapshot.id in snapshots:
                    raise SkillDefinitionError(
                        f"duplicate Skill id {snapshot.id!r} in root {root_id!r}"
                    )
                snapshots[snapshot.id] = snapshot
        registry = cls(
            skills=snapshots,
            roots={root_id: root_path for root_id, root_path, _ in roots},
        )
        registry._validate_dependencies()
        return registry

    def get(self, skill_id: str) -> SkillSnapshot | ExpertSkillSnapshot | None:
        return self._skills.get(skill_id)

    def require(self, skill_id: str) -> SkillSnapshot | ExpertSkillSnapshot:
        snapshot = self.get(skill_id)
        if snapshot is None:
            raise SkillDefinitionError(f"unknown Skill id {skill_id!r}")
        return snapshot

    def list(self) -> tuple[SkillSnapshot | ExpertSkillSnapshot, ...]:
        return tuple(self._skills.values())

    def enabled_experts(self, enabled_ids: Iterable[str]) -> tuple[ExpertSkillSnapshot, ...]:
        experts: list[ExpertSkillSnapshot] = []
        for expert_id in enabled_ids:
            snapshot = self.require(expert_id)
            if not isinstance(snapshot, ExpertSkillSnapshot):
                raise SkillDefinitionError(f"enabled expert {expert_id!r} is a plain Skill")
            experts.append(snapshot)
        return tuple(experts)

    def master_scope(self, config: Config) -> SkillScope:
        """Resolve the mode-specific Skill scope granted to Master."""

        if config.mode == "normal":
            plain_ids = [
                snapshot.id
                for snapshot in self.list()
                if not isinstance(snapshot, ExpertSkillSnapshot)
            ]
        else:
            plain_ids = []
            for skill_id in config.skills.master_allowlist:
                snapshot = self.require(skill_id)
                if isinstance(snapshot, ExpertSkillSnapshot):
                    raise SkillDefinitionError(
                        f"master_allowlist is for plain Skills; "
                        f"{skill_id!r} is an ExpertSkill"
                    )
                plain_ids.append(skill_id)
        enabled = [expert.id for expert in self.enabled_experts(config.experts.enabled)]
        return SkillScope(owner="master", skill_ids=frozenset([*plain_ids, *enabled]))

    def expert_scope(self, expert: ExpertSkillSnapshot) -> SkillScope:
        """Expose only an ExpertSkill itself and its declared plain-Skill dependencies."""

        dependencies: list[str] = []
        for dependency in expert.dependencies:
            dependency_id = resolve_dependency_id(expert, dependency)
            snapshot = self.require(dependency_id)
            if isinstance(snapshot, ExpertSkillSnapshot):
                raise SkillDefinitionError(
                    f"ExpertSkill nesting is disabled: {expert.id!r} -> {dependency_id!r}"
                )
            dependencies.append(dependency_id)
        return SkillScope(
            owner=expert.id,
            skill_ids=frozenset([expert.id, *dependencies]),
        )

    def search(
        self,
        query: str,
        *,
        scope: SkillScope,
        limit: int = 10,
    ) -> tuple[dict[str, Any], ...]:
        """Search bounded metadata without loading instructions."""

        terms = [term.casefold() for term in query.split() if term.strip()]
        ranked: list[tuple[int, str, SkillSnapshot | ExpertSkillSnapshot]] = []
        for skill_id in scope.skill_ids:
            snapshot = self._skills.get(skill_id)
            if snapshot is None:
                continue
            haystack = f"{snapshot.id} {snapshot.name} {snapshot.description}".casefold()
            score = sum(1 for term in terms if term in haystack)
            if terms and score == 0:
                continue
            ranked.append((-score, snapshot.id, snapshot))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(snapshot.descriptor() for _, _, snapshot in ranked[:limit])

    def load(
        self,
        skill_id: str,
        *,
        scope: SkillScope,
    ) -> SkillSnapshot | ExpertSkillSnapshot:
        """Load a Skill only when it belongs to the caller's frozen scope."""

        if not scope.allows(skill_id):
            raise PermissionError(f"Skill {skill_id!r} is outside scope for {scope.owner!r}")
        return self.require(skill_id)

    def read_resource(
        self,
        skill_id: str,
        resource_path: str,
        *,
        scope: SkillScope,
    ) -> str:
        """Read one UTF-8 text resource beneath an in-scope Skill directory."""

        snapshot = self.load(skill_id, scope=scope)
        relative = Path(resource_path)
        if relative.is_absolute():
            raise PermissionError("Skill resource path must be relative")
        candidate = (snapshot.directory / relative).resolve(strict=False)
        if not candidate.is_relative_to(snapshot.directory.resolve(strict=False)):
            raise PermissionError("Skill resource path escapes the Skill directory")
        if candidate == snapshot.manifest_path:
            return snapshot.instructions
        if not candidate.is_file():
            raise FileNotFoundError(f"Skill resource not found: {resource_path}")
        if candidate.stat().st_size > MAX_SKILL_INSTRUCTION_CHARS * 4:
            raise ValueError(
                f"Skill resource exceeds the {MAX_SKILL_INSTRUCTION_CHARS}-character limit"
            )
        content = candidate.read_text(encoding="utf-8")
        if len(content) > MAX_SKILL_INSTRUCTION_CHARS:
            raise ValueError(
                f"Skill resource exceeds the {MAX_SKILL_INSTRUCTION_CHARS}-character limit"
            )
        return content

    def _validate_dependencies(self) -> None:
        for snapshot in self._skills.values():
            if isinstance(snapshot, ExpertSkillSnapshot):
                self.expert_scope(snapshot)


__all__ = ["SkillRegistry"]
