"""Immutable Skill and ExpertSkill contracts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SKILL_NAME_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
SKILL_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}:[a-z][a-z0-9_-]{0,63}$"
RUNNER_ID_PATTERN = r"^[a-z][a-z0-9_.-]{0,127}$"
MCP_SERVER_NAME_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
MAX_SKILL_INSTRUCTION_CHARS = 128_000
ALLOWED_EXPERT_CAPABILITIES = frozenset(
    {
        "filesystem",
        "filesystem_read",
        "shell",
        "repo_context",
        "planning",
        "web_search",
    }
)


class SkillDefinitionError(ValueError):
    """Raised when a Skill manifest is invalid."""


class SkillManifest(BaseModel):
    """Validated YAML frontmatter shared by plain and executable Skills."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    name: str = Field(pattern=SKILL_NAME_PATTERN)
    description: str = Field(min_length=1, max_length=1_000)
    license: str | None = Field(default=None, max_length=1_000)
    compatibility: str | None = Field(default=None, max_length=2_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: str | tuple[str, ...] | None = Field(
        default=None,
        alias="allowed-tools",
    )
    kind: Literal["skill", "expert"] = "skill"
    runner: str | None = Field(default=None, pattern=RUNNER_ID_PATTERN)
    dependencies: tuple[str, ...] = Field(default=(), max_length=64)
    capabilities: tuple[str, ...] = Field(default=(), max_length=16)
    mcp_servers: tuple[str, ...] = Field(
        default=(),
        alias="mcp-servers",
        max_length=64,
    )
    model_tier: Literal["fast", "strong"] = "strong"
    concurrency_mode: Literal["parallel_safe", "exclusive"] = "exclusive"
    max_calls_per_turn: int = Field(default=1, ge=1, le=32)


class Skill(BaseModel):
    """One lazily loadable, immutable Skill discovered from a trusted root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=SKILL_ID_PATTERN)
    name: str = Field(pattern=SKILL_NAME_PATTERN)
    description: str
    kind: Literal["skill"] = "skill"
    source_root: str
    directory: Path
    manifest_path: Path
    instructions: str
    digest: str

    def descriptor(self) -> dict[str, Any]:
        """Return bounded metadata that is safe to inject into an agent prompt."""

        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "source_root": self.source_root,
            "digest": self.digest,
        }


class ExpertSkill(Skill):
    """A Skill that can also be invoked as a scoped, turn-local sub-agent."""

    kind: Literal["expert"] = "expert"
    runner: str
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    model_tier: Literal["fast", "strong"] = "strong"
    concurrency_mode: Literal["parallel_safe", "exclusive"] = "exclusive"
    max_calls_per_turn: int = 1

    def descriptor(self) -> dict[str, Any]:
        descriptor = super().descriptor()
        descriptor.update(
            {
                "runner": self.runner,
                "capabilities": list(self.capabilities),
                "mcp_servers": list(self.mcp_servers),
                "model_tier": self.model_tier,
                "concurrency_mode": self.concurrency_mode,
                "max_calls_per_turn": self.max_calls_per_turn,
            }
        )
        return descriptor


# Snapshot names make the immutable startup semantics explicit inside the host,
# while the shorter names are the public conceptual contract.
SkillSnapshot = Skill
ExpertSkillSnapshot = ExpertSkill


class SkillScope(BaseModel):
    """Immutable set of Skill ids visible to one agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: str
    skill_ids: frozenset[str]

    def allows(self, skill_id: str) -> bool:
        return skill_id in self.skill_ids


def parse_skill_manifest(
    manifest_path: Path,
    *,
    root_id: str,
    root_path: Path,
) -> SkillSnapshot | ExpertSkillSnapshot:
    """Parse and freeze one SKILL.md within a configured discovery root."""

    root = root_path.resolve(strict=False)
    path = manifest_path.resolve(strict=False)
    if path.name != "SKILL.md" or not path.is_relative_to(root):
        raise SkillDefinitionError(f"skill manifest escapes root {root_id!r}: {path}")
    text = path.read_text(encoding="utf-8")
    metadata, instructions = _split_frontmatter(text, path=path)
    try:
        manifest = SkillManifest.model_validate(metadata)
    except ValidationError as exc:
        raise SkillDefinitionError(f"invalid Skill manifest {path}: {exc}") from exc
    if not instructions.strip():
        raise SkillDefinitionError(f"Skill manifest has no instructions: {path}")
    if len(instructions) > MAX_SKILL_INSTRUCTION_CHARS:
        raise SkillDefinitionError(
            f"Skill instructions exceed {MAX_SKILL_INSTRUCTION_CHARS} characters: {path}"
        )

    skill_id = f"{root_id}:{manifest.name}"
    payload = {
        "id": skill_id,
        "metadata": manifest.model_dump(mode="json"),
        "instructions": instructions,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    common: dict[str, Any] = {
        "id": skill_id,
        "name": manifest.name,
        "description": manifest.description,
        "source_root": root_id,
        "directory": path.parent,
        "manifest_path": path,
        "instructions": instructions.strip(),
        "digest": digest,
    }
    if manifest.kind == "skill":
        expert_only = {
            "runner",
            "dependencies",
            "capabilities",
            "mcp_servers",
            "model_tier",
            "concurrency_mode",
            "max_calls_per_turn",
        }
        configured = sorted(expert_only & manifest.model_fields_set)
        if configured:
            raise SkillDefinitionError(
                f"plain Skill {skill_id!r} cannot configure: {', '.join(configured)}"
            )
        return SkillSnapshot(**common)

    if not manifest.runner:
        raise SkillDefinitionError(f"ExpertSkill {skill_id!r} requires runner")
    if len(set(manifest.dependencies)) != len(manifest.dependencies):
        raise SkillDefinitionError(f"ExpertSkill {skill_id!r} has duplicate dependencies")
    if len(set(manifest.capabilities)) != len(manifest.capabilities):
        raise SkillDefinitionError(f"ExpertSkill {skill_id!r} has duplicate capabilities")
    if len(set(manifest.mcp_servers)) != len(manifest.mcp_servers):
        raise SkillDefinitionError(f"ExpertSkill {skill_id!r} has duplicate MCP servers")
    invalid_mcp_servers = sorted(
        server
        for server in manifest.mcp_servers
        if re.fullmatch(MCP_SERVER_NAME_PATTERN, server) is None
    )
    if invalid_mcp_servers:
        raise SkillDefinitionError(
            f"ExpertSkill {skill_id!r} has invalid MCP servers: "
            + ", ".join(invalid_mcp_servers)
        )
    unknown = sorted(set(manifest.capabilities) - ALLOWED_EXPERT_CAPABILITIES)
    if unknown:
        raise SkillDefinitionError(
            f"ExpertSkill {skill_id!r} has unknown capabilities: {', '.join(unknown)}"
        )
    return ExpertSkillSnapshot(
        **common,
        kind="expert",
        runner=manifest.runner,
        dependencies=manifest.dependencies,
        capabilities=manifest.capabilities,
        mcp_servers=manifest.mcp_servers,
        model_tier=manifest.model_tier,
        concurrency_mode=manifest.concurrency_mode,
        max_calls_per_turn=manifest.max_calls_per_turn,
    )


def resolve_dependency_id(expert: ExpertSkillSnapshot, dependency: str) -> str:
    """Resolve a bare dependency name relative to its ExpertSkill root."""

    candidate = dependency.strip()
    if re.fullmatch(SKILL_ID_PATTERN, candidate):
        return candidate
    if re.fullmatch(SKILL_NAME_PATTERN, candidate):
        return f"{expert.source_root}:{candidate}"
    raise SkillDefinitionError(f"ExpertSkill {expert.id!r} has invalid dependency {dependency!r}")


def _split_frontmatter(text: str, *, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillDefinitionError(f"Skill manifest requires YAML frontmatter: {path}")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillDefinitionError(f"Skill manifest has unterminated frontmatter: {path}") from exc
    try:
        metadata = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise SkillDefinitionError(f"invalid YAML frontmatter in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SkillDefinitionError(f"Skill frontmatter must be a mapping: {path}")
    return metadata, "\n".join(lines[end + 1 :])


__all__ = [
    "ALLOWED_EXPERT_CAPABILITIES",
    "ExpertSkill",
    "ExpertSkillSnapshot",
    "MAX_SKILL_INSTRUCTION_CHARS",
    "MCP_SERVER_NAME_PATTERN",
    "RUNNER_ID_PATTERN",
    "SkillDefinitionError",
    "Skill",
    "SkillScope",
    "SkillSnapshot",
    "parse_skill_manifest",
    "resolve_dependency_id",
]
