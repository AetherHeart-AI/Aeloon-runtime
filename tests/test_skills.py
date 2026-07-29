"""Tests for Skill discovery, canonical ids, and frozen scopes."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.config import Config
from aeloon_core.harness.skill import (
    ExpertSkill,
    ExpertSkillSnapshot,
    Skill,
    SkillDefinitionError,
    SkillRegistry,
)


def test_expert_skill_is_a_skill_contract() -> None:
    assert issubclass(ExpertSkill, Skill)
    assert ExpertSkillSnapshot is ExpertSkill


def _skill(
    root: Path,
    directory: str,
    *,
    name: str,
    kind: str = "skill",
    runner: str | None = None,
    dependencies: tuple[str, ...] = (),
    mcp_servers: tuple[str, ...] = (),
) -> Path:
    path = root / directory
    path.mkdir(parents=True, exist_ok=True)
    metadata = [
        "---",
        f"name: {name}",
        f"description: {name} description",
        f"kind: {kind}",
    ]
    if runner:
        metadata.append(f"runner: {runner}")
    if dependencies:
        metadata.extend(["dependencies:", *[f"  - {item}" for item in dependencies]])
    if mcp_servers:
        metadata.extend(["mcp-servers:", *[f"  - {item}" for item in mcp_servers]])
    metadata.extend(["---", f"# {name}", "", f"Instructions for {name}."])
    (path / "SKILL.md").write_text("\n".join(metadata), encoding="utf-8")
    return path


def test_discovery_uses_builtin_workspace_and_only_explicit_roots(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace_root = workspace / ".aeloon-core" / "skills"
    team_root = tmp_path / "team"
    _skill(workspace_root, "style", name="style")
    _skill(team_root, "style", name="style")
    _skill(tmp_path / ".codex" / "skills", "hidden", name="hidden")
    config = Config(
        workspace=workspace,
        skills={
            "roots": [{"id": "team", "path": team_root}],
            "master_allowlist": ["workspace:style", "team:style"],
        },
    ).normalized()

    registry = SkillRegistry.discover(config)

    assert registry.get("builtin:research") is not None
    assert registry.get("workspace:style") is not None
    assert registry.get("team:style") is not None
    assert registry.get("workspace:hidden") is None
    assert registry.master_scope(config).skill_ids == frozenset(
        {
            "builtin:research",
            "builtin:coding",
            "workspace:style",
            "team:style",
        }
    )


def test_expert_scope_contains_only_itself_and_plain_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace" / ".aeloon-core" / "skills"
    resource_dir = _skill(root, "rules", name="rules")
    (resource_dir / "checklist.md").write_text("verify it", encoding="utf-8")
    _skill(
        root,
        "custom",
        name="custom",
        kind="expert",
        runner="builtin.prompt",
        dependencies=("rules",),
        mcp_servers=("github",),
    )
    config = Config(
        workspace=tmp_path / "workspace",
        experts={"enabled": ["workspace:custom"]},
    ).normalized()
    registry = SkillRegistry.discover(config)
    expert = registry.require("workspace:custom")

    assert isinstance(expert, ExpertSkillSnapshot)
    assert expert.mcp_servers == ("github",)
    scope = registry.expert_scope(expert)
    assert scope.skill_ids == frozenset({"workspace:custom", "workspace:rules"})
    assert (
        registry.read_resource(
            "workspace:rules",
            "checklist.md",
            scope=scope,
        )
        == "verify it"
    )
    with pytest.raises(PermissionError, match="outside scope"):
        registry.load("builtin:coding", scope=scope)
    with pytest.raises(PermissionError, match="escapes"):
        registry.read_resource("workspace:rules", "../custom/SKILL.md", scope=scope)


def test_expert_nesting_is_rejected_at_discovery(tmp_path: Path) -> None:
    root = tmp_path / "workspace" / ".aeloon-core" / "skills"
    _skill(
        root,
        "inner",
        name="inner",
        kind="expert",
        runner="builtin.prompt",
    )
    _skill(
        root,
        "outer",
        name="outer",
        kind="expert",
        runner="builtin.prompt",
        dependencies=("inner",),
    )

    with pytest.raises(SkillDefinitionError, match="nesting is disabled"):
        SkillRegistry.discover(Config(workspace=tmp_path / "workspace").normalized())


def test_master_plain_skill_allowlist_rejects_expert_ids(tmp_path: Path) -> None:
    config = Config(
        mode="expert",
        workspace=tmp_path,
        skills={"master_allowlist": ["builtin:coding"]},
    ).normalized()
    registry = SkillRegistry.discover(config)

    with pytest.raises(SkillDefinitionError, match="plain Skills"):
        registry.master_scope(config)


def test_normal_mode_exposes_every_discovered_plain_skill_to_master(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".aeloon-core" / "skills"
    _skill(root, "one", name="one")
    _skill(root, "two", name="two")
    config = Config(workspace=tmp_path).normalized()
    registry = SkillRegistry.discover(config)

    assert {"workspace:one", "workspace:two"} <= registry.master_scope(
        config
    ).skill_ids


def test_expert_mode_exposes_only_allowlisted_plain_skills_to_master(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".aeloon-core" / "skills"
    _skill(root, "one", name="one")
    _skill(root, "two", name="two")
    config = Config(
        mode="expert",
        workspace=tmp_path,
        skills={"master_allowlist": ["workspace:one"]},
    ).normalized()
    registry = SkillRegistry.discover(config)

    scope = registry.master_scope(config).skill_ids
    assert "workspace:one" in scope
    assert "workspace:two" not in scope


def test_missing_explicit_root_fails_fast(tmp_path: Path) -> None:
    config = Config(
        workspace=tmp_path,
        skills={
            "roots": [{"id": "missing", "path": tmp_path / "does-not-exist"}]
        },
    ).normalized()

    with pytest.raises(SkillDefinitionError, match="does not exist"):
        SkillRegistry.discover(config)


def test_plain_skill_accepts_standard_optional_frontmatter(tmp_path: Path) -> None:
    directory = tmp_path / ".aeloon-core" / "skills" / "portable"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        """---
name: portable
description: Portable standard Skill.
license: Apache-2.0
compatibility: Requires a text workspace.
metadata:
  author: team
allowed-tools: Read Grep
---
# Portable

These are passive instructions; allowed-tools does not grant host capabilities.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry.discover(Config(workspace=tmp_path).normalized())

    assert registry.require("workspace:portable").description == (
        "Portable standard Skill."
    )
