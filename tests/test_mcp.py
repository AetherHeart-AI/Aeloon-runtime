"""Tests for mode-aware MCP server scoping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from aeloon_core.config import Config
from aeloon_core.harness.mcp import McpConfigError, McpRegistry
from aeloon_core.harness.skill import ExpertSkillSnapshot, SkillRegistry


@dataclass(frozen=True)
class _FakeToolset:
    id: str


def _registry() -> McpRegistry:
    return McpRegistry(
        {
            "docs": _FakeToolset(id="docs"),
            "github": _FakeToolset(id="github"),
        }
    )


def test_normal_mode_exposes_every_configured_mcp_server() -> None:
    registry = _registry()

    assert len(registry.master_toolsets(Config(mode="normal"))) == 2


def test_expert_mode_exposes_only_allowlisted_master_mcp_servers() -> None:
    registry = _registry()
    config = Config(
        mode="expert",
        mcp={"master_allowlist": ["github"]},
    )

    selected = registry.master_toolsets(config)
    assert len(selected) == 1
    assert selected[0].id == "github"


def test_expert_mode_rejects_unknown_master_mcp_servers() -> None:
    registry = _registry()
    config = Config(
        mode="expert",
        mcp={"master_allowlist": ["missing"]},
    )

    with pytest.raises(McpConfigError, match="unknown MCP servers missing"):
        registry.master_toolsets(config)


def test_expert_receives_only_manifest_declared_mcp_servers(tmp_path: Path) -> None:
    directory = tmp_path / ".aeloon-core" / "skills" / "custom"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        """---
name: custom
description: Custom expert.
kind: expert
runner: builtin.prompt
mcp-servers:
  - github
---
# Custom

Use only the declared server.
""",
        encoding="utf-8",
    )
    expert = SkillRegistry.discover(
        Config(workspace=tmp_path).normalized()
    ).require("workspace:custom")
    assert isinstance(expert, ExpertSkillSnapshot)

    selected = _registry().expert_toolsets(expert)

    assert len(selected) == 1
    assert selected[0].id == "github"


def test_registry_loads_explicit_mcp_config_without_connecting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "docs": {"url": "http://127.0.0.1:8765/mcp"},
                }
            }
        ),
        encoding="utf-8",
    )

    registry = McpRegistry.discover(
        Config(workspace=tmp_path, mcp={"config_path": "mcp.json"}).normalized()
    )

    assert registry.ids() == ("docs",)
