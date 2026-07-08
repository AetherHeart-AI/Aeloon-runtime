from __future__ import annotations

import pytest

from aeloon_core.config import Config, SkillsConfig
from aeloon_core.context import apply_skill_guidance, build_initial_messages
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.skills import SkillRegistry
from aeloon_core.tools.skill import SkillTool


def write_skill(path, *, name: str, description: str | None = None, body: str = "# Body") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = [f"name: {name}"]
    if description is not None:
        frontmatter.append(f"description: {description}")
    path.write_text(
        "---\n" + "\n".join(frontmatter) + "\n---\n\n" + body + "\n",
        encoding="utf-8",
    )


def test_discovers_opencode_and_external_skill_locations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".git").mkdir()
    write_skill(
        tmp_path / ".opencode" / "skill" / "opencode-skill" / "SKILL.md",
        name="opencode-skill",
        description="OpenCode-compatible project skill.",
    )
    write_skill(
        tmp_path / ".opencode" / "skills" / "plural-skill" / "SKILL.md",
        name="plural-skill",
        description="Plural OpenCode-compatible project skill.",
    )
    write_skill(
        tmp_path / ".claude" / "skills" / "claude-skill" / "SKILL.md",
        name="claude-skill",
        description="Claude-compatible project skill.",
    )
    write_skill(
        tmp_path / ".agents" / "skills" / "agent-skill" / "SKILL.md",
        name="agent-skill",
        description="Agent-compatible project skill.",
    )

    registry = SkillRegistry.discover(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    assert [skill.name for skill in registry.all()] == [
        "agent-skill",
        "claude-skill",
        "opencode-skill",
        "plural-skill",
    ]
    assert "<available_skills>" in (registry.format_guidance() or "")


def test_external_skill_locations_can_be_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".git").mkdir()
    write_skill(
        tmp_path / ".claude" / "skills" / "claude-skill" / "SKILL.md",
        name="claude-skill",
        description="Claude-compatible project skill.",
    )
    write_skill(
        tmp_path / ".opencode" / "skills" / "opencode-skill" / "SKILL.md",
        name="opencode-skill",
        description="OpenCode-compatible project skill.",
    )

    registry = SkillRegistry.discover(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            skills=SkillsConfig(external=False),
        ).normalized()
    )

    assert [skill.name for skill in registry.all()] == ["opencode-skill"]


def test_custom_skill_paths_are_resolved_from_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write_skill(
        tmp_path / "custom" / "team-skill" / "SKILL.md",
        name="team-skill",
        description="Skill loaded from configured paths.",
    )

    registry = SkillRegistry.discover(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            skills=SkillsConfig(paths=["custom"]),
        ).normalized()
    )

    assert registry.get("team-skill") is not None


def test_discovers_global_native_skills(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    write_skill(
        home / ".aeloon-core" / "skills" / "global-native" / "SKILL.md",
        name="global-native",
        description="Global native Aeloon skill.",
    )

    registry = SkillRegistry.discover(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    assert registry.get("global-native") is not None


def test_native_skill_overrides_same_named_external_skill(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".git").mkdir()
    write_skill(
        tmp_path / ".claude" / "skills" / "shared" / "SKILL.md",
        name="shared",
        description="External copy.",
    )
    write_skill(
        tmp_path / ".opencode" / "skills" / "shared" / "SKILL.md",
        name="shared",
        description="Native copy.",
    )

    registry = SkillRegistry.discover(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    skill = registry.get("shared")
    assert skill is not None
    assert skill.description == "Native copy."
    assert [item.name for item in registry.all()] == ["shared"]


def test_multiline_folded_description_is_parsed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = tmp_path / ".opencode" / "skills" / "wrapped" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: wrapped\n"
        "description: >\n"
        "  first line of the description\n"
        "  that wraps onto a second line\n"
        "---\n\n# Body\n",
        encoding="utf-8",
    )

    registry = SkillRegistry.discover(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    skill = registry.get("wrapped")
    assert skill is not None
    assert skill.description == "first line of the description that wraps onto a second line"


def test_skills_without_descriptions_load_but_are_not_listed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write_skill(tmp_path / ".opencode" / "skills" / "manual" / "SKILL.md", name="manual")

    registry = SkillRegistry.discover(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    assert registry.get("manual") is not None
    assert "No skills are currently available." in (registry.format_guidance() or "")


def test_orchestrator_omits_skill_tool_when_no_skills(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    orchestrator = AeloonCoreOrchestrator(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    assert "skill" not in _tool_names(orchestrator)


def test_orchestrator_registers_skill_tool_when_skills_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write_skill(
        tmp_path / ".opencode" / "skills" / "release" / "SKILL.md",
        name="release",
        description="Prepare releases.",
    )

    orchestrator = AeloonCoreOrchestrator(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )

    assert "skill" in _tool_names(orchestrator)


@pytest.mark.asyncio
async def test_skill_tool_loads_content_and_samples_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skill_dir = tmp_path / ".opencode" / "skills" / "release"
    write_skill(
        skill_dir / "SKILL.md",
        name="release",
        description="Prepare releases.",
        body="## Steps\n\nUse the release checklist.",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "checklist.md").write_text("checklist", encoding="utf-8")

    registry = SkillRegistry.discover(
        Config(workspace=tmp_path, data_dir=tmp_path / "data").normalized()
    )
    result = await SkillTool(registry=registry).execute("release")

    assert '<skill_content name="release">' in result
    assert "Use the release checklist." in result
    assert f"Base directory for this skill: {skill_dir}" in result
    assert str(skill_dir / "references" / "checklist.md") in result
    assert "<file>" + str(skill_dir / "SKILL.md") + "</file>" not in result


def test_apply_skill_guidance_replaces_prior_guidance(tmp_path) -> None:
    messages = build_initial_messages(workspace=tmp_path)

    first = apply_skill_guidance(messages, "first")
    second = apply_skill_guidance(first, "second")

    assert len([message for message in second if message["role"] == "system"]) == 2
    assert "first" not in "\n".join(str(message["content"]) for message in second)
    assert "second" in str(second[1]["content"])


def _tool_names(orchestrator: AeloonCoreOrchestrator) -> set[str]:
    return {tool["function"]["name"] for tool in orchestrator.registry.get_definitions()}
