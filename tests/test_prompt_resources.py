# Full prompt snapshots intentionally preserve stable prose line lengths.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

from aeloon_core.harness import ResourceLoader, build_system_prompt, create_all_tools


def test_default_prompt_is_deterministic_and_tracks_active_tools(tmp_path: Path) -> None:
    tools = create_all_tools(tmp_path)
    resources = ResourceLoader(cwd=tmp_path, agent_dir=tmp_path / "global").reload()

    prompt = build_system_prompt(
        cwd=tmp_path,
        tools=tuple(tools[name] for name in ("read", "bash", "edit", "write")),
        resources=resources,
    )

    assert (
        prompt
        == f"""You are an expert coding assistant operating inside Aeloon, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute shell commands
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed.
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.
- Use write only for new files or complete rewrites.
- Be concise in your responses
- Show file paths clearly when working with files
Current working directory: {tmp_path}"""
    )


def test_system_override_and_append_context_skills_order(tmp_path: Path) -> None:
    global_dir = tmp_path / "global"
    workspace = tmp_path / "repo" / "nested"
    project = workspace / ".aeloon-core"
    (global_dir / "skills" / "global-skill").mkdir(parents=True)
    (global_dir / "skills" / "global-skill" / "SKILL.md").write_text(
        "---\nname: same\ndescription: global\n---\nglobal body\n",
        encoding="utf-8",
    )
    (project / "skills" / "local-skill").mkdir(parents=True)
    (project / "skills" / "local-skill" / "SKILL.md").write_text(
        "---\nname: same\ndescription: local wins\n---\nlocal body\n",
        encoding="utf-8",
    )
    (global_dir / "SYSTEM.md").write_text("GLOBAL BASE", encoding="utf-8")
    project.mkdir(parents=True, exist_ok=True)
    (project / "SYSTEM.md").write_text("PROJECT BASE", encoding="utf-8")
    (project / "APPEND_SYSTEM.md").write_text("APPENDED", encoding="utf-8")
    (tmp_path / "repo" / "AGENTS.md").write_text("ROOT RULE", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("NESTED RULE", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("FIRST MATCH", encoding="utf-8")

    resources = ResourceLoader(cwd=workspace, agent_dir=global_dir).reload()
    prompt = build_system_prompt(
        cwd=workspace,
        tools=(create_all_tools(workspace)["read"],),
        resources=resources,
    )

    assert resources.system_prompt == "PROJECT BASE"
    assert [(skill.name, skill.description) for skill in resources.skills] == [
        ("same", "local wins")
    ]
    assert "NESTED RULE" not in prompt
    positions = [
        prompt.index("PROJECT BASE"),
        prompt.index("APPENDED"),
        prompt.index("ROOT RULE"),
        prompt.index("FIRST MATCH"),
        prompt.index("<available_skills>"),
        prompt.index("Current working directory:"),
    ]
    assert positions == sorted(positions)


def test_prompt_templates_use_project_override(tmp_path: Path) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / ".aeloon-core"
    (global_dir / "prompts").mkdir(parents=True)
    (project_dir / "prompts").mkdir(parents=True)
    (global_dir / "prompts" / "review.md").write_text("global $1", encoding="utf-8")
    (project_dir / "prompts" / "review.md").write_text("local $1 $@", encoding="utf-8")

    resources = ResourceLoader(cwd=tmp_path, agent_dir=global_dir).reload()

    assert len(resources.prompt_templates) == 1
    assert resources.prompt_templates[0].format(("a", "b")) == "local a a b"
