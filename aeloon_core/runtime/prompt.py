"""Runtime-owned system prompt construction."""

# Prompt prose uses full lines to keep string snapshots stable.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from aeloon_core.core.types import Tool
from aeloon_core.runtime.resources import RuntimeResources, Skill


def format_skills_for_system_prompt(skills: tuple[Skill, ...]) -> str:
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Read the full skill file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(skill.description)}</description>",
                f"    <location>{escape(skill.file_path)}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def build_system_prompt(
    *,
    cwd: Path | str,
    tools: tuple[Tool, ...],
    resources: RuntimeResources,
    custom_prompt: str | None = None,
) -> str:
    prompt_cwd = str(Path(cwd).expanduser().resolve(strict=False)).replace("\\", "/")
    selected_names = [tool.name for tool in tools]
    append_section = "\n\n".join(value for value in resources.append_system_prompt if value)
    base_override = custom_prompt if custom_prompt is not None else resources.system_prompt
    if base_override:
        prompt = base_override
    else:
        snippets = [
            f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet
        ]
        tool_list = "\n".join(snippets) if snippets else "(none)"
        guidelines: list[str] = []

        def add(value: str) -> None:
            normalized = value.strip()
            if normalized and normalized not in guidelines:
                guidelines.append(normalized)

        if "bash" in selected_names and not any(
            name in selected_names for name in ("grep", "find", "ls")
        ):
            add("Use bash for file operations like ls, rg, find")
        for tool in tools:
            for guideline in tool.prompt_guidelines:
                add(guideline)
        add("Be concise in your responses")
        add("Show file paths clearly when working with files")
        guideline_text = "\n".join(f"- {value}" for value in guidelines)
        prompt = f"""You are an expert coding assistant operating inside Aeloon, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tool_list}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{guideline_text}"""

    if append_section:
        prompt += f"\n\n{append_section}"
    if resources.context_files:
        prompt += "\n\n<project_context>\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for path, content in resources.context_files:
            prompt += (
                f'<project_instructions path="{escape(path)}">\n'
                f"{content}\n</project_instructions>\n\n"
            )
        prompt += "</project_context>\n"
    if "read" in selected_names:
        skills = format_skills_for_system_prompt(resources.skills)
        if skills:
            prompt += f"\n{skills}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


__all__ = ["build_system_prompt", "format_skills_for_system_prompt"]
