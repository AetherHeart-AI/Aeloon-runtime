"""Built-in file search and directory listing tools."""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from aeloon_core.core.types import ToolResult, ToolUpdateCallback
from aeloon_core.tool.base import WorkspaceTool, object_schema, truncate_limited


class RipgrepTool(WorkspaceTool):
    async def _run_rg(self, arguments: list[str]) -> tuple[str, int]:
        executable = shutil.which("rg")
        if executable is None:
            raise FileNotFoundError("ripgrep (rg) is required for this tool")
        process = await asyncio.create_subprocess_exec(
            executable,
            *arguments,
            cwd=str(self.context.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode("utf-8", errors="replace")
        if process.returncode not in {0, 1}:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or "rg failed")
        return output, int(process.returncode or 0)


class GrepTool(RipgrepTool):
    name = "grep"
    label = "grep"
    description = (
        "Search file contents for a pattern. Returns matching lines with file paths "
        "and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB."
    )
    prompt_snippet = "Search file contents for patterns (respects .gitignore)"
    parameters = object_schema(
        {
            "pattern": {
                "type": "string",
                "description": "Search pattern (regex or literal string)",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory)",
            },
            "glob": {"type": "string", "description": "Filter files by glob pattern"},
            "ignoreCase": {
                "type": "boolean",
                "description": "Case-insensitive search (default: false)",
            },
            "literal": {"type": "boolean", "description": "Treat pattern as literal string"},
            "context": {"type": "number", "description": "Lines before and after each match"},
            "limit": {
                "type": "number",
                "minimum": 1,
                "description": "Maximum matches (default: 100)",
            },
        },
        ("pattern",),
    )

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        values = ["--line-number", "--color=never"]
        if arguments.get("ignoreCase"):
            values.append("--ignore-case")
        if arguments.get("literal"):
            values.append("--fixed-strings")
        if arguments.get("glob"):
            values.extend(["--glob", str(arguments["glob"])])
        if arguments.get("context") is not None:
            values.extend(["--context", str(int(arguments["context"]))])
        values.extend([str(arguments["pattern"]), str(arguments.get("path") or ".")])
        output, _ = await self._run_rg(values)
        limit = max(1, int(arguments.get("limit") or 100))
        lines = output.splitlines()
        visible, truncation = truncate_limited(lines, limit)
        return ToolResult.text(
            visible or "No matches found",
            details={"resultCount": len(lines), "truncation": truncation},
        )


class FindTool(RipgrepTool):
    name = "find"
    label = "find"
    description = (
        "Search for files by glob pattern. Returns matching paths and respects .gitignore. "
        "Output is truncated to 1000 results or 50KB."
    )
    prompt_snippet = "Find files by glob pattern (respects .gitignore)"
    parameters = object_schema(
        {
            "pattern": {"type": "string", "description": "Glob pattern to match files"},
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)",
            },
            "limit": {
                "type": "number",
                "minimum": 1,
                "description": "Maximum results (default: 1000)",
            },
        },
        ("pattern",),
    )

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        base = str(arguments.get("path") or ".")
        output, _ = await self._run_rg(["--files", "--glob", str(arguments["pattern"]), base])
        limit = max(1, int(arguments.get("limit") or 1_000))
        lines = output.splitlines()
        visible, truncation = truncate_limited(lines, limit)
        return ToolResult.text(
            visible or "No files found",
            details={"resultCount": len(lines), "truncation": truncation},
        )


class ListTool(WorkspaceTool):
    name = "ls"
    label = "ls"
    description = (
        "List directory contents sorted alphabetically, including dotfiles and '/' suffixes "
        "for directories. Output is truncated to 500 entries or 50KB."
    )
    prompt_snippet = "List directory contents"
    parameters = object_schema(
        {
            "path": {
                "type": "string",
                "description": "Directory to list (default: current directory)",
            },
            "limit": {
                "type": "number",
                "minimum": 1,
                "description": "Maximum entries (default: 500)",
            },
        }
    )

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        path = self.context.resolve(str(arguments.get("path") or "."))
        limit = max(1, int(arguments.get("limit") or 500))
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
        lines = [item.name + ("/" if item.is_dir() else "") for item in entries]
        visible, truncation = truncate_limited(lines, limit)
        return ToolResult.text(
            visible or "(empty directory)",
            details={"resultCount": len(entries), "truncation": truncation},
        )


__all__ = ["FindTool", "GrepTool", "ListTool"]
