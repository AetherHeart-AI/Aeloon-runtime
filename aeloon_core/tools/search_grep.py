"""Glob and grep tools."""

from __future__ import annotations

import asyncio
import fnmatch
import shutil
from pathlib import Path
from typing import Any

from aeloon_core.tools.base import Tool


class GlobTool(Tool):
    """Find files by glob pattern."""

    def __init__(self, *, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "glob"

    @property
    def concurrency_mode(self) -> str:
        return "read_only"

    @property
    def description(self) -> str:
        return "Find files matching a glob pattern. Uses workspace as the default root."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, for example **/*.py."},
                "root": {
                    "type": "string",
                    "description": (
                        "Optional root directory. Relative paths resolve from workspace."
                    ),
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        root: str | None = None,
        limit: int = 200,
        **kwargs: Any,
    ) -> str:
        del kwargs
        base = self._resolve(root)
        matches = sorted(path for path in base.glob(pattern) if path.exists())
        limited = matches[:limit]
        if not limited:
            return "(no matches)"
        lines = [
            str(path.relative_to(self.workspace) if _is_under(path, self.workspace) else path)
            for path in limited
        ]
        if len(matches) > len(limited):
            lines.append(f"... {len(matches) - len(limited)} more")
        return "\n".join(lines)

    def _resolve(self, root: str | None) -> Path:
        if not root:
            return self.workspace
        path = Path(root).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve(strict=False)


class GrepTool(Tool):
    """Search file contents."""

    def __init__(self, *, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "grep"

    @property
    def concurrency_mode(self) -> str:
        return "read_only"

    @property
    def description(self) -> str:
        return "Search text in files. Prefers ripgrep when it is installed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex/text pattern to search for."},
                "path": {
                    "type": "string",
                    "description": (
                        "Optional file or directory. Relative paths resolve from workspace."
                    ),
                },
                "include": {
                    "type": "string",
                    "description": "Optional glob include, for example *.py.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
        limit: int = 200,
        **kwargs: Any,
    ) -> str:
        del kwargs
        target = self._resolve(path)
        if shutil.which("rg"):
            return await self._run_rg(pattern, target, include, limit)
        return self._python_grep(pattern, target, include, limit)

    async def _run_rg(
        self,
        pattern: str,
        target: Path,
        include: str | None,
        limit: int,
    ) -> str:
        cmd = ["rg", "--line-number", "--color=never", "--no-heading"]
        if include:
            cmd.extend(["--glob", include])
        cmd.extend([pattern, str(target)])
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
        )
        stdout, stderr = await process.communicate()
        text = stdout.decode("utf-8", errors="replace")
        if process.returncode not in (0, 1):
            err = stderr.decode("utf-8", errors="replace").strip()
            return f"Error running rg: {err or process.returncode}"
        lines = text.splitlines()
        if not lines:
            return "(no matches)"
        limited = lines[:limit]
        if len(lines) > limit:
            limited.append(f"... {len(lines) - limit} more")
        return "\n".join(limited)

    def _python_grep(
        self,
        pattern: str,
        target: Path,
        include: str | None,
        limit: int,
    ) -> str:
        import re

        regex = re.compile(pattern)
        files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
        matches: list[str] = []
        for file_path in files:
            if include and not fnmatch.fnmatch(file_path.name, include):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{file_path}:{line_no}:{line}")
                    if len(matches) >= limit:
                        return "\n".join(matches)
        return "\n".join(matches) if matches else "(no matches)"

    def _resolve(self, path: str | None) -> Path:
        if not path:
            return self.workspace
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve(strict=False)


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
