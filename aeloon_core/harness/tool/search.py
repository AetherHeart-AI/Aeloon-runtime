"""Glob and grep tools."""

from __future__ import annotations

import asyncio
import fnmatch
import glob as globlib
import re
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from aeloon_core.harness.tool.base import WorkspaceTool


class ListArgs(BaseModel):
    path: str | None = Field(
        default=None,
        description="Directory to list. Relative paths resolve from workspace.",
    )
    all: bool = Field(default=False, description="Include dotfiles.")
    detail: bool = Field(default=False, description="Include mode, byte size, and timestamp.")
    limit: int = Field(default=200, ge=1, le=1000)


class ListTool(WorkspaceTool):
    """Perform a bounded, read-only directory observation."""

    name = "list"
    concurrency_mode = "read_only"
    description = "List one directory without executing a shell command."
    args_model = ListArgs

    async def execute(
        self,
        path: str | None = None,
        all: bool = False,
        detail: bool = False,
        limit: int = 200,
    ) -> str:
        directory = self._resolve(path)
        if not directory.exists():
            return f"Error: Path not found: {path or '.'}"
        if not directory.is_dir():
            return f"Error: Not a directory: {path or '.'}"
        entries = sorted(
            (
                entry
                for entry in directory.iterdir()
                if (all or not entry.name.startswith(".")) and not self._is_denied(entry)
            ),
            key=lambda entry: (not entry.is_dir(), entry.name.casefold()),
        )
        visible = entries[:limit]
        if not visible:
            return "(empty directory)"
        lines: list[str] = []
        for entry in visible:
            suffix = "/" if entry.is_dir() else ""
            if not detail:
                lines.append(entry.name + suffix)
                continue
            info = entry.lstat()
            mode = stat.filemode(info.st_mode)
            modified = datetime.fromtimestamp(info.st_mtime, tz=UTC).isoformat()
            lines.append(f"{mode} {info.st_size:>10} {modified} {entry.name}{suffix}")
        if len(entries) > len(visible):
            lines.append(f"... {len(entries) - len(visible)} more")
        return "\n".join(lines)


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob pattern, for example **/*.py.")
    root: str | None = Field(
        default=None, description="Optional root directory. Relative paths resolve from workspace."
    )
    limit: int = Field(default=200, ge=1, le=1000)


class GlobTool(WorkspaceTool):
    """Find files by glob pattern."""

    name = "glob"
    concurrency_mode = "read_only"
    description = "Find files matching a glob pattern. Uses workspace as the default root."
    args_model = GlobArgs

    async def execute(
        self,
        pattern: str,
        root: str | None = None,
        limit: int = 200,
    ) -> str:
        pattern_path = Path(pattern)
        try:
            if pattern_path.is_absolute():
                raw_matches = (Path(match) for match in globlib.glob(pattern, recursive=True))
            else:
                raw_matches = self._resolve(root).glob(pattern)
            matches = sorted(
                path
                for path in raw_matches
                if path.exists() and not self._is_denied(path)
            )
        except ValueError as exc:
            return f"Error: invalid glob pattern: {exc}"
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


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regex/text pattern to search for.")
    path: str | None = Field(
        default=None,
        description="Optional file or directory. Relative paths resolve from workspace.",
    )
    include: str | None = Field(
        default=None, description="Optional glob include, for example *.py."
    )
    limit: int = Field(default=200, ge=1, le=1000)


class GrepTool(WorkspaceTool):
    """Search file contents."""

    name = "grep"
    concurrency_mode = "read_only"
    description = "Search text in files. Prefers ripgrep when it is installed."
    args_model = GrepArgs

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
        limit: int = 200,
    ) -> str:
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
        for denied in self.denied_paths:
            if denied.is_relative_to(target):
                relative = globlib.escape(str(denied.relative_to(target)))
                cmd.extend(["--glob", f"!**/{relative}"])
                cmd.extend(["--glob", f"!**/{relative}/**"])
        # Stop option parsing before the model-authored pattern. Without this,
        # values such as --pre could turn a read-only grep into command execution.
        cmd.extend(["--", pattern, str(target)])
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
        regex = re.compile(pattern)
        files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
        matches: list[str] = []
        for file_path in files:
            if self._is_denied(file_path):
                continue
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


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
