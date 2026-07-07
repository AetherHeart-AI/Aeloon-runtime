"""Filesystem tools: read, write, edit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeloon_core.tools.base import Tool


class _WorkspaceTool(Tool):
    def __init__(self, *, workspace: Path) -> None:
        self.workspace = workspace

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve(strict=False)


class ReadTool(_WorkspaceTool):
    """Read file contents with line-numbered output."""

    name = "read"
    concurrency_mode = "read_only"
    description = "Read a UTF-8 text file. Returns numbered lines and supports offset/limit."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "offset": {
                "type": "integer",
                "description": "Line number to start from, 1-indexed.",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines.",
                "minimum": 1,
            },
        },
        "required": ["path"],
    }

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2_000

    async def execute(
        self,
        path: str,
        offset: int = 1,
        limit: int | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"
            lines = fp.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            if total == 0:
                return f"(Empty file: {path})"
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"
            start = max(offset, 1) - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [
                f"{start + index + 1}| {line}" for index, line in enumerate(lines[start:end])
            ]
            result = "\n".join(numbered)
            if len(result) > self._MAX_CHARS:
                result = result[: self._MAX_CHARS] + "\n\n... truncated ..."
            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1}.)"
            else:
                result += f"\n\n(End of file - {total} lines total)"
            return result
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text: {path}"
        except Exception as exc:
            return f"Error reading file: {exc}"


class WriteTool(_WorkspaceTool):
    """Write text content to a file."""

    name = "write"
    concurrency_mode = "mutating"
    description = "Write UTF-8 content to a file. Creates parent directories when needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        del kwargs
        try:
            fp = self._resolve(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {fp}"
        except Exception as exc:
            return f"Error writing file: {exc}"


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    if old_text in content:
        return old_text, content.count(old_text)
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates: list[str] = []
    for index in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[index : index + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))
    if candidates:
        return candidates[0], len(candidates)
    return None, 0


class EditTool(_WorkspaceTool):
    """Edit a file by replacing text."""

    name = "edit"
    concurrency_mode = "mutating"
    description = (
        "Edit a UTF-8 text file by replacing old_text with new_text. "
        "Set replace_all=true to replace every occurrence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to edit."},
            "old_text": {"type": "string", "description": "Text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of just one.",
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            match, count = _find_match(content, old_text.replace("\r\n", "\n"))
            if match is None:
                return f"Error: old_text not found in {path}"
            if count > 1 and not replace_all:
                return (
                    f"Warning: old_text appears {count} times. Provide more context "
                    "or set replace_all=true."
                )
            new_content = (
                content.replace(match, new_text.replace("\r\n", "\n"))
                if replace_all
                else content.replace(match, new_text.replace("\r\n", "\n"), 1)
            )
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")
            fp.write_bytes(new_content.encode("utf-8"))
            return f"Successfully edited {fp}"
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text: {path}"
        except Exception as exc:
            return f"Error editing file: {exc}"
