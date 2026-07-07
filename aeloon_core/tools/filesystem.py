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
            requested_end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered: list[str] = []
            output_chars = 0
            capped = False
            end = start

            for index, line in enumerate(lines[start:requested_end], start=start + 1):
                rendered = f"{index}| {line}"
                separator_chars = 1 if numbered else 0
                if numbered and output_chars + separator_chars + len(rendered) > self._MAX_CHARS:
                    capped = True
                    break
                if not numbered and len(rendered) > self._MAX_CHARS:
                    suffix = "... (line truncated)"
                    rendered = rendered[: self._MAX_CHARS - len(suffix)] + suffix
                    capped = True
                numbered.append(rendered)
                output_chars += separator_chars + len(rendered)
                end = index
                if capped:
                    break

            result = "\n".join(numbered)
            if capped:
                result += (
                    f"\n\n(Output capped at {self._MAX_CHARS} chars. "
                    f"Showing lines {offset}-{end}. Use offset={end + 1}.)"
                )
            elif end < total:
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
    description = (
        "Create a UTF-8 text file or intentionally overwrite one. Prefer read+edit for "
        "existing files. Large writes must include end_marker and append that marker as "
        "the final characters of content; the marker is stripped before writing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write."},
            "content": {
                "type": "string",
                "description": "Complete content to write. For large files, end with end_marker.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Set true only when intentionally replacing an existing file.",
            },
            "end_marker": {
                "type": "string",
                "description": (
                    "Optional completion marker. Required for large content. If set, content "
                    "must end with this exact marker and the marker is not written to disk."
                ),
                "minLength": 8,
                "maxLength": 128,
            },
        },
        "required": ["path", "content"],
    }

    _LARGE_CONTENT_CHARS = 16_000

    async def execute(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        end_marker: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            fp = self._resolve(path)
            if fp.exists() and fp.is_dir():
                return f"Error: Cannot write file because path is a directory: {path}"
            if fp.exists() and not overwrite:
                return (
                    f"Error: File already exists: {path}. Use the edit tool for existing "
                    "files, or retry write with overwrite=true if full replacement is intentional."
                )

            content_to_write = content
            if len(content) > self._LARGE_CONTENT_CHARS and not end_marker:
                return (
                    f"Error: Refusing large write of {len(content)} chars without end_marker. "
                    "For large generated files, prefer a small skeleton plus edit calls. "
                    "If a full write is intentional, pass end_marker and append it as the "
                    "final characters of content so truncated writes are rejected."
                )
            if end_marker:
                if not content.endswith(end_marker):
                    return (
                        "Error: content does not end with end_marker; refusing to write because "
                        "the model output may have been truncated. Retry by continuing the file "
                        "or use a smaller write/edit."
                    )
                content_to_write = content[: -len(end_marker)]

            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content_to_write, encoding="utf-8")
            return f"Successfully wrote {len(content_to_write)} chars to {fp}"
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
