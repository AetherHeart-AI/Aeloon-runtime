"""Small read-only filesystem observation for the Master agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from aeloon_core.harness.tool.base import WorkspaceTool


class ReadArgs(BaseModel):
    path: str = Field(description="File path to read.")
    offset: int = Field(default=1, ge=1, description="Line number to start from, 1-indexed.")
    limit: int | None = Field(default=None, ge=1, description="Maximum number of lines.")


class ReadTool(WorkspaceTool):
    """Read bounded UTF-8 text with line numbers."""

    name = "read"
    concurrency_mode = "read_only"
    description = "Read a UTF-8 text file. Returns numbered lines and supports offset/limit."
    args_model = ReadArgs

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2_000

    async def execute(
        self,
        path: str,
        offset: int = 1,
        limit: int | None = None,
    ) -> str:
        try:
            file_path = self._resolve(path)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text: {path}"
        except Exception as exc:
            return f"Error reading file: {exc}"

        total = len(lines)
        if total == 0:
            return f"(Empty file: {path})"
        if offset > total:
            return f"Error: offset {offset} is beyond end of file ({total} lines)"

        start = offset - 1
        requested_end = min(start + (limit or self._DEFAULT_LIMIT), total)
        rendered: list[str] = []
        size = 0
        end = start
        capped = False
        for line_number, line in enumerate(lines[start:requested_end], start=offset):
            item = f"{line_number}| {line}"
            separator = 1 if rendered else 0
            if rendered and size + separator + len(item) > self._MAX_CHARS:
                capped = True
                break
            if not rendered and len(item) > self._MAX_CHARS:
                suffix = "... (line truncated)"
                item = item[: self._MAX_CHARS - len(suffix)] + suffix
                capped = True
            rendered.append(item)
            size += separator + len(item)
            end = line_number
            if capped:
                break

        result = "\n".join(rendered)
        if capped:
            return (
                f"{result}\n\n(Output capped at {self._MAX_CHARS} chars. "
                f"Showing lines {offset}-{end}. Use offset={end + 1}.)"
            )
        if end < total:
            return f"{result}\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1}.)"
        return f"{result}\n\n(End of file - {total} lines total)"


__all__ = ["ReadArgs", "ReadTool"]
