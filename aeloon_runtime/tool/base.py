"""Object-oriented foundations shared by built-in tools."""

from __future__ import annotations

import asyncio
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aeloon_runtime.core.types import ToolResult, ToolUpdateCallback

DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_BYTES = 50 * 1024


@dataclass(slots=True)
class ToolContext:
    """Workspace-scoped state shared only by tools in one tool set."""

    cwd: Path
    _mutation_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    @classmethod
    def create(cls, cwd: Path | str) -> ToolContext:
        return cls(Path(cwd).expanduser().resolve(strict=False))

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        resolved = (path if path.is_absolute() else self.cwd / path).resolve(strict=False)
        if not resolved.is_relative_to(self.cwd):
            raise PermissionError(f"Path is outside the workspace: {value}")
        return resolved

    def relative(self, value: str) -> str:
        """Validate a path and return its canonical workspace-relative spelling."""

        resolved = self.resolve(value)
        relative = resolved.relative_to(self.cwd)
        return str(relative) if relative.parts else "."

    def mutation_lock(self, path: Path) -> asyncio.Lock:
        key = os.path.normcase(str(path))
        return self._mutation_locks.setdefault(key, asyncio.Lock())


class BaseTool(ABC):
    """Convenience base implementing the stable Tool metadata contract."""

    name = ""
    label = ""
    description = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    prompt_snippet = ""
    prompt_guidelines: tuple[str, ...] = ()
    execution_mode: Literal["parallel", "sequential"] = "parallel"

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return dict(arguments)

    @abstractmethod
    async def execute(
        self,
        call_id: str,
        arguments: dict[str, Any],
        on_update: ToolUpdateCallback | None,
    ) -> ToolResult: ...


class WorkspaceTool(BaseTool):
    def __init__(self, context: ToolContext) -> None:
        self.context = context


def object_schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        value["required"] = list(required)
    return value


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    return f"{size / 1024:.1f}KB"


def truncate_head(
    text: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    selected: list[str] = []
    size = 0
    truncated_by: str | None = None
    first_line_exceeds = False
    for line in lines:
        encoded_size = len((line + "\n").encode("utf-8"))
        if not selected and encoded_size > max_bytes:
            first_line_exceeds = True
            truncated_by = "bytes"
            break
        if len(selected) >= max_lines:
            truncated_by = "lines"
            break
        if size + encoded_size > max_bytes:
            truncated_by = "bytes"
            break
        selected.append(line)
        size += encoded_size
    output = "\n".join(selected)
    if text.endswith("\n") and len(selected) == len(lines):
        output += "\n"
    return output, {
        "truncated": truncated_by is not None,
        "truncatedBy": truncated_by,
        "totalLines": len(lines),
        "outputLines": len(selected),
        "maxLines": max_lines,
        "maxBytes": max_bytes,
        "firstLineExceedsLimit": first_line_exceeds,
    }


def truncate_tail(
    text: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    lines = text.splitlines()
    if len(lines) <= max_lines and len(encoded) <= max_bytes:
        return text, False
    selected = lines[-max_lines:]
    while selected and len("\n".join(selected).encode("utf-8")) > max_bytes:
        selected.pop(0)
    return "\n".join(selected), True


def truncate_limited(lines: list[str], limit: int) -> tuple[str, dict[str, Any]]:
    visible, details = truncate_head("\n".join(lines[:limit]))
    details["totalLines"] = len(lines)
    if len(lines) > limit and not details["truncated"]:
        details["truncated"] = True
        details["truncatedBy"] = "lines"
    return visible, details


def atomic_write_bytes(path: Path, value: bytes) -> None:
    """Replace one file atomically without exposing partially-written content."""

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        if path.exists():
            os.fchmod(descriptor, path.stat().st_mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "BaseTool",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "ToolContext",
    "WorkspaceTool",
    "atomic_write_bytes",
]
