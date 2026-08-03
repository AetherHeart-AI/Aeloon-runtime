"""Pi-compatible local coding tools implemented in Python."""

from __future__ import annotations

import asyncio
import base64
import difflib
import io
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from PIL import Image

from aeloon_core.harness.types import (
    AgentTool,
    ImageContent,
    TextContent,
    ToolResult,
)

DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_BYTES = 50 * 1024
DEFAULT_ACTIVE_TOOLS = ("read", "bash", "edit", "write")
ALL_TOOL_NAMES = frozenset((*DEFAULT_ACTIVE_TOOLS, "grep", "find", "ls"))
_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MUTATION_LOCKS: dict[str, asyncio.Lock] = {}


def _object_schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        value["required"] = list(required)
    return value


def _resolve(cwd: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else cwd / path).resolve(strict=False)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    return f"{size / 1024:.1f}KB"


def _truncate_head(
    text: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
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


def _truncate_tail(
    text: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    lines = text.splitlines()
    if len(lines) <= max_lines and len(encoded) <= max_bytes:
        return text, False
    selected = lines[-max_lines:]
    while selected and len("\n".join(selected).encode("utf-8")) > max_bytes:
        selected.pop(0)
    return "\n".join(selected), True


def _truncate_limited(lines: list[str], limit: int) -> tuple[str, dict[str, Any]]:
    visible, details = _truncate_head("\n".join(lines[:limit]))
    details["totalLines"] = len(lines)
    if len(lines) > limit and not details["truncated"]:
        details["truncated"] = True
        details["truncatedBy"] = "lines"
    return visible, details


def _mutation_lock(path: Path) -> asyncio.Lock:
    key = os.path.normcase(str(path))
    return _MUTATION_LOCKS.setdefault(key, asyncio.Lock())


def create_read_tool(cwd: Path, *, auto_resize_images: bool = True) -> AgentTool:
    schema = _object_schema(
        {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute)",
            },
            "offset": {
                "type": "number",
                "minimum": 1,
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "number",
                "minimum": 1,
                "description": "Maximum number of lines to read",
            },
        },
        ("path",),
    )

    async def execute(_call_id: str, params: dict[str, Any], _on_update: Any) -> ToolResult:
        raw_path = str(params["path"])
        path = _resolve(cwd, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Could not read file: {raw_path}")
        mime = _IMAGE_MIME.get(path.suffix.lower())
        if mime:
            data, processed_mime, hints = await asyncio.to_thread(
                _read_image, path, mime, auto_resize_images
            )
            note = f"Read image file [{processed_mime}]"
            if hints:
                note += "\n" + "\n".join(hints)
            return ToolResult((TextContent(note), ImageContent(data, processed_mime)))

        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        all_lines = text.splitlines() or [""]
        offset = int(params.get("offset") or 1)
        limit_value = params.get("limit")
        limit = int(limit_value) if limit_value is not None else None
        start = max(0, offset - 1)
        if start >= len(all_lines):
            raise ValueError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )
        selected = all_lines[start : start + limit if limit is not None else None]
        output, truncation = _truncate_head("\n".join(selected))
        if truncation["firstLineExceedsLimit"]:
            line_size = len(all_lines[start].encode("utf-8"))
            output = (
                f"[Line {start + 1} is {_format_size(line_size)}, exceeds "
                f"{_format_size(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n "
                f"'{start + 1}p' {raw_path} | head -c {DEFAULT_MAX_BYTES}]"
            )
        elif truncation["truncated"]:
            next_offset = start + int(truncation["outputLines"]) + 1
            output += (
                f"\n\n[Showing lines {start + 1}-{next_offset - 1}. "
                f"Continue with offset={next_offset}.]"
            )
        details = {
            "path": str(path),
            "lineRange": {"start": start + 1, "total": len(all_lines)},
            "truncation": truncation,
        }
        return ToolResult.text(output, details=details)

    return AgentTool(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images "
            "(jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is "
            "truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large "
            "files. When you need the full "
            "file, continue with offset until complete."
        ),
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of cat or sed.",),
        parameters=schema,
        execute=execute,
    )


def _read_image(path: Path, mime: str, resize: bool) -> tuple[str, str, list[str]]:
    raw = path.read_bytes()
    hints: list[str] = []
    if resize:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if max(width, height) > 2_000:
                image.thumbnail((2_000, 2_000))
                output = io.BytesIO()
                target_format = "JPEG" if mime == "image/jpeg" else "PNG"
                if target_format == "JPEG" and image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                image.save(output, format=target_format, optimize=True)
                raw = output.getvalue()
                mime = "image/jpeg" if target_format == "JPEG" else "image/png"
                hints.append(
                    f"Image resized from {width}x{height} to {image.width}x{image.height}."
                )
    return base64.b64encode(raw).decode("ascii"), mime, hints


def create_write_tool(cwd: Path) -> AgentTool:
    schema = _object_schema(
        {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative or absolute)",
            },
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        ("path", "content"),
    )

    async def execute(_call_id: str, params: dict[str, Any], _on_update: Any) -> ToolResult:
        raw_path = str(params["path"])
        content = str(params["content"])
        path = _resolve(cwd, raw_path)
        async with _mutation_lock(path):
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            encoded = content.encode("utf-8")
            await asyncio.to_thread(path.write_bytes, encoded)
        size = len(encoded)
        return ToolResult.text(f"Successfully wrote {size} bytes to {raw_path}")

    return AgentTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
        parameters=schema,
        execute=execute,
    )


def _prepare_edit_arguments(params: dict[str, Any]) -> dict[str, Any]:
    value = dict(params)
    edits = value.get("edits")
    if isinstance(edits, str):
        try:
            parsed = __import__("json").loads(edits)
            if isinstance(parsed, list):
                value["edits"] = parsed
        except ValueError:
            pass
    if isinstance(value.get("oldText"), str) and isinstance(value.get("newText"), str):
        existing = list(value.get("edits") or [])
        existing.append({"oldText": value.pop("oldText"), "newText": value.pop("newText")})
        value["edits"] = existing
    return value


def create_edit_tool(cwd: Path) -> AgentTool:
    schema = _object_schema(
        {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute)",
            },
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": _object_schema(
                    {
                        "oldText": {
                            "type": "string",
                            "description": "Exact unique text to replace",
                        },
                        "newText": {"type": "string", "description": "Replacement text"},
                    },
                    ("oldText", "newText"),
                ),
            },
        },
        ("path", "edits"),
    )

    async def execute(_call_id: str, params: dict[str, Any], _on_update: Any) -> ToolResult:
        params = _prepare_edit_arguments(params)
        raw_path = str(params["path"])
        edits = params.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError(
                "Edit tool input is invalid. edits must contain at least one replacement."
            )
        path = _resolve(cwd, raw_path)
        async with _mutation_lock(path):
            raw = await asyncio.to_thread(path.read_bytes)
            bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
            decoded = raw[len(bom) :].decode("utf-8")
            line_ending = "\r\n" if "\r\n" in decoded else "\n"
            original = decoded.replace("\r\n", "\n").replace("\r", "\n")
            replacements: list[tuple[int, int, str]] = []
            for edit in edits:
                old = str(edit.get("oldText", "")).replace("\r\n", "\n").replace("\r", "\n")
                new = str(edit.get("newText", "")).replace("\r\n", "\n").replace("\r", "\n")
                if not old:
                    raise ValueError("edits[].oldText must not be empty")
                matches = [match.start() for match in re.finditer(re.escape(old), original)]
                if len(matches) != 1:
                    raise ValueError(
                        f"Could not edit {raw_path}: oldText must match exactly once "
                        f"(found {len(matches)} matches)."
                    )
                replacements.append((matches[0], matches[0] + len(old), new))
            ordered = sorted(replacements)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False)):
                raise ValueError("Could not edit file: edits contain overlapping regions")
            changed = original
            for start, end, new in reversed(ordered):
                changed = changed[:start] + new + changed[end:]
            rendered = changed.replace("\n", line_ending)
            await asyncio.to_thread(path.write_bytes, bom + rendered.encode("utf-8"))
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                changed.splitlines(keepends=True),
                fromfile=raw_path,
                tofile=raw_path,
            )
        )
        first_line = min((original.count("\n", 0, start) + 1 for start, _, _ in ordered), default=1)
        return ToolResult.text(
            f"Successfully replaced {len(edits)} block(s) in {raw_path}.",
            details={"diff": diff, "patch": diff, "firstChangedLine": first_line},
        )

    return AgentTool(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a "
            "unique, non-overlapping region of the original file."
        ),
        prompt_snippet=(
            "Make precise file edits with exact text replacement, including multiple "
            "disjoint edits in one call"
        ),
        prompt_guidelines=(
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call "
            "with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after "
            "earlier edits are applied. Do not emit overlapping or nested edits. "
            "Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in "
            "the file. Do not pad with large unchanged regions.",
        ),
        parameters=schema,
        execute=execute,
        prepare_arguments=_prepare_edit_arguments,
    )


def create_bash_tool(cwd: Path, *, shell_path: str | None = None) -> AgentTool:
    executable = shell_path or os.environ.get("SHELL") or "/bin/bash"
    schema = _object_schema(
        {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)",
            },
        },
        ("command",),
    )

    async def execute(_call_id: str, params: dict[str, Any], on_update: Any) -> ToolResult:
        command = str(params["command"])
        timeout = params.get("timeout")
        if timeout is not None and (
            not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
            or timeout > 2_147_483.647
        ):
            raise ValueError("Invalid timeout: must be a positive finite number of seconds")
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            executable=executable,
            env=dict(os.environ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        chunks: list[str] = []

        async def pump(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                data = await stream.read(8_192)
                if not data:
                    return
                text = data.decode("utf-8", errors="replace")
                chunks.append(text)
                if on_update is not None:
                    updated = on_update(ToolResult.text(text))
                    if isinstance(updated, Awaitable):
                        await updated

        pumps = [
            asyncio.create_task(pump(process.stdout)),
            asyncio.create_task(pump(process.stderr)),
        ]
        cancelled = False
        try:
            if timeout is None:
                await process.wait()
            else:
                await asyncio.wait_for(process.wait(), float(timeout))
        except TimeoutError:
            process.kill()
            await process.wait()
            await asyncio.gather(*pumps)
            raise TimeoutError(f"Command timed out after {timeout} seconds") from None
        except asyncio.CancelledError:
            cancelled = True
            process.kill()
            await process.wait()
            raise
        finally:
            if not all(task.done() for task in pumps):
                await asyncio.gather(*pumps, return_exceptions=True)
        output = "".join(chunks).rstrip("\n")
        visible, truncated = _truncate_tail(output)
        full_output_path: str | None = None
        if truncated:
            directory = Path(tempfile.gettempdir()) / "aeloon-core"
            directory.mkdir(parents=True, exist_ok=True)
            full_path = directory / f"bash-{uuid.uuid4().hex}.log"
            full_path.write_text(output, encoding="utf-8")
            full_output_path = str(full_path)
        rendered = visible or "(no output)"
        if cancelled:
            rendered += "\n\n(command cancelled)"
        elif process.returncode:
            rendered += f"\n\nCommand exited with code {process.returncode}"
        if full_output_path:
            rendered += f"\n\n[Output truncated. Full output: {full_output_path}]"
        return ToolResult.text(
            rendered,
            details={
                "command": command,
                "output": visible,
                "exitCode": process.returncode,
                "cancelled": cancelled,
                "truncated": truncated,
                "fullOutputPath": full_output_path,
            },
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns combined output, "
            "truncated to the last 2000 lines or 50KB."
        ),
        prompt_snippet="Execute shell commands",
        parameters=schema,
        execute=execute,
    )


async def _run_rg(arguments: list[str], cwd: Path) -> tuple[str, int]:
    executable = shutil.which("rg")
    if executable is None:
        raise FileNotFoundError("ripgrep (rg) is required for this tool")
    process = await asyncio.create_subprocess_exec(
        executable,
        *arguments,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    output = stdout.decode("utf-8", errors="replace")
    if process.returncode not in {0, 1}:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or "rg failed")
    return output, int(process.returncode or 0)


def create_grep_tool(cwd: Path) -> AgentTool:
    schema = _object_schema(
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

    async def execute(_call_id: str, params: dict[str, Any], _on_update: Any) -> ToolResult:
        arguments = ["--line-number", "--color=never"]
        if params.get("ignoreCase"):
            arguments.append("--ignore-case")
        if params.get("literal"):
            arguments.append("--fixed-strings")
        if params.get("glob"):
            arguments.extend(["--glob", str(params["glob"])])
        if params.get("context") is not None:
            arguments.extend(["--context", str(int(params["context"]))])
        arguments.extend([str(params["pattern"]), str(params.get("path") or ".")])
        output, _ = await _run_rg(arguments, cwd)
        limit = max(1, int(params.get("limit") or 100))
        visible, truncation = _truncate_limited(output.splitlines(), limit)
        return ToolResult.text(visible or "No matches found", details={"truncation": truncation})

    return AgentTool(
        name="grep",
        label="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with file paths "
            "and line numbers. Respects .gitignore. Output is truncated to 100 matches "
            "or 50KB."
        ),
        prompt_snippet="Search file contents for patterns (respects .gitignore)",
        parameters=schema,
        execute=execute,
    )


def create_find_tool(cwd: Path) -> AgentTool:
    schema = _object_schema(
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

    async def execute(_call_id: str, params: dict[str, Any], _on_update: Any) -> ToolResult:
        base = str(params.get("path") or ".")
        output, _ = await _run_rg(["--files", "--glob", str(params["pattern"]), base], cwd)
        limit = max(1, int(params.get("limit") or 1_000))
        visible, truncation = _truncate_limited(output.splitlines(), limit)
        return ToolResult.text(visible or "No files found", details={"truncation": truncation})

    return AgentTool(
        name="find",
        label="find",
        description=(
            "Search for files by glob pattern. Returns matching paths and respects "
            ".gitignore. Output is truncated to 1000 results or 50KB."
        ),
        prompt_snippet="Find files by glob pattern (respects .gitignore)",
        parameters=schema,
        execute=execute,
    )


def create_ls_tool(cwd: Path) -> AgentTool:
    schema = _object_schema(
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

    async def execute(_call_id: str, params: dict[str, Any], _on_update: Any) -> ToolResult:
        path = _resolve(cwd, str(params.get("path") or "."))
        limit = max(1, int(params.get("limit") or 500))
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
        lines = [item.name + ("/" if item.is_dir() else "") for item in entries]
        visible, truncation = _truncate_limited(lines, limit)
        return ToolResult.text(visible or "(empty directory)", details={"truncation": truncation})

    return AgentTool(
        name="ls",
        label="ls",
        description=(
            "List directory contents sorted alphabetically, including dotfiles and '/' "
            "suffixes for directories. Output is truncated to 500 entries or 50KB."
        ),
        prompt_snippet="List directory contents",
        parameters=schema,
        execute=execute,
    )


def create_all_tools(
    cwd: Path | str,
    *,
    shell_path: str | None = None,
    auto_resize_images: bool = True,
) -> dict[str, AgentTool]:
    root = Path(cwd).expanduser().resolve(strict=False)
    tools = (
        create_read_tool(root, auto_resize_images=auto_resize_images),
        create_bash_tool(root, shell_path=shell_path),
        create_edit_tool(root),
        create_write_tool(root),
        create_grep_tool(root),
        create_find_tool(root),
        create_ls_tool(root),
    )
    return {tool.name: tool for tool in tools}


__all__ = [
    "ALL_TOOL_NAMES",
    "DEFAULT_ACTIVE_TOOLS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "create_all_tools",
    "create_bash_tool",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_tool",
    "create_write_tool",
]
