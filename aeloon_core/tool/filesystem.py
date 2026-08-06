"""Built-in file reading and mutation tools."""

from __future__ import annotations

import asyncio
import base64
import difflib
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from aeloon_core.core.types import ImageContent, TextContent, ToolResult, ToolUpdateCallback
from aeloon_core.tool.base import (
    DEFAULT_MAX_BYTES,
    ToolContext,
    WorkspaceTool,
    atomic_write_bytes,
    format_size,
    object_schema,
    truncate_head,
)

_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ReadTool(WorkspaceTool):
    name = "read"
    label = "read"
    description = (
        "Read the contents of a file. Supports text files and images "
        "(jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is "
        "truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large "
        "files. When you need the full file, continue with offset until complete."
    )
    prompt_snippet = "Read file contents"
    prompt_guidelines = ("Use read to examine files instead of cat or sed.",)
    parameters = object_schema(
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

    def __init__(self, context: ToolContext, *, auto_resize_images: bool = True) -> None:
        super().__init__(context)
        self.auto_resize_images = auto_resize_images

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        raw_path = str(arguments["path"])
        path = self.context.resolve(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Could not read file: {raw_path}")
        mime = _IMAGE_MIME.get(path.suffix.lower())
        if mime:
            data, processed_mime, hints, size = await asyncio.to_thread(
                _read_image, path, mime, self.auto_resize_images
            )
            note = f"Read image file [{processed_mime}]"
            if hints:
                note += "\n" + "\n".join(hints)
            return ToolResult(
                (TextContent(note), ImageContent(data, processed_mime)),
                details={"path": str(path), "sizeBytes": size, "mimeType": processed_mime},
            )

        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        size = (await asyncio.to_thread(path.stat)).st_size
        all_lines = text.splitlines() or [""]
        offset = int(arguments.get("offset") or 1)
        limit_value = arguments.get("limit")
        limit = int(limit_value) if limit_value is not None else None
        start = max(0, offset - 1)
        if start >= len(all_lines):
            raise ValueError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )
        selected = all_lines[start : start + limit if limit is not None else None]
        output, truncation = truncate_head("\n".join(selected))
        if truncation["firstLineExceedsLimit"]:
            line_size = len(all_lines[start].encode("utf-8"))
            output = (
                f"[Line {start + 1} is {format_size(line_size)}, exceeds "
                f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n "
                f"'{start + 1}p' {raw_path} | head -c {DEFAULT_MAX_BYTES}]"
            )
        elif truncation["truncated"]:
            next_offset = start + int(truncation["outputLines"]) + 1
            output += (
                f"\n\n[Showing lines {start + 1}-{next_offset - 1}. "
                f"Continue with offset={next_offset}.]"
            )
        return ToolResult.text(
            output,
            details={
                "path": str(path),
                "sizeBytes": size,
                "selectedLines": int(truncation["outputLines"]),
                "lineRange": {"start": start + 1, "total": len(all_lines)},
                "truncation": truncation,
            },
        )


class WriteTool(WorkspaceTool):
    name = "write"
    label = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    prompt_snippet = "Create or overwrite files"
    prompt_guidelines = ("Use write only for new files or complete rewrites.",)
    parameters = object_schema(
        {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative or absolute)",
            },
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        ("path", "content"),
    )

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        raw_path = str(arguments["path"])
        content = str(arguments["content"])
        path = self.context.resolve(raw_path)
        async with self.context.mutation_lock(path):
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            encoded = content.encode("utf-8")
            await asyncio.to_thread(atomic_write_bytes, path, encoded)
        return ToolResult.text(
            f"Successfully wrote {len(encoded)} bytes to {raw_path}",
            details={"path": str(path), "sizeBytes": len(encoded)},
        )


class EditTool(WorkspaceTool):
    name = "edit"
    label = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText must match a "
        "unique, non-overlapping region of the original file."
    )
    prompt_snippet = (
        "Make precise file edits with exact text replacement, including multiple "
        "disjoint edits in one call"
    )
    prompt_guidelines = (
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call "
        "with multiple entries in edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after "
        "earlier edits are applied. Do not emit overlapping or nested edits. "
        "Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in "
        "the file. Do not pad with large unchanged regions.",
    )
    parameters = object_schema(
        {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute)",
            },
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": object_schema(
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

    def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        value = dict(arguments)
        edits = value.get("edits")
        if isinstance(edits, str):
            try:
                parsed = json.loads(edits)
                if isinstance(parsed, list):
                    value["edits"] = parsed
            except ValueError:
                pass
        if isinstance(value.get("oldText"), str) and isinstance(value.get("newText"), str):
            existing = list(value.get("edits") or [])
            existing.append({"oldText": value.pop("oldText"), "newText": value.pop("newText")})
            value["edits"] = existing
        return value

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        raw_path = str(arguments["path"])
        edits = arguments.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError(
                "Edit tool input is invalid. edits must contain at least one replacement."
            )
        path = self.context.resolve(raw_path)
        async with self.context.mutation_lock(path):
            raw = await asyncio.to_thread(path.read_bytes)
            size_before = len(raw)
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
            encoded = bom + rendered.encode("utf-8")
            await asyncio.to_thread(atomic_write_bytes, path, encoded)
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
            details={
                "diff": diff,
                "patch": diff,
                "firstChangedLine": first_line,
                "path": str(path),
                "replacements": len(edits),
                "sizeBeforeBytes": size_before,
                "sizeAfterBytes": len(encoded),
            },
        )


def _read_image(path: Path, mime: str, resize: bool) -> tuple[str, str, list[str], int]:
    raw = path.read_bytes()
    original_size = len(raw)
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
    return base64.b64encode(raw).decode("ascii"), mime, hints, original_size


__all__ = ["EditTool", "ReadTool", "WriteTool"]
