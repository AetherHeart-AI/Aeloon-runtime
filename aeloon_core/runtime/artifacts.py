"""Runtime-owned final-deliverable declarations for UI clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aeloon_core.blocking import run_blocking
from aeloon_core.core.types import ToolResult, ToolUpdateCallback
from aeloon_core.tool import BaseTool

PRESENT_FILES_TOOL_NAME = "present_files"
MAX_PRESENTED_FILES = 24

_ARTIFACT_FORMATS: dict[str, tuple[str, str]] = {
    ".bmp": ("image", "image/bmp"),
    ".doc": ("document", "application/msword"),
    ".docx": (
        "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".gif": ("image", "image/gif"),
    ".htm": ("document", "text/html"),
    ".html": ("document", "text/html"),
    ".jpeg": ("image", "image/jpeg"),
    ".jpg": ("image", "image/jpeg"),
    ".markdown": ("document", "text/markdown"),
    ".md": ("document", "text/markdown"),
    ".pdf": ("pdf", "application/pdf"),
    ".png": ("image", "image/png"),
    ".ppt": ("presentation", "application/vnd.ms-powerpoint"),
    ".pptx": (
        "presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ".svg": ("image", "image/svg+xml"),
    ".webp": ("image", "image/webp"),
    ".xls": ("spreadsheet", "application/vnd.ms-excel"),
    ".xlsx": ("spreadsheet", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
}


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    name: str
    mime_type: str
    size_bytes: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PresentFilesTool(BaseTool):
    name = PRESENT_FILES_TOOL_NAME
    label = "present files"
    description = (
        "Declare completed user-facing deliverable files so clients can show them as final "
        "results. Only present the finished office document, PDF, image, Markdown, or HTML "
        "files; never present generator scripts, source code, logs, caches, or temporary files."
    )
    prompt_snippet = "Present verified final deliverable files to the user"
    prompt_guidelines = (
        "When a task produces user-facing files, finish and verify them, then call "
        "present_files exactly once with every final deliverable. Do not include generator "
        "scripts, source code, logs, caches, or temporary files.",
        "After presenting files, keep the final response to a concise delivery summary, "
        "assumptions, and unresolved items. Do not paste generation code or duplicate the "
        "files as local Markdown links unless the user explicitly asks for source code.",
    )
    execution_mode = "sequential"
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PRESENTED_FILES,
                "items": {"type": "string", "minLength": 1},
                "description": "Final user-facing files, relative to the workspace or absolute",
            }
        },
        "required": ["paths"],
        "additionalProperties": False,
    }

    def __init__(self, cwd: Path | str) -> None:
        self.root = Path(cwd).expanduser().resolve(strict=False)

    async def execute(
        self,
        _call_id: str,
        arguments: dict[str, Any],
        _on_update: ToolUpdateCallback | None,
    ) -> ToolResult:
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_PRESENTED_FILES:
            raise ValueError(
                f"paths must contain 1 to {MAX_PRESENTED_FILES} final deliverable files"
            )
        artifacts = await run_blocking(normalize_artifact_paths, self.root, paths)
        rendered = "\n".join(f"- {artifact.path}" for artifact in artifacts)
        return ToolResult.text(
            f"Presented {len(artifacts)} final deliverable file(s):\n{rendered}",
            details={"artifacts": artifacts},
        )


def normalize_artifact_paths(root: Path, values: Sequence[Any]) -> list[Artifact]:
    canonical_root = root.expanduser().resolve(strict=True)
    artifacts: list[Artifact] = []
    seen: set[Path] = set()
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Every presented path must be a non-empty string")
        requested = Path(raw).expanduser()
        candidate = requested if requested.is_absolute() else canonical_root / requested
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise ValueError(f"Presented file does not exist: {raw}") from None
        if not resolved.is_relative_to(canonical_root):
            raise ValueError("Presented files must stay inside the current workspace")
        if not resolved.is_file():
            raise ValueError(f"Presented path is not a file: {raw}")
        format_value = _ARTIFACT_FORMATS.get(resolved.suffix.lower())
        if format_value is None:
            raise ValueError(f"Unsupported deliverable format: {resolved.suffix or '(none)'}")
        if resolved in seen:
            continue
        seen.add(resolved)
        kind, mime_type = format_value
        artifacts.append(
            Artifact(
                path=resolved.relative_to(canonical_root).as_posix(),
                name=resolved.name,
                mime_type=mime_type,
                size_bytes=resolved.stat().st_size,
                kind=kind,
            )
        )
    if not artifacts:
        raise ValueError("At least one unique final deliverable file is required")
    return artifacts


def artifacts_from_tool_result(raw: Any) -> list[Artifact]:
    if not isinstance(raw, Mapping):
        return []
    details = raw.get("details")
    if not isinstance(details, Mapping):
        return []
    values = details.get("artifacts")
    if not isinstance(values, list):
        return []
    artifacts: list[Artifact] = []
    for value in values[:MAX_PRESENTED_FILES]:
        if isinstance(value, Artifact):
            artifacts.append(value)
        elif isinstance(value, Mapping):
            try:
                artifacts.append(
                    Artifact(
                        path=str(value["path"]),
                        name=str(value["name"]),
                        mime_type=str(value["mime_type"]),
                        size_bytes=int(value["size_bytes"]),
                        kind=str(value["kind"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return artifacts


__all__ = [
    "Artifact",
    "MAX_PRESENTED_FILES",
    "PRESENT_FILES_TOOL_NAME",
    "PresentFilesTool",
    "artifacts_from_tool_result",
    "normalize_artifact_paths",
]
