"""Core-owned attachment storage, Office extraction, and model tools."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from aeloon_runtime.blocking import run_blocking
from aeloon_runtime.core import ImageContent, ToolResult
from aeloon_runtime.core.types import ToolUpdateCallback
from aeloon_runtime.runtime.types import RuntimeFailure
from aeloon_runtime.tool import BaseTool

ATTACHMENT_MANIFEST_SCHEMA = "aeloon-session-attachments/v1"
ATTACHMENT_READ_LIMIT = 20_000
OFFICE_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".xlsm"})
_ATTACHMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,10}$")


@dataclass(frozen=True, slots=True)
class AttachmentDescriptor:
    id: str
    type: Literal["file", "image"]
    display_name: str
    mime_type: str
    size_bytes: int
    source_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedAttachment:
    id: str
    type: Literal["file", "image"]
    display_name: str
    mime_type: str
    size_bytes: int
    canonical_path: Path
    ownership: Literal["core_session"] = "core_session"
    lifecycle: Literal["thread"] = "thread"

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "display_name": self.display_name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "ownership": self.ownership,
            "lifecycle": self.lifecycle,
        }

    def runtime_dict(self) -> dict[str, Any]:
        return {**self.public_dict(), "canonical_path": str(self.canonical_path)}


AttachmentAccessCallback = Callable[[str, ResolvedAttachment], Awaitable[None] | None]


class AttachmentStore:
    """Persist immutable session copies and recover their ID mapping after restart."""

    def __init__(self, root: Path | str, *, image_limit: int, file_limit: int) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.image_limit = image_limit
        self.file_limit = file_limit
        self._locks: dict[str, asyncio.Lock] = {}

    async def resolve_batch(
        self,
        session_id: str,
        values: Sequence[Mapping[str, Any]],
        roots: Sequence[Path],
    ) -> tuple[ResolvedAttachment, ...]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await run_blocking(self._resolve_batch, session_id, values, roots)

    async def load(self, session_id: str) -> tuple[ResolvedAttachment, ...]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await run_blocking(self._load, session_id)

    async def remove(self, session_id: str, attachment_ids: Sequence[str]) -> None:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            await run_blocking(self._remove, session_id, set(attachment_ids))

    async def delete_session(self, session_id: str) -> None:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            directory = self._session_dir(session_id)
            if directory.exists():
                await run_blocking(shutil.rmtree, directory)
        self._locks.pop(session_id, None)

    def cleanup_orphans(self, valid_session_ids: set[str]) -> tuple[str, ...]:
        """Remove stable-copy directories that no longer have a persisted session."""

        removed: list[str] = []
        for candidate in self.root.iterdir():
            if (
                not candidate.is_dir()
                or not _ATTACHMENT_ID.fullmatch(candidate.name)
                or candidate.name in valid_session_ids
            ):
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(self.root) or resolved.parent != self.root:
                continue
            shutil.rmtree(resolved)
            removed.append(candidate.name)
        return tuple(sorted(removed))

    def _resolve_batch(
        self,
        session_id: str,
        values: Sequence[Mapping[str, Any]],
        roots: Sequence[Path],
    ) -> tuple[ResolvedAttachment, ...]:
        session_dir = self._session_dir(session_id)
        files_dir = session_dir / "files"
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        files_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        existing = {item.id: item for item in self._load(session_id)}
        resolved_roots: tuple[Path, ...] = tuple(
            root.expanduser().resolve(strict=True) for root in roots if root.exists()
        )
        descriptors: list[AttachmentDescriptor] = []
        batch_ids: set[str] = set()
        for raw in values:
            descriptor = self._descriptor(raw, resolved_roots)
            if descriptor.id in batch_ids or descriptor.id in existing:
                raise RuntimeFailure(
                    "invalid_attachment",
                    f"Attachment id is already used in this session: {descriptor.id}",
                )
            batch_ids.add(descriptor.id)
            descriptors.append(descriptor)

        staging = session_dir / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        created: list[Path] = []
        result: list[ResolvedAttachment] = []
        try:
            for descriptor in descriptors:
                suffix = descriptor.source_path.suffix.lower()
                safe_suffix = suffix if _SAFE_SUFFIX.fullmatch(suffix) else ""
                storage_name = f"{uuid.uuid4().hex}{safe_suffix}"
                staged = staging / storage_name
                shutil.copyfile(descriptor.source_path, staged)
                staged.chmod(0o600)
                if staged.stat().st_size != descriptor.size_bytes:
                    raise RuntimeFailure(
                        "invalid_attachment", "Attachment changed while Core was copying it"
                    )
                target = files_dir / storage_name
                os.replace(staged, target)
                created.append(target)
                result.append(
                    ResolvedAttachment(
                        id=descriptor.id,
                        type=descriptor.type,
                        display_name=descriptor.display_name,
                        mime_type=descriptor.mime_type,
                        size_bytes=descriptor.size_bytes,
                        canonical_path=target.resolve(strict=True),
                    )
                )
            merged = (*existing.values(), *result)
            self._write_manifest(session_dir, merged)
            return tuple(result)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _descriptor(
        self, raw: Mapping[str, Any], resolved_roots: Sequence[Path]
    ) -> AttachmentDescriptor:
        attachment_id = str(raw.get("id") or "")
        if not _ATTACHMENT_ID.fullmatch(attachment_id):
            raise RuntimeFailure("invalid_attachment", "Attachment id is invalid")
        kind = str(raw.get("type") or "")
        if kind not in {"file", "image"}:
            raise RuntimeFailure("invalid_attachment", f"Unsupported attachment type: {kind}")
        display_name = str(raw.get("display_name") or "").strip()
        if not display_name or Path(display_name).name != display_name:
            raise RuntimeFailure("invalid_attachment", "Attachment display_name is invalid")
        mime_type = str(raw.get("mime_type") or "").strip().lower()
        if not mime_type or "/" not in mime_type:
            raise RuntimeFailure("invalid_attachment", "Attachment mime_type is invalid")
        source_raw = raw.get("source_path")
        if not isinstance(source_raw, str) or not source_raw:
            raise RuntimeFailure("invalid_attachment", "Attachment source_path is required")
        try:
            source = Path(source_raw).expanduser().resolve(strict=True)
        except OSError:
            raise RuntimeFailure("invalid_attachment", "Attachment source does not exist") from None
        if not source.is_file() or not any(source.is_relative_to(root) for root in resolved_roots):
            raise RuntimeFailure(
                "invalid_attachment", "Attachment is outside the declared attachment roots"
            )
        actual_size = source.stat().st_size
        declared_size = raw.get("size_bytes")
        if not isinstance(declared_size, int) or declared_size != actual_size:
            raise RuntimeFailure(
                "invalid_attachment", "Attachment size_bytes does not match the source file"
            )
        maximum = self.image_limit if kind == "image" else self.file_limit
        if actual_size > maximum:
            raise RuntimeFailure(
                "invalid_attachment",
                f"Attachment exceeds the {maximum // (1024 * 1024)} MiB limit",
            )
        if kind == "image" and not mime_type.startswith("image/"):
            raise RuntimeFailure("invalid_attachment", "Image attachment has a non-image MIME type")
        return AttachmentDescriptor(
            id=attachment_id,
            type=kind,  # type: ignore[arg-type]
            display_name=display_name,
            mime_type=mime_type,
            size_bytes=actual_size,
            source_path=source,
        )

    def _load(self, session_id: str) -> tuple[ResolvedAttachment, ...]:
        session_dir = self._session_dir(session_id)
        manifest = session_dir / "manifest.json"
        if not manifest.exists():
            return ()
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeFailure(
                "invalid_attachment", "Session attachment manifest is corrupt"
            ) from exc
        if not isinstance(value, Mapping) or value.get("schema") != ATTACHMENT_MANIFEST_SCHEMA:
            raise RuntimeFailure("invalid_attachment", "Session attachment manifest is invalid")
        raw_items = value.get("attachments")
        if not isinstance(raw_items, list):
            raise RuntimeFailure("invalid_attachment", "Session attachment manifest is invalid")
        result: list[ResolvedAttachment] = []
        seen: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise RuntimeFailure("invalid_attachment", "Session attachment manifest is invalid")
            attachment_id = str(raw.get("id") or "")
            storage_name = str(raw.get("storage_name") or "")
            if (
                not _ATTACHMENT_ID.fullmatch(attachment_id)
                or attachment_id in seen
                or Path(storage_name).name != storage_name
            ):
                raise RuntimeFailure("invalid_attachment", "Session attachment manifest is invalid")
            seen.add(attachment_id)
            canonical = (session_dir / "files" / storage_name).resolve(strict=True)
            if (
                not canonical.is_relative_to(session_dir.resolve(strict=True))
                or not canonical.is_file()
            ):
                raise RuntimeFailure("invalid_attachment", "Session attachment file is missing")
            size_bytes = int(raw.get("size_bytes") or -1)
            if canonical.stat().st_size != size_bytes:
                raise RuntimeFailure("invalid_attachment", "Session attachment file size changed")
            kind = str(raw.get("type") or "")
            if kind not in {"file", "image"}:
                raise RuntimeFailure("invalid_attachment", "Session attachment manifest is invalid")
            result.append(
                ResolvedAttachment(
                    id=attachment_id,
                    type=kind,  # type: ignore[arg-type]
                    display_name=str(raw.get("display_name") or ""),
                    mime_type=str(raw.get("mime_type") or "application/octet-stream"),
                    size_bytes=size_bytes,
                    canonical_path=canonical,
                )
            )
        return tuple(result)

    def _remove(self, session_id: str, attachment_ids: set[str]) -> None:
        if not attachment_ids:
            return
        session_dir = self._session_dir(session_id)
        remaining: list[ResolvedAttachment] = []
        for attachment in self._load(session_id):
            if attachment.id in attachment_ids:
                attachment.canonical_path.unlink(missing_ok=True)
            else:
                remaining.append(attachment)
        self._write_manifest(session_dir, remaining)

    def _write_manifest(
        self, session_dir: Path, attachments: Sequence[ResolvedAttachment]
    ) -> None:
        files_dir = session_dir / "files"
        payload = {
            "schema": ATTACHMENT_MANIFEST_SCHEMA,
            "attachments": [
                {
                    **item.public_dict(),
                    "storage_name": item.canonical_path.relative_to(files_dir).as_posix(),
                }
                for item in attachments
            ],
        }
        temporary = session_dir / f".manifest-{uuid.uuid4().hex}.json"
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.chmod(0o600)
            os.replace(temporary, session_dir / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        if not _ATTACHMENT_ID.fullmatch(session_id):
            raise RuntimeFailure("invalid_argument", "Session id is invalid")
        return self.root / session_id


class OfficeLiteService:
    """Call the bundled Office Lite implementation in-process."""

    def __init__(self) -> None:
        self._module: ModuleType | None = None

    async def read(self, path: Path) -> tuple[str, dict[str, Any]]:
        return await run_blocking(self._load().read_document, path)

    async def render(self, path: Path, *, dpi: int = 120) -> tuple[ImageContent, ...]:
        def execute() -> tuple[ImageContent, ...]:
            with tempfile.TemporaryDirectory(prefix="aeloon-office-attachment-") as temporary:
                outputs = self._load().render_document(
                    path, Path(temporary), dpi=dpi, overwrite=True
                )
                return tuple(
                    ImageContent(base64.b64encode(output.read_bytes()).decode(), "image/png")
                    for output in outputs
                )

        return await run_blocking(execute)

    def _load(self) -> ModuleType:
        if self._module is not None:
            return self._module
        script = (
            Path(__file__).parents[1]
            / "resources"
            / "skills"
            / "aeloon-office-lite"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("aeloon_runtime_office_lite", script)
        if spec is None or spec.loader is None:
            raise RuntimeFailure("attachment_processing_failed", "Office Lite is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        return module


class AttachmentMetadataTool(BaseTool):
    name = "attachment_metadata"
    label = "attachment metadata"
    description = "Read trusted metadata for a user attachment by attachment ID."
    prompt_snippet = "Inspect user attachments only by attachment ID"
    prompt_guidelines = (
        "Never construct a workspace path from an attachment display name. Use attachment IDs.",
    )
    parameters = {
        "type": "object",
        "properties": {"attachment_id": {"type": "string", "minLength": 1}},
        "required": ["attachment_id"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        attachments: Mapping[str, ResolvedAttachment],
        on_access: AttachmentAccessCallback | None = None,
    ) -> None:
        self.attachments = attachments
        self.on_access = on_access

    async def execute(
        self,
        _call_id: str,
        arguments: dict[str, Any],
        _on_update: ToolUpdateCallback | None,
    ) -> ToolResult:
        attachment = self._attachment(arguments)
        await _notify(self.on_access, "metadata", attachment)
        return ToolResult.text(
            json.dumps(attachment.public_dict(), ensure_ascii=False),
            details={"attachment_id": attachment.id},
        )

    def _attachment(self, arguments: Mapping[str, Any]) -> ResolvedAttachment:
        attachment_id = str(arguments.get("attachment_id") or "")
        attachment = self.attachments.get(attachment_id)
        if attachment is None:
            raise ValueError(f"Unknown attachment id: {attachment_id}")
        return attachment


class AttachmentReadTool(AttachmentMetadataTool):
    name = "attachment_read"
    label = "read attachment"
    description = "Read a non-Office UTF-8 user attachment by attachment ID."
    prompt_snippet = "Read non-Office text attachments only by attachment ID"

    async def execute(
        self,
        _call_id: str,
        arguments: dict[str, Any],
        _on_update: ToolUpdateCallback | None,
    ) -> ToolResult:
        attachment = self._attachment(arguments)
        await _notify(self.on_access, "read", attachment)
        if (
            attachment.type == "image"
            or attachment.canonical_path.suffix.lower() in OFFICE_SUFFIXES
        ):
            return ToolResult.text(
                json.dumps(
                    {
                        "error": "unsupported_attachment",
                        "attachment_id": attachment.id,
                        "message": (
                            "This attachment is binary or already handled by Core Office "
                            "processing."
                        ),
                    },
                    ensure_ascii=False,
                ),
                details={"attachment_id": attachment.id},
                is_error=True,
            )
        try:
            content = await run_blocking(attachment.canonical_path.read_text, encoding="utf-8")
        except UnicodeError:
            return ToolResult.text(
                json.dumps(
                    {
                        "error": "unsupported_attachment",
                        "attachment_id": attachment.id,
                        "message": "Attachment is not UTF-8 text.",
                    },
                    ensure_ascii=False,
                ),
                details={"attachment_id": attachment.id},
                is_error=True,
            )
        if len(content) > ATTACHMENT_READ_LIMIT:
            return ToolResult.text(
                json.dumps(
                    {
                        "error": "attachment_content_too_large",
                        "attachment_id": attachment.id,
                        "limit_chars": ATTACHMENT_READ_LIMIT,
                    },
                    ensure_ascii=False,
                ),
                details={"attachment_id": attachment.id},
                is_error=True,
            )
        return ToolResult.text(content, details={"attachment_id": attachment.id})


async def _notify(
    callback: AttachmentAccessCallback | None,
    action: str,
    attachment: ResolvedAttachment,
) -> None:
    if callback is None:
        return
    result = callback(action, attachment)
    if asyncio.iscoroutine(result):
        await result


def missing_pdf_pages(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    pages: list[int] = []
    warnings = metadata.get("warnings")
    if not isinstance(warnings, list):
        return ()
    for warning in warnings:
        if not isinstance(warning, Mapping) or warning.get("code") != "pages_without_text":
            continue
        values = warning.get("pages")
        if isinstance(values, list):
            pages.extend(int(value) for value in values if isinstance(value, int) and value > 0)
    return tuple(dict.fromkeys(pages))


__all__ = [
    "AttachmentDescriptor",
    "AttachmentMetadataTool",
    "AttachmentReadTool",
    "AttachmentStore",
    "OFFICE_SUFFIXES",
    "OfficeLiteService",
    "ResolvedAttachment",
    "missing_pdf_pages",
]
