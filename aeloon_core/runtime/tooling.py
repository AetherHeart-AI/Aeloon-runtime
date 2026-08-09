"""Explicit composition of built-in and runtime-owned tools."""

from __future__ import annotations

from pathlib import Path

from aeloon_core.browser.protocol import BrowserContext
from aeloon_core.browser.tools import BROWSER_TOOL_NAMES, BrowserToolSet
from aeloon_core.core.types import Tool
from aeloon_core.runtime.artifacts import PRESENT_FILES_TOOL_NAME, PresentFilesTool
from aeloon_core.runtime.attachments import (
    AttachmentAccessCallback,
    AttachmentMetadataTool,
    AttachmentReadTool,
    ResolvedAttachment,
)
from aeloon_core.tool import BuiltinToolSet

ATTACHMENT_TOOL_NAMES = ("attachment_read", "attachment_metadata")


class RuntimeToolSet:
    required_names = (PRESENT_FILES_TOOL_NAME,)

    def __init__(
        self,
        cwd: Path | str,
        *,
        shell_path: str | None = None,
        auto_resize_images: bool = True,
        browser_context: BrowserContext | None = None,
        attachments: tuple[ResolvedAttachment, ...] = (),
        on_attachment_access: AttachmentAccessCallback | None = None,
    ) -> None:
        self.builtin = BuiltinToolSet(
            cwd,
            shell_path=shell_path,
            auto_resize_images=auto_resize_images,
        )
        self.browser = BrowserToolSet(browser_context) if browser_context is not None else None
        browser_tools = self.browser.tools if self.browser is not None else ()
        attachment_map = {attachment.id: attachment for attachment in attachments}
        self.attachment_names = ATTACHMENT_TOOL_NAMES if attachment_map else ()
        attachment_tools: tuple[Tool, ...] = (
            AttachmentReadTool(attachment_map, on_attachment_access),
            AttachmentMetadataTool(attachment_map, on_attachment_access),
        )
        values: tuple[Tool, ...] = (
            *self.builtin.tools,
            PresentFilesTool(cwd),
            *attachment_tools,
            *browser_tools,
        )
        self.tools = values
        self.by_name = {tool.name: tool for tool in values}

    @property
    def all_names(self) -> frozenset[str]:
        return frozenset(self.by_name)

    @property
    def default_active_names(self) -> tuple[str, ...]:
        browser_names = BROWSER_TOOL_NAMES if self.browser is not None else ()
        return (
            *self.builtin.default_active_names,
            *self.required_names,
            *self.attachment_names,
            *browser_names,
        )

    def active_names(
        self,
        configured: tuple[str, ...] | None,
        restored: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        selected = (
            configured
            if configured is not None
            else restored
            if restored is not None
            else self.builtin.default_active_names
        )
        browser_names = BROWSER_TOOL_NAMES if self.browser is not None else ()
        return tuple(
            dict.fromkeys(
                (*selected, *self.required_names, *self.attachment_names, *browser_names)
            )
        )


__all__ = ["ATTACHMENT_TOOL_NAMES", "RuntimeToolSet"]
