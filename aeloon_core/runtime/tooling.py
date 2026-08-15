"""Explicit composition of built-in and runtime-owned tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aeloon_core.core.types import Tool
from aeloon_core.runtime.artifacts import PRESENT_FILES_TOOL_NAME, PresentFilesTool
from aeloon_core.runtime.attachments import (
    AttachmentAccessCallback,
    AttachmentMetadataTool,
    AttachmentReadTool,
    ResolvedAttachment,
)
from aeloon_core.tool import BuiltinToolSet, WebFetchTool, WebSearchTool

ATTACHMENT_TOOL_NAMES = ("attachment_read", "attachment_metadata")


class RuntimeToolSet:
    required_names = (PRESENT_FILES_TOOL_NAME,)

    def __init__(
        self,
        cwd: Path | str,
        *,
        shell_path: str | None = None,
        auto_resize_images: bool = True,
        attachments: tuple[ResolvedAttachment, ...] = (),
        on_attachment_access: AttachmentAccessCallback | None = None,
        web_search: dict[str, Any] | None = None,
        web_fetch: dict[str, Any] | None = None,
        cloud_search: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.builtin = BuiltinToolSet(
            cwd,
            shell_path=shell_path,
            auto_resize_images=auto_resize_images,
        )
        attachment_map = {attachment.id: attachment for attachment in attachments}
        self.attachment_names = ATTACHMENT_TOOL_NAMES if attachment_map else ()
        attachment_tools: tuple[Tool, ...] = (
            AttachmentReadTool(attachment_map, on_attachment_access),
            AttachmentMetadataTool(attachment_map, on_attachment_access),
        )
        web_tools: tuple[Tool, ...] = ()
        if web_search and web_fetch:
            if web_search.get("enabled"):
                web_tools += (
                    WebSearchTool(
                        **{k: v for k, v in web_search.items() if k != "enabled"},
                        cloud_search=cloud_search,
                    ),
                )
            if web_fetch.get("enabled"):
                web_tools += (
                    WebFetchTool(**{k: v for k, v in web_fetch.items() if k != "enabled"}),
                )
        self.web_names = tuple(tool.name for tool in web_tools)
        values: tuple[Tool, ...] = (
            *self.builtin.tools,
            PresentFilesTool(cwd),
            *attachment_tools,
            *web_tools,
        )
        self.tools = values
        self.by_name = {tool.name: tool for tool in values}

    @property
    def all_names(self) -> frozenset[str]:
        return frozenset(self.by_name)

    @property
    def default_active_names(self) -> tuple[str, ...]:
        return (
            *self.builtin.default_active_names,
            *self.required_names,
            *self.attachment_names,
            *self.web_names,
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
        return tuple(
            dict.fromkeys(
                (*selected, *self.required_names, *self.attachment_names, *self.web_names)
            )
        )

    def begin_turn(self, run_id: str) -> None:
        for tool in self.tools:
            hook = getattr(tool, "begin_turn", None)
            if hook is not None:
                hook(run_id)


__all__ = ["ATTACHMENT_TOOL_NAMES", "RuntimeToolSet"]
