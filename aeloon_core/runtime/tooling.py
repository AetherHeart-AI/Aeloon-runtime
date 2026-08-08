"""Explicit composition of built-in and runtime-owned tools."""

from __future__ import annotations

from pathlib import Path

from aeloon_core.browser.protocol import BrowserContext
from aeloon_core.browser.tools import BROWSER_TOOL_NAMES, BrowserToolSet
from aeloon_core.core.types import Tool
from aeloon_core.runtime.artifacts import PRESENT_FILES_TOOL_NAME, PresentFilesTool
from aeloon_core.tool import BuiltinToolSet


class RuntimeToolSet:
    required_names = (PRESENT_FILES_TOOL_NAME,)

    def __init__(
        self,
        cwd: Path | str,
        *,
        shell_path: str | None = None,
        auto_resize_images: bool = True,
        browser_context: BrowserContext | None = None,
    ) -> None:
        self.builtin = BuiltinToolSet(
            cwd,
            shell_path=shell_path,
            auto_resize_images=auto_resize_images,
        )
        self.browser = BrowserToolSet(browser_context) if browser_context is not None else None
        browser_tools = self.browser.tools if self.browser is not None else ()
        values: tuple[Tool, ...] = (*self.builtin.tools, PresentFilesTool(cwd), *browser_tools)
        self.tools = values
        self.by_name = {tool.name: tool for tool in values}

    @property
    def all_names(self) -> frozenset[str]:
        return frozenset(self.by_name)

    @property
    def default_active_names(self) -> tuple[str, ...]:
        browser_names = BROWSER_TOOL_NAMES if self.browser is not None else ()
        return (*self.builtin.default_active_names, *self.required_names, *browser_names)

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
        return tuple(dict.fromkeys((*selected, *self.required_names, *browser_names)))


__all__ = ["RuntimeToolSet"]
