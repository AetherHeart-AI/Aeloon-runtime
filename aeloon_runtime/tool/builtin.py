"""Factory object for the built-in local tool collection."""

from __future__ import annotations

from pathlib import Path

from aeloon_runtime.core.types import Tool
from aeloon_runtime.tool.base import ToolContext
from aeloon_runtime.tool.filesystem import EditTool, ReadTool, WriteTool
from aeloon_runtime.tool.search import FindTool, GrepTool, ListTool
from aeloon_runtime.tool.shell import BashTool


class BuiltinToolSet:
    default_active_names = ("read", "bash", "edit", "write")
    all_names = frozenset((*default_active_names, "grep", "find", "ls"))

    def __init__(
        self,
        cwd: Path | str,
        *,
        shell_path: str | None = None,
        auto_resize_images: bool = True,
    ) -> None:
        context = ToolContext.create(cwd)
        self.context = context
        values: tuple[Tool, ...] = (
            ReadTool(context, auto_resize_images=auto_resize_images),
            BashTool(context, shell_path=shell_path),
            EditTool(context),
            WriteTool(context),
            GrepTool(context),
            FindTool(context),
            ListTool(context),
        )
        self.tools = values
        self.by_name = {tool.name: tool for tool in values}


__all__ = ["BuiltinToolSet"]
