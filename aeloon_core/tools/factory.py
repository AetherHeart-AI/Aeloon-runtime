"""Core tool registration."""

from __future__ import annotations

from aeloon_core.config import Config
from aeloon_core.tools.filesystem import EditTool, ReadTool, WriteTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.tools.search_grep import GlobTool, GrepTool
from aeloon_core.tools.shell import ExecTool
from aeloon_core.tools.todo import TodoWriteTool
from aeloon_core.tools.web import WebFetchTool, WebSearchTool


def register_core_tools(registry: ToolRegistry, config: Config) -> TodoWriteTool:
    """Register the standalone core tool set and return the todo tool."""

    workspace = config.workspace
    registry.register(ExecTool(workspace=workspace, timeout=config.tools.exec.timeout))
    registry.register(ReadTool(workspace=workspace))
    registry.register(WriteTool(workspace=workspace))
    registry.register(EditTool(workspace=workspace))
    registry.register(GlobTool(workspace=workspace))
    registry.register(GrepTool(workspace=workspace))
    registry.register(WebFetchTool(config=config.tools.web))
    registry.register(WebSearchTool(config=config.tools.web))
    todo_tool = TodoWriteTool(data_dir=config.data_dir)
    registry.register(todo_tool)
    return todo_tool
