"""Internal task-graph planning for tool-call execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from aeloon_core.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from aeloon_core.tools.registry import ToolRegistry


class TaskState(StrEnum):
    """Lifecycle state for an internal tool-execution task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskNode:
    """Internal representation of a single tool call in one agent turn."""

    index: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    mode: str
    deps: set[int] = field(default_factory=set)
    dependents: set[int] = field(default_factory=set)
    state: TaskState = TaskState.PENDING
    result: str | None = None
    error: str | None = None


def build_task_graph(tool_calls: list[ToolCallRequest], tools: ToolRegistry) -> list[TaskNode]:
    """Compile one LLM tool-call batch into a dependency graph by concurrency mode.

    Independent ``read_only`` calls run concurrently; any ``mutating`` or
    ``exclusive`` call is a barrier that waits for every earlier call and blocks
    every later one, so reads never observe a half-applied write.
    """

    nodes: list[TaskNode] = []
    for index, tool_call in enumerate(tool_calls):
        tool = tools.get(tool_call.name)
        mode = tool.concurrency_mode if tool is not None else "exclusive"
        assert isinstance(tool_call.arguments, dict)
        nodes.append(
            TaskNode(
                index=index,
                call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                mode=mode,
            )
        )

    for node in nodes:
        earlier = nodes[: node.index]
        # A read waits only for prior barriers; a barrier waits for everything before it.
        if node.mode == "read_only":
            node.deps = {other.index for other in earlier if other.mode != "read_only"}
        else:
            node.deps = {other.index for other in earlier}
        for dep in node.deps:
            nodes[dep].dependents.add(node.index)
    return nodes
