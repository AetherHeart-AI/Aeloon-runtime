from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aeloon_core.core import (
    AssistantMessage,
    RunController,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
    estimate_context_tokens,
)
from aeloon_core.core.events import RunEventDispatcher
from aeloon_core.core.inference_runtime import (
    normalize_inference_messages,
    project_inference_messages,
)
from aeloon_core.core.tool_runtime import ToolRuntime
from aeloon_core.tool import BaseTool


class FunctionTool(BaseTool):
    def __init__(self, name: str, execute=None, *, execution_mode="parallel") -> None:
        self.name = name
        self.label = name
        self.description = name
        self.parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self._execute = execute
        self.execution_mode = execution_mode

    async def execute(self, call_id, arguments, on_update):
        if self._execute is not None:
            return await self._execute(call_id, arguments, on_update)
        return ToolResult.text(self.name)


def _tool(name: str) -> FunctionTool:
    return FunctionTool(name)


@pytest.mark.asyncio
async def test_event_dispatcher_isolates_listeners_and_merges_hooks() -> None:
    events = RunEventDispatcher()
    observed: list[str] = []

    def broken_listener(_event: Any) -> None:
        raise RuntimeError("listener failure")

    events.subscribe(broken_listener)
    unsubscribe = events.subscribe(lambda event: observed.append(event.type))
    events.on("context", lambda _event: {"first": 1, "shared": "old"})
    events.on("context", lambda _event: {"second": 2, "shared": "new"})

    result = await events.hook("context", {"messages": []})
    unsubscribe()
    await events.emit("context", {"messages": []})

    assert result == {"first": 1, "second": 2, "shared": "new"}
    assert observed == ["context"]


@pytest.mark.asyncio
async def test_run_controller_owns_ephemeral_queue_modes_and_snapshots() -> None:
    events = RunEventDispatcher()
    updates: list[dict[str, Any]] = []
    events.subscribe(
        lambda event: updates.append(event.data) if event.type == "queue_update" else None
    )
    controller = RunController(
        steering_mode="one-at-a-time",
        follow_up_mode="all",
    )
    await controller._bind(events, lambda: None)

    await controller.steer("first")
    await controller.steer("second")
    await controller.follow_up("third")
    await controller.follow_up("fourth")

    assert [message.content for message in await controller._drain_steering()] == ["first"]
    assert [message.content for message in await controller._drain_follow_up()] == [
        "third",
        "fourth",
    ]
    assert controller.snapshot()["steer"][0]["content"] == "second"
    assert updates
    controller._release()


def test_tool_runtime_has_no_dynamic_reconfiguration_api() -> None:
    events = RunEventDispatcher()
    runtime = ToolRuntime((_tool("read"),), ("read",), events)

    assert not hasattr(runtime, "configure")
    assert runtime.tool_names == ("read",)
    assert runtime.active_names == ("read",)


@pytest.mark.asyncio
async def test_tool_runtime_can_cancel_a_sequential_tool() -> None:
    entered = asyncio.Event()

    async def execute(_call_id: str, _params: dict[str, Any], _update: Any) -> ToolResult:
        entered.set()
        await asyncio.Event().wait()
        return ToolResult.text("unreachable")

    tool = FunctionTool("wait", execute, execution_mode="sequential")
    runtime = ToolRuntime((tool,), ("wait",), RunEventDispatcher())
    execution = asyncio.create_task(
        runtime.execute_calls((ToolCall("call", "wait", {}),), is_aborted=lambda: True)
    )
    await entered.wait()

    runtime.cancel()
    messages, _terminate = await execution

    assert messages[0].is_error is True
    assert messages[0].content[0].text == "Operation aborted"


@pytest.mark.asyncio
async def test_sequential_abort_generates_results_for_every_unexecuted_call() -> None:
    aborted = False
    executed: list[str] = []

    async def execute(call_id: str, _params: dict[str, Any], _update: Any) -> ToolResult:
        nonlocal aborted
        executed.append(call_id)
        aborted = True
        return ToolResult.text("first completed")

    tool = FunctionTool("step", execute, execution_mode="sequential")
    runtime = ToolRuntime((tool,), ("step",), RunEventDispatcher())
    messages, _terminate = await runtime.execute_calls(
        (
            ToolCall("one", "step", {}),
            ToolCall("two", "step", {}),
            ToolCall("three", "step", {}),
        ),
        is_aborted=lambda: aborted,
    )

    assert executed == ["one"]
    assert [message.tool_call_id for message in messages] == ["one", "two", "three"]
    assert [message.content[0].text for message in messages[1:]] == [
        "Operation aborted before execution",
        "Operation aborted before execution",
    ]
    assert all(message.is_error for message in messages[1:])


def test_inference_projection_filters_failures_and_repairs_orphan_tool_calls() -> None:
    failed = AssistantMessage(
        (),
        provider="test",
        model="model",
        stop_reason="error",
        error_message="temporary failure",
    )
    calls = AssistantMessage(
        (ToolCall("one", "read", {}), ToolCall("two", "write", {})),
        provider="test",
        model="model",
        stop_reason="toolUse",
    )
    first_result = ToolResultMessage("one", "read", (TextContent("done"),))

    projected = normalize_inference_messages(
        (UserMessage("start"), failed, calls, first_result, UserMessage("continue"))
    )

    assert failed not in projected
    assert [message.role for message in projected] == [
        "user",
        "assistant",
        "toolResult",
        "toolResult",
        "user",
    ]
    synthetic = projected[-2]
    assert isinstance(synthetic, ToolResultMessage)
    assert synthetic.tool_call_id == "two"
    assert synthetic.is_error is True
    assert synthetic.content[0].text == "No result provided"


def test_inference_projection_rebases_compaction_boundary() -> None:
    calls = AssistantMessage(
        (ToolCall("one", "read", {}), ToolCall("two", "write", {})),
        provider="test",
        model="model",
        usage=Usage(input=899, output=1, total_tokens=900),
        stop_reason="toolUse",
    )
    failed_before_boundary = AssistantMessage(
        (TextContent("failed" * 1_000),),
        provider="test",
        model="model",
        stop_reason="error",
        error_message="temporary failure",
    )
    failed_after_boundary = AssistantMessage(
        (TextContent("failed again" * 1_000),),
        provider="test",
        model="model",
        stop_reason="aborted",
        error_message="Operation aborted",
    )
    completed = AssistantMessage(
        (TextContent("done"),),
        provider="test",
        model="model",
        usage=Usage(input=19, output=1, total_tokens=20),
    )

    projected, boundary_index = project_inference_messages(
        (
            UserMessage("old"),
            calls,
            ToolResultMessage("one", "read", (TextContent("result"),)),
            failed_before_boundary,
            UserMessage("fresh"),
            failed_after_boundary,
            completed,
        ),
        boundary_index=3,
    )

    assert [message.role for message in projected] == [
        "user",
        "assistant",
        "toolResult",
        "toolResult",
        "user",
        "assistant",
    ]
    assert boundary_index == 3
    synthetic = projected[3]
    assert isinstance(synthetic, ToolResultMessage)
    assert synthetic.tool_call_id == "two"
    assert estimate_context_tokens(projected, usage_after_index=boundary_index) == 20
