from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aeloon_core.harness.events import HarnessEventDispatcher
from aeloon_core.harness.input_queue import TurnInputQueues
from aeloon_core.harness.tool_runtime import ToolRuntime
from aeloon_core.harness.types import AgentTool, HarnessError, ToolCall, ToolResult


def _tool(name: str) -> AgentTool:
    async def execute(_call_id: str, _params: dict[str, Any], _update: Any) -> ToolResult:
        return ToolResult.text(name)

    return AgentTool(
        name=name,
        label=name,
        description=name,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=execute,
    )


@pytest.mark.asyncio
async def test_event_dispatcher_isolates_listeners_and_merges_hooks() -> None:
    events = HarnessEventDispatcher()
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
async def test_turn_input_queues_own_queue_modes_and_snapshots() -> None:
    events = HarnessEventDispatcher()
    updates: list[dict[str, Any]] = []
    events.subscribe(
        lambda event: updates.append(event.data) if event.type == "queue_update" else None
    )
    queues = TurnInputQueues(
        events,
        steering_mode="one-at-a-time",
        follow_up_mode="all",
    )

    await queues.enqueue("steer", "first")
    await queues.enqueue("steer", "second")
    await queues.enqueue("follow_up", "third")
    await queues.enqueue("follow_up", "fourth")
    await queues.enqueue("next_turn", "later")

    assert [message.content for message in await queues.drain_steering()] == ["first"]
    assert [message.content for message in await queues.drain_follow_up()] == ["third", "fourth"]
    assert [message.content for message in queues.take_next_turn()] == ["later"]
    assert queues.snapshot()["steer"][0]["content"] == "second"
    assert updates


def test_tool_runtime_rejects_invalid_reconfiguration_atomically() -> None:
    events = HarnessEventDispatcher()
    runtime = ToolRuntime((_tool("read"),), ("read",), events)

    with pytest.raises(HarnessError) as invalid:
        runtime.configure((_tool("write"),), ("missing",))

    assert invalid.value.code == "invalid_argument"
    assert runtime.tool_names == ("read",)
    assert runtime.active_names == ("read",)


@pytest.mark.asyncio
async def test_tool_runtime_can_cancel_a_sequential_tool() -> None:
    entered = asyncio.Event()

    async def execute(_call_id: str, _params: dict[str, Any], _update: Any) -> ToolResult:
        entered.set()
        await asyncio.Event().wait()
        return ToolResult.text("unreachable")

    tool = AgentTool(
        name="wait",
        label="wait",
        description="wait",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=execute,
        execution_mode="sequential",
    )
    runtime = ToolRuntime((tool,), ("wait",), HarnessEventDispatcher())
    execution = asyncio.create_task(
        runtime.execute_calls((ToolCall("call", "wait", {}),), is_aborted=lambda: True)
    )
    await entered.wait()

    runtime.cancel()
    messages, _terminate = await execution

    assert messages[0].is_error is True
    assert messages[0].content[0].text == "Operation aborted"
