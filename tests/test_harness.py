from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from aeloon_core.core import (
    AssistantMessage,
    AssistantStreamEvent,
    RunController,
    RunError,
    RunRequest,
    StreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    run_agent,
)
from aeloon_core.runtime.providers import DEEPSEEK_V4_FLASH
from aeloon_core.runtime.providers.testing import ScriptedProvider
from aeloon_core.tool import BaseTool


class FunctionTool(BaseTool):
    def __init__(self, name, execute, *, execution_mode="parallel"):
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
        return await self._execute(call_id, arguments, on_update)


def _answer(text: str, *, stop: str = "stop", calls: tuple[ToolCall, ...] = ()):
    return AssistantMessage(
        (TextContent(text), *calls),
        provider="deepseek",
        model="deepseek/deepseek-v4-flash",
        stop_reason=stop,
    )


def _tool(name: str, execute) -> Tool:
    return FunctionTool(name, execute)


def _request(
    provider,
    *,
    text: str = "go",
    tools: tuple[Tool, ...] = (),
    active: tuple[str, ...] | None = None,
    messages=(),
    options: StreamOptions | None = None,
    run_id: str = "run",
) -> RunRequest:
    return RunRequest(
        run_id=run_id,
        inference=provider,
        model=DEEPSEEK_V4_FLASH,
        messages=tuple(messages),
        input=(UserMessage(text),),
        system_prompt="SYSTEM",
        tools=tools,
        active_tool_names=active,
        stream_options=options or StreamOptions(),
    )


@pytest.mark.asyncio
async def test_loop_executes_tools_and_emits_run_lifecycle() -> None:
    async def execute(_call_id, _params, _update):
        return ToolResult.text("tool output")

    provider = ScriptedProvider(
        [
            _answer("working", stop="toolUse", calls=(ToolCall("c1", "demo", {}),)),
            _answer("done"),
        ]
    )
    events: list[str] = []
    result = await run_agent(
        _request(provider, tools=(_tool("demo", execute),), active=("demo",)),
        emit=lambda event: events.append(event.type),
    )

    assert result.final_message.text == "done"
    assert len(provider.requests) == 2
    injected = provider.requests[1][1].messages[-1]
    assert isinstance(injected, ToolResultMessage)
    assert injected.content[0].text == "tool output"
    assert events.count("turn_start") == 2
    assert events.count("turn_end") == 2
    assert events.index("tool_execution_start") < events.index("tool_execution_end")
    assert events[-1] == "settled"
    assert "agent_end" in events


@pytest.mark.asyncio
async def test_tool_failures_are_returned_to_model_and_do_not_end_loop() -> None:
    async def fail(_call_id, _params, _update):
        raise RuntimeError("boom")

    provider = ScriptedProvider(
        [
            _answer("", stop="toolUse", calls=(ToolCall("c1", "fail", {}),)),
            _answer("recovered"),
        ]
    )
    result = await run_agent(_request(provider, tools=(_tool("fail", fail),), active=("fail",)))

    returned = provider.requests[1][1].messages[-1]
    assert result.final_message.text == "recovered"
    assert isinstance(returned, ToolResultMessage)
    assert returned.is_error is True
    assert "RuntimeError: boom" in returned.content[0].text


@pytest.mark.asyncio
async def test_tools_run_in_parallel_but_results_keep_call_order() -> None:
    async def slow(_call_id, _params, _update):
        await asyncio.sleep(0.1)
        return ToolResult.text("slow")

    async def fast(_call_id, _params, _update):
        await asyncio.sleep(0.02)
        return ToolResult.text("fast")

    provider = ScriptedProvider(
        [
            _answer(
                "",
                stop="toolUse",
                calls=(ToolCall("slow", "slow", {}), ToolCall("fast", "fast", {})),
            ),
            _answer("done"),
        ]
    )
    started = time.monotonic()
    await run_agent(
        _request(
            provider,
            tools=(_tool("slow", slow), _tool("fast", fast)),
            active=("slow", "fast"),
        )
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.18
    results = [
        message
        for message in provider.requests[1][1].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [message.tool_call_id for message in results] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_length_stop_skips_entire_tool_batch() -> None:
    calls = 0

    async def execute(_call_id, _params, _update):
        nonlocal calls
        calls += 1
        return ToolResult.text("unexpected")

    provider = ScriptedProvider(
        [
            _answer(
                "",
                stop="length",
                calls=(ToolCall("one", "demo", {}), ToolCall("two", "demo", {})),
            ),
            _answer("retried"),
        ]
    )
    await run_agent(_request(provider, tools=(_tool("demo", execute),), active=("demo",)))

    assert calls == 0
    results = [
        message
        for message in provider.requests[1][1].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 2
    assert all(message.is_error for message in results)
    assert all("arguments may be truncated" in message.content[0].text for message in results)


@pytest.mark.asyncio
async def test_steer_is_injected_after_current_tool_batch() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(_call_id, _params, _update):
        entered.set()
        await release.wait()
        return ToolResult.text("finished")

    provider = ScriptedProvider(
        [
            _answer("", stop="toolUse", calls=(ToolCall("c", "wait", {}),)),
            _answer("done"),
        ]
    )
    controller = RunController()
    task = asyncio.create_task(
        run_agent(
            _request(provider, tools=(_tool("wait", execute),), active=("wait",)),
            controller=controller,
        )
    )
    await entered.wait()
    await controller.steer("change direction")
    release.set()
    await task

    roles = [message.role for message in provider.requests[1][1].messages[-2:]]
    assert roles == ["toolResult", "user"]
    assert provider.requests[1][1].messages[-1].content == "change direction"


class GateProvider:
    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.requests = []

    def stream(self, model, context, _options) -> AsyncIterator[AssistantStreamEvent]:
        return self._stream(model, context)

    async def _stream(self, model, context):
        self.requests.append(context)
        yield AssistantStreamEvent("start")
        if len(self.requests) == 1:
            await self.gate.wait()
        text = "first" if len(self.requests) == 1 else "second"
        message = AssistantMessage((TextContent(text),), provider=model.provider, model=model.id)
        yield AssistantStreamEvent("text_delta", delta=text)
        yield AssistantStreamEvent("done", message=message)


@pytest.mark.asyncio
async def test_follow_up_cancel_and_controller_lifecycle() -> None:
    provider = GateProvider()
    controller = RunController()
    task = asyncio.create_task(run_agent(_request(provider), controller=controller))
    await asyncio.sleep(0)
    await controller.follow_up("after completion")
    provider.gate.set()
    result = await task
    assert result.final_message.text == "second"
    assert provider.requests[1].messages[-1].content == "after completion"
    with pytest.raises(RunError) as inactive:
        await controller.steer("too late")
    assert inactive.value.code == "invalid_state"

    blocked = GateProvider()
    aborting = RunController()
    prompt_task = asyncio.create_task(
        run_agent(_request(blocked, run_id="abort"), controller=aborting)
    )
    await asyncio.sleep(0)
    abort_task = asyncio.create_task(aborting.cancel())
    aborted = await prompt_task
    await abort_task
    assert aborted.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_provider_retry_emits_events() -> None:
    class RetryingProvider:
        def stream(self, model, _context, options):
            async def events():
                yield AssistantStreamEvent("start")
                callback = options.metadata["on_retry"]
                await callback({"stage": "start", "attempt": 1, "delayMs": 0})
                await callback({"stage": "end", "attempt": 1, "delayMs": 0})
                yield AssistantStreamEvent("done", message=_answer("done"))

            return events()

    events: list[str] = []
    result = await run_agent(
        _request(RetryingProvider()),
        emit=lambda event: events.append(event.type),
    )

    assert result.final_message.text == "done"
    assert "auto_retry_start" in events
    assert "auto_retry_end" in events


@pytest.mark.asyncio
async def test_provider_and_tool_hooks_can_patch_or_block() -> None:
    executed = 0

    async def execute(_call_id, _params, _update):
        nonlocal executed
        executed += 1
        return ToolResult.text("original")

    provider = ScriptedProvider(
        [
            _answer("", stop="toolUse", calls=(ToolCall("c", "demo", {}),)),
            _answer("done"),
        ]
    )
    hooks = {
        "before_inference_context": (
            lambda event: (
                {
                    "context": {
                        **event.data["context"],
                        "systemPrompt": "PATCHED SYSTEM",
                    }
                }
                if len(provider.requests) == 0
                else None
            ),
        ),
        "tool_call": (lambda _event: {"block": True, "reason": "blocked"},),
        "tool_result": (lambda _event: {"content": [{"type": "text", "text": "patched result"}]},),
    }

    await run_agent(
        _request(provider, tools=(_tool("demo", execute),), active=("demo",)),
        hooks=hooks,
    )

    assert provider.requests[0][1].system_prompt == "PATCHED SYSTEM"
    assert executed == 0
    result = provider.requests[1][1].messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert result.content[0].text == "patched result"


@pytest.mark.asyncio
async def test_repeated_and_concurrent_runs_do_not_share_state() -> None:
    first_provider = ScriptedProvider([_answer("one")])
    second_provider = ScriptedProvider([_answer("two")])

    first, second = await asyncio.gather(
        run_agent(_request(first_provider, text="first", run_id="first")),
        run_agent(_request(second_provider, text="second", run_id="second")),
    )

    assert [message.content for message in first_provider.requests[0][1].messages] == ["first"]
    assert [message.content for message in second_provider.requests[0][1].messages] == ["second"]
    assert first.final_message.text == "one"
    assert second.final_message.text == "two"
