from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aeloon_core.harness import (
    DEEPSEEK_V4_FLASH,
    AgentHarness,
    AgentTool,
    AssistantMessage,
    AssistantStreamEvent,
    HarnessError,
    JsonlSessionRepository,
    ResourceLoader,
    ScriptedProvider,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)


def _answer(text: str, *, stop: str = "stop", calls: tuple[ToolCall, ...] = ()):
    return AssistantMessage(
        (TextContent(text), *calls),
        provider="deepseek",
        model="deepseek-v4-flash",
        stop_reason=stop,
    )


def _tool(name: str, execute) -> AgentTool:
    return AgentTool(
        name=name,
        label=name,
        description=name,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=execute,
    )


@pytest.mark.asyncio
async def test_loop_executes_tools_and_preserves_pi_event_lifecycle(tmp_path: Path) -> None:
    async def execute(_call_id, _params, _update):
        return ToolResult.text("tool output")

    provider = ScriptedProvider(
        [
            _answer("working", stop="toolUse", calls=(ToolCall("c1", "demo", {}),)),
            _answer("done"),
        ]
    )
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        tools=(_tool("demo", execute),),
        active_tool_names=("demo",),
    )
    events: list[str] = []
    harness.subscribe(lambda event: events.append(event.type))

    result = await harness.prompt("start")

    assert result.text == "done"
    assert len(provider.requests) == 2
    injected = provider.requests[1][1].messages[-1]
    assert isinstance(injected, ToolResultMessage)
    assert injected.content[0].text == "tool output"
    assert events[0:3] == ["before_agent_start", "agent_start", "turn_start"]
    assert events.count("turn_start") == 2
    assert events.count("turn_end") == 2
    assert events.index("tool_execution_start") < events.index("tool_execution_end")
    assert events[-2:] == ["settled", "queue_update"] or events[-1] == "settled"
    assert "agent_end" in events


@pytest.mark.asyncio
async def test_tool_failures_are_returned_to_model_and_do_not_end_loop(tmp_path: Path) -> None:
    async def fail(_call_id, _params, _update):
        raise RuntimeError("boom")

    provider = ScriptedProvider(
        [
            _answer("", stop="toolUse", calls=(ToolCall("c1", "fail", {}),)),
            _answer("recovered"),
        ]
    )
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        tools=(_tool("fail", fail),),
        active_tool_names=("fail",),
    )

    result = await harness.prompt("go")

    returned = provider.requests[1][1].messages[-1]
    assert result.text == "recovered"
    assert isinstance(returned, ToolResultMessage)
    assert returned.is_error is True
    assert "RuntimeError: boom" in returned.content[0].text


@pytest.mark.asyncio
async def test_tools_run_in_parallel_but_results_keep_call_order(tmp_path: Path) -> None:
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
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        tools=(_tool("slow", slow), _tool("fast", fast)),
        active_tool_names=("slow", "fast"),
    )

    started = time.monotonic()
    await harness.prompt("go")
    elapsed = time.monotonic() - started

    assert elapsed < 0.18
    results = [
        message
        for message in provider.requests[1][1].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [message.tool_call_id for message in results] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_length_stop_skips_entire_tool_batch(tmp_path: Path) -> None:
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
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        tools=(_tool("demo", execute),),
        active_tool_names=("demo",),
    )

    await harness.prompt("go")

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
async def test_steer_is_injected_after_current_tool_batch(tmp_path: Path) -> None:
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
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        tools=(_tool("wait", execute),),
        active_tool_names=("wait",),
    )
    task = asyncio.create_task(harness.prompt("go"))
    await entered.wait()
    await harness.steer("change direction")
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
async def test_follow_up_busy_next_turn_abort_and_idle(tmp_path: Path) -> None:
    provider = GateProvider()
    harness = AgentHarness(provider=provider, model=DEEPSEEK_V4_FLASH, cwd=str(tmp_path))
    task = asyncio.create_task(harness.prompt("initial"))
    await asyncio.sleep(0)
    with pytest.raises(HarnessError) as busy:
        await harness.prompt("too soon")
    assert busy.value.code == "busy"
    await harness.follow_up("after completion")
    provider.gate.set()
    result = await task
    assert result.text == "second"
    assert provider.requests[1].messages[-1].content == "after completion"
    assert harness.is_idle

    await harness.next_turn("queued")
    provider.gate.set()
    await harness.prompt("explicit")
    assert [message.content for message in provider.requests[2].messages[-2:]] == [
        "queued",
        "explicit",
    ]

    blocked = GateProvider()
    aborting = AgentHarness(provider=blocked, model=DEEPSEEK_V4_FLASH, cwd=str(tmp_path))
    prompt_task = asyncio.create_task(aborting.prompt("wait"))
    await asyncio.sleep(0)
    abort_task = asyncio.create_task(aborting.abort())
    aborted = await prompt_task
    await abort_task
    assert aborted.stop_reason == "aborted"
    assert aborting.is_idle


@pytest.mark.asyncio
async def test_provider_retry_phase_emits_events(tmp_path: Path) -> None:
    class RetryingProvider:
        def stream(self, model, _context, options):
            async def events():
                yield AssistantStreamEvent("start")
                callback = options.metadata["on_retry"]
                await callback({"stage": "start", "attempt": 1, "delayMs": 0, "error": "retry"})
                assert harness.phase == "retry"
                await callback({"stage": "end", "attempt": 1, "delayMs": 0, "error": None})
                message = _answer("done")
                yield AssistantStreamEvent("done", message=message)

            return events()

    harness = AgentHarness(provider=RetryingProvider(), model=DEEPSEEK_V4_FLASH, cwd=str(tmp_path))
    events: list[str] = []
    harness.subscribe(lambda event: events.append(event.type))

    result = await harness.prompt("go")

    assert result.text == "done"
    assert "auto_retry_start" in events
    assert "auto_retry_end" in events
    assert harness.phase == "idle"


@pytest.mark.asyncio
async def test_resources_and_active_tools_refresh_at_turn_boundary(tmp_path: Path) -> None:
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("FIRST RULE", encoding="utf-8")
    provider = ScriptedProvider([_answer("one"), _answer("two")])
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        resource_loader=ResourceLoader(cwd=tmp_path, agent_dir=tmp_path / "global"),
    )
    events: list[str] = []
    harness.subscribe(lambda event: events.append(event.type))

    await harness.prompt("first")
    instructions.write_text("SECOND RULE", encoding="utf-8")
    await harness.set_active_tools(("read", "grep"))
    await harness.prompt("second")

    assert "FIRST RULE" in provider.requests[0][1].system_prompt
    assert "SECOND RULE" in provider.requests[1][1].system_prompt
    assert "- grep: Search file contents for patterns" in provider.requests[1][1].system_prompt
    assert "- bash:" not in provider.requests[1][1].system_prompt
    assert "resources_update" in events
    assert "tools_update" in events


@pytest.mark.asyncio
async def test_provider_and_tool_hooks_can_patch_or_block(tmp_path: Path) -> None:
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
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        tools=(_tool("demo", execute),),
        active_tool_names=("demo",),
    )
    harness.on(
        "before_provider_payload",
        lambda event: (
            {
                "payload": {
                    **event.data["payload"],
                    "systemPrompt": "PATCHED SYSTEM",
                }
            }
            if len(provider.requests) == 0
            else None
        ),
    )
    harness.on("tool_call", lambda _event: {"block": True, "reason": "blocked"})
    harness.on(
        "tool_result",
        lambda _event: {"content": [{"type": "text", "text": "patched result"}]},
    )

    await harness.prompt("go")

    assert provider.requests[0][1].system_prompt == "PATCHED SYSTEM"
    assert executed == 0
    result = provider.requests[1][1].messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert result.content[0].text == "patched result"


@pytest.mark.asyncio
async def test_new_session_does_not_override_configured_thinking_or_empty_tools(
    tmp_path: Path,
) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    provider = ScriptedProvider([_answer("done")])
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        session=session,
        thinking_level="high",
        active_tool_names=(),
    )

    await harness.prompt("go")

    assert provider.requests[0][2].thinking_level == "high"
    assert provider.requests[0][1].tools == ()
