from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from aeloon_core.minimal_context import LAZY_TOOL_RESULT_MARKER, MinimalContextProcessor
from aeloon_core.profile_delegation import (
    DELEGATE_JOIN_CHARS,
    DelegationBranch,
    DelegationResult,
    joined_tool_result,
)
from aeloon_core.profile_runtime import (
    CONTROL_TOOL_DEFINITIONS,
    DELEGATE_TASK_CHARS,
    DelegateTaskArguments,
)
from aeloon_core.profiles import RuntimeAgentSpec, RuntimeProfileSpec
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.state import LightweightState, ProfileRef, RunStatus
from aeloon_core.state_machine import run_agent_loop
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import TokenLedger
from aeloon_core.turn_events import TurnEventProgress


def _control_call(name: str, arguments: dict[str, Any], *, usage: int) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=f"{name}-1", name=name, arguments=arguments)],
        finish_reason="tool_calls",
        usage={"total_tokens": usage},
    )


def _profile(*, delegated_tools: tuple[str, ...] = ()) -> RuntimeProfileSpec:
    return RuntimeProfileSpec(
        profile_schema_version=1,
        compiled_api_version=1,
        profile_id="parallel-research",
        revision=1,
        description="Parallel research test profile",
        default_agent_id="lead",
        max_handoffs=2,
        master_prompt="Always select lead.",
        shared_prompt="Use evidence and report uncertainty.",
        agents=(
            RuntimeAgentSpec(
                id="lead",
                description="Coordinate and synthesize",
                tools=(),
                prompt="Delegate independent work, then complete.",
            ),
            RuntimeAgentSpec(
                id="researcher",
                description="Gather evidence",
                tools=delegated_tools,
                prompt="Research only the assigned question.",
            ),
        ),
        artifact_id="parallel-artifact",
        generation=1,
        control_protocol_version=2,
    )


class BarrierProvider(LLMProvider):
    """Hold both branch calls at a barrier and release them in reverse order."""

    supports_concurrent_calls = True

    def __init__(self) -> None:
        super().__init__()
        self.branch_started = 0
        self.both_started = asyncio.Event()
        self.release_branches = asyncio.Event()
        self.release_alpha = asyncio.Event()
        self.beta_finished = asyncio.Event()
        self.finish_order: list[str] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format is not None:
            return LLMResponse(content='{"agent_id":"lead"}', usage={"total_tokens": 1})

        if any(
            "DELEGATED BRANCH PROTOCOL" in str(message.get("content") or "") for message in messages
        ):
            task = "alpha" if "alpha evidence" in str(messages[-1]["content"]) else "beta"
            self.branch_started += 1
            if self.branch_started == 2:
                self.both_started.set()
            await self.release_branches.wait()
            if task == "alpha":
                await self.release_alpha.wait()
                usage = 5
            else:
                self.finish_order.append("beta")
                self.beta_finished.set()
                usage = 7
            if task == "alpha":
                self.finish_order.append("alpha")
            return LLMResponse(content=f"report-{task}", usage={"total_tokens": usage})

        if any(
            message.get("role") == "tool" and message.get("name") == "delegate_tasks"
            for message in messages
        ):
            return _control_call(
                "complete_task",
                {"final_content": "joined answer"},
                usage=3,
            )
        return _control_call(
            "delegate_tasks",
            {
                "tasks": [
                    {"agent_id": "researcher", "task": "alpha evidence"},
                    {"agent_id": "researcher", "task": "beta evidence"},
                ]
            },
            usage=2,
        )


class DelegationProgress:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.completed: list[tuple[str, str]] = []
        self.joins: list[tuple[int, int]] = []

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        del text, tool_hint

    async def on_profile_delegate_branch_start(
        self,
        branch_id: str,
        label: str,
        agent_id: str,
        task: str,
    ) -> None:
        del branch_id, agent_id, task
        self.started.append(label)

    async def on_profile_delegate_branch_complete(
        self,
        branch_id: str,
        label: str,
        agent_id: str,
        **kwargs: Any,
    ) -> None:
        del branch_id, agent_id
        self.completed.append((label, str(kwargs["status"])))

    async def on_profile_delegate_join(self, source_agent_id: str, **kwargs: Any) -> None:
        del source_agent_id
        self.joins.append((int(kwargs["succeeded"]), int(kwargs["branch_count"])))


@pytest.mark.asyncio
async def test_delegate_tasks_runs_branches_concurrently_and_joins_in_input_order() -> None:
    provider = BarrierProvider()
    progress = DelegationProgress()
    run = asyncio.create_task(
        run_agent_loop(
            provider=provider,
            model="test-model",
            tools=ToolRegistry(),
            messages=[{"role": "user", "content": "research this"}],
            profile=_profile(),
            on_progress=progress,
        )
    )

    await asyncio.wait_for(provider.both_started.wait(), timeout=1)
    assert provider.branch_started == 2
    assert not run.done()

    provider.release_branches.set()
    await asyncio.wait_for(provider.beta_finished.wait(), timeout=1)
    assert not run.done()
    provider.release_alpha.set()
    state = await asyncio.wait_for(run, timeout=1)

    assert state.metadata.status == RunStatus.COMPLETED
    assert state.metadata.final_content == "joined answer"
    assert provider.finish_order == ["beta", "alpha"]
    delegate_result = next(
        message["content"]
        for message in state.messages
        if message.get("role") == "tool" and message.get("name") == "delegate_tasks"
    )
    payload = json.loads(delegate_result.split("\n", 1)[1])
    assert [branch["subagent"] for branch in payload["branches"]] == [
        "researcher#1",
        "researcher#2",
    ]
    assert [branch["report"] for branch in payload["branches"]] == [
        "report-alpha",
        "report-beta",
    ]
    assert progress.started == ["researcher#1", "researcher#2"]
    assert progress.completed == [
        ("researcher#2", "completed"),
        ("researcher#1", "completed"),
    ]
    assert progress.joins == [(2, 2)]
    assert state.delegation_count == 1
    assert state.token_ledger.totals["total_tokens"] == 18
    assert state.token_ledger.for_component(
        "subagent:delegate-1-1:researcher#1:worker"
    ) == {"total_tokens": 5}
    assert state.token_ledger.for_component(
        "subagent:delegate-1-2:researcher#2:worker"
    ) == {"total_tokens": 7}
    assert state.token_ledger.is_conserved()


class EmptyArgs(BaseModel):
    pass


class ExclusiveTool(Tool):
    name = "unsafe"
    description = "An unsafe delegated operation."
    args_model = EmptyArgs

    async def execute(self) -> str:
        raise AssertionError("unsafe delegated tool must not execute")


class UnsafeDelegateProvider(LLMProvider):
    supports_concurrent_calls = True

    def __init__(self) -> None:
        super().__init__()
        self.branch_calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format is not None:
            return LLMResponse(content='{"agent_id":"lead"}')
        if any(
            "DELEGATED BRANCH PROTOCOL" in str(message.get("content") or "") for message in messages
        ):
            self.branch_calls += 1
            return LLMResponse(content="must not run")
        if any(
            message.get("role") == "tool" and message.get("name") == "delegate_tasks"
            for message in messages
        ):
            return _control_call(
                "complete_task",
                {"final_content": "unsafe delegation rejected"},
                usage=1,
            )
        return _control_call(
            "delegate_tasks",
            {
                "tasks": [
                    {"agent_id": "researcher", "task": "first"},
                    {"agent_id": "researcher", "task": "second"},
                ]
            },
            usage=1,
        )


@pytest.mark.asyncio
async def test_delegate_tasks_rejects_non_read_only_roles_before_fork() -> None:
    provider = UnsafeDelegateProvider()
    tools = ToolRegistry()
    tools.register(ExclusiveTool())

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "research safely"}],
        profile=_profile(delegated_tools=("unsafe",)),
    )

    assert provider.branch_calls == 0
    assert state.delegation_count == 0
    assert state.control_protocol_retries == 1
    assert state.metadata.final_content == "unsafe delegation rejected"
    rejected = next(
        message["content"]
        for message in state.messages
        if message.get("role") == "tool" and message.get("name") == "delegate_tasks"
    )
    assert "non-read-only tools: unsafe" in rejected


class ValueArgs(BaseModel):
    value: str


class ReadOnlyLookup(Tool):
    name = "lookup"
    description = "Look up one value without mutation."
    args_model = ValueArgs
    concurrency_mode = "read_only"

    async def execute(self, value: str) -> str:
        return f"evidence:{value}"


class ToolBranchProvider(LLMProvider):
    supports_concurrent_calls = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format is not None:
            return LLMResponse(content='{"agent_id":"lead"}')
        is_branch = any(
            "DELEGATED BRANCH PROTOCOL" in str(message.get("content") or "")
            for message in messages
        )
        if is_branch:
            tool_result = next(
                (
                    str(message.get("content") or "")
                    for message in reversed(messages)
                    if message.get("role") == "tool"
                ),
                None,
            )
            if tool_result is not None:
                return LLMResponse(content=f"report:{tool_result}")
            assigned = (
                "alpha"
                if "alpha evidence" in str(messages[-1]["content"])
                else "beta"
            )
            return _control_call("lookup", {"value": assigned}, usage=1)
        if any(
            message.get("role") == "tool" and message.get("name") == "delegate_tasks"
            for message in messages
        ):
            return _control_call(
                "complete_task",
                {"final_content": "tool reports joined"},
                usage=1,
            )
        return _control_call(
            "delegate_tasks",
            {
                "tasks": [
                    {"agent_id": "researcher", "task": "alpha evidence"},
                    {"agent_id": "researcher", "task": "beta evidence"},
                ]
            },
            usage=1,
        )


@pytest.mark.asyncio
async def test_delegated_tool_events_are_prefixed_labeled_and_merged() -> None:
    tools = ToolRegistry()
    tools.register(ReadOnlyLookup())
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)
    state = await run_agent_loop(
        provider=ToolBranchProvider(),
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "research with tools"}],
        profile=_profile(delegated_tools=("lookup",)),
        on_progress=progress,
    )

    tool_blocks = [
        payload["block"]
        for event, payload in events
        if event == "chat.block.add" and payload.get("block", {}).get("type") == "tool_call"
    ]
    assert {block["subagent_label"] for block in tool_blocks} == {
        "researcher#1",
        "researcher#2",
    }
    assert all(str(block["id"]).startswith("delegate-1-") for block in tool_blocks)
    update_ids = {
        payload["block_id"]
        for event, payload in events
        if event == "chat.block.update" and "result" in payload.get("patch", {})
    }
    assert update_ids == {block["id"] for block in tool_blocks}
    assert state.tools_used == ["lookup", "lookup"]
    assert state.metadata.final_content == "tool reports joined"
    delegate_result = next(
        message["content"]
        for message in state.messages
        if message.get("role") == "tool" and message.get("name") == "delegate_tasks"
    )
    payload = json.loads(delegate_result.split("\n", 1)[1])
    assert [branch["report"] for branch in payload["branches"]] == [
        "report:evidence:alpha",
        "report:evidence:beta",
    ]
    subagent_usage = {
        component: usage
        for component, usage in state.token_ledger.by_component.items()
        if component.startswith("subagent:")
    }
    assert subagent_usage
    for component, usage in subagent_usage.items():
        assert progress.usage_by_component[component] == usage


class CleanupProvider(LLMProvider):
    supports_concurrent_calls = True

    def __init__(self) -> None:
        super().__init__()
        self.branch_started = 0
        self.both_started = asyncio.Event()
        self.sibling_cancelled = asyncio.Event()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format is not None:
            return LLMResponse(content='{"agent_id":"lead"}')
        is_branch = any(
            "DELEGATED BRANCH PROTOCOL" in str(message.get("content") or "")
            for message in messages
        )
        if is_branch:
            alpha = "alpha evidence" in str(messages[-1]["content"])
            self.branch_started += 1
            if self.branch_started == 2:
                self.both_started.set()
            await self.both_started.wait()
            if alpha:
                return LLMResponse(content="alpha report")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.sibling_cancelled.set()
                raise
        return _control_call(
            "delegate_tasks",
            {
                "tasks": [
                    {"agent_id": "researcher", "task": "alpha evidence"},
                    {"agent_id": "researcher", "task": "beta evidence"},
                ]
            },
            usage=1,
        )


class FailingCompletionProgress:
    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        del text, tool_hint

    async def on_profile_delegate_branch_complete(
        self,
        branch_id: str,
        label: str,
        agent_id: str,
        **kwargs: Any,
    ) -> None:
        del branch_id, agent_id, kwargs
        if label == "researcher#1":
            raise RuntimeError("progress callback failed")


@pytest.mark.asyncio
async def test_branch_callback_failure_cancels_and_awaits_running_siblings() -> None:
    provider = CleanupProvider()

    with pytest.raises(RuntimeError, match="progress callback failed"):
        await asyncio.wait_for(
            run_agent_loop(
                provider=provider,
                model="test-model",
                tools=ToolRegistry(),
                messages=[{"role": "user", "content": "research"}],
                profile=_profile(),
                on_progress=FailingCompletionProgress(),
            ),
            timeout=1,
        )

    assert provider.sibling_cancelled.is_set()


class HangingLifecycleProgress:
    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        del text, tool_hint

    async def on_profile_delegate_branch_start(self, *args: Any) -> None:
        del args
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_branch_lifecycle_callback_has_its_own_deadline(monkeypatch) -> None:
    monkeypatch.setattr(
        "aeloon_core.profile_delegation.DELEGATE_LIFECYCLE_TIMEOUT_SECONDS",
        0.01,
    )
    provider = UnsafeDelegateProvider()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            run_agent_loop(
                provider=provider,
                model="test-model",
                tools=ToolRegistry(),
                messages=[{"role": "user", "content": "research"}],
                profile=_profile(),
                on_progress=HangingLifecycleProgress(),
            ),
            timeout=1,
        )

    assert provider.branch_calls == 0


def test_joined_reports_bypass_generic_tool_preview_for_coordinator() -> None:
    joined = "alpha evidence " + ("a" * 1_500) + " beta evidence " + ("b" * 1_500)
    messages = [
        {"role": "user", "content": "research"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {"name": "delegate_tasks", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "delegate-1",
            "name": "delegate_tasks",
            "content": joined,
        },
    ]
    state = LightweightState.from_messages(messages)
    state.profile_ref = ProfileRef(
        profile_id="research",
        revision=1,
        artifact_id="artifact",
        generation=1,
    )

    context = MinimalContextProcessor(max_tool_result_chars=1_200).process(
        state=state,
        messages=messages,
        tools=[],
    )

    delegated_result = next(
        message for message in context.messages if message.get("name") == "delegate_tasks"
    )
    assert delegated_result["content"] == joined
    assert LAZY_TOOL_RESULT_MARKER not in delegated_result["content"]
    assert context.lazy_references == ()


def test_prior_turn_joined_reports_return_to_generic_lazy_preview() -> None:
    joined = "past evidence " + ("x" * 3_000)
    messages = [
        {"role": "user", "content": "first research turn"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {"name": "delegate_tasks", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "delegate-1",
            "name": "delegate_tasks",
            "content": joined,
        },
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up"},
    ]
    state = LightweightState.from_messages(messages)
    state.profile_ref = ProfileRef(
        profile_id="research",
        revision=1,
        artifact_id="artifact",
        generation=1,
    )

    context = MinimalContextProcessor(max_tool_result_chars=1_200).process(
        state=state,
        messages=messages,
        tools=[],
    )

    delegated_result = next(
        message for message in context.messages if message.get("name") == "delegate_tasks"
    )
    assert LAZY_TOOL_RESULT_MARKER in delegated_result["content"]
    assert len(context.lazy_references) == 1


def test_joined_reports_are_fairly_bounded_and_remain_valid_json() -> None:
    agent = _profile().agent("researcher")
    results = [
        DelegationResult(
            branch=DelegationBranch(
                branch_id=f"delegate-1-{index}",
                label=f"researcher#{index}",
                index=index,
                task=DelegateTaskArguments(
                    agent_id="researcher",
                    task=(f"track {index} " * 200),
                ),
                agent=agent,
                report_chars=6_000,
            ),
            status="completed",
            report=(f'START-{index} ' + ('"\\ ' * 2_000) + f' END-{index}'),
            duration_ms=1,
            tools_used=("lookup",),
            ledger=TokenLedger(),
        )
        for index in range(1, 5)
    ]

    joined = joined_tool_result(results)

    assert len(joined) <= DELEGATE_JOIN_CHARS
    payload = json.loads(joined.split("\n", 1)[1])
    reports = [branch["report"] for branch in payload["branches"]]
    assert len(reports) == 4
    assert all("... [truncated] ..." in report for report in reports)
    assert all(f"START-{index}" in reports[index - 1] for index in range(1, 5))
    assert all(f"END-{index}" in reports[index - 1] for index in range(1, 5))
    assert max(map(len, reports)) - min(map(len, reports)) <= 1
    assert all(len(branch["task"]) <= 240 for branch in payload["branches"])


def test_delegate_tool_schema_matches_runtime_boundaries() -> None:
    definition = next(
        item
        for item in CONTROL_TOOL_DEFINITIONS
        if item["function"]["name"] == "delegate_tasks"
    )
    tasks = definition["function"]["parameters"]["properties"]["tasks"]
    item = tasks["items"]["properties"]

    assert tasks["minItems"] == 2
    assert tasks["maxItems"] == 4
    assert tasks["uniqueItems"] is True
    assert item["agent_id"]["pattern"] == "^[a-z][a-z0-9_-]{0,63}$"
    assert item["task"]["minLength"] == 1
    assert item["task"]["maxLength"] == DELEGATE_TASK_CHARS
    assert item["task"]["pattern"] == "\\S"


class HangingLookup(Tool):
    name = "hang"
    description = "Wait until cancelled."
    args_model = ValueArgs
    concurrency_mode = "read_only"

    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def execute(self, value: str) -> str:
        del value
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class QuickLookup(Tool):
    name = "quick"
    description = "Return immediately."
    args_model = ValueArgs
    concurrency_mode = "read_only"

    async def execute(self, value: str) -> str:
        return f"quick:{value}"


class TimeoutProvider(LLMProvider):
    supports_concurrent_calls = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format is not None:
            return LLMResponse(content='{"agent_id":"lead"}', usage={"total_tokens": 1})
        is_branch = any(
            "DELEGATED BRANCH PROTOCOL" in str(message.get("content") or "")
            for message in messages
        )
        if is_branch:
            if any(
                "alpha evidence" in str(message.get("content") or "")
                for message in messages
            ):
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="quick-1",
                            name="quick",
                            arguments={"value": "alpha"},
                        ),
                        ToolCallRequest(
                            id="hang-1",
                            name="hang",
                            arguments={"value": "alpha"},
                        ),
                    ],
                    finish_reason="tool_calls",
                    usage={"total_tokens": 4},
                )
            return LLMResponse(content="report-beta", usage={"total_tokens": 3})
        if any(
            message.get("role") == "tool" and message.get("name") == "delegate_tasks"
            for message in messages
        ):
            return _control_call(
                "complete_task",
                {"final_content": "partial reports joined"},
                usage=2,
            )
        return _control_call(
            "delegate_tasks",
            {
                "tasks": [
                    {"agent_id": "researcher", "task": "alpha evidence"},
                    {"agent_id": "researcher", "task": "beta evidence"},
                ]
            },
            usage=2,
        )


@pytest.mark.asyncio
async def test_timed_out_branch_isolated_and_preserves_observed_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        "aeloon_core.profile_delegation.DELEGATE_BRANCH_TIMEOUT_SECONDS",
        0.02,
    )
    hanging = HangingLookup()
    tools = ToolRegistry()
    tools.register(QuickLookup())
    tools.register(hanging)
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    progress = TurnEventProgress(session_id="session-1", emit=emit)

    state = await run_agent_loop(
        provider=TimeoutProvider(),
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "research with a timeout"}],
        profile=_profile(delegated_tools=("quick", "hang")),
        on_progress=progress,
    )

    assert state.metadata.final_content == "partial reports joined"
    assert hanging.cancelled.is_set()
    delegate_result = next(
        message["content"]
        for message in state.messages
        if message.get("role") == "tool" and message.get("name") == "delegate_tasks"
    )
    payload = json.loads(delegate_result.split("\n", 1)[1])
    assert [branch["status"] for branch in payload["branches"]] == [
        "failed",
        "completed",
    ]
    assert "timed out after 0.02 seconds" in payload["branches"][0]["report"]
    assert payload["branches"][0]["tools_used"] == ["quick"]
    assert state.tools_used == ["quick"]
    delegated_blocks = [
        block
        for block in progress.blocks
        if block.get("subagent_label") == "researcher#1"
        and block.get("type") == "tool_call"
    ]
    assert {block["name"]: block["status"] for block in delegated_blocks} == {
        "quick": "done",
        "hang": "error",
    }
    updated_ids = {
        payload["block_id"]
        for event, payload in events
        if event == "chat.block.update"
        and "result" in payload.get("patch", {})
    }
    assert {block["id"] for block in delegated_blocks} <= updated_ids
    assert state.token_ledger.totals["total_tokens"] == 12
    assert state.token_ledger.for_component(
        "subagent:delegate-1-1:researcher#1:worker"
    ) == {
        "total_tokens": 4
    }
    assert state.token_ledger.is_conserved()


class RepeatingDelegateProvider(LLMProvider):
    supports_concurrent_calls = True

    def __init__(self) -> None:
        super().__init__()
        self.parent_calls = 0
        self.branch_calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format is not None:
            return LLMResponse(content='{"agent_id":"lead"}')
        if any(
            "DELEGATED BRANCH PROTOCOL" in str(message.get("content") or "")
            for message in messages
        ):
            self.branch_calls += 1
            return LLMResponse(content=f"report-{self.branch_calls}")

        self.parent_calls += 1
        if self.parent_calls <= 2:
            return _control_call(
                "delegate_tasks",
                {
                    "tasks": [
                        {"agent_id": "researcher", "task": "alpha evidence"},
                        {"agent_id": "researcher", "task": "beta evidence"},
                    ]
                },
                usage=1,
            )
        return _control_call(
            "complete_task",
            {"final_content": "used the first reports"},
            usage=1,
        )


@pytest.mark.asyncio
async def test_identical_successful_delegation_is_guarded_without_reforking() -> None:
    provider = RepeatingDelegateProvider()
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    state = await run_agent_loop(
        provider=provider,
        model="test-model",
        tools=ToolRegistry(),
        messages=[{"role": "user", "content": "research once"}],
        profile=_profile(),
        on_progress=TurnEventProgress(session_id="session-1", emit=emit),
    )

    assert state.metadata.final_content == "used the first reports"
    assert provider.branch_calls == 2
    assert state.delegation_count == 1
    assert state.last_delegation_succeeded is True
    guard = next(
        payload
        for event, payload in events
        if event == "chat.guard.decision" and payload["event"] == "duplicate_delegation"
    )
    assert guard["action"] == "return_to_model"
    duplicate_result = [
        message["content"]
        for message in state.messages
        if message.get("role") == "tool"
        and message.get("name") == "delegate_tasks"
        and str(message.get("content") or "").startswith("Error:")
    ]
    assert len(duplicate_result) == 1
