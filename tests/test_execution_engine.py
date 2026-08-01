"""Tests for the Python-owned pi-core execution boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.harness.capabilities import FileSystem, Shell
from aeloon_core.harness.execution import (
    AgentRunSpec,
    AgentRunStatus,
    HarnessAgentRuntime,
    deserialize_messages,
    output_tools,
    serialize_messages,
)
from aeloon_core.harness.provider import ScriptedPiModel
from aeloon_core.harness.tool import FunctionTool, Tool, ToolRegistry


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


class CompleteWorkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    artifacts: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()


class _RecordingRead(Tool):
    name = "read_test"
    description = "Read one test path."
    args_model = _ReadArgs
    concurrency_mode = "read_only"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, path: str) -> str:
        self.calls.append(path)
        return f"contents:{path}"


def _response(*parts: dict[str, Any], usage: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "content": list(parts),
        "usage": {
            "input": 1,
            "output": 1,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 2,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            **(usage or {}),
        },
    }


def _call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"type": "toolCall", "name": name, "arguments": arguments, "id": call_id}


def _text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def _spec(
    responses: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    request_limit: int | None = 4,
) -> AgentRunSpec:
    return AgentRunSpec(
        role="expert",
        model=ScriptedPiModel(tuple(responses)),
        instructions="Complete explicitly.",
        prompt="read and finish",
        history=[],
        tools=registry,
        output_type=output_tools(
            (CompleteWorkArgs, "complete_work", "Finish with a structured report."),
        ),
        terminal_models={"complete_work": CompleteWorkArgs},
        request_limit=request_limit,
        transition_trace_enabled=True,
    )


@pytest.mark.asyncio
async def test_runtime_executes_python_tool_through_pi_core_and_returns_typed_output() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    outcome = await HarnessAgentRuntime().run(
        _spec(
            [
                _response(_call("read_test", {"path": "README.md"}, "read-1")),
                _response(
                    _call(
                        "complete_work",
                        {
                            "summary": "done",
                            "evidence": [
                                {
                                    "kind": "file",
                                    "locator": "README.md:1",
                                    "claim": "README was inspected",
                                    "status": "observed",
                                }
                            ],
                        },
                        "done-1",
                    )
                ),
            ],
            registry,
        )
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert isinstance(outcome.output, CompleteWorkArgs)
    assert outcome.output.summary == "done"
    assert read.calls == ["README.md"]
    assert outcome.tools_used == ["read_test"]
    assert deserialize_messages(serialize_messages(outcome.messages)) == outcome.messages
    assert {record.component for record in outcome.transitions} == {"pi-core"}


@pytest.mark.asyncio
async def test_mixed_terminal_batch_is_retried_before_any_side_effect() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    outcome = await HarnessAgentRuntime().run(
        _spec(
            [
                _response(
                    _call("read_test", {"path": "secret.txt"}, "read-1"),
                    _call("complete_work", {"summary": "unsafe"}, "done-1"),
                ),
                _response(_call("complete_work", {"summary": "safe"}, "done-2")),
            ],
            registry,
        )
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert outcome.output.summary == "safe"
    assert read.calls == []


@pytest.mark.asyncio
async def test_malformed_batch_is_retried_as_a_unit() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    outcome = await HarnessAgentRuntime().run(
        _spec(
            [
                _response(
                    _call("read_test", {}, "invalid"),
                    _call("read_test", {"path": "must-not-run"}, "valid"),
                ),
                _response(_call("complete_work", {"summary": "recovered"}, "done")),
            ],
            registry,
        )
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert read.calls == []


@pytest.mark.asyncio
async def test_request_limit_returns_exact_pi_history_without_tool_effects() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    outcome = await HarnessAgentRuntime().run(
        _spec(
            [_response(_call("read_test", {"path": "must-not-run"}, "read"))],
            registry,
            request_limit=1,
        )
    )

    assert outcome.status is AgentRunStatus.LIMIT_EXCEEDED
    assert outcome.failure is not None
    assert read.calls == []
    assert outcome.messages
    assert deserialize_messages(serialize_messages(outcome.messages)) == outcome.messages


@pytest.mark.asyncio
async def test_bounded_run_reserves_final_request_for_structured_output() -> None:
    read = _RecordingRead()
    registry = ToolRegistry()
    registry.register(read)
    outcome = await HarnessAgentRuntime().run(
        _spec(
            [
                _response(_call("read_test", {"path": "README.md"}, "read-1")),
                _response(_call("complete_work", {"summary": "checkpoint"}, "done")),
            ],
            registry,
            request_limit=2,
        )
    )

    requests = [record for record in outcome.transitions if record.node == "model_request"]
    assert outcome.status is AgentRunStatus.COMPLETED
    assert outcome.output.summary == "checkpoint"
    assert read.calls == ["README.md"]
    assert requests[0].decision["tool_names"] == ["read_test", "complete_work"]
    assert requests[1].decision["tool_names"] == ["complete_work"]
    assert outcome.usage["requests"] == 2


@pytest.mark.asyncio
async def test_plain_text_cannot_complete_structured_expert() -> None:
    registry = ToolRegistry()
    outcome = await HarnessAgentRuntime().run(
        _spec([_response(_text("I am done"))], registry, request_limit=1)
    )

    assert outcome.status is AgentRunStatus.LIMIT_EXCEEDED
    assert outcome.output is None


@pytest.mark.asyncio
async def test_read_only_tools_run_concurrently_and_mutation_is_a_barrier() -> None:
    class EmptyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    active = 0
    max_active = 0
    order: list[str] = []

    async def read(name: str) -> str:
        nonlocal active, max_active
        order.append(f"{name}:start")
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        order.append(f"{name}:end")
        return name

    async def read_a() -> str:
        return await read("a")

    async def read_b() -> str:
        return await read("b")

    async def mutate() -> str:
        order.append("write:start")
        await asyncio.sleep(0)
        order.append("write:end")
        return "written"

    registry = ToolRegistry()
    for name, handler, mode in (
        ("read_a", read_a, "read_only"),
        ("read_b", read_b, "read_only"),
        ("write_test", mutate, "mutating"),
    ):
        registry.register(
            FunctionTool(
                name=name,
                description=name,
                args_model=EmptyArgs,
                handler=handler,
                concurrency_mode=mode,
            )
        )

    outcome = await HarnessAgentRuntime().run(
        _spec(
            [
                _response(_call("read_a", {}, "a"), _call("read_b", {}, "b")),
                _response(
                    _call("read_a", {}, "a2"),
                    _call("write_test", {}, "w"),
                    _call("read_b", {}, "b2"),
                ),
                _response(_call("complete_work", {"summary": "done"}, "done")),
            ],
            registry,
        )
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert max_active == 2
    assert order[4:] == [
        "a:start",
        "a:end",
        "write:start",
        "write:end",
        "b:start",
        "b:end",
    ]


@pytest.mark.asyncio
async def test_pi_core_filesystem_and_shell_tools_run_inside_workspace(
    tmp_path: Path,
) -> None:
    outcome = await HarnessAgentRuntime().run(
        AgentRunSpec(
            role="master",
            model=ScriptedPiModel(
                (
                    _response(
                        _call(
                            "write_file",
                            {"path": "note.txt", "content": "hello"},
                            "w",
                        )
                    ),
                    _response(_call("read_file", {"path": "note.txt"}, "r")),
                    _response(_call("run_command", {"command": "pwd"}, "b")),
                    _response(_text("done")),
                )
            ),
            instructions="Exercise each direct capability.",
            prompt="work",
            history=[],
            tools=ToolRegistry(),
            output_type=str,
            terminal_models={},
            request_limit=4,
            capabilities=(
                FileSystem(root_dir=tmp_path),
                Shell(cwd=tmp_path, default_timeout=5),
            ),
        )
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert outcome.output == "done"
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"
    assert outcome.tools_used == ["write_file", "read_file", "run_command"]


@pytest.mark.asyncio
async def test_pi_core_filesystem_blocks_nested_protected_paths(tmp_path: Path) -> None:
    protected = tmp_path / ".git" / "refs" / "heads" / "main"
    protected.parent.mkdir(parents=True)
    protected.write_text("original", encoding="utf-8")
    outcome = await HarnessAgentRuntime().run(
        AgentRunSpec(
            role="expert",
            model=ScriptedPiModel(
                (
                    _response(
                        _call(
                            "write_file",
                            {"path": ".git/refs/heads/main", "content": "changed"},
                            "blocked",
                        )
                    ),
                    _response(
                        _call("complete_work", {"summary": "protected"}, "done")
                    ),
                )
            ),
            instructions="Respect protected paths.",
            prompt="work",
            history=[],
            tools=ToolRegistry(),
            output_type=output_tools(
                (CompleteWorkArgs, "complete_work", "Finish with a structured report."),
            ),
            terminal_models={"complete_work": CompleteWorkArgs},
            request_limit=3,
            capabilities=(FileSystem(root_dir=tmp_path),),
        )
    )

    assert outcome.status is AgentRunStatus.COMPLETED
    assert protected.read_text(encoding="utf-8") == "original"
