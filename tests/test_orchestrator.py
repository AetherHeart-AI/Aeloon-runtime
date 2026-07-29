"""Tests for conversation-scoped Ultra Master orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from aeloon_core.config import Config
from aeloon_core.harness.mcp import McpRegistry
from aeloon_core.orchestrator import AeloonCoreOrchestrator


def _config(tmp_path: Path, **overrides: Any) -> Config:
    payload: dict[str, Any] = {
        "workspace": tmp_path / "workspace",
        "data_dir": tmp_path / "data",
        "agents": {"defaults": {"context_compaction": {"enabled": False}}},
    }
    payload.update(overrides)
    return Config(**payload).normalized()


def test_orchestrator_uses_skills_and_has_no_role_or_workflow_catalog(
    tmp_path: Path,
) -> None:
    def respond(_messages: list[ModelMessage], _info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))

    assert [expert.id for expert in app.experts] == [
        "builtin:research",
        "builtin:coding",
    ]
    assert not hasattr(app, "roles")
    assert not hasattr(app, "catalog")
    assert not hasattr(app, "workflows")


@pytest.mark.asyncio
async def test_plain_text_turn_is_persisted_as_master_history(tmp_path: Path) -> None:
    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], _info: Any) -> ModelResponse:
        requests.append(messages)
        return ModelResponse(parts=[TextPart(f"answer-{len(requests)}")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))
    first = await app.run_turn("first request")
    second = await app.run_turn("second request", session_id=first.session_id)

    assert first.final_content == "answer-1"
    assert second.final_content == "answer-2"
    assert len(requests[1]) > len(requests[0])
    history = app.sessions.history(first.session_id)
    assert [turn["user_prompt"] for turn in history] == [
        "first request",
        "second request",
    ]
    assert [turn["final_content"] for turn in history] == ["answer-1", "answer-2"]
    assert [turn["status"] for turn in history] == ["completed", "completed"]
    await app.close()


@pytest.mark.asyncio
async def test_master_limit_returns_and_persists_partial_turn(
    tmp_path: Path,
) -> None:
    requests = 0

    def respond(_messages: list[ModelMessage], info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert [tool.name for tool in info.model_request_parameters.function_tools] == []
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "read_file",
                    {"path": "README.md"},
                    "tool-after-limit",
                )
            ]
        )

    config = _config(
        tmp_path,
        agents={
            "defaults": {
                "max_iterations": 1,
                "context_compaction": {"enabled": False},
            }
        },
    )
    app = AeloonCoreOrchestrator(config, model=FunctionModel(respond))

    result = await app.run_turn("inspect")

    assert requests == 1
    assert result.status == "partial"
    assert "partial session history were preserved" in result.final_content
    assert result.messages
    persisted = app.sessions.history(result.session_id)
    assert persisted[-1]["status"] == "partial"
    assert persisted[-1]["final_content"] == result.final_content
    assert persisted[-1]["messages"] == result.messages

    app.model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart("resumed")])
    )
    resumed = await app.run_turn("continue", session_id=result.session_id)

    assert resumed.status == "completed"
    assert resumed.final_content == "resumed"
    assert [turn["status"] for turn in app.sessions.history(result.session_id)] == [
        "partial",
        "completed",
    ]
    await app.close()


@pytest.mark.asyncio
async def test_master_exposes_ultra_skill_and_expert_tools(tmp_path: Path) -> None:
    exposed: list[str] = []

    async def respond(_messages: list[ModelMessage], info: Any) -> ModelResponse:
        exposed.extend(tool.name for tool in info.model_request_parameters.function_tools)
        return ModelResponse(parts=[TextPart("done")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))
    result = await app.run_turn("inspect")

    assert result.final_content == "done"
    assert {
        "read_file",
        "write_file",
        "run_command",
        "inventory_agent_context",
        "write_plan",
        "skill_search",
        "skill_load",
        "skill_read",
        "expert_run",
    } <= set(exposed)
    assert not {"workflow_execute", "run_workflow", "spawn_worker"} & set(exposed)
    await app.close()


@pytest.mark.asyncio
async def test_normal_master_exposes_tools_from_every_configured_mcp_server(
    tmp_path: Path,
) -> None:
    exposed: list[str] = []

    async def lookup(query: str) -> str:
        return query

    async def respond(_messages: list[ModelMessage], info: Any) -> ModelResponse:
        exposed.extend(tool.name for tool in info.model_request_parameters.function_tools)
        return ModelResponse(parts=[TextPart("done")])

    docs = FunctionToolset([lookup], id="docs").prefixed("docs")
    app = AeloonCoreOrchestrator(
        _config(tmp_path, mode="normal"),
        model=FunctionModel(respond),
        mcp=McpRegistry({"docs": docs}),
    )
    await app.run_turn("inspect")

    assert "docs_lookup" in exposed
    await app.close()


@pytest.mark.asyncio
async def test_enabled_experts_are_injected_without_internal_dependencies(
    tmp_path: Path,
) -> None:
    instructions: list[str] = []

    async def respond(_messages: list[ModelMessage], info: Any) -> ModelResponse:
        instructions.append(str(info.instructions))
        return ModelResponse(parts=[TextPart("done")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))
    await app.run_turn("delegate if useful")

    prompt = "\n".join(instructions)
    assert "full-capability Master" in prompt
    assert '"id": "builtin:coding"' in prompt
    assert '"id": "builtin:research"' in prompt
    assert "Experts cannot call other experts" in prompt
    assert "generic DAG" in prompt
    await app.close()


@pytest.mark.asyncio
async def test_master_can_call_custom_prompt_expert_and_aggregates_usage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".aeloon-core" / "skills" / "echo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: echo
description: Return a bounded expert report.
kind: expert
runner: builtin.prompt
capabilities: []
max_calls_per_turn: 1
---
# Echo expert

Return the assigned outcome as a structured report.
""",
        encoding="utf-8",
    )

    async def respond(messages: list[ModelMessage], info: Any) -> ModelResponse:
        function_tools = [tool.name for tool in info.model_request_parameters.function_tools]
        output_tools = [tool.name for tool in info.model_request_parameters.output_tools]
        if "expert_run" in function_tools:
            already_called = any(
                isinstance(message, ModelRequest)
                and any(
                    isinstance(part, ToolReturnPart) and part.tool_name == "expert_run"
                    for part in message.parts
                )
                for message in messages
            )
            if already_called:
                return ModelResponse(parts=[TextPart("expert complete")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "expert_run",
                        {"expert_id": "workspace:echo", "task": "Return verified result"},
                        "expert-call",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tools[0],
                    {
                        "status": "completed",
                        "final_content": "verified result",
                        "artifacts": [],
                        "evidence": [],
                        "findings": [],
                        "unresolved": [],
                    },
                    "expert-output",
                )
            ]
        )

    config = _config(
        tmp_path,
        experts={
            "enabled": ["workspace:echo"],
            "max_calls_per_turn": 2,
        },
    )
    app = AeloonCoreOrchestrator(config, model=FunctionModel(respond))
    result = await app.run_turn("Use the echo expert")

    assert result.final_content == "expert complete"
    assert result.tools_used == ["expert_run"]
    assert result.usage["requests"] == 3
    await app.close()
