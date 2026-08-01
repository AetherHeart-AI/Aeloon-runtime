"""Tests for conversation-scoped Ultra Master orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from aeloon_core.config import Config
from aeloon_core.harness.expert import ExpertResult, ExpertRunnerRegistry
from aeloon_core.harness.mcp import McpRegistry
from aeloon_core.harness.provider import ScriptedPiModel
from aeloon_core.harness.tool import FunctionTool
from aeloon_core.orchestrator import AeloonCoreOrchestrator


def _response(*parts: dict[str, Any]) -> dict[str, Any]:
    return {"content": list(parts)}


def _call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"type": "toolCall", "name": name, "arguments": arguments, "id": call_id}


def _text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def _model(*responses: dict[str, Any]) -> ScriptedPiModel:
    return ScriptedPiModel(tuple(responses))


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
    app = AeloonCoreOrchestrator(_config(tmp_path), model=_model(_response(_text("done"))))

    assert [expert.id for expert in app.experts] == [
        "builtin:research",
        "builtin:coding",
    ]
    assert not hasattr(app, "roles")
    assert not hasattr(app, "catalog")
    assert not hasattr(app, "workflows")


@pytest.mark.asyncio
async def test_plain_text_turn_is_persisted_as_master_history(tmp_path: Path) -> None:
    app = AeloonCoreOrchestrator(
        _config(tmp_path),
        model=_model(_response(_text("answer-1"))),
    )
    first = await app.run_turn("first request")
    app.model = _model(_response(_text("answer-2")))
    second = await app.run_turn("second request", session_id=first.session_id)

    assert first.final_content == "answer-1"
    assert second.final_content == "answer-2"
    first_request = next(item for item in first.transitions if item["node"] == "model_request")
    second_request = next(item for item in second.transitions if item["node"] == "model_request")
    assert second_request["before_digest"] != first_request["before_digest"]
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
    config = _config(
        tmp_path,
        agents={
            "defaults": {
                "max_iterations": 1,
                "context_compaction": {"enabled": False},
            }
        },
    )
    app = AeloonCoreOrchestrator(
        config,
        model=_model(
            _response(
                _call("read_file", {"path": "README.md"}, "tool-after-limit")
            )
        ),
    )

    result = await app.run_turn("inspect")

    requests = [item for item in result.transitions if item["node"] == "model_request"]
    assert len(requests) == 1
    assert requests[0]["decision"]["tool_names"] == []
    assert result.status == "partial"
    assert "partial session history were preserved" in result.final_content
    assert result.messages
    persisted = app.sessions.history(result.session_id)
    assert persisted[-1]["status"] == "partial"
    assert persisted[-1]["final_content"] == result.final_content
    assert persisted[-1]["messages"] == result.messages

    app.model = _model(_response(_text("resumed")))
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
    app = AeloonCoreOrchestrator(
        _config(tmp_path),
        model=_model(_response(_text("done"))),
    )
    result = await app.run_turn("inspect")
    request = next(item for item in result.transitions if item["node"] == "model_request")
    exposed = set(request["decision"]["tool_names"])

    assert result.final_content == "done"
    assert {
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "start_command",
        "check_command",
        "stop_command",
        "inventory_agent_context",
        "list_directory",
        "find_files",
        "search_files",
        "create_directory",
        "file_info",
        "write_plan",
        "skill_search",
        "skill_load",
        "skill_read",
        "expert_run",
    } <= exposed
    assert not {"workflow_execute", "run_workflow", "spawn_worker"} & exposed
    await app.close()


@pytest.mark.asyncio
async def test_normal_master_exposes_tools_from_every_configured_mcp_server(
    tmp_path: Path,
) -> None:
    class LookupArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str

    async def lookup(query: str) -> str:
        return query

    docs = FunctionTool(
        name="docs_lookup",
        description="Look up documentation.",
        args_model=LookupArgs,
        handler=lookup,
        concurrency_mode="read_only",
    )
    app = AeloonCoreOrchestrator(
        _config(tmp_path, mode="normal"),
        model=_model(_response(_text("done"))),
        mcp=McpRegistry({"docs": docs}),
    )
    result = await app.run_turn("inspect")
    request = next(item for item in result.transitions if item["node"] == "model_request")

    assert "docs_lookup" in request["decision"]["tool_names"]
    await app.close()


@pytest.mark.asyncio
async def test_enabled_experts_are_injected_without_internal_dependencies(
    tmp_path: Path,
) -> None:
    app = AeloonCoreOrchestrator(
        _config(tmp_path),
        model=_model(_response(_text("done"))),
    )
    await app.run_turn("delegate if useful")

    from aeloon_core.harness.agent.prompt import master_system_prompt

    prompt = master_system_prompt(
        expert_descriptors=[expert.descriptor() for expert in app.experts],
        plain_skill_ids=sorted(
            app.master_skill_scope.skill_ids - {expert.id for expert in app.experts}
        ),
        mode=app.config.mode,
        mcp_server_ids=list(app.master_mcp_ids),
        capability_names=["filesystem", "shell", "repo_context", "planning"],
    )
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

    class FakePromptRunner:
        async def run(self, request, context) -> ExpertResult:
            del request, context
            return ExpertResult(
                status="completed",
                final_content="verified result",
                usage={"requests": 1},
            )

    config = _config(
        tmp_path,
        experts={
            "enabled": ["workspace:echo"],
            "max_calls_per_turn": 2,
        },
    )
    app = AeloonCoreOrchestrator(
        config,
        model=_model(
            _response(
                _call(
                    "expert_run",
                    {"expert_id": "workspace:echo", "task": "Return verified result"},
                    "expert-call",
                )
            ),
            _response(_text("expert complete")),
        ),
    )
    app.runners = ExpertRunnerRegistry({"builtin.prompt": FakePromptRunner()})
    result = await app.run_turn("Use the echo expert")

    assert result.final_content == "expert complete"
    assert result.tools_used == ["expert_run"]
    assert result.usage["requests"] == 3
    await app.close()
