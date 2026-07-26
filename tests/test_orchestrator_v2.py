from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from aeloon_core.config import Config
from aeloon_core.orchestrator import AeloonCoreOrchestrator


def _config(tmp_path: Path) -> Config:
    return Config(
        workspace=tmp_path / "workspace",
        data_dir=tmp_path / "data",
        agents={"defaults": {"context_compaction": {"enabled": False}}},
    ).normalized()


def test_orchestrator_has_no_durable_child_control_plane(tmp_path: Path) -> None:
    def respond(_messages: list[ModelMessage], _info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))

    assert not hasattr(app, "workers")
    assert not hasattr(app, "worker_control")
    assert not hasattr(app, "worker_manager")
    assert not hasattr(app, "flow_store")
    assert not hasattr(app, "flow_control")
    assert [worker.id for worker in app.worker_types.list()] == [
        "builder",
        "explorer",
        "researcher",
        "reviewer",
    ]


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
    await app.close()


@pytest.mark.asyncio
async def test_master_exposes_observation_and_harness_tools_only(tmp_path: Path) -> None:
    exposed: list[str] = []

    async def respond(_messages: list[ModelMessage], info: Any) -> ModelResponse:
        exposed.extend(
            tool.name for tool in info.model_request_parameters.function_tools
        )
        return ModelResponse(parts=[TextPart("done")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))
    result = await app.run_turn("inspect")

    assert result.final_content == "done"
    assert {"list", "read", "glob", "grep", "run_workflow"} <= set(exposed)
    assert not {
        "spawn_worker",
        "resume_worker",
        "create_flow",
        "advance_flow",
        "finish_turn",
    } & set(exposed)
    await app.close()


@pytest.mark.asyncio
async def test_worker_definitions_are_injected_as_ephemeral_responsibilities(
    tmp_path: Path,
) -> None:
    instructions: list[str] = []

    async def respond(_messages: list[ModelMessage], info: Any) -> ModelResponse:
        instructions.append(str(info.instructions))
        return ModelResponse(parts=[TextPart("done")])

    app = AeloonCoreOrchestrator(_config(tmp_path), model=FunctionModel(respond))
    await app.run_turn("delegate if needed")

    prompt = "\n".join(instructions)
    assert "All child-agent work is ephemeral" in prompt
    assert "run_workflow" in prompt
    assert '"id": "builder"' in prompt
    assert "finish inside the current turn" in prompt
    await app.close()
