from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.default_profile import (
    coding_profile_source,
    materialize_coding_profile,
)
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.profiles import parse_profile
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)


def test_bundled_coding_profile_declares_focused_roles_and_scopes() -> None:
    profile = parse_profile(coding_profile_source())

    assert profile.id == "coding"
    assert profile.default_agent == "implementer"
    assert [agent.id for agent in profile.agents] == [
        "planner",
        "implementer",
        "reviewer",
    ]
    assert profile.agent("planner").tools == (
        "read",
        "glob",
        "grep",
        "webfetch",
        "websearch",
    )
    assert {"write", "edit", "exec", "todowrite"} <= set(
        profile.agent("implementer").tools
    )
    assert "write" not in profile.agent("reviewer").tools
    assert "complete_task" in profile.shared_prompt


def test_materialized_profile_never_overwrites_workspace_edits(tmp_path: Path) -> None:
    target = materialize_coding_profile(tmp_path)
    assert target.read_text() == coding_profile_source()

    target.write_text("user-owned profile\n")
    assert materialize_coding_profile(tmp_path) == target
    assert target.read_text() == "user-owned profile\n"


@pytest.mark.asyncio
async def test_default_turn_bootstraps_and_pins_bundled_coding_profile(
    tmp_path: Path,
) -> None:
    config = Config.model_validate(
        {
            "workspace": tmp_path / "workspace",
            "data_dir": tmp_path / "data",
            "skills": {"enabled": False},
            "agents": {
                "defaults": {
                    "model": "test-model",
                    "context_compaction": {"enabled": False},
                }
            },
        }
    ).normalized()
    config.workspace.mkdir()
    orchestrator = AeloonCoreOrchestrator(config)
    orchestrator.provider = ScriptedProvider(
        [
            LLMResponse(content='{"agent_id":"implementer"}'),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="complete-1",
                        name="complete_task",
                        arguments={"final_content": "coding profile ready"},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    result = await orchestrator.run_turn("inspect the project", session_id="session-1")

    assert result.final_content == "coding profile ready"
    assert result.profile is not None
    assert result.profile["profile_id"] == "coding"
    assert result.profile["revision"] == 1
    status = orchestrator.profile_store.status("coding")
    assert status["active"] is True
    inspected = orchestrator.profile_store.inspect(status["artifact_id"])
    assert inspected["approval"]["approved_by"] == "aeloon-core:builtin:coding"
    source_path = config.workspace / ".aeloon-core/profiles/coding/PROFILE.md"
    assert source_path.read_text() == coding_profile_source()

    orchestrator.provider = ScriptedProvider(
        [
            LLMResponse(content='{"agent_id":"implementer"}'),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="complete-2",
                        name="complete_task",
                        arguments={"final_content": "still ready"},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )
    second = await orchestrator.run_turn("check again", session_id="session-2")
    assert second.profile is not None
    assert second.profile["generation"] == result.profile["generation"]
    assert len(list(orchestrator.profile_store.audit_dir.glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_default_loader_preserves_operator_active_coding_artifact(
    tmp_path: Path,
) -> None:
    config = Config.model_validate(
        {
            "workspace": tmp_path / "workspace",
            "data_dir": tmp_path / "data",
            "skills": {"enabled": False},
            "agents": {
                "defaults": {
                    "model": "test-model",
                    "context_compaction": {"enabled": False},
                }
            },
        }
    ).normalized()
    config.workspace.mkdir()
    orchestrator = AeloonCoreOrchestrator(config)
    custom_source = coding_profile_source().replace("revision: 1", "revision: 9", 1)
    custom_source = custom_source.replace(
        "You are part of a coding team operating directly in the user's workspace.",
        "Use the operator-managed coding instructions.",
        1,
    )
    artifact = await orchestrator.profile_store.compile(custom_source)
    orchestrator.profile_store.approve(artifact["artifact_id"], approved_by="operator")
    orchestrator.profile_store.activate(artifact["artifact_id"])
    orchestrator.provider = ScriptedProvider(
        [
            LLMResponse(content='{"agent_id":"implementer"}'),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="complete-custom",
                        name="complete_task",
                        arguments={"final_content": "custom ready"},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    result = await orchestrator.run_turn("use my profile", session_id="session-custom")

    assert result.profile is not None
    assert result.profile["revision"] == 9
    assert result.profile["artifact_id"] == artifact["artifact_id"]
    assert orchestrator.profile_store.status("coding")["generation"] == 1
