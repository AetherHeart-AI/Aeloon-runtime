from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.default_profile import (
    builtin_profile_source,
    coding_profile_source,
    load_builtin_profile,
    materialize_builtin_profile,
    materialize_coding_profile,
    research_profile_source,
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


def orchestrator_for_paths(workspace: Path, data_dir: Path) -> AeloonCoreOrchestrator:
    config = Config.model_validate(
        {
            "workspace": workspace,
            "data_dir": data_dir,
            "skills": {"enabled": False},
        }
    ).normalized()
    return AeloonCoreOrchestrator(config)


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


def test_bundled_research_profile_declares_parallel_read_only_team() -> None:
    profile = parse_profile(research_profile_source())

    assert profile.id == "research"
    assert profile.default_agent == "research_lead"
    assert [agent.id for agent in profile.agents] == [
        "research_lead",
        "source_scout",
        "fact_checker",
    ]
    assert all(agent.tools == ("websearch", "webfetch") for agent in profile.agents)
    assert "delegate_tasks" in profile.agent("research_lead").prompt
    assert builtin_profile_source("research") == research_profile_source()


def test_unknown_bundled_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown built-in profile"):
        builtin_profile_source("not-built-in")


def test_materialized_profile_never_overwrites_workspace_edits(tmp_path: Path) -> None:
    target = materialize_coding_profile(tmp_path)
    assert target.read_text() == coding_profile_source()

    target.write_text("user-owned profile\n")
    assert materialize_coding_profile(tmp_path) == target
    assert target.read_text() == "user-owned profile\n"


def test_materialized_research_profile_never_overwrites_workspace_edits(
    tmp_path: Path,
) -> None:
    target = materialize_builtin_profile(tmp_path, "research")
    assert target.read_text() == research_profile_source()

    target.write_text("user-owned research profile\n")
    assert materialize_builtin_profile(tmp_path, "research") == target
    assert target.read_text() == "user-owned research profile\n"


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
async def test_research_turn_bootstraps_bundled_profile_in_any_workspace(
    tmp_path: Path,
) -> None:
    config = Config.model_validate(
        {
            "workspace": tmp_path / "unrelated-workspace",
            "data_dir": tmp_path / "data",
            "skills": {"enabled": False},
            "agents": {
                "defaults": {
                    "model": "test-model",
                    "profile_id": "research",
                    "context_compaction": {"enabled": False},
                }
            },
        }
    ).normalized()
    config.workspace.mkdir()
    orchestrator = AeloonCoreOrchestrator(config)
    orchestrator.provider = ScriptedProvider(
        [
            LLMResponse(content='{"agent_id":"research_lead"}'),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="complete-research",
                        name="complete_task",
                        arguments={"final_content": "research profile ready"},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    result = await orchestrator.run_turn("stable lookup", session_id="research-session")

    assert result.final_content == "research profile ready"
    assert result.profile is not None
    assert result.profile["profile_id"] == "research"
    status = orchestrator.profile_store.status("research")
    inspected = orchestrator.profile_store.inspect(status["artifact_id"])
    assert inspected["approval"]["approved_by"] == "aeloon-core:builtin:research"
    source_path = config.workspace / ".aeloon-core/profiles/research/PROFILE.md"
    assert source_path.read_text() == research_profile_source()


@pytest.mark.asyncio
async def test_builtin_research_artifact_does_not_depend_on_bootstrap_workspace(
    tmp_path: Path,
) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    data_dir = tmp_path / "data"

    first_orchestrator = orchestrator_for_paths(workspace_a, data_dir)
    first = await load_builtin_profile(
        first_orchestrator.profile_store,
        workspace=workspace_a,
        profile_id="research",
    )
    shutil.rmtree(workspace_a)
    second_orchestrator = orchestrator_for_paths(workspace_b, data_dir)
    second = await load_builtin_profile(
        second_orchestrator.profile_store,
        workspace=workspace_b,
        profile_id="research",
    )

    assert second.artifact_id == first.artifact_id
    assert second.generation == first.generation
    assert second.profile_id == "research"


@pytest.mark.asyncio
async def test_builtin_research_bootstrap_survives_workspace_mirror_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_materialize(workspace: Path, profile_id: str) -> Path:
        raise OSError("read-only workspace")

    monkeypatch.setattr(
        "aeloon_core.default_profile.materialize_builtin_profile",
        fail_to_materialize,
    )
    orchestrator = orchestrator_for_paths(tmp_path / "workspace", tmp_path / "data")
    store = orchestrator.profile_store

    profile = await load_builtin_profile(
        store,
        workspace=tmp_path / "workspace",
        profile_id="research",
    )

    assert profile.profile_id == "research"
    assert store.status("research")["active"] is True


@pytest.mark.asyncio
async def test_builtin_loader_preserves_operator_active_research_artifact(
    tmp_path: Path,
) -> None:
    orchestrator = orchestrator_for_paths(tmp_path / "workspace", tmp_path / "data")
    store = orchestrator.profile_store
    custom_source = research_profile_source().replace("revision: 1", "revision: 9", 1)
    custom_source = custom_source.replace(
        "You are an evidence-driven research team.",
        "Use the operator-managed research instructions.",
        1,
    )
    artifact = await store.compile(custom_source)
    store.approve(artifact["artifact_id"], approved_by="operator")
    store.activate(artifact["artifact_id"])

    profile = await load_builtin_profile(
        store,
        workspace=tmp_path / "workspace",
        profile_id="research",
    )

    assert profile.revision == 9
    assert profile.artifact_id == artifact["artifact_id"]
    assert store.status("research")["generation"] == 1


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
