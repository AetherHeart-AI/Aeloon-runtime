from __future__ import annotations

import shlex
import sys
from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

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
        self.calls.append(messages)
        del (
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
            response_format,
        )
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)


def config_for(tmp_path, *, transition_trace_enabled: bool = True) -> Config:
    return Config.model_validate(
        {
            "workspace": tmp_path,
            "data_dir": tmp_path / "data",
            "skills": {"enabled": False},
            "agents": {
                "defaults": {
                    "model": "test-model",
                    "profile_id": None,
                    "context_compaction": {"enabled": False},
                    "uasm": {
                        "transition_trace_enabled": transition_trace_enabled,
                    },
                }
            },
        }
    ).normalized()


def profile_source(*, revision: int, prompt: str) -> str:
    return f"""---
schema_version: 1
id: runtime-team
revision: {revision}
description: Runtime team
default_agent: operator
max_handoffs: 2
agents:
  - id: operator
    description: Operate the task
    tools: []
---

## Shared
Stay in scope.

## Master
Select operator.

## Agent: operator
{prompt}
"""


@pytest.mark.asyncio
async def test_orchestrator_persists_uasm_usage_and_independent_trace(tmp_path) -> None:
    orchestrator = AeloonCoreOrchestrator(config_for(tmp_path))
    orchestrator.provider = ScriptedProvider(
        [LLMResponse(content="done", usage={"total_tokens": 7})]
    )

    result = await orchestrator.run_turn("answer", session_id="session-1")

    assert result.status == "completed"
    assert result.turn_id
    assert result.usage["totals"]["total_tokens"] == 7
    assert result.transitions
    assert orchestrator.sessions.history("session-1")[0]["usage"] == result.usage
    assert orchestrator.sessions.history("session-1")[0]["turn_id"] == result.turn_id
    persisted_transitions = orchestrator.sessions.transition_history("session-1")
    assert len(persisted_transitions) == len(result.transitions)
    assert [record["sequence"] for record in persisted_transitions] == [
        record["sequence"] for record in result.transitions
    ]
    assert all(record["type"] == "transition" for record in persisted_transitions)
    assert all(record["turn_id"] == result.turn_id for record in persisted_transitions)
    assert len(orchestrator.sessions.list_sessions()) == 1
    assert orchestrator.sessions.list_sessions()[0].turns == 1


@pytest.mark.asyncio
async def test_orchestrator_uses_state_machine_as_the_only_runtime(tmp_path) -> None:
    orchestrator = AeloonCoreOrchestrator(
        config_for(tmp_path, transition_trace_enabled=False)
    )
    orchestrator.provider = ScriptedProvider([LLMResponse(content="done")])

    result = await orchestrator.run_turn("answer", session_id="session-1")

    assert result.status == "completed"
    assert result.final_content == "done"
    assert result.transitions == []
    assert orchestrator.sessions.transition_history("session-1") == []


@pytest.mark.asyncio
async def test_orchestrator_trace_io_failure_is_observability_only(
    tmp_path,
    monkeypatch,
) -> None:
    orchestrator = AeloonCoreOrchestrator(config_for(tmp_path))
    orchestrator.provider = ScriptedProvider([LLMResponse(content="done")])

    def fail_trace(**_kwargs: Any) -> None:
        raise OSError("trace disk full")

    monkeypatch.setattr(orchestrator.sessions, "append_transition", fail_trace)

    result = await orchestrator.run_turn("answer", session_id="session-1")

    assert result.status == "completed"
    assert result.final_content == "done"
    assert orchestrator.sessions.history("session-1")[0]["final_content"] == "done"
    assert orchestrator.sessions.transition_history("session-1") == []


@pytest.mark.asyncio
async def test_orchestrator_pins_active_profile_once_per_turn(tmp_path) -> None:
    config = config_for(tmp_path).model_copy(
        update={
            "agents": config_for(tmp_path).agents.model_copy(
                update={
                    "defaults": config_for(tmp_path).agents.defaults.model_copy(
                        update={"profile_id": "runtime-team"}
                    )
                }
            )
        }
    )
    orchestrator = AeloonCoreOrchestrator(config)
    first = await orchestrator.profile_store.compile(
        profile_source(revision=1, prompt="Use version one behavior.")
    )
    second = await orchestrator.profile_store.compile(
        profile_source(revision=2, prompt="Use version two behavior.")
    )
    orchestrator.profile_store.approve(first["artifact_id"])
    orchestrator.profile_store.approve(second["artifact_id"])
    orchestrator.profile_store.activate(first["artifact_id"])

    class ActivatingProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__(
                [
                    LLMResponse(content='{"agent_id":"operator"}'),
                    LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCallRequest(
                                id="complete-1",
                                name="complete_task",
                                arguments={"final_content": "first done"},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                ]
            )

        async def chat(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def]
            if not self.calls:
                orchestrator.profile_store.activate(second["artifact_id"])
            return await super().chat(messages, *args, **kwargs)

    activating_provider = ActivatingProvider()
    orchestrator.provider = activating_provider
    first_result = await orchestrator.run_turn("answer", session_id="session-1")

    assert first_result.profile == {
        "profile_id": "runtime-team",
        "revision": 1,
        "artifact_id": first["artifact_id"],
        "generation": 1,
    }
    assert "version one" in activating_provider.calls[1][-1]["content"]
    assert orchestrator.sessions.history("session-1")[0]["profile"] == first_result.profile

    orchestrator.provider = ScriptedProvider(
        [
            LLMResponse(content='{"agent_id":"operator"}'),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="complete-2",
                        name="complete_task",
                        arguments={"final_content": "second done"},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )
    second_result = await orchestrator.run_turn("answer again", session_id="session-1")

    assert second_result.profile == {
        "profile_id": "runtime-team",
        "revision": 2,
        "artifact_id": second["artifact_id"],
        "generation": 2,
    }


@pytest.mark.asyncio
async def test_agent_exec_cannot_cross_profile_operator_store_boundary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = config_for(workspace).model_copy(
        update={"data_dir": (tmp_path / "operator-data").resolve()}
    )
    orchestrator = AeloonCoreOrchestrator(config)
    artifact = await orchestrator.profile_store.compile(
        profile_source(revision=1, prompt="Operate safely.")
    )
    code = (
        "from pathlib import Path; "
        "from aeloon_core.profile_artifacts import ProfileArtifactStore; "
        f"ProfileArtifactStore(data_dir=Path({str(config.data_dir)!r})).approve("
        f"{artifact['artifact_id']!r})"
    )
    exec_tool = orchestrator.profile_registry.get("exec")
    assert exec_tool is not None

    result = await exec_tool.execute(
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    )

    assert "Exit code: 0" not in result
    assert orchestrator.profile_store.inspect(artifact["artifact_id"])["state"] == "validated"


@pytest.mark.asyncio
async def test_no_profile_turn_preserves_canonical_system_messages(tmp_path) -> None:
    orchestrator = AeloonCoreOrchestrator(config_for(tmp_path))
    orchestrator.sessions.append_turn(
        session_id="session-legacy",
        user_prompt="old",
        final_content="old result",
        tools_used=[],
        messages=[
            {"role": "user", "content": "old"},
            {
                "role": "system",
                "content": "PROFILE CONTROL PROTOCOL: stale correction",
            },
            {"role": "assistant", "content": "old result"},
        ],
    )
    provider = ScriptedProvider([LLMResponse(content="new result")])
    orchestrator.provider = provider

    result = await orchestrator.run_turn("new", session_id="session-legacy")

    assert result.final_content == "new result"
    assert any(
        str(message.get("content") or "").startswith("PROFILE CONTROL PROTOCOL:")
        for message in provider.calls[0]
    )
