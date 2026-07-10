from __future__ import annotations

from typing import Any

import pytest

from aeloon_core.config import Config
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.providers.base import LLMProvider, LLMResponse


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses

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
        del (
            messages,
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


def config_for(tmp_path, *, uasm_enabled: bool) -> Config:
    return Config.model_validate(
        {
            "workspace": tmp_path,
            "data_dir": tmp_path / "data",
            "skills": {"enabled": False},
            "agents": {
                "defaults": {
                    "model": "test-model",
                    "context_compaction": {"enabled": False},
                    "uasm": {
                        "enabled": uasm_enabled,
                        "temporary_guard_enabled": False,
                        "minimal_context_enabled": False,
                    },
                }
            },
        }
    ).normalized()


@pytest.mark.asyncio
async def test_orchestrator_persists_uasm_usage_and_independent_trace(tmp_path) -> None:
    orchestrator = AeloonCoreOrchestrator(config_for(tmp_path, uasm_enabled=True))
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
async def test_orchestrator_keeps_legacy_kernel_as_default(tmp_path) -> None:
    orchestrator = AeloonCoreOrchestrator(config_for(tmp_path, uasm_enabled=False))
    orchestrator.provider = ScriptedProvider([LLMResponse(content="legacy done")])

    result = await orchestrator.run_turn("answer", session_id="session-1")

    assert result.status == "legacy"
    assert result.final_content == "legacy done"
    assert result.transitions == []
    assert orchestrator.sessions.transition_history("session-1") == []


@pytest.mark.asyncio
async def test_orchestrator_trace_io_failure_is_observability_only(
    tmp_path,
    monkeypatch,
) -> None:
    orchestrator = AeloonCoreOrchestrator(config_for(tmp_path, uasm_enabled=True))
    orchestrator.provider = ScriptedProvider([LLMResponse(content="done")])

    def fail_trace(**_kwargs: Any) -> None:
        raise OSError("trace disk full")

    monkeypatch.setattr(orchestrator.sessions, "append_transition", fail_trace)

    result = await orchestrator.run_turn("answer", session_id="session-1")

    assert result.status == "completed"
    assert result.final_content == "done"
    assert orchestrator.sessions.history("session-1")[0]["final_content"] == "done"
    assert orchestrator.sessions.transition_history("session-1") == []
