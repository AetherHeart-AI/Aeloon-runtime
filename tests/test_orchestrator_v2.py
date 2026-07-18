"""Master/Worker capability isolation and v2 execution integration tests."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.config import (
    AgentDefaultsConfig,
    AgentsConfig,
    Config,
    ContextCompactionConfig,
    SkillsConfig,
)
from aeloon_core.context import strip_skill_tool_history
from aeloon_core.context_compaction import COMPACTION_MARKER
from aeloon_core.master_tools import build_master_scheduler_tools
from aeloon_core.minimal_context import LAZY_TOOL_RESULT_MARKER, MinimalContextProcessor
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.state import LightweightState


class ScriptedProvider(LLMProvider):
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        super().__init__()
        self.calls = deque(calls)
        self.max_token_limits: list[int | None] = []

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
        self.max_token_limits.append(max_tokens)
        del (
            messages,
            tools,
            model,
            temperature,
            reasoning_effort,
            tool_choice,
            response_format,
        )
        name, arguments = self.calls.popleft()
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id=f"call-{name}", name=name, arguments=arguments)],
            finish_reason="tool_calls",
        )


class HighUsageProvider(LLMProvider):
    """Return a huge first-round usage sample without a huge next request."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.max_token_limits: list[int | None] = []

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
            temperature,
            reasoning_effort,
            tool_choice,
            response_format,
        )
        self.call_count += 1
        self.max_token_limits.append(max_tokens)
        if self.call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call-todo",
                        name="todowrite",
                        arguments={
                            "todos": [
                                {"content": "finish the objective", "status": "in_progress"}
                            ]
                        },
                    )
                ],
                finish_reason="tool_calls",
                usage={"total_tokens": 127_999},
            )
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call-complete",
                    name="complete_work",
                    arguments={"summary": "finished after multiple model rounds"},
                )
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 1_000},
        )


class CompactionAwareProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.domain_calls = 0
        self.summary_calls = 0

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
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
            response_format,
        )
        if not tools:
            self.summary_calls += 1
            return LLMResponse(
                content="Earlier Worker progress was compressed into this checkpoint.",
                finish_reason="stop",
                usage={"total_tokens": 50},
            )
        self.domain_calls += 1
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id=f"call-complete-{self.domain_calls}",
                    name="complete_work",
                    arguments={"summary": f"completed run {self.domain_calls}"},
                )
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 100},
        )


def _config(tmp_path: Path, *, skills: bool = False) -> Config:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Config(
        workspace=workspace,
        data_dir=tmp_path / "data",
        skills=SkillsConfig(enabled=skills, external=False, claude_code=False),
    ).normalized()


@pytest.mark.asyncio
async def test_master_and_worker_tool_surfaces_are_disjoint(tmp_path: Path) -> None:
    app = AeloonCoreOrchestrator(_config(tmp_path))
    master = build_master_scheduler_tools(
        control=app.worker_control,
        base_session_id="master",
        base_turn_id="turn",
        flow_control=app.flow_control,
    )
    for name in ("list", "read", "glob", "grep"):
        tool = app.master_observation_tools.get(name)
        assert tool is not None
        master.register(tool)
    master_names = {
        definition["function"]["name"] for definition in master.get_definitions()
    }

    spawned = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="build the result",
        idempotency_key="spawn",
        detached=True,
    )
    run = app.workers.get_run(spawned["run_id"])
    worker, _ = await app._build_worker_tools(run)
    worker_names = {
        definition["function"]["name"] for definition in worker.get_definitions()
    }

    assert {"list", "read", "glob", "grep"}.issubset(master_names)
    assert {
        "create_flow",
        "add_flow_nodes",
        "advance_flow",
        "revise_flow_node",
        "complete_flow",
    }.issubset(master_names)
    assert master_names.isdisjoint(
        {"write", "str_replace", "exec", "webfetch", "websearch", "skill"}
    )
    assert {"write", "str_replace", "exec", "webfetch", "websearch"}.issubset(
        worker_names
    )
    assert {"complete_work", "request_master"}.issubset(worker_names)
    assert worker_names.isdisjoint(
        {
            "discover_worker_types",
            "list_workers",
            "inspect_worker",
            "spawn_worker",
            "reuse_worker",
            "await_workers",
            "resume_worker",
            "cancel_worker",
            "archive_worker",
            "create_flow",
            "add_flow_nodes",
            "advance_flow",
            "revise_flow_node",
            "complete_flow",
        }
    )
    assert "skill" not in worker_names


@pytest.mark.asyncio
async def test_stale_master_registry_cannot_schedule_low_level_worker(
    tmp_path: Path,
) -> None:
    app = AeloonCoreOrchestrator(_config(tmp_path))
    app.flow_store.begin_turn("master", "current-turn")
    stale = build_master_scheduler_tools(
        control=app.worker_control,
        base_session_id="master",
        base_turn_id="stale-turn",
        flow_control=app.flow_control,
        execution_guard=lambda _tool: app.flow_store.refresh_turn_lease(
            "master",
            "stale-turn",
        ),
    )

    result = await stale.execute(
        "spawn_worker",
        {
            "worker_type_id": "builder",
            "objective": "must not start",
            "idempotency_key": "stale-spawn",
            "detached": True,
        },
    )

    assert "TOOL_EXECUTION_ERROR" in result
    assert app.worker_control.list_workers("master") == []


@pytest.mark.asyncio
async def test_todowrite_and_terminal_state_are_per_worker_run(tmp_path: Path) -> None:
    app = AeloonCoreOrchestrator(_config(tmp_path))
    first = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="first",
        idempotency_key="first",
        detached=True,
    )
    second = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="second",
        idempotency_key="second",
        detached=True,
    )
    _, first_claimed = app.workers.try_start_run(first["run_id"])
    _, second_claimed = app.workers.try_start_run(second["run_id"])
    assert first_claimed is True
    assert second_claimed is True
    first_tools, first_terminal = await app._build_worker_tools(
        app.workers.get_run(first["run_id"])
    )
    second_tools, second_terminal = await app._build_worker_tools(
        app.workers.get_run(second["run_id"])
    )

    first_todo = first_tools.get("todowrite")
    second_todo = second_tools.get("todowrite")
    assert first_todo is not None
    assert second_todo is not None
    assert first_todo is not second_todo
    assert first_todo.run_id == first["run_id"]  # type: ignore[attr-defined]
    assert second_todo.run_id == second["run_id"]  # type: ignore[attr-defined]
    assert first_terminal is not second_terminal
    await first_tools.execute(
        "todowrite",
        {"todos": [{"content": "first item", "status": "pending"}]},
    )
    await second_tools.execute(
        "todowrite",
        {"todos": [{"content": "second item", "status": "completed"}]},
    )
    first_file = app.config.data_dir / "todos" / f"{first['run_id']}.json"
    second_file = app.config.data_dir / "todos" / f"{second['run_id']}.json"
    assert "first item" in first_file.read_text(encoding="utf-8")
    assert "second item" in second_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_worker_waits_then_resumes_from_exact_checkpoint(tmp_path: Path) -> None:
    app = AeloonCoreOrchestrator(_config(tmp_path))
    app._file_tool_limit = 32_000
    app.provider = ScriptedProvider(
        [
            (
                "request_master",
                {"summary": "inspected the code", "question": "Which behavior wins?"},
            ),
            (
                "complete_work",
                {
                    "summary": "implemented the selected behavior",
                    "artifacts": ["src/result.py"],
                    "evidence": ["tests pass"],
                },
            ),
        ]
    )

    spawned = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="implement the behavior",
        idempotency_key="spawn",
    )
    waiting = (
        await app.worker_control.await_workers(
            [spawned["run_id"]], timeout=2, base_session_id="master"
        )
    )[0]

    assert waiting["status"] == "waiting_for_context"
    assert waiting["waiting_request"]["question"] == "Which behavior wins?"
    checkpoint = app.workers.load_checkpoint(spawned["run_id"])
    assert checkpoint is not None
    assert checkpoint["snapshot_digest"] == spawned["snapshot"]["digest"]

    resumed = await app.worker_control.resume_worker(
        spawned["run_id"],
        response="Use behavior A",
        idempotency_key="resume",
        base_session_id="master",
    )
    completed = (
        await app.worker_control.await_workers(
            [resumed["run_id"]], timeout=2, base_session_id="master"
        )
    )[0]

    assert completed["status"] == "completed"
    assert completed["source_run_id"] == spawned["run_id"]
    assert completed["summary"] == "implemented the selected behavior"
    assert completed["artifacts"] == ["src/result.py"]
    assert app.workers.get_run(spawned["run_id"]).status.value == "waiting_for_context"
    resumed_checkpoint = app.workers.load_checkpoint(resumed["run_id"])
    assert resumed_checkpoint is not None
    serialized = str(resumed_checkpoint["messages"])
    assert "Which behavior wins?" in serialized
    assert "Use behavior A" in serialized
    assert app.worker_control.default_budget.max_tokens is None
    assert app.worker_control.default_budget.max_tool_calls is None
    assert app.provider.max_token_limits == [None, None]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_worker_cumulative_usage_does_not_consume_context_window(
    tmp_path: Path,
) -> None:
    app = AeloonCoreOrchestrator(_config(tmp_path))
    app._file_tool_limit = 32_000
    app.provider = HighUsageProvider()

    spawned = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="builder",
        objective="finish across multiple model rounds",
        idempotency_key="spawn",
    )
    completed = (
        await app.worker_control.await_workers(
            [spawned["run_id"]], timeout=2, base_session_id="master"
        )
    )[0]

    assert completed["status"] == "completed"
    assert completed["summary"] == "finished after multiple model rounds"
    assert completed["usage"]["totals"]["total_tokens"] > 128_000
    assert app.provider.call_count == 2  # type: ignore[attr-defined]
    assert app.provider.max_token_limits == [None, None]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_worker_uses_shared_context_compaction_pipeline(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config(
        workspace=workspace,
        data_dir=tmp_path / "data",
        agents=AgentsConfig(
            defaults=AgentDefaultsConfig(
                context_window_tokens=4_000,
                context_compaction=ContextCompactionConfig(
                    trigger_ratio=0.1,
                    preserve_recent_turns=1,
                    summary_max_tokens=256,
                ),
            )
        ),
        skills=SkillsConfig(enabled=False, external=False, claude_code=False),
    ).normalized()
    app = AeloonCoreOrchestrator(config)
    app._file_tool_limit = 32_000
    app.provider = CompactionAwareProvider()

    first = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="explorer",
        objective="inspect enough material to establish prior Worker context",
        idempotency_key="first",
    )
    first_result = (
        await app.worker_control.await_workers(
            [first["run_id"]], timeout=2, base_session_id="master"
        )
    )[0]
    assert first_result["status"] == "completed"

    second = await app.worker_control.reuse_worker(
        base_session_id="master",
        worker_id=first["worker_id"],
        objective="continue from the prior work and finish the next outcome",
        idempotency_key="second",
    )
    second_result = (
        await app.worker_control.await_workers(
            [second["run_id"]], timeout=2, base_session_id="master"
        )
    )[0]

    assert second_result["status"] == "completed"
    assert app.provider.domain_calls == 2  # type: ignore[attr-defined]
    assert app.provider.summary_calls >= 1  # type: ignore[attr-defined]
    checkpoint = app.workers.load_checkpoint(second["run_id"])
    assert checkpoint is not None
    assert COMPACTION_MARKER in str(checkpoint["messages"])


@pytest.mark.asyncio
async def test_skill_catalog_and_tool_belong_only_to_worker(tmp_path: Path) -> None:
    config = _config(tmp_path, skills=True)
    skill = config.workspace / ".aeloon-core" / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: demo\ndescription: Demo workflow\n---\nFollow the trusted workflow.\n",
        encoding="utf-8",
    )
    app = AeloonCoreOrchestrator(config)
    spawned = await app.worker_control.spawn_worker(
        base_session_id="master",
        worker_type_id="explorer",
        objective="inspect",
        idempotency_key="spawn",
        detached=True,
    )
    tools, _ = await app._build_worker_tools(app.workers.get_run(spawned["run_id"]))

    assert app.master_observation_tools.get("skill") is None
    assert tools.get("skill") is not None
    assert "demo" in str(app.skills.format_guidance())


def test_current_run_keeps_full_skill_result_and_later_run_can_lazy_load() -> None:
    large_skill = "trusted workflow\n" * 100
    messages = [
        {"role": "system", "content": "worker"},
        {"role": "user", "content": "objective"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "skill-call", "function": {"name": "skill"}}],
        },
        {
            "role": "tool",
            "name": "skill",
            "tool_call_id": "skill-call",
            "content": large_skill,
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "skill", "description": "load", "parameters": {}},
        }
    ]
    state = LightweightState.from_messages(messages, active_tools=["skill"], max_iterations=5)
    processor = MinimalContextProcessor(max_tool_result_chars=100)

    current = processor.process(state=state, messages=messages, tools=tools)
    assert current.messages[-1]["content"] == large_skill

    later_messages = [*messages, {"role": "user", "content": "next objective"}]
    later = processor.process(state=state, messages=later_messages, tools=tools)
    skill_result = next(
        message
        for message in later.messages
        if message.get("role") == "tool" and message.get("name") == "skill"
    )
    assert str(skill_result["content"]).startswith(LAZY_TOOL_RESULT_MARKER)


def test_v2_master_strips_old_skill_calls_and_results() -> None:
    messages = [
        {"role": "system", "content": "master"},
        {"role": "user", "content": "old task"},
        {
            "role": "assistant",
            "content": "I loaded a workflow.",
            "tool_calls": [
                {
                    "id": "skill-call",
                    "function": {"name": "skill", "arguments": '{"name":"demo"}'},
                },
                {
                    "id": "read-call",
                    "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                },
            ],
        },
        {
            "role": "tool",
            "name": "skill",
            "tool_call_id": "skill-call",
            "content": "full old Skill body",
        },
        {
            "role": "tool",
            "name": "read",
            "tool_call_id": "read-call",
            "content": "README contents",
        },
    ]

    cleaned = strip_skill_tool_history(messages)

    serialized = str(cleaned)
    assert "full old Skill body" not in serialized
    assert "skill-call" not in serialized
    assert "read-call" in serialized
    assert "README contents" in serialized
    assert "I loaded a workflow." in serialized
