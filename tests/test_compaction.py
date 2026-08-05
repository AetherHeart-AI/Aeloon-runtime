from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aeloon_core.config import Config
from aeloon_core.core import (
    DEEPSEEK_V4_FLASH,
    AssistantMessage,
    ContextPolicy,
    ScriptedProvider,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
    estimate_context_tokens,
)
from aeloon_core.core.compaction import should_compact
from aeloon_core.runtime import JsonlSessionRepository
from aeloon_core.runtime.agent import SessionAgent
from aeloon_core.runtime.compaction import CompactionSettings, prepare_compaction


def _assistant(text: str, tokens: int = 0) -> AssistantMessage:
    return AssistantMessage(
        (TextContent(text),),
        provider="deepseek",
        model="deepseek-v4-flash",
        usage=Usage(input=tokens, output=1, total_tokens=tokens + 1 if tokens else 0),
    )


def _session_agent(
    tmp_path: Path,
    session,
    provider,
    *,
    model=DEEPSEEK_V4_FLASH,
    settings: CompactionSettings | None = None,
) -> SessionAgent:
    selected = settings or CompactionSettings()
    config = Config(
        workspace=tmp_path,
        data_dir=tmp_path,
        agent={
            "model": model.id,
            "compaction": {
                "enabled": selected.enabled,
                "reserve_tokens": selected.reserve_tokens,
                "keep_recent_tokens": selected.keep_recent_tokens,
            },
        },
    ).normalized()

    class StaticCatalog:
        async def model(self, _model_id):
            return model

        def provider(self, _model):
            return provider

    return SessionAgent(config=config, session=session, catalog=StaticCatalog())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_turn_safe_compaction_and_previous_summary_merge(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(4):
        await session.append_message(UserMessage(f"user {index} " + "x" * 80))
        await session.append_message(_assistant(f"assistant {index} " + "y" * 80))
    settings = CompactionSettings(reserve_tokens=64, keep_recent_tokens=20)

    preparation = await prepare_compaction(session, settings)

    assert preparation is not None
    assert preparation.messages_to_summarize[0].role == "user"
    assert preparation.retained_tail[0].role == "user"
    assert len(preparation.messages_to_summarize) % 2 == 0

    provider = ScriptedProvider([_assistant("structured checkpoint", tokens=12)])
    agent = _session_agent(tmp_path, session, provider, settings=settings)
    result = await agent.compact()
    assert result.summary.startswith("structured checkpoint")
    assert (await session.find_entries("compaction"))[-1]["fromHook"] is False

    await session.append_message(UserMessage("new older message " + "z" * 100))
    await session.append_message(_assistant("new older answer " + "z" * 100))
    await session.append_message(UserMessage("new recent"))
    await session.append_message(_assistant("new recent answer"))
    second = await prepare_compaction(session, settings)
    assert second is not None
    assert second.previous_summary == result.summary
    assert any(
        isinstance(message, UserMessage) and message.content.startswith("user 3")
        for message in second.messages_to_summarize
    )


def test_token_estimation_prefers_last_valid_assistant_usage_plus_tail() -> None:
    messages = [
        UserMessage("x" * 400),
        _assistant("answer", tokens=1_000),
        UserMessage("y" * 40),
    ]
    assert estimate_context_tokens(messages) == 1_001 + 10
    assert should_compact(90, 100, ContextPolicy(reserve_tokens=16))
    assert not should_compact(
        90,
        100,
        ContextPolicy(enabled=False, reserve_tokens=16),
    )


@pytest.mark.asyncio
async def test_explicit_compaction_uses_runtime_compactor_port(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"user {index} " + "x" * 50))
        await session.append_message(_assistant(f"assistant {index}"))
    provider = ScriptedProvider([_assistant("from runtime compactor")])
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        settings=CompactionSettings(keep_recent_tokens=5),
    )

    result = await agent.compact("focus")

    assert result.summary == "from runtime compactor"
    assert len(provider.requests) == 1
    entry = (await session.find_entries("compaction"))[-1]
    assert entry["fromHook"] is False


@pytest.mark.asyncio
async def test_context_overflow_compacts_once_then_retries(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"old user {index} " + "x" * 80))
        await session.append_message(_assistant(f"old answer {index} " + "y" * 80))
    overflow = AssistantMessage(
        (),
        provider="deepseek",
        model="deepseek-v4-flash",
        stop_reason="error",
        error_message="maximum context length exceeded",
    )
    provider = ScriptedProvider(
        [overflow, _assistant("overflow checkpoint", tokens=20), _assistant("recovered")]
    )
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        settings=CompactionSettings(keep_recent_tokens=10),
    )
    events: list[tuple[str, dict]] = []
    agent.subscribe(lambda event: events.append((event.type, event.data)))

    result = await agent.prompt("continue", run_id="overflow")

    assert result.final_message.text == "recovered"
    assert len(provider.requests) == 3
    assert len(await session.find_entries("compaction")) == 1
    assert any(kind == "compaction_start" and data["reason"] == "overflow" for kind, data in events)
    assert any(kind == "compaction_end" and data["willRetry"] is True for kind, data in events)


@pytest.mark.asyncio
async def test_threshold_compaction_runs_automatically_after_turn(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"old user {index} " + "x" * 80))
        await session.append_message(_assistant(f"old answer {index} " + "y" * 80))
    provider = ScriptedProvider(
        [_assistant("main answer", tokens=90), _assistant("automatic checkpoint", tokens=5)]
    )
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=100),
        settings=CompactionSettings(reserve_tokens=10, keep_recent_tokens=10),
    )
    events: list[tuple[str, dict]] = []
    agent.subscribe(lambda event: events.append((event.type, event.data)))

    result = await agent.prompt("new prompt", run_id="threshold")

    assert result.final_message.text == "main answer"
    assert len(await session.find_entries("compaction")) == 1
    assert any(
        kind == "compaction_start" and data["reason"] == "threshold" for kind, data in events
    )


@pytest.mark.asyncio
async def test_compaction_records_files_read_and_modified(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("inspect and edit"))
    await session.append_message(
        AssistantMessage(
            (
                ToolCall("read", "read", {"path": "src/a.py"}),
                ToolCall("write", "write", {"path": "src/b.py", "content": "x"}),
            ),
            provider="deepseek",
            model="deepseek-v4-flash",
            stop_reason="toolUse",
        )
    )
    await session.append_message(UserMessage("recent"))
    await session.append_message(_assistant("recent answer"))
    agent = _session_agent(
        tmp_path,
        session,
        ScriptedProvider([_assistant("checkpoint")]),
        settings=CompactionSettings(keep_recent_tokens=1),
    )

    result = await agent.compact()

    assert result.details == {
        "readFiles": ["src/a.py"],
        "modifiedFiles": ["src/b.py"],
    }
    assert "Files read:\n- src/a.py" in result.summary
    assert "Files modified:\n- src/b.py" in result.summary
