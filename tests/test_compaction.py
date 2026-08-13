from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from aeloon_core.config import Config
from aeloon_core.core import (
    AssistantMessage,
    AssistantStreamEvent,
    ContextPolicy,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    estimate_context_tokens,
)
from aeloon_core.core.compaction import is_context_overflow, should_compact
from aeloon_core.runtime import JsonlSessionRepository
from aeloon_core.runtime.agent import SessionAgent
from aeloon_core.runtime.compaction import (
    CompactionPreparation,
    CompactionSettings,
    compact_preparation,
    prepare_compaction,
    serialize_conversation,
)
from aeloon_core.runtime.providers import DEEPSEEK_V4_FLASH
from aeloon_core.runtime.providers.testing import ScriptedProvider


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

    class StaticManager:
        async def model(self, _model_id):
            return model

        def inference(self, _model):
            return provider

        async def close(self):
            return None

    return SessionAgent(
        config=config,
        session=session,
        provider_manager=StaticManager(),  # type: ignore[arg-type]
    )


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
    assert preparation.is_split_turn is True
    assert preparation.retained_tail[0].role == "assistant"
    assert [message.role for message in preparation.turn_prefix_messages] == ["user"]
    assert len(preparation.messages_to_summarize) % 2 == 0

    provider = ScriptedProvider(
        [
            _assistant("structured checkpoint", tokens=12),
            _assistant("turn prefix checkpoint", tokens=4),
        ]
    )
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
        isinstance(message, AssistantMessage) and message.text.startswith("assistant 3")
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
        [
            overflow,
            _assistant("overflow checkpoint", tokens=20),
            _assistant("overflow turn prefix", tokens=5),
            _assistant("recovered"),
        ]
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
    assert len(provider.requests) == 4
    assert len(await session.find_entries("compaction")) == 1
    assert any(kind == "compaction_start" and data["reason"] == "overflow" for kind, data in events)
    assert any(kind == "compaction_end" and data["willRetry"] is True for kind, data in events)


@pytest.mark.asyncio
async def test_ollama_overflow_forces_compaction_below_keep_recent(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    model = Model(
        "ollama/local",
        "Local",
        "ollama",
        context_window=128_000,
        max_output_tokens=32_768,
    )
    await session.append_message(UserMessage("analyze the document"))
    for index in range(3):
        await session.append_message(
            AssistantMessage(
                (ToolCall(f"call-{index}", "bash", {"command": f"step {index}"}),),
                provider=model.provider,
                model=model.id,
                usage=Usage(input=29_985, output=1, total_tokens=29_986),
                stop_reason="toolUse",
            )
        )
        await session.append_message(
            ToolResultMessage(
                f"call-{index}",
                "bash",
                (TextContent("x" * 6_000),),
            )
        )
    settings = CompactionSettings()
    assert await prepare_compaction(session, settings) is None

    overflow = AssistantMessage(
        (),
        provider=model.provider,
        model=model.id,
        stop_reason="error",
        error_message=(
            'InferenceError: ollama returned HTTP 400: {"error":{"code":400,'
            '"message":"request (32992 tokens) exceeds the available context size '
            '(32768 tokens), try increasing it","type":"exceed_context_size_error",'
            '"n_prompt_tokens":32992,"n_ctx":32768}}'
        ),
    )
    provider = ScriptedProvider(
        [overflow, _assistant("forced checkpoint"), _assistant("recovered")]
    )
    agent = _session_agent(tmp_path, session, provider, model=model, settings=settings)

    result = await agent.prompt("continue", run_id="ollama-overflow")

    assert result.final_message.text == "recovered"
    assert len(provider.requests) == 3
    compactions = await session.find_entries("compaction")
    assert len(compactions) == 1
    assert compactions[0]["details"]["reason"] == "overflow"


@pytest.mark.asyncio
async def test_next_prompt_recovers_after_compaction_failure(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"old {index} " + "x" * 100))
        await session.append_message(_assistant(f"answer {index} " + "y" * 100))
    overflow = AssistantMessage(
        (),
        provider="deepseek",
        model="deepseek-v4-flash",
        stop_reason="error",
        error_message="prompt is too long",
    )
    summary_failure = replace(overflow, error_message="invalid summary response")
    provider = ScriptedProvider(
        [
            overflow,
            summary_failure,
            overflow,
            _assistant("checkpoint"),
            _assistant("turn prefix checkpoint"),
            _assistant("recovered"),
        ]
    )
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        settings=CompactionSettings(keep_recent_tokens=10),
    )

    first = await agent.prompt("first", run_id="first")
    second = await agent.prompt("second", run_id="second")

    assert first.stop_reason == "error"
    assert second.final_message.text == "recovered"
    assert len(await session.find_entries("compaction")) == 1
    errors = [
        entry
        for entry in await session.find_entries("message")
        if entry["message"].get("stopReason") == "error"
    ]
    assert len(errors) == 2


@pytest.mark.asyncio
async def test_next_prompt_recovers_after_post_compaction_retry_failure(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"old {index} " + "x" * 100))
        await session.append_message(_assistant(f"answer {index} " + "y" * 100))
    overflow = AssistantMessage(
        (),
        provider="deepseek",
        model="deepseek-v4-flash",
        stop_reason="error",
        error_message="maximum context length exceeded",
    )
    retry_failure = replace(overflow, error_message="bad request")
    provider = ScriptedProvider(
        [
            overflow,
            _assistant("checkpoint"),
            _assistant("turn prefix checkpoint"),
            retry_failure,
            _assistant("later recovered"),
        ]
    )
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        settings=CompactionSettings(keep_recent_tokens=10),
    )

    first = await agent.prompt("first", run_id="first")
    second = await agent.prompt("second", run_id="second")

    assert first.stop_reason == "error"
    assert second.final_message.text == "later recovered"
    assert len(await session.find_entries("compaction")) == 1


@pytest.mark.asyncio
async def test_threshold_compaction_runs_automatically_after_turn(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"old user {index} " + "x" * 80))
        await session.append_message(_assistant(f"old answer {index} " + "y" * 80))
    provider = ScriptedProvider(
        [
            _assistant("main answer", tokens=950),
            _assistant("automatic checkpoint", tokens=5),
            _assistant("automatic turn prefix", tokens=3),
        ]
    )
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=1_000),
        settings=CompactionSettings(reserve_tokens=100, keep_recent_tokens=10),
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
async def test_failed_attempts_do_not_raise_threshold_estimate(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(
            AssistantMessage(
                (TextContent(f"partial {index} " + "x" * 10_000),),
                provider="deepseek",
                model="deepseek-v4-flash",
                usage=Usage(input=950, output=0, total_tokens=950),
                stop_reason="error",
                error_message="service unavailable",
            )
        )
    provider = ScriptedProvider([_assistant("recovered")])
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=1_000),
        settings=CompactionSettings(reserve_tokens=100, keep_recent_tokens=10),
    )
    events: list[tuple[str, dict]] = []
    agent.subscribe(lambda event: events.append((event.type, event.data)))

    result = await agent.prompt("continue", run_id="filtered-threshold")

    assert result.final_message.text == "recovered"
    assert len(provider.requests) == 1
    assert [message.role for message in provider.requests[0][1].messages] == ["user"]
    assert await session.find_entries("compaction") == []
    assert not any(kind == "compaction_start" for kind, _data in events)


@pytest.mark.parametrize(
    ("input_chars", "should_fail"),
    [(356, False), (360, False), (361, True)],
)
@pytest.mark.asyncio
async def test_single_input_uses_context_window_minus_reserve(
    tmp_path: Path,
    input_chars: int,
    should_fail: bool,
) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    provider = ScriptedProvider([] if should_fail else [_assistant("accepted")])
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=100),
        settings=CompactionSettings(enabled=False, reserve_tokens=10),
    )
    events: list[tuple[str, dict]] = []
    agent.subscribe(lambda event: events.append((event.type, event.data)))

    result = await agent.prompt("x" * input_chars, run_id=f"input-{input_chars}")

    assert (result.stop_reason == "error") is should_fail
    assert len(provider.requests) == (0 if should_fail else 1)
    entries = await session.find_entries("message")
    assert [entry["message"]["role"] for entry in entries] == ["user", "assistant"]
    if not should_fail:
        assert result.final_message.text == "accepted"
        return

    assert result.usage.total_tokens == 0
    assert result.final_message.content == ()
    assert result.final_message.error_message == (
        "User input exceeds usable context budget: estimated 91 tokens, budget 90 tokens "
        "(context window 100, reserve 10). "
        "Historical compaction cannot reduce a single oversized input."
    )
    assert entries[-1]["message"]["stopReason"] == "error"
    lifecycle = [
        kind
        for kind, _data in events
        if kind
        in {
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
            "settled",
        }
    ]
    assert lifecycle == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
        "settled",
    ]


@pytest.mark.asyncio
async def test_oversized_single_input_does_not_compact_large_history(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("old history " + "x" * 10_000))
    await session.append_message(_assistant("measured", tokens=950))
    provider = ScriptedProvider([])
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=100),
        settings=CompactionSettings(reserve_tokens=10, keep_recent_tokens=1),
    )
    events: list[tuple[str, dict]] = []
    agent.subscribe(lambda event: events.append((event.type, event.data)))

    result = await agent.prompt("x" * 361, run_id="oversized-with-history")

    assert result.stop_reason == "error"
    assert provider.requests == []
    assert await session.find_entries("compaction") == []
    assert not any(kind == "compaction_start" for kind, _data in events)
    entries = await session.find_entries("message")
    assert [entry["message"]["role"] for entry in entries[-2:]] == ["user", "assistant"]
    assert entries[-1]["message"]["stopReason"] == "error"


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
        ScriptedProvider([_assistant("checkpoint"), _assistant("turn prefix checkpoint")]),
        settings=CompactionSettings(keep_recent_tokens=1),
    )

    result = await agent.compact()

    assert result.details == {
        "readFiles": ["src/a.py"],
        "modifiedFiles": ["src/b.py"],
    }
    assert "Files read:\n- src/a.py" in result.summary
    assert "Files modified:\n- src/b.py" in result.summary


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("prompt is too long: 210000 tokens", True),
        ("Your input exceeds the context window of this model", True),
        ("the request exceeds the available context size", True),
        ("ThrottlingException: Too many tokens, please retry", False),
        ("rate limit: too many tokens", False),
    ],
)
def test_context_overflow_patterns_exclude_rate_limits(message: str, expected: bool) -> None:
    assert is_context_overflow(message) is expected


def test_context_overflow_detects_silent_and_length_zero_output() -> None:
    assert is_context_overflow(
        AssistantMessage(
            (TextContent("completed"),),
            provider="test",
            model="model",
            usage=Usage(input=101, output=1, total_tokens=102),
        ),
        100,
    )
    assert is_context_overflow(
        AssistantMessage(
            (),
            provider="test",
            model="model",
            usage=Usage(input=99, output=0, total_tokens=99),
            stop_reason="length",
        ),
        100,
    )


@pytest.mark.asyncio
async def test_compaction_always_splits_at_assistant_cut_point(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("small prefix"))
    await session.append_message(_assistant("kept suffix"))

    preparation = await prepare_compaction(
        session,
        CompactionSettings(keep_recent_tokens=1),
    )

    assert preparation is not None
    assert preparation.is_split_turn is True
    assert preparation.messages_to_summarize == ()
    assert [message.role for message in preparation.turn_prefix_messages] == ["user"]
    assert [message.role for message in preparation.retained_tail] == ["assistant"]


@pytest.mark.asyncio
async def test_compaction_can_split_a_single_oversized_tool_turn(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("large request " + "u" * 50_000))
    await session.append_message(
        AssistantMessage(
            (ToolCall("large", "read", {"path": "huge.log"}),),
            provider="deepseek",
            model="deepseek-v4-flash",
            stop_reason="toolUse",
        )
    )
    await session.append_message(
        ToolResultMessage("large", "read", (TextContent("z" * 50_000),))
    )
    await session.append_message(_assistant("recent suffix"))

    preparation = await prepare_compaction(
        session,
        CompactionSettings(keep_recent_tokens=100),
    )

    assert preparation is not None
    assert preparation.is_split_turn is True
    assert preparation.messages_to_summarize == ()
    assert [message.role for message in preparation.turn_prefix_messages] == [
        "user",
        "assistant",
        "toolResult",
    ]
    assert preparation.retained_tail[0].role == "assistant"


def test_summary_serialization_truncates_individual_tool_results() -> None:
    serialized = serialize_conversation(
        (ToolResultMessage("large", "read", (TextContent("z" * 50_000),)),)
    )

    assert serialized.count("z") <= 2_000
    assert "more characters truncated" in serialized


@pytest.mark.asyncio
async def test_rolling_summary_chunks_oversized_input_and_uses_unique_sessions() -> None:
    class SummaryProvider:
        def __init__(self) -> None:
            self.requests = []

        def stream(self, model, context, options):
            async def events():
                self.requests.append((model, context, options))
                message = AssistantMessage(
                    (TextContent(f"summary {len(self.requests)}"),),
                    provider=model.provider,
                    model=model.id,
                    usage=Usage(input=10, output=2, total_tokens=12),
                )
                yield AssistantStreamEvent("start")
                yield AssistantStreamEvent("text_delta", delta=message.text, content_index=0)
                yield AssistantStreamEvent("done", message=message)

            return events()

    provider = SummaryProvider()
    model = Model(
        "test/model",
        "Test",
        "test",
        context_window=2_000,
        max_output_tokens=1_000,
    )
    messages = tuple(UserMessage(f"turn {index} " + "x" * 20_000) for index in range(3))
    preparation = CompactionPreparation(
        first_kept_entry_id="kept",
        messages_to_summarize=messages,
        turn_prefix_messages=(),
        is_split_turn=False,
        retained_tail=(),
        tokens_before=20_000,
        previous_summary=None,
        read_files=(),
        modified_files=(),
    )

    result = await compact_preparation(
        preparation,
        inference=provider,
        model=model,
        stream_options=StreamOptions(max_retries=0),
        settings=CompactionSettings(reserve_tokens=400),
    )

    assert len(provider.requests) == 3
    assert len({request[1].session_id for request in provider.requests}) == 3
    assert {request[2].max_tokens for request in provider.requests} == {320}
    assert result.summary == "summary 3"
    assert result.usage.total_tokens == 36
    assert all(
        len(request[1].system_prompt) + len(request[1].messages[0].content) <= 1_580 * 4
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_cancelling_preflight_compaction_stops_summary_and_inference(
    tmp_path: Path,
) -> None:
    class BlockingSummaryProvider:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.requests = 0

        def stream(self, _model, _context, _options):
            async def events():
                self.requests += 1
                yield AssistantStreamEvent("start")
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()

            return events()

    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("old " + "x" * 100))
    await session.append_message(_assistant("measured", tokens=950))
    provider = BlockingSummaryProvider()
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=1_000),
        settings=CompactionSettings(reserve_tokens=100, keep_recent_tokens=1),
    )

    operation = asyncio.create_task(agent.prompt("cancel", run_id="cancel"))
    await provider.entered.wait()
    await agent.abort()
    result = await operation

    assert result.stop_reason == "aborted"
    assert provider.cancelled.is_set()
    assert provider.requests == 1
    assert await session.find_entries("compaction") == []


@pytest.mark.asyncio
async def test_successful_silent_overflow_compacts_without_retrying_answer(
    tmp_path: Path,
) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    answer = AssistantMessage(
        (TextContent("answer"),),
        provider="deepseek",
        model="deepseek-v4-flash",
        usage=Usage(input=10_001, output=1, total_tokens=10_002),
    )
    provider = ScriptedProvider([answer, _assistant("checkpoint")])
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=10_000),
        settings=CompactionSettings(reserve_tokens=1_000, keep_recent_tokens=1),
    )
    events: list[tuple[str, dict]] = []
    agent.subscribe(lambda event: events.append((event.type, event.data)))

    result = await agent.prompt("x" * 500, run_id="silent")

    assert result.final_message.text == "answer"
    assert len(provider.requests) == 2
    assert len(await session.find_entries("compaction")) == 1
    assert any(
        kind == "compaction_end"
        and data["reason"] == "overflow"
        and data["willRetry"] is False
        for kind, data in events
    )


@pytest.mark.asyncio
async def test_length_zero_output_overflow_compacts_then_retries(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    for index in range(3):
        await session.append_message(UserMessage(f"old {index} " + "x" * 100))
        await session.append_message(_assistant(f"answer {index} " + "y" * 100))
    length_overflow = AssistantMessage(
        (),
        provider="deepseek",
        model="deepseek-v4-flash",
        usage=Usage(input=9_900, output=0, total_tokens=9_900),
        stop_reason="length",
    )
    provider = ScriptedProvider(
        [
            length_overflow,
            _assistant("checkpoint"),
            _assistant("turn prefix checkpoint"),
            _assistant("recovered"),
        ]
    )
    agent = _session_agent(
        tmp_path,
        session,
        provider,
        model=replace(DEEPSEEK_V4_FLASH, context_window=10_000),
        settings=CompactionSettings(reserve_tokens=1_000, keep_recent_tokens=10),
    )

    result = await agent.prompt("continue", run_id="length")

    assert result.final_message.text == "recovered"
    assert len(provider.requests) == 4
    assert len(await session.find_entries("compaction")) == 1
