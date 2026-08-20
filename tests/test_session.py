from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aeloon_runtime.config import Config
from aeloon_runtime.core import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from aeloon_runtime.runtime import JsonlSessionRepository
from aeloon_runtime.runtime.agent import SessionAgent
from aeloon_runtime.runtime.providers import DEEPSEEK_V4_FLASH
from aeloon_runtime.runtime.providers.testing import ScriptedProvider


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        (TextContent(text),),
        provider="deepseek",
        model="deepseek-v4-flash",
        usage=Usage(input=4, output=2, total_tokens=6),
    )


@pytest.mark.asyncio
async def test_session_stats_report_context_mix_and_cache_hits(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("u" * 8))
    await session.append_message(
        AssistantMessage(
            (TextContent("a" * 8),),
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=Usage(
                input=6,
                output=2,
                cache_read=4,
                cache_write=2,
                total_tokens=12,
                cost={"total": 1.25},
            ),
        )
    )
    await session.append_message(ToolResultMessage("call", "read", (TextContent("t" * 8),)))
    await session.append_message(UserMessage("u" * 4))

    stats = await session.stats(context_window=100)

    assert stats["messageCount"] == 4
    assert stats["totalTokens"] == 12
    assert stats["costTotal"] == 1.25
    assert stats["contextWindow"] == {
        "usedTokens": 15,
        "windowTokens": 100,
        "remainingTokens": 85,
        "usagePercent": 15.0,
    }
    message_types = stats["messageTypes"]
    assert sum(item["estimatedTokens"] for item in message_types.values()) == 15
    assert message_types["system"] == {
        "messageCount": 0,
        "estimatedTokens": 8,
        "percentage": 53.34,
    }
    assert sum(item["percentage"] for item in message_types.values()) == 100.0
    assert message_types["user"]["messageCount"] == 2
    assert message_types["user"]["percentage"] == 20.0
    assert message_types["assistant"]["messageCount"] == 1
    assert message_types["toolResult"]["messageCount"] == 1
    assert stats["cache"] == {
        "inputTokens": 6,
        "readTokens": 4,
        "writeTokens": 2,
        "cacheableTokens": 10,
        "hitTokenPercent": 40.0,
        "requestCount": 1,
        "hitRequestCount": 1,
        "hitRequestPercent": 100.0,
    }


@pytest.mark.asyncio
async def test_session_stats_handle_empty_context_and_unknown_window(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)

    stats = await session.stats()

    assert stats["contextWindow"] == {
        "usedTokens": 0,
        "windowTokens": None,
        "remainingTokens": None,
        "usagePercent": None,
    }
    assert all(
        item == {"messageCount": 0, "estimatedTokens": 0, "percentage": 0.0}
        for item in stats["messageTypes"].values()
    )
    assert stats["cache"]["hitTokenPercent"] == 0.0
    assert stats["cache"]["hitRequestPercent"] == 0.0


@pytest.mark.asyncio
async def test_incremental_stats_match_full_recomputation(tmp_path: Path) -> None:
    repository = JsonlSessionRepository(tmp_path)
    session = await repository.create(cwd=tmp_path, session_id="incremental-stats")
    await session.append_message(UserMessage("first"))
    await session.append_message(_assistant("answer"))
    assert await session.stats(context_window=1_000) == await session._stats_full(
        context_window=1_000
    )

    await session.append_message(UserMessage("second"))
    await session.append_message(_assistant("second answer"))
    assert await session.stats(context_window=1_000) == await session._stats_full(
        context_window=1_000
    )

    old_leaf = await session.get_leaf_id()
    abandoned = await session.append_message(UserMessage("abandoned"))
    await session.set_leaf_id(old_leaf)
    await session.append_message(UserMessage("alternate"))
    branched = await session.stats(context_window=1_000)
    assert branched == await session._stats_full(context_window=1_000)
    assert branched["messageCount"] == 6
    assert all(
        message.content != "abandoned" for message in (await session.build_context()).messages
    )
    assert abandoned != await session.get_leaf_id()

    retained = await session.append_message(UserMessage("retained"))
    await session.append_compaction(
        summary="checkpoint",
        tokens_before=100,
        first_kept_entry_id=retained,
    )
    assert await session.stats(context_window=1_000) == await session._stats_full(
        context_window=1_000
    )

    reopened = await repository.open("incremental-stats")
    assert await reopened.stats(context_window=1_000) == await reopened._stats_full(
        context_window=1_000
    )


@pytest.mark.asyncio
async def test_jsonl_v3_is_message_durable_and_recovers_truncated_tail(tmp_path: Path) -> None:
    repository = JsonlSessionRepository(tmp_path)
    session = await repository.create(cwd=tmp_path, session_id="durable")
    await session.append_message(UserMessage("hello"))
    await session.append_message(_assistant("world"))

    lines = session.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["version"] == 3
    assert [json.loads(line)["type"] for line in lines[1:]] == ["message", "message"]
    with session.path.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"message"')

    reopened = await repository.open("durable")
    context = await reopened.build_context()
    assert [message.role for message in context.messages] == ["user", "assistant"]
    assert context.messages[-1].text == "world"


@pytest.mark.asyncio
async def test_jsonl_appends_coalesce_syncs_and_terminal_entries_force_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = JsonlSessionRepository(tmp_path)
    session = await repository.create(cwd=tmp_path, session_id="batched-sync")
    syncs = 0

    def record_sync(_path: Path) -> None:
        nonlocal syncs
        syncs += 1

    monkeypatch.setattr("aeloon_runtime.runtime.session._fsync_path", record_sync)
    for index in range(100):
        await session.append_message(UserMessage(str(index)))
    await asyncio.sleep(0.05)
    assert syncs == 1

    await session.append_run_end(run_id="run", status="completed")
    assert syncs == 2
    reopened = await repository.open("batched-sync")
    assert len(await reopened.get_entries()) == 101


@pytest.mark.asyncio
async def test_session_tree_leaf_navigation_and_branch_recovery(tmp_path: Path) -> None:
    repository = JsonlSessionRepository(tmp_path)
    session = await repository.create(cwd=tmp_path, session_id="tree")
    first = await session.append_message(UserMessage("first"))
    answer = await session.append_message(_assistant("answer"))
    abandoned = await session.append_message(UserMessage("abandoned"))
    await session.set_leaf_id(answer)
    alternate = await session.append_message(UserMessage("alternate"))
    await session.set_label(alternate, "chosen")
    await session.set_name("demo")

    assert first != answer != abandoned != alternate
    assert await session.get_label(alternate) == "chosen"
    assert await session.get_name() == "demo"
    branch = await session.get_branch()
    branch_messages = [entry for entry in branch if entry["type"] == "message"]
    assert [entry["message"]["content"] for entry in branch_messages] == [
        "first",
        [{"type": "text", "text": "answer"}],
        "alternate",
    ]
    assert all(entry["id"] != abandoned for entry in branch)

    reopened = await repository.open("tree")
    assert await reopened.get_leaf_id() == await session.get_leaf_id()
    assert [message.role for message in (await reopened.build_context()).messages] == [
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.asyncio
async def test_compaction_summary_precedes_retained_tail_in_context(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    old_user = await session.append_message(UserMessage("old"))
    await session.append_message(_assistant("old answer"))
    retained_user = await session.append_message(UserMessage("recent"))
    await session.append_message(_assistant("recent answer"))
    await session.append_compaction(
        summary="checkpoint",
        tokens_before=100,
        first_kept_entry_id=retained_user,
    )

    messages = (await session.build_context()).messages
    assert old_user not in {entry["id"] for entry in await session.get_branch()}
    assert messages[0].role == "user"
    assert "<summary>\ncheckpoint\n</summary>" in messages[0].content
    assert messages[1].content == "recent"
    assert messages[2].text == "recent answer"


@pytest.mark.asyncio
async def test_compaction_boundary_ignores_stale_usage_in_effective_stats(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    retained = await session.append_message(UserMessage("recent"))
    await session.append_message(
        AssistantMessage(
            (TextContent("old measured answer"),),
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=Usage(input=890, output=10, total_tokens=900),
        )
    )
    await session.append_compaction(
        summary="checkpoint",
        tokens_before=900,
        first_kept_entry_id=retained,
    )

    context = await session.build_context()
    stale_stats = await session.stats(context_window=1_000)

    assert context.compaction_boundary_ms is not None
    assert isinstance(stale_stats["contextWindow"]["usedTokens"], int)
    assert stale_stats["contextWindow"]["usedTokens"] < 900

    await session.append_message(UserMessage("next"))
    await session.append_message(
        AssistantMessage(
            (TextContent("fresh"),),
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=Usage(input=20, output=10, total_tokens=30),
        )
    )
    fresh_stats = await session.stats(context_window=1_000)
    assert fresh_stats["contextWindow"]["usedTokens"] == 30


@pytest.mark.asyncio
async def test_failed_assistant_is_durable_but_filtered_from_next_inference(
    tmp_path: Path,
) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    failed = AssistantMessage(
        (),
        provider="deepseek",
        model="deepseek-v4-flash",
        stop_reason="error",
        error_message="bad request",
    )
    provider = ScriptedProvider([failed, _assistant("recovered")])

    class StaticManager:
        async def model(self, _model_id):
            return DEEPSEEK_V4_FLASH

        def inference(self, _model):
            return provider

        async def close(self):
            return None

    agent = SessionAgent(
        config=Config(
            workspace=tmp_path,
            data_dir=tmp_path,
            agent={
                "model": DEEPSEEK_V4_FLASH.id,
                "compaction": {"enabled": False},
            },
        ).normalized(),
        session=session,
        provider_manager=StaticManager(),  # type: ignore[arg-type]
    )

    first = await agent.prompt("first", run_id="first")
    second = await agent.prompt("second", run_id="second")

    assert first.stop_reason == "error"
    assert second.final_message.text == "recovered"
    persisted = await session.find_entries("message")
    assert any(
        entry["message"].get("stopReason") == "error"
        for entry in persisted
        if entry["message"].get("role") == "assistant"
    )
    replayed = provider.requests[1][1].messages
    assert [message.role for message in replayed] == ["user", "user"]


@pytest.mark.asyncio
async def test_repository_list_uses_separate_harness_sessions_directory(tmp_path: Path) -> None:
    legacy = tmp_path / "sessions"
    legacy.mkdir()
    (legacy / "old.json").write_text("{}", encoding="utf-8")
    repository = JsonlSessionRepository(tmp_path)
    session = await repository.create(cwd=tmp_path, session_id="new")

    listed = await repository.list(cwd=tmp_path)

    assert repository.directory == tmp_path / "harness-sessions"
    assert [item.id for item in listed] == ["new"]
    assert session.path.parent == repository.directory


@pytest.mark.asyncio
async def test_tree_navigation_can_summarize_and_label_abandoned_branch(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_message(UserMessage("root"))
    target = await session.append_message(_assistant("root answer"))
    await session.append_message(UserMessage("branch work"))
    await session.append_message(_assistant("branch result"))
    provider = ScriptedProvider([_assistant("branch checkpoint")])

    class StaticManager:
        async def model(self, _model_id):
            return DEEPSEEK_V4_FLASH

        def inference(self, _model):
            return provider

        async def close(self):
            return None

    agent = SessionAgent(
        config=Config(
            workspace=tmp_path,
            data_dir=tmp_path,
            agent={"model": DEEPSEEK_V4_FLASH.id},
        ).normalized(),
        session=session,
        provider_manager=StaticManager(),  # type: ignore[arg-type]
    )

    result = await agent.navigate_tree(target, summarize=True, label="abandoned")

    assert result["cancelled"] is False
    summary_id = result["summaryEntryId"]
    assert summary_id is not None
    summary = await session.get_entry(summary_id)
    assert summary is not None
    assert "branch checkpoint" in summary["summary"]
    assert await session.get_label(summary_id) == "abandoned"
    context = await session.build_context()
    assert "summary of a branch" in context.messages[-1].content
    assert "branch checkpoint" in context.messages[-1].content


@pytest.mark.asyncio
async def test_model_thinking_and_active_tool_changes_restore_from_tree(tmp_path: Path) -> None:
    repository = JsonlSessionRepository(tmp_path)
    session = await repository.create(cwd=tmp_path, session_id="settings")
    await session.append_model_change("deepseek", "deepseek-v4-pro")
    await session.append_thinking_level_change("high")
    await session.append_active_tools_change(("read", "grep"))
    await session.append_message(UserMessage("hello"))

    context = await (await repository.open("settings")).build_context()

    assert context.model == ("deepseek", "deepseek-v4-pro")
    assert context.thinking_level == "high"
    assert context.active_tool_names == ("read", "grep")

    retained = await session.append_message(UserMessage("retained"))
    await session.append_message(replace(_assistant("retained answer"), model="deepseek-v4-pro"))
    await session.append_compaction(
        summary="settings checkpoint",
        tokens_before=50,
        first_kept_entry_id=retained,
    )
    compacted = await (await repository.open("settings")).build_context()
    assert compacted.model == ("deepseek", "deepseek-v4-pro")
    assert compacted.thinking_level == "high"
    assert compacted.active_tool_names == ("read", "grep")
