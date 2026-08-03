from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aeloon_core.harness import (
    DEEPSEEK_V4_FLASH,
    AgentHarness,
    AssistantMessage,
    JsonlSessionRepository,
    ScriptedProvider,
    TextContent,
    Usage,
    UserMessage,
)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        (TextContent(text),),
        provider="deepseek",
        model="deepseek-v4-flash",
        usage=Usage(input=4, output=2, total_tokens=6),
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
    harness = AgentHarness(
        provider=provider,
        model=DEEPSEEK_V4_FLASH,
        cwd=str(tmp_path),
        session=session,
    )

    result = await harness.navigate_tree(target, summarize=True, label="abandoned")

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
