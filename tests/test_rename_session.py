from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.config import Config, save_config
from aeloon_core.core import AssistantMessage, TextContent
from aeloon_core.rpc import AeloonRpcAdapter
from aeloon_core.runtime import ProviderManager, RuntimeService
from aeloon_core.runtime.providers.testing import ScriptedProvider
from aeloon_core.runtime.rename import fallback_session_title, normalize_session_title


def test_normalize_session_title_removes_model_formatting() -> None:
    assert normalize_session_title('Title: "修复图片附件识别。"') == "修复图片附件识别"
    assert normalize_session_title("# Session title: Improve sidebar\nExtra") == "Improve sidebar"
    assert normalize_session_title("  ") is None


def test_fallback_session_title_uses_non_generic_user_prompt() -> None:
    assert fallback_session_title("修复新消息没有生成标题。") == "修复新消息没有生成标题"
    assert fallback_session_title("New chat") is None


@pytest.mark.asyncio
async def test_first_completed_turn_generates_one_semantic_session_title(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            agent={"model": "deepseek/deepseek-v4-flash"},
        ),
        config_path,
    )
    providers: list[ScriptedProvider] = []

    def provider_factory(**_kwargs):
        provider = ScriptedProvider(
            [
                AssistantMessage(
                    (TextContent("I fixed the attachment pipeline."),),
                    "deepseek",
                    "deepseek/deepseek-v4-flash",
                ),
                AssistantMessage(
                    (TextContent("修复图片附件识别"),),
                    "deepseek",
                    "deepseek/deepseek-v4-flash",
                ),
            ]
        )
        providers.append(provider)
        return provider

    runtime = RuntimeService(
        config_path=config_path,
        provider_manager_factory=lambda config: ProviderManager(
            config,
            driver_factories={"deepseek": lambda *_args: provider_factory()},
        ),
    )
    rpc = AeloonRpcAdapter(runtime)
    metadata = await rpc.dispatch(
        "session.create",
        {"session_id": "rename-thread", "workspace": str(tmp_path), "title": "New chat"},
    )
    started = await rpc.dispatch(
        "turn.start",
        {
            "session_id": metadata["session_id"],
            "input": {"kind": "prompt", "text": "检查图片为什么没有被模型识别"},
        },
    )
    operation = runtime._operation({"operation_id": started["operation_id"]})
    assert operation.task is not None
    await operation.task

    snapshot = await rpc.dispatch("session.get", {"session_id": metadata["session_id"]})
    assert snapshot["metadata"]["title"] == "修复图片附件识别"
    assert len(providers[0].requests) == 2
    rename_context = providers[0].requests[1][1]
    assert rename_context.tools == ()
    assert "检查图片为什么没有被模型识别" in str(rename_context.messages[0].content)
    assert all(
        entry.get("message", {}).get("content") != "修复图片附件识别"
        for entry in await (await runtime.repository.open(metadata["session_id"])).get_entries()
    )
    renamed_events = [event for event in rpc._events if event["name"] == "session.renamed"]
    assert renamed_events[-1]["payload"] == {
        "title": "修复图片附件识别",
        "source": "automatic",
    }


@pytest.mark.asyncio
async def test_failed_title_inference_falls_back_to_user_prompt(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            agent={"model": "deepseek/deepseek-v4-flash"},
        ),
        config_path,
    )

    def provider_factory(**_kwargs):
        return ScriptedProvider(
            [
                AssistantMessage(
                    (TextContent("The user-facing operation completed."),),
                    "deepseek",
                    "deepseek/deepseek-v4-flash",
                ),
                AssistantMessage(
                    (),
                    "deepseek",
                    "deepseek/deepseek-v4-flash",
                    stop_reason="error",
                    error_message="Title generation unavailable",
                ),
            ]
        )

    runtime = RuntimeService(
        config_path=config_path,
        provider_manager_factory=lambda config: ProviderManager(
            config,
            driver_factories={"deepseek": lambda *_args: provider_factory()},
        ),
    )
    rpc = AeloonRpcAdapter(runtime)
    metadata = await rpc.dispatch(
        "session.create",
        {"session_id": "fallback-title", "workspace": str(tmp_path), "title": "New chat"},
    )
    started = await rpc.dispatch(
        "turn.start",
        {
            "session_id": metadata["session_id"],
            "input": {"kind": "prompt", "text": "修复新消息没有生成标题。"},
        },
    )
    operation = runtime._operation({"operation_id": started["operation_id"]})
    assert operation.task is not None
    await operation.task

    snapshot = await rpc.dispatch("session.get", {"session_id": metadata["session_id"]})
    assert snapshot["metadata"]["title"] == "修复新消息没有生成标题"
    assert rpc._events[-2]["name"] == "session.renamed"
    assert rpc._events[-2]["payload"] == {
        "title": "修复新消息没有生成标题",
        "source": "automatic",
    }


@pytest.mark.asyncio
async def test_later_completed_turn_retries_a_still_generic_title(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            agent={"model": "deepseek/deepseek-v4-flash"},
        ),
        config_path,
    )
    provider_scripts = [
        [
            AssistantMessage(
                (TextContent("Hello."),),
                "deepseek",
                "deepseek/deepseek-v4-flash",
            ),
            AssistantMessage(
                (),
                "deepseek",
                "deepseek/deepseek-v4-flash",
                stop_reason="error",
                error_message="Title generation unavailable",
            ),
        ],
        [
            AssistantMessage(
                (TextContent("The title retry is fixed."),),
                "deepseek",
                "deepseek/deepseek-v4-flash",
            ),
            AssistantMessage(
                (TextContent("修复自动标题重试"),),
                "deepseek",
                "deepseek/deepseek-v4-flash",
            ),
        ],
    ]

    def provider_factory(*_args):
        return ScriptedProvider(provider_scripts.pop(0))

    runtime = RuntimeService(
        config_path=config_path,
        provider_manager_factory=lambda config: ProviderManager(
            config,
            driver_factories={"deepseek": provider_factory},
        ),
    )
    rpc = AeloonRpcAdapter(runtime)
    metadata = await rpc.dispatch(
        "session.create",
        {"session_id": "retry-title", "workspace": str(tmp_path), "title": "New chat"},
    )

    first = await rpc.dispatch(
        "turn.start",
        {
            "session_id": metadata["session_id"],
            "input": {"kind": "prompt", "text": "New chat"},
        },
    )
    first_operation = runtime._operation({"operation_id": first["operation_id"]})
    assert first_operation.task is not None
    await first_operation.task
    session = await runtime.repository.open(metadata["session_id"])
    assert await session.get_name() is None

    second = await rpc.dispatch(
        "turn.start",
        {
            "session_id": metadata["session_id"],
            "input": {"kind": "prompt", "text": "修复自动标题重试"},
        },
    )
    second_operation = runtime._operation({"operation_id": second["operation_id"]})
    assert second_operation.task is not None
    await second_operation.task

    session = await runtime.repository.open(metadata["session_id"])
    assert await session.get_name() == "修复自动标题重试"


@pytest.mark.asyncio
async def test_manual_session_rename_emits_rpc_event(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    runtime = RuntimeService(config_path=config_path)
    rpc = AeloonRpcAdapter(runtime)
    metadata = await rpc.dispatch(
        "session.create", {"session_id": "manual-rename-thread", "workspace": str(tmp_path)}
    )

    result = await rpc.dispatch(
        "session.rename",
        {"session_id": metadata["session_id"], "title": "Manual title"},
    )

    assert result["title"] == "Manual title"
    assert rpc._events[-1]["name"] == "session.renamed"
    assert rpc._events[-1]["payload"] == {
        "title": "Manual title",
        "source": "manual",
    }
