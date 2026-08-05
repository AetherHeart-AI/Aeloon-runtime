from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.bridge import BridgeRpcAdapter
from aeloon_core.config import Config, save_config
from aeloon_core.core import (
    AssistantMessage,
    ScriptedProvider,
    TextContent,
)
from aeloon_core.runtime import ProviderCatalog, RuntimeService
from aeloon_core.runtime.rename import normalize_session_title


def test_normalize_session_title_removes_model_formatting() -> None:
    assert normalize_session_title('Title: "修复图片附件识别。"') == "修复图片附件识别"
    assert normalize_session_title("# Session title: Improve sidebar\nExtra") == "Improve sidebar"
    assert normalize_session_title("  ") is None


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
        catalog_factory=lambda config: ProviderCatalog(
            config,
            local_provider_factory=provider_factory,
        ),
    )
    bridge = BridgeRpcAdapter(runtime)
    metadata = await bridge.dispatch(
        "session.create", {"workspace": str(tmp_path), "title": "New chat"}
    )
    started = await bridge.dispatch(
        "turn.start",
        {
            "session_id": metadata["session_id"],
            "input": {"kind": "prompt", "text": "检查图片为什么没有被模型识别"},
        },
    )
    operation = runtime._operation({"operation_id": started["operation_id"]})
    assert operation.task is not None
    await operation.task

    snapshot = await bridge.dispatch("session.get", {"session_id": metadata["session_id"]})
    assert snapshot["metadata"]["title"] == "修复图片附件识别"
    assert len(providers[0].requests) == 2
    rename_context = providers[0].requests[1][1]
    assert rename_context.tools == ()
    assert "检查图片为什么没有被模型识别" in str(rename_context.messages[0].content)
    assert all(
        entry.get("message", {}).get("content") != "修复图片附件识别"
        for entry in await (await runtime.repository.open(metadata["session_id"])).get_entries()
    )
    renamed_events = [event for event in bridge._events if event["name"] == "session.renamed"]
    assert renamed_events[-1]["payload"] == {
        "title": "修复图片附件识别",
        "source": "automatic",
    }


@pytest.mark.asyncio
async def test_manual_session_rename_emits_bridge_event(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    runtime = RuntimeService(config_path=config_path)
    bridge = BridgeRpcAdapter(runtime)
    metadata = await bridge.dispatch("session.create", {"workspace": str(tmp_path)})

    result = await bridge.dispatch(
        "session.rename",
        {"session_id": metadata["session_id"], "title": "Manual title"},
    )

    assert result["title"] == "Manual title"
    assert bridge._events[-1]["name"] == "session.renamed"
    assert bridge._events[-1]["payload"] == {
        "title": "Manual title",
        "source": "manual",
    }
