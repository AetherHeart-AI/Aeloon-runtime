from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.config import Config, save_config
from aeloon_core.core import (
    AssistantMessage,
    TextContent,
    ToolCall,
)
from aeloon_core.rpc import AeloonRpcAdapter
from aeloon_core.runtime import (
    PRESENT_FILES_TOOL_NAME,
    Artifact,
    JsonlSessionRepository,
    PresentFilesTool,
    ProviderManager,
    RuntimeService,
)
from aeloon_core.runtime.providers.testing import ScriptedProvider
from aeloon_core.tool import BuiltinToolSet

ALL_TOOL_NAMES = BuiltinToolSet.all_names
DEFAULT_ACTIVE_TOOLS = BuiltinToolSet.default_active_names


@pytest.mark.asyncio
async def test_present_files_normalizes_supported_deliverables(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    presentation = output / "Quarterly.PPTX"
    presentation.write_bytes(b"presentation")
    document = output / "brief.md"
    document.write_text("# Brief\n", encoding="utf-8")
    tool = PresentFilesTool(tmp_path)

    result = await tool.execute(
        "present",
        {"paths": ["output/Quarterly.PPTX", str(document), "output/Quarterly.PPTX"]},
        None,
    )

    assert tool.name == PRESENT_FILES_TOOL_NAME
    assert result.details["artifacts"] == [
        Artifact(
            path="output/Quarterly.PPTX",
            name="Quarterly.PPTX",
            mime_type=("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
            size_bytes=len(b"presentation"),
            kind="presentation",
        ),
        Artifact(
            path="output/brief.md",
            name="brief.md",
            mime_type="text/markdown",
            size_bytes=len("# Brief\n"),
            kind="document",
        ),
    ]


@pytest.mark.asyncio
async def test_present_files_rejects_unsafe_or_intermediate_paths(tmp_path: Path) -> None:
    tool = PresentFilesTool(tmp_path)
    (tmp_path / "generator.py").write_text("print('generate')", encoding="utf-8")
    directory = tmp_path / "folder"
    directory.mkdir()
    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"pdf")
    (tmp_path / "outside-link.pdf").symlink_to(outside)

    with pytest.raises(ValueError, match="Unsupported deliverable format"):
        await tool.execute("source", {"paths": ["generator.py"]}, None)
    with pytest.raises(ValueError, match="not a file"):
        await tool.execute("directory", {"paths": ["folder"]}, None)
    with pytest.raises(ValueError, match="inside the current workspace"):
        await tool.execute("outside", {"paths": [str(outside)]}, None)
    with pytest.raises(ValueError, match="inside the current workspace"):
        await tool.execute("symlink", {"paths": ["outside-link.pdf"]}, None)
    with pytest.raises(ValueError, match="does not exist"):
        await tool.execute("missing", {"paths": ["missing.pdf"]}, None)
    with pytest.raises(ValueError, match="1 to 24"):
        await tool.execute("many", {"paths": ["missing.pdf"] * 25}, None)


@pytest.mark.asyncio
async def test_artifact_delivery_is_display_only_session_state(tmp_path: Path) -> None:
    session = await JsonlSessionRepository(tmp_path).create(cwd=tmp_path)
    await session.append_artifact_delivery(
        run_id="run",
        tool_call_id="present",
        artifacts=[{"path": "report.pdf", "name": "report.pdf"}],
    )

    assert (await session.get_entries())[-1]["type"] == "artifact_delivery"
    assert (await session.build_context()).messages == ()
    assert (await session.stats())["messageCount"] == 0


@pytest.mark.asyncio
async def test_runtime_projects_presented_files_live_and_from_history(tmp_path: Path) -> None:
    report = tmp_path / "report.pptx"
    report.write_bytes(b"presentation")
    (tmp_path / "generator.py").write_text("print('generate')", encoding="utf-8")
    config_path = tmp_path / "config.json"
    save_config(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            agent={"model": "deepseek/deepseek-v4-flash"},
        ),
        config_path,
    )
    provider = ScriptedProvider(
        [
            AssistantMessage(
                (ToolCall("present", PRESENT_FILES_TOOL_NAME, {"paths": ["report.pptx"]}),),
                "deepseek",
                "deepseek/deepseek-v4-flash",
                stop_reason="toolUse",
            ),
            AssistantMessage(
                (TextContent("The presentation is ready."),),
                "deepseek",
                "deepseek/deepseek-v4-flash",
            ),
        ]
    )
    runtime = RuntimeService(
        config_path=config_path,
        provider_manager_factory=lambda config: ProviderManager(
            config,
            driver_factories={"deepseek": lambda *_args: provider},
        ),
    )
    rpc = AeloonRpcAdapter(runtime)
    session = await rpc.dispatch(
        "session.create", {"session_id": "artifact-thread", "workspace": str(tmp_path)}
    )
    await rpc.dispatch(
        "session.configure",
        {"session_id": session["session_id"], "active_tools": []},
    )
    started = await rpc.dispatch(
        "turn.start",
        {
            "session_id": session["session_id"],
            "input": {"kind": "prompt", "text": "Create a presentation"},
        },
    )
    operation = runtime._operation({"operation_id": started["operation_id"]})
    assert operation.task is not None
    await operation.task

    first_context = provider.requests[0][1]
    assert [tool["name"] for tool in first_context.tools] == [PRESENT_FILES_TOOL_NAME]
    assert "present_files exactly once" in first_context.system_prompt

    completed = next(
        event
        for event in rpc._events
        if event["name"] == "tool.completed" and event["operation_id"] == started["operation_id"]
    )
    assert completed["payload"]["patch"]["artifacts"][0]["path"] == "report.pptx"
    snapshot = await rpc.dispatch("session.get", {"session_id": session["session_id"]})
    artifacts = snapshot["timeline"][0]["blocks"][0]["artifacts"]
    assert [artifact["path"] for artifact in artifacts] == ["report.pptx"]
    assert "generator.py" not in str(artifacts)
    entries = await (await runtime.repository.open(session["session_id"])).get_entries()
    delivery = next(entry for entry in entries if entry["type"] == "artifact_delivery")
    assert delivery["runId"] == started["operation_id"]
    assert delivery["toolCallId"] == "present"

    catalog = await rpc.dispatch("catalog.get")
    present = next(item for item in catalog["tools"] if item["id"] == PRESENT_FILES_TOOL_NAME)
    assert present["description"] == "Runtime-managed final deliverable tool"
    assert snapshot["state"]["active_tools"] == [PRESENT_FILES_TOOL_NAME]
    assert PRESENT_FILES_TOOL_NAME not in ALL_TOOL_NAMES
    assert DEFAULT_ACTIVE_TOOLS == ("read", "bash", "edit", "write")


def test_rpc_catalogue_advertises_structured_artifacts() -> None:
    assert PRESENT_FILES_TOOL_NAME == "present_files"
