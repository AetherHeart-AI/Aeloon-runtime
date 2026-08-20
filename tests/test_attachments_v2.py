from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from aeloon_runtime.config import Config, CustomProviderConfig, ProviderModelConfig, save_config
from aeloon_runtime.core import AssistantMessage, ImageContent, Model, TextContent
from aeloon_runtime.rpc import AeloonRpcAdapter
from aeloon_runtime.runtime import ProviderManager, RuntimeService
from aeloon_runtime.runtime.attachments import (
    AttachmentMetadataTool,
    AttachmentReadTool,
    AttachmentStore,
)
from aeloon_runtime.runtime.providers.testing import ScriptedProvider
from aeloon_runtime.runtime.types import RuntimeFailure


def _descriptor(path: Path, *, attachment_id: str = "attachment-1") -> dict[str, object]:
    return {
        "id": attachment_id,
        "type": "file",
        "display_name": "中文报告.pdf" if path.suffix == ".pdf" else "说明.txt",
        "mime_type": "application/pdf" if path.suffix == ".pdf" else "text/plain",
        "size_bytes": path.stat().st_size,
        "source_path": str(path),
    }


@pytest.mark.asyncio
async def test_attachment_store_is_atomic_private_and_restart_safe(tmp_path: Path) -> None:
    source_root = tmp_path / "workbench" / "attachments" / "upload"
    source_root.mkdir(parents=True)
    source = source_root / "content.txt"
    source.write_text("trusted attachment", encoding="utf-8")
    store_root = tmp_path / "core" / "session-attachments"
    store = AttachmentStore(store_root, image_limit=1024, file_limit=1024)

    resolved = await store.resolve_batch(
        "thread-1", [_descriptor(source)], (source_root.parent,)
    )

    assert resolved[0].canonical_path.read_text(encoding="utf-8") == "trusted attachment"
    assert resolved[0].canonical_path != source
    manifest = (store_root / "thread-1" / "manifest.json").read_text(encoding="utf-8")
    assert str(source) not in manifest
    assert "source_path" not in manifest
    restarted = AttachmentStore(store_root, image_limit=1024, file_limit=1024)
    assert (await restarted.load("thread-1"))[0].id == "attachment-1"

    with pytest.raises(RuntimeFailure, match="already used"):
        await restarted.resolve_batch("thread-1", [_descriptor(source)], (source_root.parent,))

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(RuntimeFailure, match="outside"):
        await restarted.resolve_batch(
            "thread-1", [_descriptor(outside, attachment_id="attachment-2")], (source_root.parent,)
        )
    assert [item.id for item in await restarted.load("thread-1")] == ["attachment-1"]


def test_attachment_store_cleans_only_sessions_missing_from_repository(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "stable", image_limit=1024, file_limit=1024)
    kept = store.root / "session-kept"
    orphan = store.root / "session-orphan"
    unrelated = store.root / "not valid!"
    kept.mkdir()
    orphan.mkdir()
    unrelated.mkdir()

    assert store.cleanup_orphans({"session-kept"}) == ("session-orphan",)
    assert kept.is_dir()
    assert not orphan.exists()
    assert unrelated.is_dir()


@pytest.mark.asyncio
async def test_unknown_attachment_type_is_rejected(tmp_path: Path) -> None:
    runtime = RuntimeService(
        config_path=tmp_path / "config.json",
        data_dir=tmp_path / "data",
    )
    try:
        with pytest.raises(RuntimeFailure, match="Unsupported attachment type: unsupported"):
            await runtime._resolve_attachments(
                "thread",
                [{"type": "unsupported"}],
                (),
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_attachment_tools_accept_only_ids_and_hide_paths(tmp_path: Path) -> None:
    source = tmp_path / "content.txt"
    source.write_text("hello by id", encoding="utf-8")
    store = AttachmentStore(tmp_path / "store", image_limit=1024, file_limit=1024)
    attachment = (
        await store.resolve_batch("thread", [_descriptor(source)], (tmp_path,))
    )[0]
    values = {attachment.id: attachment}

    read = await AttachmentReadTool(values).execute(
        "read", {"attachment_id": attachment.id}, None
    )
    metadata = await AttachmentMetadataTool(values).execute(
        "metadata", {"attachment_id": attachment.id}, None
    )

    assert read.content[0].text == "hello by id"
    assert str(attachment.canonical_path) not in metadata.content[0].text
    with pytest.raises(ValueError, match="Unknown attachment id"):
        await AttachmentReadTool(values).execute(
            "bad", {"attachment_id": str(attachment.canonical_path)}, None
        )


def _pdf(path: Path, *, text: str | None, pages: int = 1) -> None:
    document = canvas.Canvas(str(path))
    for page in range(pages):
        if text is not None:
            document.drawString(72, 720, f"{text} {page + 1}")
        else:
            document.rect(72, 600, 200, 100)
        document.showPage()
    document.save()


async def _run_pdf_turn(
    tmp_path: Path, *, scanned_pages: int = 0
) -> tuple[RuntimeService, AeloonRpcAdapter, ScriptedProvider, dict[str, object]]:
    attachment_root = tmp_path / "workbench" / "attachments" / "upload"
    attachment_root.mkdir(parents=True)
    source = attachment_root / "content.pdf"
    _pdf(source, text=None if scanned_pages else "Deterministic PDF text", pages=scanned_pages or 1)
    model = Model(
        "studio/vision",
        "Vision",
        "studio",
        input=("text", "image"),
        context_window=128_000,
    )
    provider = ScriptedProvider(
        [AssistantMessage((TextContent("done"),), "studio", "studio/vision")],
        models=(model,),
        provider_id="studio",
    )
    config_path = tmp_path / "config.json"
    save_config(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "core",
            agent={"model": "studio/vision"},
            providers={
                **Config().providers,
                "studio": CustomProviderConfig(
                    name="Studio",
                    endpoint="http://127.0.0.1:8000/v1",
                    backend="vllm",
                    models=[
                        ProviderModelConfig(id="vision", supports_image=True)
                    ],
                ),
            },
        ),
        config_path,
    )
    runtime = RuntimeService(
        config_path=config_path,
        provider_manager_factory=lambda config: ProviderManager(
            config, driver_factories={"custom": lambda *_args: provider}
        ),
    )
    rpc = AeloonRpcAdapter(runtime)
    await rpc.dispatch(
        "session.create", {"session_id": "pdf-thread", "workspace": str(tmp_path)}
    )
    started = await rpc.dispatch(
        "turn.start",
        {
            "session_id": "pdf-thread",
            "input": {
                "kind": "prompt",
                "text": "总结附件",
                "attachments": [_descriptor(source)],
            },
        },
        attachment_roots=(attachment_root.parent,),
    )
    operation = runtime._operation({"operation_id": started["operation_id"]})
    assert operation.task is not None
    await operation.task
    return runtime, rpc, provider, started


@pytest.mark.asyncio
async def test_chinese_pdf_is_extracted_without_model_path_guessing(tmp_path: Path) -> None:
    runtime, rpc, provider, started = await _run_pdf_turn(tmp_path)
    try:
        assert started["attachment_ids"] == ["attachment-1"]
        context = provider.requests[0][1]
        rendered = str(context.messages)
        assert "Deterministic PDF text" in rendered
        assert "Attachment id=attachment-1 display_name=中文报告.pdf" in rendered
        assert "workbench/workspace" not in rendered
        assert "source_path" not in rendered
        attachment_events = [
            event
            for event in rpc._events
            if event["name"] == "log.entry"
            and event["payload"].get("category") == "attachment"
        ]
        assert {event["payload"]["action"] for event in attachment_events} >= {
            "resolved",
            "office_read",
        }
        assert all(event["payload"]["core_commit"] for event in attachment_events)
    finally:
        await rpc.close()


@pytest.mark.asyncio
async def test_every_scanned_pdf_page_is_sent_to_vision(tmp_path: Path) -> None:
    runtime, rpc, provider, _started = await _run_pdf_turn(tmp_path, scanned_pages=2)
    try:
        content = provider.requests[0][1].messages[-1].content
        assert isinstance(content, tuple)
        assert len([item for item in content if isinstance(item, ImageContent)]) == 2
        rendered = next(
            event
            for event in rpc._events
            if event["name"] == "log.entry"
            and event["payload"].get("action") == "vision_rendered"
        )
        assert rendered["payload"]["pages"] == [1, 2]
    finally:
        await rpc.close()
