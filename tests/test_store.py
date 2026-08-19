from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from pathlib import Path

import pytest

from aeloon_core.store import SCHEMA_VERSION, AsyncRuntimeStore, RuntimeStore


def test_store_migrates_and_persists_projection(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite"
    store = RuntimeStore(path)
    assert int(store._db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    project = store.add_project(name="Demo", path=str(tmp_path / "project"), is_git=False)
    thread = store.create_thread(
        project_id=project["id"],
        title="First",
        kind="standard",
        workspace=project["path"],
        thread_id="thread-1",
    )
    assert store.list_projects() == [project]
    assert store.get_thread("thread-1") == thread
    assert store.set_thread_archived("thread-1", True)["archived_at"]
    store.close()

    reopened = RuntimeStore(path)
    assert reopened.list_threads(archived=True)[0]["id"] == "thread-1"
    reopened.close()


def test_store_publishes_new_project_and_thread_atomically(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    project, thread = store.create_project_and_thread(
        project_name="Atomic",
        project_path=str(tmp_path / "project"),
        project_is_git=False,
        project_id="project-atomic",
        thread_title="First",
        thread_kind="standard",
        thread_workspace=str(tmp_path / "project"),
        thread_id="thread-atomic",
    )
    assert store.get_project(project["id"]) == project
    assert store.get_thread(thread["id"]) == thread

    with pytest.raises(sqlite3.IntegrityError):
        store.create_project_and_thread(
            project_name="Rolled back",
            project_path=str(tmp_path / "rolled-back"),
            project_is_git=False,
            project_id="project-rolled-back",
            thread_title="Invalid",
            thread_kind="not-a-thread-kind",
            thread_workspace=str(tmp_path),
        )
    assert store.get_project("project-rolled-back") is None
    store.close()


def test_store_cleans_only_unreferenced_attachment_blobs(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    root = tmp_path / "attachments"
    kept = store.add_attachment(
        name="kept.txt",
        mime_type="text/plain",
        data=b"kept",
        root=root,
    )
    orphan = root / "orphan.blob"
    orphan.write_bytes(b"orphan")
    assert store.cleanup_orphan_attachments(root) == [str(orphan)]
    assert Path(kept["storage_path"]).read_bytes() == b"kept"
    assert Path(kept["storage_path"]).name == f"{hashlib.sha256(b'kept').hexdigest()}.blob"
    store.close()


def test_store_content_addresses_duplicate_attachment_blobs(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    root = tmp_path / "attachments"
    first = store.add_attachment(
        name="first.txt", mime_type="text/plain", data=b"same", root=root
    )
    second = store.add_attachment(
        name="second.txt", mime_type="text/plain", data=b"same", root=root
    )
    assert first["storage_path"] == second["storage_path"]
    assert Path(first["storage_path"]).name == f"{hashlib.sha256(b'same').hexdigest()}.blob"
    assert store.delete_attachment(first["id"]) is True
    assert Path(second["storage_path"]).exists()
    assert store.delete_attachment(second["id"]) is True
    assert not Path(second["storage_path"]).exists()
    store.close()


def test_store_deletes_attachment_blobs_only_after_last_thread_reference(
    tmp_path: Path,
) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    project = store.add_project(name="Demo", path=str(tmp_path / "project"), is_git=False)
    first = store.create_thread(
        project_id=project["id"], title="First", kind="standard", workspace=str(tmp_path)
    )
    second = store.create_thread(
        project_id=project["id"], title="Second", kind="standard", workspace=str(tmp_path)
    )
    attachment = store.add_attachment(
        name="shared.txt",
        mime_type="text/plain",
        data=b"shared",
        root=tmp_path / "attachments",
    )
    reference = {"id": attachment["id"], "name": attachment["name"]}
    store.create_turn(thread_id=first["id"], core_turn_id="turn-first")
    store.update_turn_input("turn-first", user_text="first", attachments=[reference])
    store.create_turn(thread_id=second["id"], core_turn_id="turn-second")
    store.update_turn_input("turn-second", user_text="second", attachments=[reference])

    assert store.delete_thread(first["id"]) is True
    assert Path(attachment["storage_path"]).exists()
    assert store.get_attachment(attachment["id"]) is not None

    assert store.delete_thread(second["id"]) is True
    assert not Path(attachment["storage_path"]).exists()
    assert store.get_attachment(attachment["id"]) is None
    store.close()


def test_store_project_delete_cleans_attachment_metadata_and_blobs(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    project = store.add_project(name="Demo", path=str(tmp_path / "project"), is_git=False)
    thread = store.create_thread(
        project_id=project["id"], title="First", kind="standard", workspace=str(tmp_path)
    )
    attachment = store.add_attachment(
        name="project.txt",
        mime_type="text/plain",
        data=b"project",
        root=tmp_path / "attachments",
    )
    store.create_turn(thread_id=thread["id"], core_turn_id="turn-project")
    store.update_turn_input(
        "turn-project", user_text="project", attachments=[{"id": attachment["id"]}]
    )

    assert store.remove_project(project["id"]) is True
    assert store.get_project(project["id"]) is None
    assert store.get_attachment(attachment["id"]) is None
    assert not Path(attachment["storage_path"]).exists()
    store.close()


def test_store_projects_turn_events_into_durable_snapshot(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    project = store.add_project(name="Demo", path=str(tmp_path), is_git=False)
    thread = store.create_thread(
        project_id=project["id"], title="First", kind="standard", workspace=str(tmp_path)
    )
    store.ensure_turn(thread["id"], "op-1")
    store.update_turn_input("op-1", user_text="hello", attachments=[])
    store.project_event(
        thread["id"],
        {"seq": 1, "name": "operation.started", "operation_id": "op-1", "payload": {}},
    )
    store.project_event(
        thread["id"],
        {
            "seq": 2,
            "name": "content.started",
            "operation_id": "op-1",
            "payload": {"block": {"id": "block-1", "type": "text", "content": ""}},
        },
    )
    store.project_event(
        thread["id"],
        {
            "seq": 3,
            "name": "content.delta",
            "operation_id": "op-1",
            "payload": {"block_id": "block-1", "delta": "hello"},
        },
    )
    store.project_event(
        thread["id"],
        {
            "seq": 4,
            "name": "operation.completed",
            "operation_id": "op-1",
            "payload": {},
        },
    )
    turn = store.list_turns(thread["id"])[0]
    assert turn["user_text"] == "hello"
    assert turn["status"] == "completed"
    assert turn["blocks"][0]["content"] == "hello"
    assert store.get_thread(thread["id"])["is_unread"] is True
    assert len(store.list_events(thread["id"])) == 4
    store.close()


@pytest.mark.asyncio
async def test_async_store_exposes_serialized_reads(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    project = store.add_project(name="Demo", path=str(tmp_path), is_git=False)
    store.create_thread(
        project_id=project["id"], title="First", kind="standard", workspace=str(tmp_path)
    )
    store.close()
    async_store = AsyncRuntimeStore(tmp_path / "runtime.sqlite")
    assert len(await async_store.list_projects()) == 1
    assert len(await async_store.list_threads()) == 1
    await async_store.close()


@pytest.mark.asyncio
async def test_async_store_can_share_the_runtime_connection(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    async_store = AsyncRuntimeStore(store)
    project = await async_store.add_project(
        name="Shared", path=str(tmp_path), is_git=False
    )
    assert store.get_project(project["id"]) == project
    await async_store.close()
    # The composition root owns the shared connection and closes it after the
    # worker has drained; AsyncRuntimeStore.close() must not close it early.
    store.close()


def test_store_refresh_project_persists_git_status(tmp_path: Path) -> None:
    project_path = tmp_path / "workspace"
    project_path.mkdir()
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    project = store.add_project(name="Demo", path=str(project_path), is_git=False)
    subprocess.run(["git", "init", "-q", str(project_path)], check=True)
    refreshed = store.refresh_project(project["id"])
    assert refreshed is not None
    assert refreshed["is_git"] is True
    assert refreshed["name"] == "workspace"
    store.close()

    reopened = RuntimeStore(tmp_path / "runtime.sqlite")
    persisted = reopened.get_project(project["id"])
    assert persisted is not None
    assert persisted["is_git"] is True
    assert persisted["name"] == "workspace"
    reopened.close()
