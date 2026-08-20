from __future__ import annotations

import json

# ruff: noqa: E501
import sqlite3
import subprocess
from pathlib import Path

import pytest

import aeloon_runtime.migration as migration
from aeloon_runtime.migration import migrate_workbench
from aeloon_runtime.store import RuntimeStore


def _create_legacy_db(path: Path, *, project_path: str = "/tmp", thread: tuple[str, str, str] | None = None) -> None:
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, path TEXT, is_git INTEGER, created_at TEXT);
        CREATE TABLE threads (id TEXT PRIMARY KEY, project_id TEXT, title TEXT, kind TEXT,
          workspace TEXT, model_id TEXT, thinking_level TEXT, branch TEXT, status TEXT,
          pinned_at TEXT, archived_at TEXT, is_unread INTEGER, sort_order REAL,
          created_at TEXT, updated_at TEXT);
        """
    )
    db.execute(
        "INSERT INTO projects VALUES ('p1', 'Demo', ?, 1, '2026-01-01T00:00:00Z')",
        (project_path,),
    )
    if thread is not None:
        thread_id, kind, workspace = thread
        db.execute(
            "INSERT INTO threads VALUES (?, 'p1', 'Thread', ?, ?, '', 'off', 'aeloon/old', 'idle', NULL, NULL, 0, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (thread_id, kind, workspace),
        )
    db.commit()
    db.close()


def _git_project(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Migration Test"], check=True)
    (path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return path


def test_migrate_workbench_copies_projection_and_is_repeatable(tmp_path: Path) -> None:
    old = tmp_path / "old-workbench"
    old.mkdir()
    old_core = tmp_path / "old-core"
    (old_core / "harness-sessions").mkdir(parents=True)
    (old_core / "harness-sessions" / "t1.jsonl").write_text(
        '{"type":"session","version":3,"id":"t1","timestamp":"2026-01-01T00:00:00Z","cwd":"/tmp"}\n',
        encoding="utf-8",
    )
    source = old / "workbench.sqlite"
    db = sqlite3.connect(source)
    db.executescript(
        """
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, path TEXT, is_git INTEGER, created_at TEXT);
        CREATE TABLE threads (id TEXT PRIMARY KEY, project_id TEXT, title TEXT, kind TEXT,
          workspace TEXT, model_id TEXT, thinking_level TEXT, branch TEXT, status TEXT,
          pinned_at TEXT, archived_at TEXT, is_unread INTEGER, sort_order REAL,
          created_at TEXT, updated_at TEXT);
        INSERT INTO projects VALUES ('p1', 'Demo', '/tmp', 1, '2026-01-01T00:00:00Z');
        INSERT INTO threads VALUES ('t1', 'p1', 'Thread', 'standard', '/tmp', '', 'off', NULL, 'idle', NULL, NULL, 0, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
        """
    )
    db.commit()
    db.close()
    result = migrate_workbench(
        from_workbench=old,
        from_core=old_core,
        data_dir=tmp_path / "runtime",
        roots_output=tmp_path / "roots.json",
    )
    assert result["projects"] == 1
    store = RuntimeStore(tmp_path / "runtime" / "runtime.sqlite")
    assert store.get_thread("t1")["title"] == "Thread"
    store.close()
    assert (tmp_path / "runtime" / "harness-sessions" / "t1.jsonl").is_file()
    second = migrate_workbench(
        from_workbench=old,
        from_core=old_core,
        data_dir=tmp_path / "runtime",
        roots_output=tmp_path / "roots.json",
    )
    assert second["migrated"] is True
    assert Path(str(second["backup"])).is_dir()
    (tmp_path / "roots.json").unlink()
    third = migrate_workbench(
        from_workbench=old,
        from_core=old_core,
        data_dir=tmp_path / "runtime",
        roots_output=tmp_path / "roots.json",
    )
    assert third["migrated"] is True
    assert json.loads((tmp_path / "roots.json").read_text(encoding="utf-8"))["roots"] == [
        str(Path("/tmp").resolve())
    ]


def test_migrate_workbench_indexes_legacy_attachment_files_and_rewrites_paths(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old-workbench"
    old.mkdir()
    attachment_id = "11111111-1111-4111-8111-111111111111"
    attachment_dir = old / "attachments" / attachment_id
    attachment_dir.mkdir(parents=True)
    content = b"legacy attachment\n"
    legacy_path = attachment_dir / "content.txt"
    legacy_path.write_bytes(content)
    source = old / "workbench.sqlite"
    db = sqlite3.connect(source)
    db.executescript(
        """
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, path TEXT, is_git INTEGER, created_at TEXT);
        CREATE TABLE threads (id TEXT PRIMARY KEY, project_id TEXT, title TEXT, kind TEXT,
          workspace TEXT, model_id TEXT, thinking_level TEXT, branch TEXT, status TEXT,
          pinned_at TEXT, archived_at TEXT, is_unread INTEGER, sort_order REAL,
          created_at TEXT, updated_at TEXT);
        CREATE TABLE turns (id TEXT PRIMARY KEY, thread_id TEXT, core_turn_id TEXT UNIQUE,
          user_text TEXT, attachments_json TEXT, status TEXT, error TEXT,
          blocks_json TEXT, usage_json TEXT, created_at TEXT, updated_at TEXT);
        INSERT INTO projects VALUES ('p1', 'Demo', '/tmp', 0, '2026-01-01T00:00:00Z');
        INSERT INTO threads VALUES ('t1', 'p1', 'Thread', 'standard', '/tmp', '', 'off', NULL,
          'completed', NULL, NULL, 0, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
        """
    )
    db.execute(
        "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, NULL, '[]', '{}', ?, ?)",
        (
            "turn-row",
            "t1",
            "turn-1",
            "attached",
            json.dumps(
                [
                    {
                        "id": attachment_id,
                        "type": "file",
                        "name": "notes.txt",
                        "mime_type": "text/plain",
                        "size_bytes": len(content),
                        "path": str(legacy_path),
                    }
                ]
            ),
            "completed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ),
    )
    db.commit()
    db.close()

    result = migrate_workbench(
        from_workbench=old,
        from_core=tmp_path / "old-core",
        data_dir=tmp_path / "runtime-attachments",
        roots_output=tmp_path / "roots-attachments.json",
    )
    assert result["migrated"] is True
    store = RuntimeStore(tmp_path / "runtime-attachments" / "runtime.sqlite")
    attachment = store.get_attachment(attachment_id)
    assert attachment is not None
    assert Path(str(attachment["storage_path"])).name.endswith(".blob")
    assert Path(str(attachment["storage_path"])).read_bytes() == content
    turn = store.get_turn("turn-1")
    assert turn is not None
    assert turn["attachments"] == [
        {
            "id": attachment_id,
            "type": "file",
            "display_name": "notes.txt",
            "mime_type": "text/plain",
            "size_bytes": len(content),
        }
    ]
    store.close()


def test_attachment_migration_does_not_read_an_external_legacy_path(tmp_path: Path) -> None:
    source_root = tmp_path / "old-workbench"
    source_root.mkdir()
    external = tmp_path / "outside.txt"
    external.write_text("must not be imported\n", encoding="utf-8")
    target = tmp_path / "runtime.sqlite"
    db = sqlite3.connect(target)
    db.execute("CREATE TABLE attachments (id TEXT PRIMARY KEY, storage_path TEXT)")
    db.execute(
        "INSERT INTO attachments VALUES (?, ?)",
        ("attachment-1", str(external)),
    )
    db.commit()
    db.close()

    with pytest.raises(RuntimeError, match="Legacy attachment is missing"):
        migration._rewrite_attachment_paths(
            target,
            tmp_path / "attachments",
            source_root=source_root,
        )
    assert not (tmp_path / "attachments").exists()


def test_migrate_workbench_moves_managed_worktree_and_records_backup(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.name", "Migration Test"], check=True)
    (project / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-qm", "base"], check=True)
    old_worktree = tmp_path / "old-worktree"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-b", "aeloon/old", str(old_worktree)],
        check=True,
    )
    old = tmp_path / "old-workbench"
    old.mkdir()
    source = old / "workbench.sqlite"
    db = sqlite3.connect(source)
    db.executescript(
        """
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, path TEXT, is_git INTEGER, created_at TEXT);
        CREATE TABLE threads (id TEXT PRIMARY KEY, project_id TEXT, title TEXT, kind TEXT,
          workspace TEXT, model_id TEXT, thinking_level TEXT, branch TEXT, status TEXT,
          pinned_at TEXT, archived_at TEXT, is_unread INTEGER, sort_order REAL,
          created_at TEXT, updated_at TEXT);
        """,
    )
    db.execute(
        "INSERT INTO projects VALUES ('p1', 'Demo', ?, 1, '2026-01-01T00:00:00Z')",
        (str(project),),
    )
    db.execute(
        "INSERT INTO threads VALUES ('t1', 'p1', 'Thread', 'worktree', ?, '', 'off', 'aeloon/old', 'idle', NULL, NULL, 0, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (str(old_worktree),),
    )
    db.commit()
    db.close()
    result = migrate_workbench(
        from_workbench=old,
        from_core=tmp_path / "old-core",
        data_dir=tmp_path / "runtime",
        roots_output=tmp_path / "roots.json",
    )
    new_worktree = tmp_path / "runtime" / "worktrees" / "p1" / "t1"
    assert result["threads"] == 1
    assert new_worktree.is_dir()
    assert not old_worktree.exists()
    store = RuntimeStore(tmp_path / "runtime" / "runtime.sqlite")
    assert store.get_thread("t1")["workspace"] == str(new_worktree)
    store.close()


def test_migrate_workbench_reads_uncheckpointed_wal_and_cleans_malformed_input(tmp_path: Path) -> None:
    old = tmp_path / "old-workbench"
    old.mkdir()
    source = old / "workbench.sqlite"
    db = sqlite3.connect(source)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA wal_autocheckpoint=0")
    db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, path TEXT, is_git INTEGER, created_at TEXT)")
    db.execute("INSERT INTO projects VALUES ('p1', 'WAL', '/tmp', 1, '2026-01-01T00:00:00Z')")
    db.commit()
    assert source.with_name("workbench.sqlite-wal").exists()
    result = migrate_workbench(
        from_workbench=old,
        from_core=tmp_path / "old-core",
        data_dir=tmp_path / "runtime",
        roots_output=tmp_path / "roots.json",
    )
    assert result["projects"] == 1
    db.close()
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "workbench.sqlite").write_text("not sqlite", encoding="utf-8")
    with pytest.raises(sqlite3.DatabaseError):
        migrate_workbench(
            from_workbench=malformed,
            from_core=tmp_path / "old-core-2",
            data_dir=tmp_path / "runtime-malformed",
            roots_output=tmp_path / "roots-malformed.json",
        )
    assert not (tmp_path / "runtime-malformed" / "runtime.sqlite").exists()
    assert not (tmp_path / "runtime-malformed" / ".incomplete").exists()


def test_migrate_workbench_rejects_dirty_or_missing_worktree_without_partial_targets(tmp_path: Path) -> None:
    project = _git_project(tmp_path / "project")
    old_worktree = tmp_path / "old-worktree"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-b", "aeloon/old", str(old_worktree)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    (old_worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    old = tmp_path / "old-workbench"
    old.mkdir()
    _create_legacy_db(old / "workbench.sqlite", project_path=str(project), thread=("t1", "worktree", str(old_worktree)))
    (old / "attachments").mkdir()
    (old / "attachments" / "a.blob").write_bytes(b"attachment")
    with pytest.raises(RuntimeError, match="dirty"):
        migrate_workbench(
            from_workbench=old,
            from_core=tmp_path / "old-core",
            data_dir=tmp_path / "runtime-dirty",
            roots_output=tmp_path / "roots-dirty.json",
        )
    assert old_worktree.exists()
    assert not (tmp_path / "runtime-dirty" / "runtime.sqlite").exists()
    assert not (tmp_path / "runtime-dirty" / "attachments").exists()
    assert not (tmp_path / "runtime-dirty" / ".incomplete").exists()

    (old_worktree / "dirty.txt").unlink()
    old_worktree.rename(tmp_path / "missing-worktree")
    with pytest.raises(RuntimeError, match="missing"):
        migrate_workbench(
            from_workbench=old,
            from_core=tmp_path / "old-core-2",
            data_dir=tmp_path / "runtime-missing",
            roots_output=tmp_path / "roots-missing.json",
        )
    assert not (tmp_path / "runtime-missing" / "runtime.sqlite").exists()
    assert not (tmp_path / "runtime-missing" / ".incomplete").exists()


def test_migrate_workbench_rolls_back_a_worktree_when_projection_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _git_project(tmp_path / "project")
    old_worktree = tmp_path / "old-worktree"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-b", "aeloon/old", str(old_worktree)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    old = tmp_path / "old-workbench"
    old.mkdir()
    _create_legacy_db(old / "workbench.sqlite", project_path=str(project), thread=("t1", "worktree", str(old_worktree)))
    monkeypatch.setattr(migration, "_verify_projection", lambda *_args: (_ for _ in ()).throw(RuntimeError("verify")))
    with pytest.raises(RuntimeError, match="verify"):
        migrate_workbench(
            from_workbench=old,
            from_core=tmp_path / "old-core",
            data_dir=tmp_path / "runtime",
            roots_output=tmp_path / "roots.json",
        )
    assert old_worktree.is_dir()
    assert not (tmp_path / "runtime" / "worktrees").exists()
    assert not (tmp_path / "runtime" / "runtime.sqlite").exists()
    assert not (tmp_path / "runtime" / ".incomplete").exists()


def test_migrate_workbench_recovers_interrupted_worktree_journal_and_retries(
    tmp_path: Path,
) -> None:
    project = _git_project(tmp_path / "project")
    old_worktree = tmp_path / "old-worktree"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-b", "aeloon/old", str(old_worktree)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    old = tmp_path / "old-workbench"
    old.mkdir()
    _create_legacy_db(old / "workbench.sqlite", project_path=str(project), thread=("t1", "worktree", str(old_worktree)))

    runtime_data = tmp_path / "runtime"
    new_worktree = runtime_data / "worktrees" / "p1" / "t1"
    new_worktree.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "-C", str(project), "worktree", "move", str(old_worktree), str(new_worktree)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    store = RuntimeStore(runtime_data / "runtime.sqlite")
    store.add_project(name="Demo", path=str(project), is_git=True)
    project_id = store.list_projects()[0]["id"]
    store.create_thread(
        project_id=project_id,
        title="Thread",
        kind="worktree",
        workspace=str(new_worktree),
        branch="aeloon/old",
        thread_id="t1",
    )
    store.close()
    runtime_data.mkdir(parents=True, exist_ok=True)
    (runtime_data / ".incomplete").write_text('{"pid": 1}\n', encoding="utf-8")
    (runtime_data / "migration.jsonl").write_text(
        json.dumps(
            {
                "stage": "worktree_moved",
                "thread_id": "t1",
                "project_path": str(project),
                "old": str(old_worktree),
                "new": str(new_worktree),
                "branch": "aeloon/old",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = migrate_workbench(
        from_workbench=old,
        from_core=tmp_path / "old-core",
        data_dir=runtime_data,
        roots_output=tmp_path / "roots.json",
    )
    assert result["migrated"] is True
    assert not old_worktree.exists()
    assert new_worktree.is_dir()
    assert not (runtime_data / ".incomplete").exists()
