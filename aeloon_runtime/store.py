"""SQLite-backed Runtime projection store.

The store keeps the Workbench projection in the Runtime process. Methods are
small, synchronous units protected by one connection lock; AsyncRuntimeStore
provides the gateway-facing awaitable facade without allowing concurrent
SQLite writes to interleave.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RuntimeStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self.path.chmod(0o600)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._db.execute("BEGIN")
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def _migrate(self) -> None:
        with self._lock:
            version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Runtime store schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version < 1:
                self._db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                      id TEXT PRIMARY KEY,
                      name TEXT NOT NULL,
                      path TEXT NOT NULL UNIQUE,
                      is_git INTEGER NOT NULL DEFAULT 0,
                      created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS threads (
                      id TEXT PRIMARY KEY,
                      project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                      title TEXT NOT NULL,
                      kind TEXT NOT NULL CHECK(kind IN ('standard', 'worktree')),
                      workspace TEXT NOT NULL,
                      model_id TEXT NOT NULL DEFAULT '',
                      thinking_level TEXT NOT NULL DEFAULT 'off',
                      branch TEXT,
                      status TEXT NOT NULL DEFAULT 'idle',
                      pinned_at TEXT,
                      archived_at TEXT,
                      is_unread INTEGER NOT NULL DEFAULT 0,
                      sort_order REAL NOT NULL DEFAULT 0,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS turns (
                      id TEXT PRIMARY KEY,
                      thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                      core_turn_id TEXT NOT NULL UNIQUE,
                      user_text TEXT NOT NULL,
                      attachments_json TEXT NOT NULL DEFAULT '[]',
                      status TEXT NOT NULL,
                      error TEXT,
                      blocks_json TEXT NOT NULL DEFAULT '[]',
                      usage_json TEXT NOT NULL DEFAULT '{}',
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS events (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                      seq INTEGER NOT NULL,
                      name TEXT NOT NULL,
                      event_json TEXT NOT NULL,
                      created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS events_thread_idx ON events(thread_id, id);
                    """
                )
                version = 1
            if version < 2:
                columns = {row[1] for row in self._db.execute("PRAGMA table_info(threads)")}
                if "sort_order" not in columns:
                    self._db.execute(
                        "ALTER TABLE threads ADD COLUMN sort_order REAL NOT NULL DEFAULT 0"
                    )
                version = 2
            if version < 3:
                self._db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS attachments (
                      id TEXT PRIMARY KEY,
                      name TEXT NOT NULL,
                      mime_type TEXT,
                      size_bytes INTEGER NOT NULL,
                      storage_path TEXT NOT NULL,
                      created_at TEXT NOT NULL
                    )
                    """
                )
                version = 3
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM projects ORDER BY created_at, id").fetchall()
        return [self._project(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._project(row) if row else None

    def add_project(
        self,
        *,
        name: str,
        path: str,
        is_git: bool,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        value = {
            "id": project_id or str(uuid.uuid4()),
            "name": name,
            "path": path,
            "is_git": is_git,
            "created_at": _now(),
        }
        with self.transaction() as db:
            db.execute(
                "INSERT INTO projects (id, name, path, is_git, created_at) VALUES (?, ?, ?, ?, ?)",
                (value["id"], value["name"], value["path"], int(is_git), value["created_at"]),
            )
        return value

    def refresh_project(self, project_id: str) -> dict[str, Any] | None:
        """Re-detect the project display name and Git status, then persist both."""

        project = self.get_project(project_id)
        if project is None:
            return None
        path = Path(project["path"])
        name = path.name or str(path)
        is_git = (path / ".git").exists()
        with self.transaction() as db:
            db.execute(
                "UPDATE projects SET name = ?, is_git = ? WHERE id = ?",
                (name, int(is_git), project_id),
            )
        return self.get_project(project_id)

    def create_project_and_thread(
        self,
        *,
        project_name: str,
        project_path: str,
        project_is_git: bool,
        project_id: str,
        thread_title: str,
        thread_kind: str,
        thread_workspace: str,
        thread_branch: str | None = None,
        thread_model_id: str = "",
        thread_thinking_level: str = "off",
        thread_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Commit a new project and its first thread as one projection unit.

        Runtime creates the session/worktree before calling this method.  The
        SQLite commit is therefore the final publication step; a crash cannot
        leave a project row without the thread that caused it to exist.
        """

        now = _now()
        project = {
            "id": project_id,
            "name": project_name,
            "path": project_path,
            "is_git": bool(project_is_git),
            "created_at": now,
        }
        thread = {
            "id": thread_id or str(uuid.uuid4()),
            "project_id": project_id,
            "title": thread_title,
            "kind": thread_kind,
            "workspace": thread_workspace,
            "model_id": thread_model_id,
            "thinking_level": thread_thinking_level,
            "branch": thread_branch,
            "status": "idle",
            "pinned_at": None,
            "archived_at": None,
            "is_unread": False,
            "sort_order": 0.0,
            "created_at": now,
            "updated_at": now,
        }
        with self.transaction() as db:
            db.execute(
                "INSERT INTO projects (id, name, path, is_git, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    project["id"],
                    project["name"],
                    project["path"],
                    int(project["is_git"]),
                    project["created_at"],
                ),
            )
            db.execute(
                """INSERT INTO threads
                (id, project_id, title, kind, workspace, model_id, thinking_level, branch,
                 status, pinned_at, archived_at, is_unread, sort_order, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(
                    thread[field]
                    for field in (
                        "id",
                        "project_id",
                        "title",
                        "kind",
                        "workspace",
                        "model_id",
                        "thinking_level",
                        "branch",
                        "status",
                        "pinned_at",
                        "archived_at",
                        "is_unread",
                        "sort_order",
                        "created_at",
                        "updated_at",
                    )
                ),
            )
        return project, thread

    def remove_project(self, project_id: str) -> bool:
        attachment_paths: list[str] = []
        with self.transaction() as db:
            thread_rows = db.execute(
                "SELECT id FROM threads WHERE project_id = ?", (project_id,)
            ).fetchall()
            attachment_ids = self._attachment_ids_for_threads(
                db, [str(row[0]) for row in thread_rows]
            )
            cursor = db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            if cursor.rowcount and attachment_ids:
                attachment_paths = self._delete_unreferenced_attachments(db, attachment_ids)
        for path in attachment_paths:
            Path(path).unlink(missing_ok=True)
        return cursor.rowcount > 0

    def list_threads(
        self, project_id: str | None = None, *, archived: bool | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            values.append(project_id)
        if archived is True:
            clauses.append("archived_at IS NOT NULL")
        elif archived is False:
            clauses.append("archived_at IS NULL")
        query = "SELECT * FROM threads"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY sort_order ASC, created_at ASC, id ASC"
        with self._lock:
            rows = self._db.execute(query, values).fetchall()
        return [self._thread(row) for row in rows]

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        return self._thread(row) if row else None

    def create_thread(
        self,
        *,
        project_id: str,
        title: str,
        kind: str,
        workspace: str,
        branch: str | None = None,
        model_id: str = "",
        thinking_level: str = "off",
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        value = {
            "id": thread_id or str(uuid.uuid4()),
            "project_id": project_id,
            "title": title,
            "kind": kind,
            "workspace": workspace,
            "model_id": model_id,
            "thinking_level": thinking_level,
            "branch": branch,
            "status": "idle",
            "pinned_at": None,
            "archived_at": None,
            "is_unread": False,
            "sort_order": 0.0,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self.transaction() as db:
            minimum = db.execute(
                "SELECT MIN(sort_order) FROM threads WHERE project_id = ?", (project_id,)
            ).fetchone()[0]
            value["sort_order"] = float(minimum) - 1 if minimum is not None else 0.0
            db.execute(
                """INSERT INTO threads
                (id, project_id, title, kind, workspace, model_id, thinking_level, branch,
                 status, pinned_at, archived_at, is_unread, sort_order, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(
                    value[field]
                    for field in (
                        "id",
                        "project_id",
                        "title",
                        "kind",
                        "workspace",
                        "model_id",
                        "thinking_level",
                        "branch",
                        "status",
                        "pinned_at",
                        "archived_at",
                        "is_unread",
                        "sort_order",
                        "created_at",
                        "updated_at",
                    )
                ),
            )
        return value

    def delete_thread(self, thread_id: str) -> bool:
        attachment_paths: list[str] = []
        with self.transaction() as db:
            attachment_ids = self._attachment_ids_for_threads(db, [thread_id])
            cursor = db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            if cursor.rowcount and attachment_ids:
                attachment_paths = self._delete_unreferenced_attachments(db, attachment_ids)
        for path in attachment_paths:
            Path(path).unlink(missing_ok=True)
        return cursor.rowcount > 0

    def create_turn(
        self,
        *,
        thread_id: str,
        core_turn_id: str,
        user_text: str = "",
        attachments: list[dict[str, Any]] | None = None,
        status: str = "queued",
        created_at: str | None = None,
    ) -> dict[str, Any]:
        now = created_at or _now()
        value = {
            "id": str(uuid.uuid4()),
            "thread_id": thread_id,
            "core_turn_id": core_turn_id,
            "user_text": user_text,
            "attachments": attachments or [],
            "status": status,
            "error": None,
            "blocks": [],
            "usage": {},
            "created_at": now,
            "updated_at": now,
        }
        with self.transaction() as db:
            db.execute(
                """INSERT INTO turns
                   (id, thread_id, core_turn_id, user_text, attachments_json, status,
                    error, blocks_json, usage_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    value["id"],
                    thread_id,
                    core_turn_id,
                    user_text,
                    json.dumps(value["attachments"]),
                    status,
                    None,
                    "[]",
                    "{}",
                    now,
                    now,
                ),
            )
            db.execute(
                "UPDATE threads SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, thread_id),
            )
        return value

    def ensure_turn(
        self, thread_id: str, core_turn_id: str, *, status: str = "queued"
    ) -> dict[str, Any]:
        existing = self.get_turn(core_turn_id)
        if existing is not None:
            return existing
        return self.create_turn(thread_id=thread_id, core_turn_id=core_turn_id, status=status)

    def get_turn(self, core_turn_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM turns WHERE core_turn_id = ?", (core_turn_id,)
            ).fetchone()
        return self._turn(row) if row else None

    def list_turns(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM turns WHERE thread_id = ? ORDER BY created_at, id", (thread_id,)
            ).fetchall()
        return [self._turn(row) for row in rows]

    def update_turn_input(
        self,
        core_turn_id: str,
        *,
        user_text: str,
        attachments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        with self.transaction() as db:
            db.execute(
                "UPDATE turns SET user_text = ?, attachments_json = ?, updated_at = ? "
                "WHERE core_turn_id = ?",
                (user_text, json.dumps(attachments), _now(), core_turn_id),
            )
        return self.get_turn(core_turn_id)

    def project_event(self, thread_id: str, event: dict[str, Any]) -> None:
        """Persist an event and update its durable turn/thread projection."""

        with self.transaction() as db:
            db.execute(
                "INSERT INTO events (thread_id, seq, name, event_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    thread_id,
                    int(event.get("seq", 0)),
                    str(event.get("name", "")),
                    json.dumps(event),
                    event.get("time", _now()),
                ),
            )
            operation_id = event.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id:
                return
            row = db.execute(
                "SELECT * FROM turns WHERE core_turn_id = ?", (operation_id,)
            ).fetchone()
            if row is None:
                return
            current = self._turn(row)
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            status = current["status"]
            error = current["error"]
            name = str(event.get("name", ""))
            if name == "operation.started":
                status, error = "active", None
            elif name == "operation.cancelling":
                status, error = "cancelling", None
            elif name == "operation.completed":
                status, error = "completed", None
            elif name == "operation.failed":
                status = "failed"
                error = str(payload.get("error") or "Operation failed")
            elif name == "operation.cancelled":
                status, error = "cancelled", None
            blocks = list(current["blocks"])
            if name in {"content.started", "tool.started"} and isinstance(
                payload.get("block"), dict
            ):
                block = dict(payload["block"])
                blocks = [item for item in blocks if item.get("id") != block.get("id")] + [block]
            elif name == "content.delta":
                block_id = str(payload.get("block_id") or "")
                delta = str(payload.get("delta") or "")
                blocks = [
                    {**item, "content": f"{item.get('content', '')}{delta}"}
                    if item.get("id") == block_id
                    else item
                    for item in blocks
                ]
            elif name in {"content.updated", "content.completed", "tool.updated", "tool.completed"}:
                block_id = str(payload.get("block_id") or "")
                patch = payload.get("patch")
                if isinstance(patch, dict):
                    blocks = [
                        {**item, **patch} if item.get("id") == block_id else item
                        for item in blocks
                    ]
            usage = current["usage"]
            if name == "usage.updated" and isinstance(payload.get("usage"), dict):
                usage = dict(payload["usage"])
            updated_at = str(event.get("time") or _now())
            db.execute(
                """UPDATE turns SET status = ?, error = ?, blocks_json = ?, usage_json = ?,
                   updated_at = ? WHERE core_turn_id = ?""",
                (status, error, json.dumps(blocks), json.dumps(usage), updated_at, operation_id),
            )
            thread_status = "idle" if status == "cancelled" else status
            db.execute(
                "UPDATE threads SET status = ?, is_unread = CASE WHEN ? THEN 1 ELSE is_unread END, "
                "updated_at = ? WHERE id = ?",
                (thread_status, name == "operation.completed", updated_at, thread_id),
            )

    def update_thread_workspace(
        self, thread_id: str, *, workspace: str, branch: str | None = None
    ) -> dict[str, Any] | None:
        """Update a migrated/managed thread's canonical workspace atomically."""

        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE threads SET workspace = ?, branch = ?, updated_at = ? WHERE id = ?",
                (workspace, branch, _now(), thread_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_thread(thread_id)

    def set_thread_title(self, thread_id: str, title: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            db.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), thread_id),
            )
        return self.get_thread(thread_id)

    def set_thread_pinned(self, thread_id: str, pinned: bool) -> dict[str, Any] | None:
        with self.transaction() as db:
            if pinned:
                db.execute(
                    "UPDATE threads SET pinned_at = ?, updated_at = ? "
                    "WHERE id = ? AND archived_at IS NULL",
                    (_now(), _now(), thread_id),
                )
            else:
                db.execute(
                    "UPDATE threads SET pinned_at = NULL, updated_at = ? WHERE id = ?",
                    (_now(), thread_id),
                )
        return self.get_thread(thread_id)

    def set_thread_archived(self, thread_id: str, archived: bool) -> dict[str, Any] | None:
        with self.transaction() as db:
            db.execute(
                """UPDATE threads
                   SET archived_at = ?,
                       pinned_at = CASE WHEN ? THEN NULL ELSE pinned_at END,
                       is_unread = CASE WHEN ? THEN 0 ELSE is_unread END,
                       updated_at = ?
                 WHERE id = ?""",
                (_now() if archived else None, archived, archived, _now(), thread_id),
            )
        return self.get_thread(thread_id)

    def set_thread_read(self, thread_id: str, read: bool) -> dict[str, Any] | None:
        with self.transaction() as db:
            if read:
                db.execute(
                    "UPDATE threads SET status = CASE WHEN status = 'completed' THEN 'idle' "
                    "ELSE status END, is_unread = 0, updated_at = ? WHERE id = ?",
                    (_now(), thread_id),
                )
            else:
                db.execute(
                    "UPDATE threads SET is_unread = 1, updated_at = ? "
                    "WHERE id = ? AND archived_at IS NULL",
                    (_now(), thread_id),
                )
        return self.get_thread(thread_id)

    def configure_thread(
        self,
        thread_id: str,
        *,
        model_id: str | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_thread(thread_id)
        if current is None:
            return None
        with self.transaction() as db:
            db.execute(
                """UPDATE threads
                   SET model_id = COALESCE(?, model_id),
                       thinking_level = COALESCE(?, thinking_level),
                       updated_at = ?
                 WHERE id = ?""",
                (model_id, thinking_level, _now(), thread_id),
            )
        return self.get_thread(thread_id)

    def reorder_threads(self, project_id: str, thread_ids: list[str]) -> list[dict[str, Any]]:
        if not thread_ids:
            raise ValueError("thread_ids must not be empty")
        if len(set(thread_ids)) != len(thread_ids):
            raise ValueError("thread_ids must not contain duplicates")
        with self.transaction() as db:
            rows = db.execute(
                "SELECT id, project_id, archived_at, pinned_at FROM threads "
                f"WHERE id IN ({','.join('?' for _ in thread_ids)})",
                thread_ids,
            ).fetchall()
            by_id = {str(row["id"]): row for row in rows}
            if len(by_id) != len(thread_ids):
                raise ValueError("Thread not found")
            for thread_id in thread_ids:
                row = by_id[thread_id]
                if row["project_id"] != project_id:
                    raise ValueError("Threads cannot be reordered across projects")
                if row["archived_at"] is not None:
                    raise ValueError("Archived threads cannot be reordered")
                if row["pinned_at"] is not None:
                    raise ValueError("Pinned threads cannot be reordered")
            for index, thread_id in enumerate(thread_ids):
                db.execute(
                    "UPDATE threads SET sort_order = ?, updated_at = ? "
                    "WHERE id = ? AND project_id = ?",
                    (float(index), _now(), thread_id, project_id),
                )
        return self.list_threads(project_id)

    def record_event(self, thread_id: str, event: dict[str, Any]) -> None:
        with self.transaction() as db:
            db.execute(
                """INSERT INTO events
                   (thread_id, seq, name, event_json, created_at)
                 VALUES (?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    int(event.get("seq", 0)),
                    str(event.get("name", "")),
                    json.dumps(event),
                    event.get("time", _now()),
                ),
            )

    def list_events(self, thread_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT event_json FROM events WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def add_attachment(
        self,
        *,
        name: str,
        mime_type: str | None,
        data: bytes,
        root: Path,
    ) -> dict[str, Any]:
        # Serialize blob deduplication, metadata commit, and delete_attachment
        # under one lock.  A duplicate upload must not commit a row after a
        # concurrent delete has checked references and removed the shared blob.
        with self._lock:
            attachment_id = str(uuid.uuid4())
            directory = Path(root).expanduser().resolve(strict=False)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest = hashlib.sha256(data).hexdigest()
            target = directory / f"{digest}.blob"
            temporary = directory / f".{digest}.{uuid.uuid4().hex}.tmp"
            temporary.write_bytes(data)
            temporary.chmod(0o600)
            created_target = False
            try:
                if (
                    target.exists()
                    and target.is_file()
                    and hashlib.sha256(target.read_bytes()).hexdigest() == digest
                ):
                    # Content-addressing permits duplicate uploads to share one
                    # private blob.  Do not replace or delete a blob already
                    # referenced by another attachment row.
                    temporary.unlink(missing_ok=True)
                else:
                    os.replace(temporary, target)
                    target.chmod(0o600)
                    created_target = True
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            value = {
                "id": attachment_id,
                "name": name,
                "mime_type": mime_type,
                "size_bytes": len(data),
                "storage_path": str(target),
                "created_at": _now(),
            }
            try:
                with self.transaction() as db:
                    db.execute(
                        """INSERT INTO attachments
                           (id, name, mime_type, size_bytes, storage_path, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        tuple(
                            value[field]
                            for field in (
                                "id",
                                "name",
                                "mime_type",
                                "size_bytes",
                                "storage_path",
                                "created_at",
                            )
                        ),
                    )
            except Exception:
                if created_target:
                    target.unlink(missing_ok=True)
                raise
            return value

    def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM attachments WHERE id = ?", (attachment_id,)
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        return value

    def delete_attachment(self, attachment_id: str) -> bool:
        # Keep the reference check and blob unlink under the same Store lock.
        # Otherwise a concurrent duplicate upload can commit a new metadata
        # row between this transaction and unlink(), leaving that row pointing
        # at a deleted content-addressed blob.
        with self._lock:
            value = self.get_attachment(attachment_id)
            if value is None:
                return False
            storage_path = str(value["storage_path"])
            with self.transaction() as db:
                db.execute("DELETE FROM attachments WHERE id = ?", (attachment_id,))
                still_referenced = db.execute(
                    "SELECT 1 FROM attachments WHERE storage_path = ? LIMIT 1",
                    (storage_path,),
                ).fetchone()
            if still_referenced is None:
                Path(storage_path).unlink(missing_ok=True)
            return True

    def cleanup_orphan_attachments(self, root: Path) -> list[str]:
        """Remove blob files left behind by an interrupted upload.

        Uploads write their blob before committing the metadata row. A crash
        between those operations must not accumulate unreachable private files
        across Runtime starts, so cleanup is restricted to this attachment
        directory and to the ``*.blob`` naming convention.
        """

        directory = Path(root).expanduser().resolve(strict=False)
        if not directory.is_dir():
            return []
        with self._lock:
            referenced = {
                str(Path(str(row[0])).expanduser().resolve(strict=False))
                for row in self._db.execute("SELECT storage_path FROM attachments")
            }
        removed: list[str] = []
        for candidate in directory.glob("*.blob"):
            if str(candidate.expanduser().resolve(strict=False)) in referenced:
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue
            removed.append(str(candidate))
        return removed

    @staticmethod
    def _project(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "path": row["path"],
            "is_git": bool(row["is_git"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _thread(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "title": row["title"],
            "kind": row["kind"],
            "workspace": row["workspace"],
            "model_id": row["model_id"],
            "thinking_level": row["thinking_level"],
            "branch": row["branch"],
            "status": row["status"],
            "pinned_at": row["pinned_at"],
            "archived_at": row["archived_at"],
            "is_unread": bool(row["is_unread"]),
            "sort_order": row["sort_order"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _turn(row: sqlite3.Row) -> dict[str, Any]:
        def decode(name: str, fallback: Any) -> Any:
            try:
                value = json.loads(row[name])
            except (TypeError, json.JSONDecodeError):
                return fallback
            return value

        return {
            "id": row["id"],
            "thread_id": row["thread_id"],
            "core_turn_id": row["core_turn_id"],
            "user_text": row["user_text"],
            "attachments": decode("attachments_json", []),
            "status": row["status"],
            "error": row["error"],
            "blocks": decode("blocks_json", []),
            "usage": decode("usage_json", {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _attachment_ids_for_threads(
        db: sqlite3.Connection, thread_ids: list[str]
    ) -> set[str]:
        if not thread_ids:
            return set()
        placeholders = ",".join("?" for _ in thread_ids)
        rows = db.execute(
            f"SELECT attachments_json FROM turns WHERE thread_id IN ({placeholders})",
            thread_ids,
        ).fetchall()
        attachment_ids: set[str] = set()
        for row in rows:
            try:
                values = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value:
                    attachment_ids.add(value)
                elif isinstance(value, dict) and isinstance(value.get("id"), str):
                    attachment_ids.add(value["id"])
        return attachment_ids

    @classmethod
    def _delete_unreferenced_attachments(
        cls, db: sqlite3.Connection, candidates: set[str]
    ) -> list[str]:
        if not candidates:
            return []
        referenced = cls._attachment_ids_for_threads(
            db,
            [str(row[0]) for row in db.execute("SELECT id FROM threads").fetchall()],
        )
        deletable = candidates - referenced
        if not deletable:
            return []
        placeholders = ",".join("?" for _ in deletable)
        rows = db.execute(
            f"SELECT id, storage_path FROM attachments WHERE id IN ({placeholders})",
            sorted(deletable),
        ).fetchall()
        db.execute(
            f"DELETE FROM attachments WHERE id IN ({placeholders})", sorted(deletable)
        )
        paths: list[str] = []
        for _attachment_id, storage_path in rows:
            value = str(storage_path)
            if value in paths:
                continue
            if db.execute(
                "SELECT 1 FROM attachments WHERE storage_path = ? LIMIT 1", (value,)
            ).fetchone() is None:
                paths.append(value)
        return paths


class AsyncRuntimeStore:
    def __init__(self, path: Path | str | RuntimeStore) -> None:
        # Composition roots that already own a RuntimeStore can attach the
        # serial worker without opening a second SQLite connection.  This is
        # important for the gateway: all production calls must pass through
        # one ordered worker, while migration/tests may still use the
        # synchronous store directly.
        self._owns_store = not isinstance(path, RuntimeStore)
        self._store = path if isinstance(path, RuntimeStore) else RuntimeStore(path)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aeloon-store")
        self._closed = False

    async def _call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("Runtime store is closed")
        loop = asyncio.get_running_loop()
        worker_future = self._worker.submit(functools.partial(function, *args, **kwargs))
        future = asyncio.wrap_future(worker_future, loop=loop)
        try:
            return await future
        except asyncio.CancelledError:
            # A cancelled request must not let a SQLite write continue behind
            # the caller's back.  Wait for the single worker to finish before
            # propagating cancellation to preserve operation ordering.
            with contextlib.suppress(BaseException):
                await asyncio.shield(asyncio.wrap_future(worker_future, loop=loop))
            raise

    async def close(self) -> None:
        if self._closed:
            return
        # Submit a final barrier even when the synchronous store is owned by a
        # composition root.  This drains every queued operation before the
        # worker is stopped without closing a connection still exposed to the
        # root's migration/test hooks.
        await self._call(self._store.close if self._owns_store else (lambda: None))
        self._closed = True
        self._worker.shutdown(wait=True)

    async def list_projects(self) -> list[dict[str, Any]]:
        return await self._call(self._store.list_projects)

    async def list_threads(
        self, project_id: str | None = None, *, archived: bool | None = None
    ) -> list[dict[str, Any]]:
        return await self._call(self._store.list_threads, project_id, archived=archived)

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.get_thread, thread_id)

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.get_project, project_id)

    async def add_project(
        self,
        *,
        name: str,
        path: str,
        is_git: bool,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            self._store.add_project,
            name=name,
            path=path,
            is_git=is_git,
            project_id=project_id,
        )

    async def refresh_project(self, project_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.refresh_project, project_id)

    async def remove_project(self, project_id: str) -> bool:
        return await self._call(self._store.remove_project, project_id)

    async def create_project_and_thread(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return await self._call(self._store.create_project_and_thread, **kwargs)

    async def create_thread(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call(self._store.create_thread, **kwargs)

    async def delete_thread(self, thread_id: str) -> bool:
        return await self._call(self._store.delete_thread, thread_id)

    async def update_thread_workspace(
        self, thread_id: str, *, workspace: str, branch: str | None = None
    ) -> dict[str, Any] | None:
        return await self._call(
            self._store.update_thread_workspace,
            thread_id,
            workspace=workspace,
            branch=branch,
        )

    async def set_thread_title(self, thread_id: str, title: str) -> dict[str, Any] | None:
        return await self._call(self._store.set_thread_title, thread_id, title)

    async def set_thread_pinned(self, thread_id: str, pinned: bool) -> dict[str, Any] | None:
        return await self._call(self._store.set_thread_pinned, thread_id, pinned)

    async def set_thread_archived(self, thread_id: str, archived: bool) -> dict[str, Any] | None:
        return await self._call(self._store.set_thread_archived, thread_id, archived)

    async def set_thread_read(self, thread_id: str, read: bool) -> dict[str, Any] | None:
        return await self._call(self._store.set_thread_read, thread_id, read)

    async def configure_thread(
        self,
        thread_id: str,
        *,
        model_id: str | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any] | None:
        return await self._call(
            self._store.configure_thread,
            thread_id,
            model_id=model_id,
            thinking_level=thinking_level,
        )

    async def reorder_threads(self, project_id: str, thread_ids: list[str]) -> list[dict[str, Any]]:
        return await self._call(self._store.reorder_threads, project_id, thread_ids)

    async def create_turn(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call(self._store.create_turn, **kwargs)

    async def ensure_turn(
        self, thread_id: str, core_turn_id: str, *, status: str = "queued"
    ) -> dict[str, Any]:
        return await self._call(
            self._store.ensure_turn, thread_id, core_turn_id, status=status
        )

    async def get_turn(self, core_turn_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.get_turn, core_turn_id)

    async def list_turns(self, thread_id: str) -> list[dict[str, Any]]:
        return await self._call(self._store.list_turns, thread_id)

    async def update_turn_input(
        self, core_turn_id: str, *, user_text: str, attachments: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        return await self._call(
            self._store.update_turn_input,
            core_turn_id,
            user_text=user_text,
            attachments=attachments,
        )

    async def project_event(self, thread_id: str, event: dict[str, Any]) -> None:
        await self._call(self._store.project_event, thread_id, event)

    async def record_event(self, thread_id: str, event: dict[str, Any]) -> None:
        await self._call(self._store.record_event, thread_id, event)

    async def list_events(self, thread_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        return await self._call(self._store.list_events, thread_id, limit)

    async def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        return await self._call(self._store.get_attachment, attachment_id)

    async def add_attachment(
        self,
        *,
        name: str,
        mime_type: str | None,
        data: bytes,
        root: Path,
    ) -> dict[str, Any]:
        return await self._call(
            self._store.add_attachment,
            name=name,
            mime_type=mime_type,
            data=data,
            root=root,
        )

    async def delete_attachment(self, attachment_id: str) -> bool:
        return await self._call(self._store.delete_attachment, attachment_id)

    async def cleanup_orphan_attachments(self, root: Path) -> list[str]:
        return await self._call(self._store.cleanup_orphan_attachments, root)
