"""Crash-safe, conservative Workbench -> Runtime data migration.

The migration intentionally keeps the old trees untouched until the new
projection, attachments and managed worktrees have all been verified.  A
0600 JSONL journal makes a killed process recoverable without guessing which
worktrees were already moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from aeloon_core.git_workspace import git_command
from aeloon_core.store import RuntimeStore


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def migrate_workbench(
    *,
    from_workbench: Path,
    from_core: Path,
    data_dir: Path,
    roots_output: Path,
) -> dict[str, object]:
    """Migrate old Workbench/Core state into a Runtime data directory.

    The operation is idempotent after ``migration.complete``.  If a previous
    process died, the journal is replayed backwards before a fresh attempt;
    this leaves the source tree in its original shape and avoids starting a
    Runtime against a half-migrated projection.
    """

    from_workbench = _resolve_migration_path(from_workbench, "Workbench source")
    from_core = _resolve_migration_path(from_core, "Core source")
    data_dir = _resolve_migration_path(data_dir, "Runtime data directory")
    roots_output = _resolve_migration_path(roots_output, "Launcher roots output")
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_dir.chmod(0o700)
    complete_path = data_dir / "migration.complete"
    if complete_path.is_file() and not (data_dir / ".incomplete").exists():
        try:
            value = json.loads(complete_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                # The completion marker and launcher-roots file are published
                # as the final migration pair.  If a user or an interrupted
                # backup cleanup removed only the roots file, reconstruct it
                # from the committed projection before treating the migration
                # as complete.
                if not roots_output.is_file() and (data_dir / "runtime.sqlite").is_file():
                    store = RuntimeStore(data_dir / "runtime.sqlite")
                    try:
                        roots = {
                            str(Path(str(item["path"])).expanduser().resolve(strict=False))
                            for item in store.list_projects()
                        }
                        roots.update(
                            str(Path(str(item["workspace"])).expanduser().resolve(strict=False))
                            for item in store.list_threads()
                            if item.get("kind") == "worktree"
                        )
                    finally:
                        store.close()
                    _write_json(roots_output, {"roots": sorted(roots), "version": 1})
                return {**value, "migrated": True, "data_dir": str(data_dir)}
        except (OSError, json.JSONDecodeError):
            raise RuntimeError(
                "migration.complete is corrupt; remove it after inspection"
            ) from None

    lock_path = data_dir / ".migration.lock"
    _acquire_lock(lock_path)
    incomplete = data_dir / ".incomplete"
    journal = data_dir / "migration.jsonl"
    target_db = data_dir / "runtime.sqlite"
    recovering = incomplete.exists() and journal.exists()
    created_target = not target_db.exists() or recovering
    worktrees_preexisting = (data_dir / "worktrees").exists() and not recovering
    roots_preexisting = roots_output.exists()
    copied_paths: list[Path] = []
    moved: list[dict[str, str]] = []
    store: RuntimeStore | None = None
    backup_path: Path | None = None
    roots_backup: bytes | None = None
    roots_backup_mode: int | None = None
    try:
        if recovering:
            _recover_worktrees(journal, target_db)
            # A journaled attempt owns every opaque target it created. Remove
            # those leftovers before copying the source again; otherwise a
            # retry could silently retain stale attachments or session logs.
            for path in (
                target_db,
                target_db.with_name(f"{target_db.name}-wal"),
                target_db.with_name(f"{target_db.name}-shm"),
                data_dir / "attachments",
                data_dir / "harness-sessions",
                data_dir / "config.json",
                data_dir / "worktrees",
            ):
                if path.exists():
                    copied_paths.append(path)
            target_db.unlink(missing_ok=True)
            target_db.with_name(f"{target_db.name}-wal").unlink(missing_ok=True)
            target_db.with_name(f"{target_db.name}-shm").unlink(missing_ok=True)
        _write_json(incomplete, {"started_at": time.time(), "pid": os.getpid()})
        _append_stage(journal, "start", {"from_workbench": str(from_workbench)})

        source_db = _find_database(from_workbench)
        if source_db is not None:
            _copy_sqlite(source_db, target_db)
            _append_stage(journal, "database_copied", {"source": str(source_db)})
        store = RuntimeStore(target_db)
        projects = store.list_projects()
        threads = store.list_threads()
        # Keep a semantic fingerprint before the migration rewrites legacy
        # attachment descriptors and moves managed worktrees.  The database
        # bytes are intentionally not compared: those two operations are
        # expected to change paths, while ids, turns, events and content must
        # remain identical.
        projection_before = _projection_signature(store)

        attachment_source = from_workbench / "attachments"
        attachment_target = data_dir / "attachments"
        sessions_target = data_dir / "harness-sessions"
        sessions_was_present = sessions_target.exists()
        session_source = from_core / "harness-sessions"
        if not session_source.is_dir():
            # Older Core snapshots used the shorter ``sessions`` directory;
            # Runtime's repository has always exposed ``harness-sessions``.
            session_source = from_core / "sessions"
        _copy_tree_if_present(session_source, sessions_target)
        if not sessions_was_present and sessions_target.exists():
            copied_paths.append(sessions_target)
        config = from_core / "config.json"
        if config.is_file() and not (data_dir / "config.json").exists():
            shutil.copy2(config, data_dir / "config.json")
            (data_dir / "config.json").chmod(0o600)
            copied_paths.append(data_dir / "config.json")
        copied_paths.extend(
            _rewrite_attachment_paths(
                target_db,
                attachment_target,
                source_root=attachment_source,
            )
        )
        copied_paths.extend(
            _import_legacy_attachments(
                target_db,
                source_root=attachment_source,
                target_root=attachment_target,
            )
        )
        _append_stage(journal, "opaque_data_copied", {})

        for thread in threads:
            if thread.get("kind") != "worktree":
                continue
            old = Path(str(thread["workspace"])).expanduser().resolve(strict=False)
            project = next(
                (item for item in projects if item["id"] == thread["project_id"]), None
            )
            if project is None or not old.exists():
                raise RuntimeError(f"Managed worktree is missing: {old}")
            status_code, status_output, status_error = git_command(
                old, ["status", "--porcelain"]
            )
            if status_code != 0:
                raise RuntimeError(status_error[-4000:] or "Could not inspect managed worktree")
            if status_output.strip():
                raise RuntimeError(f"Managed worktree is dirty: {old}")
            tree_hash = _git_tree_hash(old)
            new = (data_dir / "worktrees" / str(project["id"]) / str(thread["id"])).resolve(
                strict=False
            )
            new.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            new.parent.chmod(0o700)
            if new.exists():
                raise RuntimeError(f"Migration target already exists: {new}")
            code, output, error = git_command(
                Path(str(project["path"])), ["worktree", "move", str(old), str(new)]
            )
            if code != 0:
                raise RuntimeError(error[-4000:] or output[-4000:] or "Could not move worktree")
            entry = {
                "thread_id": str(thread["id"]),
                "project_path": str(project["path"]),
                "old": str(old),
                "new": str(new),
                "branch": str(thread.get("branch") or "") or None,
                "tree_hash": tree_hash,
            }
            moved.append(entry)
            _append_stage(journal, "worktree_moved", entry)
            updated = store.update_thread_workspace(
                str(thread["id"]), workspace=str(new), branch=entry["branch"]
            )
            if updated is None:
                raise RuntimeError(f"Thread disappeared during migration: {thread['id']}")

        _verify_projection(store, moved, projection_before)
        _verify_attachment_blobs(attachment_target)
        roots = sorted(
            {str(Path(str(item["path"])).expanduser().resolve(strict=False)) for item in projects}
            | {entry["new"] for entry in moved}
        )
        if roots_preexisting:
            roots_backup = roots_output.read_bytes()
            roots_backup_mode = roots_output.stat().st_mode & 0o777
        roots_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_json(roots_output, {"roots": roots, "version": 1})
        _append_stage(journal, "verified", {"projects": len(projects), "threads": len(threads)})

        backup = _backup_old_data(from_workbench, from_core, data_dir)
        backup_path = backup
        result: dict[str, object] = {
            "migrated": True,
            "projects": len(projects),
            "threads": len(threads),
            "data_dir": str(data_dir),
            "backup": str(backup),
        }
        _append_stage(journal, "complete", result)
        _write_json(complete_path, {**result, "completed_at": time.time()})
        incomplete.unlink(missing_ok=True)
        return result
    except Exception:
        # Restore external worktrees before deleting the projection that
        # describes them.  The old tree remains the source of truth if any
        # later verification, backup, or roots write fails.
        _rollback_entries(moved)
        if store is not None:
            for entry in moved:
                store.update_thread_workspace(
                    entry["thread_id"], workspace=entry["old"], branch=entry.get("branch") or None
                )
            store.close()
            store = None
        if created_target:
            target_db.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                (target_db.parent / f"{target_db.name}{suffix}").unlink(missing_ok=True)
        for path in reversed(copied_paths):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        if not worktrees_preexisting:
            shutil.rmtree(data_dir / "worktrees", ignore_errors=True)
        if not roots_preexisting:
            roots_output.unlink(missing_ok=True)
        elif roots_backup is not None:
            roots_output.write_bytes(roots_backup)
            if roots_backup_mode is not None:
                roots_output.chmod(roots_backup_mode)
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
        incomplete.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
        raise
    finally:
        if store is not None:
            store.close()
        _release_lock(lock_path)


def _acquire_lock(path: Path) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            return _acquire_lock(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"Migration lock exists: {path}") from None
        raise RuntimeError(f"Migration already running (pid {pid})") from None
    else:
        os.write(fd, json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode())
        os.close(fd)
        path.chmod(0o600)


def _release_lock(path: Path) -> None:
    path.unlink(missing_ok=True)


def _append_stage(path: Path, stage: str, detail: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "at": time.time(), **detail}) + "\n")
    path.chmod(0o600)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _recover_worktrees(journal: Path, target_db: Path) -> None:
    entries: list[dict[str, str]] = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("stage") == "worktree_moved":
            entries.append(value)
    _rollback_entries(entries, repair=True)
    if target_db.exists() and entries:
        store = RuntimeStore(target_db)
        try:
            for entry in entries:
                store.update_thread_workspace(
                    entry["thread_id"], workspace=entry["old"], branch=entry.get("branch") or None
                )
        finally:
            store.close()
    journal.unlink(missing_ok=True)


def _rollback_entries(entries: list[dict[str, str]], *, repair: bool = False) -> None:
    for entry in reversed(entries):
        old = Path(entry["old"])
        new = Path(entry["new"])
        if not new.exists():
            continue
        project = Path(entry["project_path"])
        code, output, error = git_command(project, ["worktree", "move", str(new), str(old)])
        if code != 0 and repair:
            # Git's worktree administration can retain the moved path after a
            # killed migration. Repair the registration only while recovering
            # an interrupted journal, then retry the reversible move.
            git_command(project, ["worktree", "repair", str(new)])
            code, output, error = git_command(
                project, ["worktree", "move", str(new), str(old)]
            )
        if code != 0:
            raise RuntimeError(error[-4000:] or output[-4000:] or "Could not roll back worktree")


def _verify_projection(
    store: RuntimeStore,
    moved: list[dict[str, str]],
    before: dict[str, object] | None = None,
) -> None:
    if before is not None:
        after = _projection_signature(store)
        if after != before:
            raise RuntimeError("Migrated projects/threads/turns/events projection is inconsistent")
    for entry in moved:
        thread = store.get_thread(entry["thread_id"])
        if thread is None or thread["workspace"] != entry["new"]:
            raise RuntimeError(f"Migrated thread projection is inconsistent: {entry['thread_id']}")
        if not Path(entry["new"]).is_dir():
            raise RuntimeError(f"Migrated worktree is not a directory: {entry['new']}")
        if _git_tree_hash(Path(entry["new"])) != entry.get("tree_hash"):
            raise RuntimeError(f"Migrated worktree file hash changed: {entry['thread_id']}")


def _git_tree_hash(worktree: Path) -> str:
    """Hash the committed file tree, independent of Git's worktree metadata."""

    code, head, _error = git_command(worktree, ["rev-parse", "HEAD"])
    head_value = head.strip() if code == 0 else "<unborn>"
    code, files, error = git_command(worktree, ["ls-files", "-s"])
    if code != 0:
        raise RuntimeError(error[-4000:] or "Could not hash managed worktree")
    payload = f"{head_value}\n{files}".encode()
    return hashlib.sha256(payload).hexdigest()


def _projection_signature(store: RuntimeStore) -> dict[str, object]:
    """Return the migration-stable projection identity and content signature.

    Workspace paths and attachment storage paths are deliberately omitted:
    both are rewritten by migration.  Turn attachment ids/types/sizes remain
    in the signature, so a dropped or duplicated upload is still detected.
    Event payloads are hashed canonically rather than compared by SQLite row
    ids, which can change when a legacy database is copied and migrated.
    """

    projects = [
        tuple(project.get(field) for field in ("id", "name", "path", "is_git", "created_at"))
        for project in store.list_projects()
    ]
    threads = []
    turns = []
    events = []
    for thread in store.list_threads():
        threads.append(
            tuple(
                thread.get(field)
                for field in (
                    "id",
                    "project_id",
                    "title",
                    "kind",
                    "model_id",
                    "thinking_level",
                    "branch",
                    "status",
                    "pinned_at",
                    "archived_at",
                    "is_unread",
                    "sort_order",
                    "created_at",
                )
            )
        )
        for turn in store.list_turns(str(thread["id"])):
            attachments = []
            for item in turn.get("attachments", []):
                if isinstance(item, dict):
                    attachments.append(
                        (
                            item.get("id"),
                            str(item.get("type") or "file"),
                            item.get("display_name") or item.get("name"),
                            item.get("mime_type"),
                            item.get("size_bytes"),
                        )
                    )
                elif isinstance(item, str):
                    attachments.append((item, "file", None, None, None))
                else:
                    attachments.append((None, "file", None, None, None))
            turns.append(
                (
                    turn.get("id"),
                    turn.get("thread_id"),
                    turn.get("core_turn_id"),
                    turn.get("user_text"),
                    tuple(attachments),
                    turn.get("status"),
                    turn.get("error"),
                    _canonical_json_hash(turn.get("blocks", [])),
                    _canonical_json_hash(turn.get("usage", {})),
                    turn.get("created_at"),
                    turn.get("updated_at"),
                )
            )
        for event in store.list_events(str(thread["id"]), limit=10**9):
            events.append(
                (
                    str(thread["id"]),
                    int(event.get("seq", 0)),
                    str(event.get("name", "")),
                    _canonical_json_hash(event),
                )
            )
    return {
        "projects": tuple(projects),
        "threads": tuple(threads),
        "turns": tuple(turns),
        "events": tuple(events),
    }


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _verify_attachment_blobs(root: Path) -> None:
    """Validate every imported content-addressed blob before publishing roots."""

    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Attachment target is not a private directory: {root}")
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".blob":
            raise RuntimeError(f"Unexpected attachment target: {candidate}")
        digest = candidate.stem
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Invalid attachment blob name: {candidate.name}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Attachment blob hash mismatch: {candidate.name}")
        if candidate.stat().st_mode & 0o077:
            raise RuntimeError(f"Attachment blob is not private: {candidate}")


def _backup_old_data(from_workbench: Path, from_core: Path, data_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = data_dir / f"v2-backup-{timestamp}"
    suffix = 1
    while backup.exists():
        backup = data_dir / f"v2-backup-{timestamp}-{suffix}"
        suffix += 1
    backup.mkdir(mode=0o700)
    _copy_non_worktree_tree(from_workbench, backup / "workbench")
    _copy_non_worktree_tree(from_core, backup / "core")
    return backup


def _copy_non_worktree_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_symlink():
        raise RuntimeError(f"Migration source contains a symbolic link: {source}")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if source.is_file():
        shutil.copy2(source, target / source.name)
        (target / source.name).chmod(0o600)
        return
    for child in source.iterdir():
        if child.is_symlink():
            raise RuntimeError(f"Migration source contains a symbolic link: {child}")
        if child.name in {"worktrees", ".git"}:
            continue
        destination = target / child.name
        if child.is_dir():
            _copy_non_worktree_tree(child, destination)
        elif child.is_file():
            shutil.copy2(child, destination)
            destination.chmod(0o600)


def _find_database(root: Path) -> Path | None:
    candidates = (root / "workbench.sqlite", root / "database.sqlite", root)
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError(f"Migration source contains a symbolic link: {candidate}")
        if candidate.is_file() and candidate.suffix in {".sqlite", ".db"}:
            return candidate
    return None


def _copy_sqlite(source: Path, target: Path) -> None:
    source_db = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        target.unlink(missing_ok=True)
        target_db = sqlite3.connect(target)
        try:
            source_db.backup(target_db)
            target_db.commit()
        finally:
            target_db.close()
    finally:
        source_db.close()
    target.chmod(0o600)


def _rewrite_attachment_paths(
    target_db: Path,
    attachment_root: Path,
    *,
    source_root: Path | None = None,
) -> list[Path]:
    copied: list[Path] = []
    if not target_db.exists():
        return copied
    root_created = not attachment_root.exists()
    db = sqlite3.connect(target_db)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(attachments)")}
        if not {"id", "storage_path"}.issubset(columns):
            return copied
        rows = db.execute("SELECT id, storage_path FROM attachments").fetchall()
        for attachment_id, old_path in rows:
            source = Path(str(old_path))
            if source_root is not None:
                # The old database contains absolute paths, but migration is
                # only allowed to read the Workbench attachment tree.  A
                # stale/corrupt row must not turn into an arbitrary local file
                # read just because that path happens to exist.
                source = _legacy_attachment_path(
                    source_root,
                    str(attachment_id),
                    str(old_path),
                )
            if not source.is_file():
                raise RuntimeError(f"Legacy attachment is missing: {old_path}")
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            candidate = attachment_root / f"{digest}.blob"
            if (
                not candidate.exists()
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest
            ):
                candidate.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if root_created:
                    copied.append(attachment_root)
                    root_created = False
                candidate.write_bytes(data)
                candidate.chmod(0o600)
                if candidate not in copied:
                    copied.append(candidate)
            if candidate.exists():
                db.execute(
                    "UPDATE attachments SET storage_path = ? WHERE id = ?",
                    (str(candidate), attachment_id),
                )
        db.commit()
    finally:
        db.close()
    return copied


def _import_legacy_attachments(
    target_db: Path,
    *,
    source_root: Path,
    target_root: Path,
) -> list[Path]:
    """Import file/image descriptors embedded in legacy turn JSON.

    Workbench stored attachment metadata only in ``turns.attachments_json``
    and kept the bytes under one UUID directory per upload.  Runtime v3 has a
    first-class attachment index and content-addressed blobs, so migration
    creates those rows and rewrites each descriptor to metadata-only form.
    """

    if not target_db.exists():
        return []
    copied: list[Path] = []
    source_root = source_root.expanduser().resolve(strict=False)
    target_root = target_root.expanduser().resolve(strict=False)
    root_created = not target_root.exists()
    db = sqlite3.connect(target_db)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(turns)")}
        if "attachments_json" not in columns:
            return copied
        rows = db.execute("SELECT core_turn_id, attachments_json FROM turns").fetchall()
        for core_turn_id, raw in rows:
            try:
                values = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(values, list):
                continue
            changed = False
            normalized: list[object] = []
            for value in values:
                if not isinstance(value, dict):
                    normalized.append(value)
                    continue
                attachment_id = value.get("id")
                attachment_type = str(value.get("type") or "file")
                if (
                    not isinstance(attachment_id, str)
                    or not attachment_id
                    or attachment_type not in {"file", "image"}
                ):
                    normalized.append(value)
                    continue
                source = _legacy_attachment_path(
                    source_root,
                    attachment_id,
                    str(value.get("path") or value.get("source_path") or ""),
                )
                if not source.is_file():
                    raise RuntimeError(f"Legacy attachment is missing: {attachment_id}")
                data = source.read_bytes()
                mime_type = str(value.get("mime_type") or "application/octet-stream")
                limit = 10 * 1024 * 1024 if mime_type.startswith("image/") else 25 * 1024 * 1024
                if len(data) > limit:
                    raise RuntimeError(f"Legacy attachment exceeds Runtime limit: {attachment_id}")
                digest = hashlib.sha256(data).hexdigest()
                target = target_root / f"{digest}.blob"
                target_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                target_root.chmod(0o700)
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    if root_created:
                        copied.append(target_root)
                        root_created = False
                    target.write_bytes(data)
                    target.chmod(0o600)
                    if target not in copied:
                        copied.append(target)
                name = str(value.get("display_name") or value.get("name") or source.name)
                created_at = str(value.get("created_at") or _now())
                db.execute(
                    """INSERT INTO attachments
                       (id, name, mime_type, size_bytes, storage_path, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                         name = excluded.name,
                         mime_type = excluded.mime_type,
                         size_bytes = excluded.size_bytes,
                         storage_path = excluded.storage_path""",
                    (attachment_id, name, mime_type, len(data), str(target), created_at),
                )
                rewritten = {
                    "id": attachment_id,
                    "type": attachment_type,
                    "display_name": name,
                    "mime_type": mime_type,
                    "size_bytes": len(data),
                }
                normalized.append(rewritten)
                changed = True
            if changed:
                db.execute(
                    "UPDATE turns SET attachments_json = ? WHERE core_turn_id = ?",
                    (json.dumps(normalized), core_turn_id),
                )
        db.commit()
    finally:
        db.close()
    return copied


def _legacy_attachment_path(source_root: Path, attachment_id: str, raw_path: str) -> Path:
    if source_root.is_symlink():
        raise RuntimeError(f"Migration source contains a symbolic link: {source_root}")
    candidates: list[Path] = []
    if raw_path:
        candidate = Path(raw_path).expanduser()
        candidates.append(candidate if candidate.is_absolute() else source_root / candidate)
    directory = source_root / attachment_id
    if directory.is_dir():
        candidates.extend(sorted(item for item in directory.iterdir() if item.is_file()))
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if _is_within(resolved, source_root) and resolved.is_file():
            return resolved
    return source_root / attachment_id / "missing"


def _copy_tree_if_present(source: Path, target: Path) -> None:
    if not source.is_dir() or target.exists():
        return
    if source.is_symlink():
        raise RuntimeError(f"Migration source contains a symbolic link: {source}")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    for child in source.iterdir():
        if child.is_symlink():
            raise RuntimeError(f"Migration source contains a symbolic link: {child}")
        destination = target / child.name
        if child.is_dir():
            _copy_tree_if_present(child, destination)
        else:
            shutil.copy2(child, destination)
            destination.chmod(0o600)
    target.chmod(0o700)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_migration_path(value: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link: {path}")
    return path.resolve(strict=False)


__all__ = ["migrate_workbench"]
