"""Transactional runtime for streamed model-authored file writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback retains the in-process lock.
    fcntl = None  # type: ignore[assignment]


class WriteRuntimeError(RuntimeError):
    """A streamed write could not be staged or committed safely."""


@dataclass(frozen=True)
class WriteLimits:
    max_files: int = 64
    max_file_bytes: int = 16 * 1024 * 1024
    max_batch_bytes: int = 64 * 1024 * 1024
    max_path_chars: int = 1_024


@dataclass
class StagedWriteFile:
    file_id: str
    path: str
    mode: str
    target: Path
    staging_path: Path
    baseline_exists: bool
    baseline_size: int | None
    baseline_mtime_ns: int | None
    baseline_sha256: str | None
    bytes_written: int = 0
    sha256: str = ""

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.file_id,
            "path": self.path,
            "mode": self.mode,
            "bytes": self.bytes_written,
            "sha256": self.sha256,
        }


@dataclass
class StagedWriteBatch:
    transaction_id: str
    staging_dir: Path
    files: list[StagedWriteFile]
    complete: bool = False

    def manifest(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "files": [item.manifest() for item in self.files],
        }


@dataclass
class _OpenFile:
    staged: StagedWriteFile
    handle: object
    digest: object


class WriteAttempt:
    """One provider-attempt staging area; it never mutates target files."""

    def __init__(
        self,
        coordinator: WriteCoordinator,
        transaction_id: str,
        staging_dir: Path,
    ) -> None:
        self.coordinator = coordinator
        self.transaction_id = transaction_id
        self.staging_dir = staging_dir
        self.files: list[StagedWriteFile] = []
        self._paths: set[Path] = set()
        self._open: _OpenFile | None = None
        self._batch_bytes = 0
        self._closed = False

    def start_file(self, *, file_id: str, path: str, mode: str) -> None:
        if self._closed:
            raise WriteRuntimeError("write attempt is already closed")
        if self._open is not None:
            raise WriteRuntimeError("nested WRITE blocks are not allowed")
        if len(self.files) >= self.coordinator.limits.max_files:
            raise WriteRuntimeError("write batch exceeds the file-count limit")
        if mode not in {"create", "overwrite"}:
            raise WriteRuntimeError("WRITE mode must be 'create' or 'overwrite'")
        target = self.coordinator.resolve_target(path)
        if target in self._paths:
            raise WriteRuntimeError(f"duplicate write target after normalization: {path}")
        self._paths.add(target)
        if target.is_symlink():
            raise WriteRuntimeError(f"refusing to write through a symlink: {path}")
        exists = target.exists()
        if exists and not target.is_file():
            raise WriteRuntimeError(f"write target is not a regular file: {path}")
        if mode == "create" and exists:
            raise WriteRuntimeError(f"create target already exists: {path}")
        if mode == "overwrite" and not exists:
            raise WriteRuntimeError(f"overwrite target does not exist: {path}")
        stat = target.stat() if exists else None
        stage_path = self.staging_dir / f"{len(self.files):04d}-{uuid.uuid4().hex}.body"
        handle = stage_path.open("xb")
        os.chmod(stage_path, 0o600)
        staged = StagedWriteFile(
            file_id=file_id,
            path=path,
            mode=mode,
            target=target,
            staging_path=stage_path,
            baseline_exists=exists,
            baseline_size=stat.st_size if stat else None,
            baseline_mtime_ns=stat.st_mtime_ns if stat else None,
            baseline_sha256=self.coordinator.hash_file(target) if stat else None,
        )
        self.files.append(staged)
        self._open = _OpenFile(staged=staged, handle=handle, digest=hashlib.sha256())

    def write_text(self, text: str) -> None:
        if self._open is None:
            raise WriteRuntimeError("received WRITE body without an open file")
        try:
            payload = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise WriteRuntimeError("WRITE body is not valid UTF-8") from exc
        item = self._open.staged
        next_file_size = item.bytes_written + len(payload)
        next_batch_size = self._batch_bytes + len(payload)
        if next_file_size > self.coordinator.limits.max_file_bytes:
            raise WriteRuntimeError(f"file exceeds byte limit: {item.path}")
        if next_batch_size > self.coordinator.limits.max_batch_bytes:
            raise WriteRuntimeError("write batch exceeds the byte limit")
        self._open.handle.write(payload)  # type: ignore[attr-defined]
        self._open.digest.update(payload)  # type: ignore[attr-defined]
        item.bytes_written = next_file_size
        self._batch_bytes = next_batch_size

    def end_file(self) -> None:
        if self._open is None:
            raise WriteRuntimeError("END WRITE received without an open file")
        handle = self._open.handle
        handle.flush()  # type: ignore[attr-defined]
        os.fsync(handle.fileno())  # type: ignore[attr-defined]
        handle.close()  # type: ignore[attr-defined]
        self._open.staged.sha256 = self._open.digest.hexdigest()  # type: ignore[attr-defined]
        self._open = None

    def finish_batch(self) -> StagedWriteBatch:
        if self._closed:
            raise WriteRuntimeError("write attempt is already closed")
        if self._open is not None:
            raise WriteRuntimeError("write batch ended before the current file")
        if not self.files:
            raise WriteRuntimeError("write batch contains no files")
        self._closed = True
        return StagedWriteBatch(
            transaction_id=self.transaction_id,
            staging_dir=self.staging_dir,
            files=list(self.files),
            complete=True,
        )

    def abort(self) -> None:
        if self._open is not None:
            try:
                self._open.handle.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._open = None
        self._closed = True
        shutil.rmtree(self.staging_dir, ignore_errors=True)


class WriteCoordinator:
    """Own path authorization, staging, commit, rollback, and crash recovery."""

    def __init__(
        self,
        *,
        workspace: Path,
        data_dir: Path,
        denied_paths: Iterable[Path] = (),
        limits: WriteLimits | None = None,
        formatter: Callable[[Path, str], None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.data_dir = Path(data_dir).expanduser().resolve(strict=False)
        self.denied_paths = tuple(
            Path(path).expanduser().resolve(strict=False) for path in denied_paths
        )
        self.limits = limits or WriteLimits()
        self.formatter = formatter
        workspace_key = hashlib.sha256(str(self.workspace).encode()).hexdigest()[:20]
        self.staging_root = self.data_dir / "write-staging" / workspace_key
        self.journal_root = self.data_dir / "write-journals" / workspace_key
        self.lock_path = self.data_dir / "write-locks" / f"{workspace_key}.lock"
        self.staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.journal_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._commit_lock = asyncio.Lock()
        self.recover_incomplete_commits()

    def start_attempt(self, transaction_id: str) -> WriteAttempt:
        safe = "".join(ch for ch in transaction_id if ch.isalnum() or ch in {"-", "_"})
        if not safe:
            raise WriteRuntimeError("invalid write transaction id")
        staging_dir = Path(tempfile.mkdtemp(prefix=f"{safe}-", dir=self.staging_root))
        os.chmod(staging_dir, 0o700)
        return WriteAttempt(self, transaction_id, staging_dir)

    def resolve_target(self, raw_path: str) -> Path:
        if not raw_path or len(raw_path) > self.limits.max_path_chars:
            raise WriteRuntimeError("WRITE path is empty or too long")
        if "\x00" in raw_path:
            raise WriteRuntimeError("WRITE path contains NUL")
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            raise WriteRuntimeError(f"absolute WRITE paths are not allowed: {raw_path}")
        if ".." in path.parts:
            raise WriteRuntimeError(f"WRITE path may not contain '..': {raw_path}")
        self._reject_symlink_components(path)
        resolved = (self.workspace / path).resolve(strict=False)
        if not resolved.is_relative_to(self.workspace):
            raise WriteRuntimeError(f"WRITE path escapes workspace: {raw_path}")
        for denied in self.denied_paths:
            if resolved == denied or resolved.is_relative_to(denied):
                raise WriteRuntimeError(f"WRITE path is protected: {raw_path}")
        return resolved

    def _reject_symlink_components(self, relative: Path) -> None:
        current = self.workspace
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise WriteRuntimeError(f"WRITE path traverses symlink: {relative}")

    async def commit_batch(self, batch: StagedWriteBatch) -> dict[str, object]:
        if not batch.complete:
            raise WriteRuntimeError("refusing to commit an incomplete write batch")
        async with self._commit_lock:
            with self._file_lock():
                try:
                    self._preflight(batch)
                    self._format(batch)
                    journal_path, journal = self._prepare_commit(batch)
                    try:
                        files = journal["files"]
                        assert isinstance(files, list)
                        for entry in files:
                            assert isinstance(entry, dict)
                            target = self.resolve_target(str(entry["path"]))
                            if str(target) != entry["target"]:
                                raise WriteRuntimeError(
                                    f"WRITE target changed before commit: {entry['path']}"
                                )
                            self._revalidate_entry(entry)
                            entry["started"] = True
                            self._write_journal(journal_path, journal)
                            os.replace(entry["temporary"], entry["target"])
                            self._fsync_directory(Path(str(entry["target"])).parent)
                            entry["committed"] = True
                            self._write_journal(journal_path, journal)
                    except BaseException:
                        self._rollback_journal(journal)
                        raise
                    journal["completed"] = True
                    self._write_journal(journal_path, journal)
                    try:
                        self._cleanup_journal_files(journal)
                        journal_path.unlink(missing_ok=True)
                        self._fsync_directory(self.journal_root)
                    except OSError:
                        # A completed journal is safe to clean on the next startup.
                        pass
                    return batch.manifest()
                finally:
                    shutil.rmtree(batch.staging_dir, ignore_errors=True)

    def discard_batch(self, batch: StagedWriteBatch) -> None:
        shutil.rmtree(batch.staging_dir, ignore_errors=True)

    def _preflight(self, batch: StagedWriteBatch) -> None:
        for item in batch.files:
            target = self.resolve_target(item.path)
            if target != item.target:
                raise WriteRuntimeError(f"WRITE target changed during generation: {item.path}")
            exists = target.exists()
            if item.mode == "create" and exists:
                raise WriteRuntimeError(f"create target appeared during generation: {item.path}")
            if item.mode == "overwrite":
                if not exists or not target.is_file() or target.is_symlink():
                    raise WriteRuntimeError(f"overwrite target changed type: {item.path}")
                stat = target.stat()
                if (
                    stat.st_size != item.baseline_size
                    or stat.st_mtime_ns != item.baseline_mtime_ns
                    or self.hash_file(target) != item.baseline_sha256
                ):
                    raise WriteRuntimeError(
                        f"overwrite target changed during generation: {item.path}"
                    )

    def _format(self, batch: StagedWriteBatch) -> None:
        if self.formatter is None:
            return
        for item in batch.files:
            self.formatter(item.staging_path, item.path)
            payload = item.staging_path.read_bytes()
            if len(payload) > self.limits.max_file_bytes:
                raise WriteRuntimeError(f"formatter output exceeds byte limit: {item.path}")
            item.bytes_written = len(payload)
            item.sha256 = hashlib.sha256(payload).hexdigest()
        if sum(item.bytes_written for item in batch.files) > self.limits.max_batch_bytes:
            raise WriteRuntimeError("formatted batch exceeds the byte limit")

    def _prepare_commit(self, batch: StagedWriteBatch) -> tuple[Path, dict[str, object]]:
        entries: list[dict[str, object]] = []
        created_dirs: list[str] = []
        try:
            for item in batch.files:
                created_dirs.extend(str(path) for path in self._ensure_parent(item.target.parent))
                temporary = item.target.parent / (
                    f".aeloon-write-{batch.transaction_id}-{uuid.uuid4().hex}"
                )
                backup = item.target.parent / (
                    f".aeloon-backup-{batch.transaction_id}-{uuid.uuid4().hex}"
                )
                shutil.copyfile(item.staging_path, temporary)
                os.chmod(
                    temporary,
                    (item.target.stat().st_mode & 0o777) if item.target.exists() else 0o644,
                )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                if item.target.exists():
                    shutil.copy2(item.target, backup)
                    with backup.open("rb") as handle:
                        os.fsync(handle.fileno())
                entries.append(
                    {
                        "target": str(item.target),
                        "path": item.path,
                        "temporary": str(temporary),
                        "backup": str(backup),
                        "existed": item.target.exists(),
                        "mode": item.mode,
                        "baseline_size": item.baseline_size,
                        "baseline_mtime_ns": item.baseline_mtime_ns,
                        "baseline_sha256": item.baseline_sha256,
                        "started": False,
                        "committed": False,
                    }
                )
        except BaseException:
            self._cleanup_journal_files({"files": entries})
            for raw in reversed(created_dirs):
                try:
                    Path(raw).rmdir()
                except OSError:
                    pass
            raise
        journal: dict[str, object] = {
            "version": 1,
            "transaction_id": batch.transaction_id,
            "files": entries,
            "created_dirs": list(dict.fromkeys(created_dirs)),
            "completed": False,
        }
        journal_path = self.journal_root / f"{batch.transaction_id}-{uuid.uuid4().hex}.json"
        try:
            self._write_journal(journal_path, journal)
        except BaseException:
            self._cleanup_journal_files(journal)
            for raw in reversed(created_dirs):
                try:
                    Path(raw).rmdir()
                except OSError:
                    pass
            raise
        return journal_path, journal

    def _write_journal(self, path: Path, journal: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(journal, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory(path.parent)

    def recover_incomplete_commits(self) -> None:
        with self._file_lock():
            for path in self.journal_root.glob("*.json"):
                try:
                    journal = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(journal, dict):
                        if journal.get("completed"):
                            self._cleanup_journal_files(journal)
                        else:
                            self._rollback_journal(journal)
                    path.unlink(missing_ok=True)
                except Exception:
                    # Leave an unreadable journal in place for operator inspection.
                    continue

    def _rollback_journal(self, journal: dict[str, object]) -> None:
        entries = journal.get("files")
        if not isinstance(entries, list):
            return
        for raw in reversed(entries):
            if not isinstance(raw, dict) or not raw.get("started"):
                continue
            target = Path(str(raw.get("target")))
            backup = Path(str(raw.get("backup")))
            temporary = Path(str(raw.get("temporary")))
            if not target.resolve(strict=False).is_relative_to(self.workspace):
                continue
            if raw.get("existed") and (
                not backup.name.startswith(".aeloon-backup-")
                or backup.parent.resolve(strict=False) != target.parent.resolve(strict=False)
            ):
                raise WriteRuntimeError("invalid backup path in WRITE recovery journal")
            if raw.get("existed") and backup.exists():
                os.replace(backup, target)
                self._fsync_directory(target.parent)
            elif not raw.get("existed") and not temporary.exists():
                target.unlink(missing_ok=True)
                self._fsync_directory(target.parent)
        self._cleanup_journal_files(journal)
        created_dirs = journal.get("created_dirs")
        if isinstance(created_dirs, list):
            for raw in reversed(created_dirs):
                try:
                    Path(str(raw)).rmdir()
                except OSError:
                    pass

    def _ensure_parent(self, parent: Path) -> list[Path]:
        missing: list[Path] = []
        current = parent
        while not current.exists() and current != self.workspace:
            missing.append(current)
            current = current.parent
        if current.exists() and current.is_symlink():
            raise WriteRuntimeError(f"WRITE parent became a symlink: {parent}")
        parent.mkdir(parents=True, exist_ok=True)
        return list(reversed(missing))

    def _revalidate_entry(self, entry: dict[str, object]) -> None:
        target = Path(str(entry["target"]))
        if entry.get("mode") == "create":
            if target.exists() or target.is_symlink():
                raise WriteRuntimeError(
                    f"create target appeared before commit: {entry.get('path')}"
                )
            return
        if not target.is_file() or target.is_symlink():
            raise WriteRuntimeError(f"overwrite target changed type: {entry.get('path')}")
        stat = target.stat()
        if (
            stat.st_size != entry.get("baseline_size")
            or stat.st_mtime_ns != entry.get("baseline_mtime_ns")
            or self.hash_file(target) != entry.get("baseline_sha256")
        ):
            raise WriteRuntimeError(f"overwrite target changed before commit: {entry.get('path')}")

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @contextmanager
    def _file_lock(self):
        with self.lock_path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _cleanup_journal_files(self, journal: dict[str, object]) -> None:
        entries = journal.get("files")
        if not isinstance(entries, list):
            return
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            for key, prefix in (("temporary", ".aeloon-write-"), ("backup", ".aeloon-backup-")):
                candidate = Path(str(raw.get(key)))
                if candidate.name.startswith(prefix) and candidate.parent.resolve(
                    strict=False
                ).is_relative_to(self.workspace):
                    candidate.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "StagedWriteBatch",
    "StagedWriteFile",
    "WriteAttempt",
    "WriteCoordinator",
    "WriteLimits",
    "WriteRuntimeError",
]
