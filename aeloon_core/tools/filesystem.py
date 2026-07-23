"""Filesystem tools: read, atomic write, and exact string replacement."""

from __future__ import annotations

import codecs
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aeloon_core.tools.base import WorkspaceTool


class ReadArgs(BaseModel):
    path: str = Field(description="File path to read.")
    offset: int = Field(default=1, ge=1, description="Line number to start from, 1-indexed.")
    limit: int | None = Field(default=None, ge=1, description="Maximum number of lines.")


class ReadTool(WorkspaceTool):
    """Read file contents with line-numbered output."""

    name = "read"
    concurrency_mode = "read_only"
    description = "Read a UTF-8 text file. Returns numbered lines and supports offset/limit."
    args_model = ReadArgs

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2_000

    async def execute(
        self,
        path: str,
        offset: int = 1,
        limit: int | None = None,
    ) -> str:
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"
            lines = fp.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            if total == 0:
                return f"(Empty file: {path})"
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"
            start = max(offset, 1) - 1
            requested_end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered: list[str] = []
            output_chars = 0
            capped = False
            end = start

            for index, line in enumerate(lines[start:requested_end], start=start + 1):
                rendered = f"{index}| {line}"
                separator_chars = 1 if numbered else 0
                if numbered and output_chars + separator_chars + len(rendered) > self._MAX_CHARS:
                    capped = True
                    break
                if not numbered and len(rendered) > self._MAX_CHARS:
                    suffix = "... (line truncated)"
                    rendered = rendered[: self._MAX_CHARS - len(suffix)] + suffix
                    capped = True
                numbered.append(rendered)
                output_chars += separator_chars + len(rendered)
                end = index
                if capped:
                    break

            result = "\n".join(numbered)
            if capped:
                result += (
                    f"\n\n(Output capped at {self._MAX_CHARS} chars. "
                    f"Showing lines {offset}-{end}. Use offset={end + 1}.)"
                )
            elif end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1}.)"
            else:
                result += f"\n\n(End of file - {total} lines total)"
            return result
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text: {path}"
        except Exception as exc:
            return f"Error reading file: {exc}"


# Soft per-call write/edit size guidance when model metadata is unavailable.
# Model metadata reports tokens; tool descriptions count characters. This value
# is advertised as a preference only — runtime does not hard-reject on it.
# The real host ceiling is the independent 16 MiB UTF-8 file size limit.
DEFAULT_MAX_ARGUMENT_CHARS = 128_000
ESTIMATED_OUTPUT_CHARS_PER_TOKEN = 4
_MAX_FILE_BYTES = 16 * 1024 * 1024
HOST_MAX_ARGUMENT_CHARS = _MAX_FILE_BYTES
_MAX_ARGUMENT_CHARS = DEFAULT_MAX_ARGUMENT_CHARS  # backward-compatible alias


def resolve_max_argument_chars(model_max_output_tokens: int | None = None) -> int:
    """Return soft per-call write/edit character guidance for tool descriptions.

    Defaults to 128,000 characters when metadata is unavailable. Otherwise,
    convert the model's token ceiling to an estimated character capacity and
    constrain it only by the independent host resource ceiling. Callers use
    this for prompting the model, not for runtime rejection.
    """

    if model_max_output_tokens is None:
        return DEFAULT_MAX_ARGUMENT_CHARS
    try:
        model_limit = int(model_max_output_tokens)
    except (TypeError, ValueError):
        return DEFAULT_MAX_ARGUMENT_CHARS
    if model_limit <= 0:
        return DEFAULT_MAX_ARGUMENT_CHARS
    model_character_limit = model_limit * ESTIMATED_OUTPUT_CHARS_PER_TOKEN
    return min(HOST_MAX_ARGUMENT_CHARS, model_character_limit)


def _render_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def _error(
    code: str,
    message: str,
    *,
    next_action: str,
    **details: Any,
) -> str:
    fields = [f"{key}={_render_value(value)}" for key, value in details.items()]
    fields.append(f"next_action={next_action!r}")
    return f"Error [{code}]: {message}; " + "; ".join(fields)


class _ToolFailure(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        next_action: str,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.next_action = next_action
        self.details = details

    def render(self) -> str:
        return _error(
            self.code,
            self.message,
            next_action=self.next_action,
            **self.details,
        )


@dataclass(frozen=True)
class _FileBaseline:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class _AtomicCommit:
    total_bytes: int
    sha256: str


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_snapshot(path: Path, *, display_path: str) -> tuple[bytes, _FileBaseline]:
    try:
        initial = path.lstat()
    except FileNotFoundError as exc:
        raise _ToolFailure(
            "FILE_NOT_FOUND",
            "Target file does not exist.",
            path=display_path,
            next_action="Create a new file with write, or choose an existing file.",
        ) from exc
    if stat.S_ISLNK(initial.st_mode):
        raise _ToolFailure(
            "PATH_SYMLINK",
            "Symbolic links are not valid mutation targets.",
            path=display_path,
            next_action="Choose a regular file path, or the real target path of a symlink.",
        )
    if not stat.S_ISREG(initial.st_mode):
        raise _ToolFailure(
            "NOT_FILE",
            "Mutation target is not a regular file.",
            path=display_path,
            next_action="Choose a regular file path.",
        )
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise _ToolFailure(
            "CONCURRENT_MODIFICATION",
            "Target disappeared while it was being read.",
            path=display_path,
            next_action="Read the current file state and retry.",
        ) from exc
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise _ToolFailure(
            "CONCURRENT_MODIFICATION",
            "Target disappeared while it was being read.",
            path=display_path,
            next_action="Read the current file state and retry.",
        ) from exc
    if _stat_fingerprint(initial) != _stat_fingerprint(current):
        raise _ToolFailure(
            "CONCURRENT_MODIFICATION",
            "Target changed while it was being read.",
            path=display_path,
            next_action="Read the current file and retry with its latest contents.",
        )
    return raw, _FileBaseline(
        device=current.st_dev,
        inode=current.st_ino,
        mode=current.st_mode,
        size=current.st_size,
        mtime_ns=current.st_mtime_ns,
        ctime_ns=current.st_ctime_ns,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validate_utf8(content: bytes) -> None:
    """Validate UTF-8 without retaining a second full decoded copy in memory."""

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    view = memoryview(content)
    chunk_size = 64 * 1024
    for offset in range(0, len(view), chunk_size):
        decoder.decode(view[offset : offset + chunk_size], final=False)
    decoder.decode(b"", final=True)


def _baseline_matches(path: Path, expected: _FileBaseline | None) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return expected is None
    if expected is None or not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
        return False
    metadata = _stat_fingerprint(current)
    expected_metadata = (
        expected.device,
        expected.inode,
        expected.mode,
        expected.size,
        expected.mtime_ns,
        expected.ctime_ns,
    )
    if metadata != expected_metadata:
        return False
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.lstat()
    except FileNotFoundError:
        return False
    after_metadata = _stat_fingerprint(after)
    return after_metadata == expected_metadata and digest == expected.sha256


def _atomic_replace(
    path: Path,
    chunks: Iterable[bytes],
    *,
    display_path: str,
    baseline: _FileBaseline | None,
    revalidate_path: Callable[[], None],
) -> _AtomicCommit:
    """Stage content beside path, fsync it, then atomically replace after revalidation."""

    temp_path: Path | None = None
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            if baseline is not None:
                os.fchmod(handle.fileno(), stat.S_IMODE(baseline.mode))
            for chunk in chunks:
                if not chunk:
                    continue
                written = handle.write(chunk)
                if written != len(chunk):
                    raise OSError(
                        f"short temporary-file write: expected {len(chunk)} bytes, wrote {written}"
                    )
                digest.update(chunk)
                total_bytes += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        revalidate_path()
        if not _baseline_matches(path, baseline):
            raise _ToolFailure(
                "CONCURRENT_MODIFICATION",
                "Target changed before the atomic commit.",
                path=display_path,
                next_action="Read the current file and retry against its latest contents.",
            )
        os.replace(temp_path, path)
        temp_path = None
        return _AtomicCommit(total_bytes=total_bytes, sha256=digest.hexdigest())
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()


def _display_path(tool: WorkspaceTool, target: Path) -> str:
    """Prefer a workspace-relative path; fall back to the absolute path."""

    resolved = target.resolve(strict=False)
    if resolved.is_relative_to(tool.workspace):
        return resolved.relative_to(tool.workspace).as_posix()
    return str(resolved)


def _validate_mutation_path(tool: WorkspaceTool, path: str) -> Path:
    if not path or path == ".":
        raise _ToolFailure(
            "PATH_INVALID",
            "A non-empty file path is required.",
            path=path,
            next_action="Provide a relative path from the workspace or an absolute file path.",
        )
    if "\x00" in path:
        raise _ToolFailure(
            "PATH_INVALID",
            "File paths cannot contain NUL bytes.",
            path=path,
            next_action="Remove the NUL byte and provide a valid file path.",
        )
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _ToolFailure(
            "PATH_INVALID",
            "File path cannot be encoded as UTF-8.",
            path=path,
            next_action="Remove invalid surrogate characters from the path and retry.",
        ) from exc

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        target = candidate
        parts = list(candidate.parts)
        # On POSIX the first part is the root ("/"); on Windows it is the drive.
        if len(parts) <= 1:
            raise _ToolFailure(
                "PATH_INVALID",
                "A file path is required, not a filesystem root.",
                path=path,
                next_action="Provide a path to a file, not a root directory.",
            )
        current = Path(parts[0])
        walk_parts = parts[1:]
    else:
        target = tool.workspace.joinpath(candidate)
        current = tool.workspace
        walk_parts = [part for part in candidate.parts if part not in {"", "."}]
        if not walk_parts:
            raise _ToolFailure(
                "PATH_INVALID",
                "A file path is required, not the workspace root.",
                path=path,
                next_action="Provide a relative path to a file or an absolute file path.",
            )

    for index, part in enumerate(walk_parts):
        current = current / part
        try:
            entry = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(entry.st_mode):
            raise _ToolFailure(
                "PATH_SYMLINK",
                "Symbolic links are not allowed in mutation paths.",
                path=path,
                next_action="Choose a path containing only real directories, or the real target path.",
            )
        is_target = index == len(walk_parts) - 1
        if is_target and not stat.S_ISREG(entry.st_mode):
            raise _ToolFailure(
                "NOT_FILE",
                "Mutation target is not a regular file.",
                path=path,
                next_action="Choose a regular file path.",
            )
        if not is_target and not stat.S_ISDIR(entry.st_mode):
            raise _ToolFailure(
                "PATH_INVALID",
                "A parent component is not a directory.",
                path=path,
                next_action="Choose a path whose parent components are directories.",
            )

    resolved = target.resolve(strict=False)
    for denied in tool.denied_paths:
        if resolved == denied or resolved.is_relative_to(denied):
            raise _ToolFailure(
                "PATH_PROTECTED",
                "Mutation path is protected from agent tools.",
                path=path,
                next_action="Choose an unprotected path.",
            )
    return target


def _prepare_parent(tool: WorkspaceTool, path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _validate_mutation_path(tool, path)


def _build_write_args_model(max_chars: int) -> type[BaseModel]:
    limit = max(1, int(max_chars))

    class WriteArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        path: str = Field(
            description=(
                "File path to create or overwrite. Relative paths resolve from the workspace; "
                "absolute paths are allowed."
            ),
        )
        content: str = Field(
            description=(
                f"Complete UTF-8 file content. Prefer staying near {limit:,} characters when "
                "practical; larger content is accepted until the 16 MiB file-size limit."
            ),
        )

    WriteArgs.__name__ = "WriteArgs"
    WriteArgs.__qualname__ = "WriteArgs"
    return WriteArgs


# Default schema used when tools are constructed without an explicit budget.
WriteArgs = _build_write_args_model(DEFAULT_MAX_ARGUMENT_CHARS)


class WriteTool(WorkspaceTool):
    """Atomically create or overwrite a UTF-8 text file."""

    name = "write"
    concurrency_mode = "mutating"
    description = (
        f"Atomically create or overwrite a complete UTF-8 file (prefer ~"
        f"{DEFAULT_MAX_ARGUMENT_CHARS:,} characters; hard limit 16 MiB). "
        "Use str_replace for targeted edits."
    )
    args_model = WriteArgs

    def __init__(
        self,
        *,
        workspace: Path,
        denied_paths: Iterable[Path] = (),
        max_content_chars: int | None = None,
    ) -> None:
        super().__init__(workspace=workspace, denied_paths=denied_paths)
        self.configure_max_content_chars(
            DEFAULT_MAX_ARGUMENT_CHARS if max_content_chars is None else max_content_chars
        )

    def configure_max_content_chars(self, max_content_chars: int) -> None:
        """Update soft per-call size guidance advertised in the tool schema."""

        limit = max(1, int(max_content_chars))
        self.max_content_chars = limit
        self.args_model = _build_write_args_model(limit)
        self.description = (
            f"Atomically create or overwrite a complete UTF-8 file (prefer ~{limit:,} "
            "characters; hard limit 16 MiB). Use str_replace for targeted edits."
        )

    async def execute(
        self,
        path: str,
        content: str,
    ) -> str:
        try:
            try:
                chunk = content.encode("utf-8")
            except UnicodeEncodeError:
                return _error(
                    "INVALID_UTF8",
                    "write content cannot be encoded as UTF-8.",
                    path=path,
                    field="content",
                    next_action="Remove invalid surrogate characters and retry.",
                )

            try:
                target = _validate_mutation_path(self, path)
                _, baseline = _read_snapshot(target, display_path=path)
            except _ToolFailure as exc:
                if exc.code != "FILE_NOT_FOUND":
                    raise
                baseline = None
            if len(chunk) > _MAX_FILE_BYTES:
                return _error(
                    "FILE_TOO_LARGE",
                    "write would exceed the maximum file size.",
                    path=path,
                    actual=len(chunk),
                    limit=_MAX_FILE_BYTES,
                    next_action="Reduce the content or split it across separate files.",
                )

            _prepare_parent(self, path, target)
            commit = _atomic_replace(
                target,
                (chunk,),
                display_path=path,
                baseline=baseline,
                revalidate_path=lambda: _validate_mutation_path(self, path),
            )
            display_path = _display_path(self, target)
            return (
                "Successfully wrote file; "
                f"path={display_path!r}; chars={len(content)}; bytes={len(chunk)}; "
                f"total_bytes={commit.total_bytes}; sha256={commit.sha256}"
            )
        except _ToolFailure as exc:
            return exc.render()
        except OSError as exc:
            return _error(
                "IO_ERROR",
                "Filesystem operation failed while writing the file.",
                path=path,
                actual=f"{type(exc).__name__}: {exc}",
                next_action="Resolve the filesystem error and retry; the target was not committed.",
            )


_NEWLINE_PATTERN = re.compile(br"\r?\n")
_CANONICAL_NEWLINE_PATTERN = br"(?:\r\n|(?<!\r)\n)"
_STAGING_CHUNK_BYTES = 64 * 1024


def _literal_pattern(content: bytes) -> bytes:
    """Match canonical literal bytes without consuming the CR half of CRLF."""

    return br"\r(?!\n)".join(re.escape(part) for part in content.split(b"\r"))


def _exact_newline_pattern(normalized_old: bytes) -> re.Pattern[bytes]:
    return re.compile(
        _CANONICAL_NEWLINE_PATTERN.join(
            _literal_pattern(part) for part in normalized_old.split(b"\n")
        )
    )


def _preferred_newline(content: bytes, start: int = 0, end: int | None = None) -> bytes:
    crlf_count = content.count(b"\r\n", start, end)
    newline_count = content.count(b"\n", start, end)
    if newline_count and crlf_count >= newline_count - crlf_count:
        return b"\r\n"
    return b"\n"


def _render_replacement(
    replacement_parts: tuple[bytes, ...],
    content: bytes,
    start: int,
    end: int,
    global_newline: bytes,
) -> bytes:
    if len(replacement_parts) == 1:
        return replacement_parts[0]

    local_newline = _preferred_newline(content, start, end)
    has_local_newline = content.count(b"\n", start, end) > 0
    fallback = local_newline if has_local_newline else global_newline
    original_endings = _NEWLINE_PATTERN.finditer(content, start, end)
    output = bytearray(replacement_parts[0])
    for part in replacement_parts[1:]:
        original = next(original_endings, None)
        newline = original.group() if original is not None else fallback
        if newline == b"\n" and output.endswith(b"\r"):
            # Keep a literal trailing CR distinct from the following logical newline.
            newline = b"\r\n"
        output.extend(newline)
        output.extend(part)
    return bytes(output)


def _byte_chunks(content: bytes, start: int, end: int) -> Iterator[bytes]:
    offset = start
    while offset < end:
        chunk_end = min(offset + _STAGING_CHUNK_BYTES, end)
        if chunk_end < end and content[chunk_end - 1 : chunk_end + 1] == b"\r\n":
            chunk_end += 1
        yield content[offset:chunk_end]
        offset = chunk_end


def _protect_literal_cr_boundaries(chunks: Iterable[bytes]) -> Iterator[bytes]:
    previous_ends_with_cr = False
    for chunk in chunks:
        if not chunk:
            continue
        if previous_ends_with_cr and chunk.startswith(b"\n"):
            chunk = b"\r\n" + chunk[1:]
        yield chunk
        previous_ends_with_cr = chunk.endswith(b"\r")


def _replacement_chunks(
    content: bytes,
    pattern: re.Pattern[bytes],
    replacement_parts: tuple[bytes, ...],
    global_newline: bytes,
    *,
    replace_all: bool,
) -> Iterator[bytes]:
    """Yield replacement output without retaining the expanded file in memory."""

    def pieces() -> Iterator[bytes]:
        previous_end = 0
        for match in pattern.finditer(content):
            yield from _byte_chunks(content, previous_end, match.start())
            yield _render_replacement(
                replacement_parts,
                content,
                match.start(),
                match.end(),
                global_newline,
            )
            previous_end = match.end()
            if not replace_all:
                break
        yield from _byte_chunks(content, previous_end, len(content))

    yield from _protect_literal_cr_boundaries(pieces())


def _build_str_replace_args_model(max_chars: int) -> type[BaseModel]:
    limit = max(1, int(max_chars))

    class StrReplaceArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        path: str = Field(
            description=(
                "UTF-8 file path to edit or create. Relative paths resolve from the workspace; "
                "absolute paths are allowed."
            ),
        )
        old_str: str = Field(
            description=(
                "Exact text to replace; CRLF and LF are treated as equivalent. "
                "Use an empty string to create a file that does not exist. "
                f"Prefer keeping replacements near {limit:,} characters when practical."
            ),
        )
        new_str: str = Field(
            description=(
                f"Replacement text. Prefer staying near {limit:,} characters when practical; "
                "larger text is accepted until the 16 MiB file-size limit."
            ),
        )
        replace_all: bool = Field(
            default=False, description="Replace every occurrence instead of just one."
        )

    StrReplaceArgs.__name__ = "StrReplaceArgs"
    StrReplaceArgs.__qualname__ = "StrReplaceArgs"
    return StrReplaceArgs


StrReplaceArgs = _build_str_replace_args_model(DEFAULT_MAX_ARGUMENT_CHARS)


class StrReplaceTool(WorkspaceTool):
    """Atomically replace exact text or create a missing UTF-8 file."""

    name = "str_replace"
    concurrency_mode = "mutating"
    description = (
        "Atomically replace an exact old_str in a UTF-8 file. An empty old_str creates a missing "
        "file from new_str, but refuses an existing target. For non-empty old_str, CRLF and LF "
        "are equivalent and existing line endings are preserved. The match must be unique unless "
        "replace_all=true."
    )
    args_model = StrReplaceArgs

    def __init__(
        self,
        *,
        workspace: Path,
        denied_paths: Iterable[Path] = (),
        max_content_chars: int | None = None,
    ) -> None:
        super().__init__(workspace=workspace, denied_paths=denied_paths)
        self.configure_max_content_chars(
            DEFAULT_MAX_ARGUMENT_CHARS if max_content_chars is None else max_content_chars
        )

    def configure_max_content_chars(self, max_content_chars: int) -> None:
        """Update soft per-call size guidance advertised in the tool schema."""

        limit = max(1, int(max_content_chars))
        self.max_content_chars = limit
        self.args_model = _build_str_replace_args_model(limit)
        self.description = (
            "Atomically replace an exact old_str in a UTF-8 file. An empty old_str creates a "
            "missing file from new_str, but refuses an existing target. For non-empty old_str, "
            "CRLF and LF are equivalent and existing line endings are preserved. The match must "
            f"be unique unless replace_all=true. Prefer replacements near {limit:,} characters; "
            "hard limit is 16 MiB for the resulting file."
        )

    async def execute(
        self,
        path: str,
        old_str: str,
        new_str: str,
        replace_all: bool = False,
    ) -> str:
        try:
            for field_name, value in (("old_str", old_str), ("new_str", new_str)):
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    return _error(
                        "INVALID_UTF8",
                        f"{field_name} cannot be encoded as UTF-8.",
                        path=path,
                        field=field_name,
                        next_action="Remove invalid surrogate characters and retry.",
                    )

            target = _validate_mutation_path(self, path)
            if not old_str:
                try:
                    target.lstat()
                except FileNotFoundError:
                    pass
                else:
                    return _error(
                        "TARGET_EXISTS",
                        "An empty old_str only creates a missing file.",
                        path=path,
                        next_action=(
                            "Use write to overwrite the complete file, or provide a non-empty "
                            "old_str for a targeted edit."
                        ),
                    )

                encoded_new = new_str.encode("utf-8")
                if len(encoded_new) > _MAX_FILE_BYTES:
                    return _error(
                        "FILE_TOO_LARGE",
                        "str_replace would exceed the maximum file size.",
                        path=path,
                        actual=len(encoded_new),
                        limit=_MAX_FILE_BYTES,
                        next_action="Reduce the content or split it across separate files.",
                    )

                _prepare_parent(self, path, target)
                commit = _atomic_replace(
                    target,
                    (encoded_new,),
                    display_path=path,
                    baseline=None,
                    revalidate_path=lambda: _validate_mutation_path(self, path),
                )
                display_path = _display_path(self, target)
                return (
                    "Successfully created file; "
                    f"path={display_path!r}; chars={len(new_str)}; bytes={len(encoded_new)}; "
                    f"total_bytes={commit.total_bytes}; sha256={commit.sha256}"
                )

            raw, baseline = _read_snapshot(target, display_path=path)
            try:
                _validate_utf8(raw)
            except UnicodeDecodeError:
                return _error(
                    "INVALID_UTF8",
                    "Target file is not valid UTF-8 text.",
                    path=path,
                    next_action="Choose a UTF-8 text file or repair its encoding first.",
                )

            normalized_old = old_str.replace("\r\n", "\n").encode("utf-8")
            normalized_new = new_str.replace("\r\n", "\n").encode("utf-8")
            if normalized_old == normalized_new:
                return _error(
                    "NO_CHANGE",
                    "Replacement would not change the file.",
                    path=path,
                    next_action="Provide a new_str that differs from old_str.",
                )

            pattern = _exact_newline_pattern(normalized_old)
            replacement_parts = tuple(normalized_new.split(b"\n"))
            global_newline = _preferred_newline(raw)
            match_count = 0
            first_span: tuple[int, int] | None = None
            line_numbers: list[int] = []
            line_number = 1
            line_cursor = 0
            for match in pattern.finditer(raw):
                match_count += 1
                if first_span is None:
                    first_span = match.span()
                if len(line_numbers) < 5:
                    line_number += raw.count(b"\n", line_cursor, match.start())
                    line_numbers.append(line_number)
                    line_cursor = match.start()

            if match_count == 0 or first_span is None:
                return _error(
                    "MATCH_NOT_FOUND",
                    "old_str was not found by exact matching.",
                    path=path,
                    actual=0,
                    next_action="Read the file and copy an exact old_str, including whitespace.",
                )
            if match_count > 1 and not replace_all:
                return _error(
                    "MATCH_NOT_UNIQUE",
                    "old_str matches more than once.",
                    path=path,
                    actual=match_count,
                    line_numbers=line_numbers,
                    next_action="Provide more surrounding context or set replace_all=true.",
                )

            commit = _atomic_replace(
                target,
                _replacement_chunks(
                    raw,
                    pattern,
                    replacement_parts,
                    global_newline,
                    replace_all=replace_all,
                ),
                display_path=path,
                baseline=baseline,
                revalidate_path=lambda: _validate_mutation_path(self, path),
            )
            display_path = _display_path(self, target)
            return (
                "Successfully replaced text; "
                f"path={display_path!r}; replacements={match_count if replace_all else 1}; "
                f"total_bytes={commit.total_bytes}; sha256={commit.sha256}"
            )
        except _ToolFailure as exc:
            return exc.render()
        except UnicodeDecodeError:
            return _error(
                "INVALID_UTF8",
                "Target file is not valid UTF-8 text.",
                path=path,
                next_action="Choose a UTF-8 text file or repair its encoding first.",
            )
        except OSError as exc:
            return _error(
                "IO_ERROR",
                "Filesystem operation failed while replacing text.",
                path=path,
                actual=f"{type(exc).__name__}: {exc}",
                next_action="Resolve the filesystem error and retry; the target was not committed.",
            )
