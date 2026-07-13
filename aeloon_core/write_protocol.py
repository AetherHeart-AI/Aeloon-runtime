"""Incrementally demultiplex visible model text and framed file bodies."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from aeloon_core.write_runtime import StagedWriteBatch, WriteAttempt, WriteRuntimeError

OPEN_PREFIX = "<<<AELOON_WRITE_V1 "
HEADER_SUFFIX = ">>>\n"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class WriteProtocolError(WriteRuntimeError):
    """The model output violated the framed-write protocol."""


def protocol_guidance(transaction_id: str) -> str:
    return f"""[aeloon-core:write-protocol-v1]
You may create or fully replace UTF-8 workspace files without placing their bodies in JSON.
Use this exact framing, replacing metadata values as needed:
<<<AELOON_WRITE_V1 {{"tx":"{transaction_id}","id":"f1","path":"relative/path","mode":"create"}}>>>
raw file body<<<END_AELOON_WRITE_V1:{transaction_id}:f1>>>
After all files, emit exactly:
<<<END_AELOON_WRITE_BATCH_V1:{transaction_id}>>>
Modes are create and overwrite. The batch must be the response's only action: do not mix it
with native tool/control calls, and emit only whitespace after the batch marker. Missing markers,
invalid paths, truncation, or mixed actions cause the entire batch to be discarded. For small
changes to existing files, continue to use edit."""


@dataclass(frozen=True)
class DemuxResult:
    visible_content: str | None
    batch: StagedWriteBatch | None


class WriteFrameDecoder:
    """A bounded incremental parser whose BODY state writes directly to staging."""

    def __init__(
        self,
        *,
        transaction_id: str,
        attempt: WriteAttempt,
        on_visible: Callable[[str], None] | None = None,
        max_header_chars: int = 4_096,
    ) -> None:
        self.transaction_id = transaction_id
        self.attempt = attempt
        self.on_visible = on_visible
        self.max_header_chars = max_header_chars
        self._state = "text"
        self._buffer = ""
        self._visible: list[str] = []
        self._end_marker: str | None = None
        self._saw_file = False
        self._batch: StagedWriteBatch | None = None
        self._failed = False

    @property
    def saw_write(self) -> bool:
        return self._saw_file

    @property
    def visible_content(self) -> str | None:
        raw = "".join(self._visible)
        return raw if raw.strip() else None

    def feed(self, text: str) -> str:
        if self._failed:
            return ""
        visible_start = len(self._visible)
        self._buffer += text
        try:
            self._drain()
        except Exception:
            self._failed = True
            self.attempt.abort()
            raise
        return "".join(self._visible[visible_start:])

    def finalize(self, *, finish_reason: str, has_tool_calls: bool) -> DemuxResult:
        if self._failed:
            raise WriteProtocolError("write stream is already invalid")
        try:
            self._drain(final=True)
            if self._state == "think":
                self._buffer = ""
                self._state = "text"
            elif self._state == "body":
                raise WriteProtocolError("WRITE body is missing its END marker")
            elif self._state == "done" and self._buffer.strip():
                raise WriteProtocolError("non-whitespace content follows the batch marker")
            elif self._state == "text" and self._buffer:
                self._emit(self._buffer)
                self._buffer = ""

            if self._saw_file:
                if finish_reason != "stop":
                    raise WriteProtocolError(
                        f"WRITE batch ended with finish_reason={finish_reason!r}, not 'stop'"
                    )
                if has_tool_calls:
                    raise WriteProtocolError("WRITE batch may not be mixed with native tool calls")
                if self._batch is None:
                    raise WriteProtocolError("WRITE batch is missing its batch END marker")
            else:
                self.attempt.abort()
            return DemuxResult(visible_content=self.visible_content, batch=self._batch)
        except Exception:
            self._failed = True
            self.attempt.abort()
            raise

    def abort(self) -> None:
        self._failed = True
        self.attempt.abort()

    def _drain(self, *, final: bool = False) -> None:
        while self._buffer:
            if self._state == "body":
                assert self._end_marker is not None
                index = self._buffer.find(self._end_marker)
                if index >= 0:
                    if index:
                        self.attempt.write_text(self._buffer[:index])
                    self.attempt.end_file()
                    self._buffer = self._buffer[index + len(self._end_marker) :]
                    self._end_marker = None
                    self._state = "text"
                    continue
                keep = self._suffix_prefix_len(self._buffer, self._end_marker)
                emit = self._buffer[:-keep] if keep else self._buffer
                if emit:
                    self.attempt.write_text(emit)
                self._buffer = self._buffer[-keep:] if keep else ""
                return

            if self._state == "think":
                index = self._buffer.find(THINK_CLOSE)
                if index >= 0:
                    self._buffer = self._buffer[index + len(THINK_CLOSE) :]
                    self._state = "text"
                    continue
                if final:
                    self._buffer = ""
                    return
                keep = self._suffix_prefix_len(self._buffer, THINK_CLOSE)
                self._buffer = self._buffer[-keep:] if keep else ""
                return

            if self._state == "done":
                if self._buffer.strip():
                    raise WriteProtocolError("non-whitespace content follows the batch marker")
                return

            batch_marker = f"<<<END_AELOON_WRITE_BATCH_V1:{self.transaction_id}>>>"
            candidates = [
                (self._buffer.find(THINK_OPEN), "think"),
                (self._buffer.find(OPEN_PREFIX), "write"),
                (self._buffer.find(batch_marker), "batch"),
            ]
            found = [(index, kind) for index, kind in candidates if index >= 0]
            if not found:
                if final:
                    self._emit(self._buffer)
                    self._buffer = ""
                    return
                keep = max(
                    self._suffix_prefix_len(self._buffer, marker)
                    for marker in (THINK_OPEN, OPEN_PREFIX, batch_marker)
                )
                emit = self._buffer[:-keep] if keep else self._buffer
                if emit:
                    self._emit(emit)
                self._buffer = self._buffer[-keep:] if keep else ""
                return

            index, kind = min(found, key=lambda item: item[0])
            if index:
                self._emit(self._buffer[:index])
                self._buffer = self._buffer[index:]
            if kind == "think":
                self._buffer = self._buffer[len(THINK_OPEN) :]
                self._state = "think"
                continue
            if kind == "batch":
                if not self._saw_file:
                    raise WriteProtocolError("batch END marker appeared before any WRITE block")
                self._buffer = self._buffer[len(batch_marker) :]
                self._batch = self.attempt.finish_batch()
                self._state = "done"
                continue

            header_end = self._buffer.find(HEADER_SUFFIX, len(OPEN_PREFIX))
            if header_end < 0:
                if len(self._buffer) > self.max_header_chars:
                    raise WriteProtocolError("WRITE header exceeds the size limit")
                if final:
                    raise WriteProtocolError("WRITE header is incomplete")
                return
            raw_header = self._buffer[len(OPEN_PREFIX) : header_end]
            self._buffer = self._buffer[header_end + len(HEADER_SUFFIX) :]
            self._start_file(raw_header)

    def _start_file(self, raw_header: str) -> None:
        try:
            metadata = json.loads(raw_header)
        except json.JSONDecodeError as exc:
            raise WriteProtocolError(f"invalid WRITE header JSON: {exc.msg}") from exc
        if not isinstance(metadata, dict):
            raise WriteProtocolError("WRITE header must be a JSON object")
        tx = metadata.get("tx")
        file_id = metadata.get("id")
        path = metadata.get("path")
        mode = metadata.get("mode")
        if tx != self.transaction_id:
            raise WriteProtocolError("WRITE header transaction id does not match this call")
        if not isinstance(file_id, str) or not file_id or ":" in file_id:
            raise WriteProtocolError("WRITE id must be a non-empty string without ':'")
        if not isinstance(path, str) or not isinstance(mode, str):
            raise WriteProtocolError("WRITE path and mode must be strings")
        self.attempt.start_file(file_id=file_id, path=path, mode=mode)
        self._end_marker = f"<<<END_AELOON_WRITE_V1:{self.transaction_id}:{file_id}>>>"
        self._saw_file = True
        self._state = "body"

    def _emit(self, text: str) -> None:
        if not text:
            return
        self._visible.append(text)
        if self.on_visible is not None:
            self.on_visible(text)

    @staticmethod
    def _suffix_prefix_len(text: str, prefix: str) -> int:
        for size in range(min(len(text), len(prefix) - 1), 0, -1):
            if prefix.startswith(text[-size:]):
                return size
        return 0


__all__ = [
    "DemuxResult",
    "WriteFrameDecoder",
    "WriteProtocolError",
    "protocol_guidance",
]
