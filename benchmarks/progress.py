"""Human-readable benchmark progress on stderr."""

from __future__ import annotations

import logging
import sys
import threading
import time
from types import TracebackType
from typing import TextIO

LOGGER_NAME = "aeloon.benchmarks"
_OUTPUT_LOCK = threading.RLock()
_ACTIVE_BARS: list[ProgressBar] = []


class _DynamicStderrHandler(logging.Handler):
    """Resolve stderr at emit time so redirects and test capture keep working."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = sys.stderr
            with _OUTPUT_LOCK:
                active = [
                    bar
                    for bar in _ACTIVE_BARS
                    if bar._interactive and bar._stream is stream
                ]
                for bar in active:
                    bar._clear_locked()
                print(self.format(record), file=stream, flush=True)
                for bar in active:
                    bar._render_locked()
        except Exception:
            self.handleError(record)


class ProgressBar:
    """Thread-safe terminal progress with readable redirected output."""

    _BAR_WIDTH = 24

    def __init__(self, description: str, *, total: int, unit: str = "case") -> None:
        if total < 0:
            raise ValueError("progress total cannot be negative")
        self.description = description
        self.total = total
        self.unit = unit
        self.completed = 0
        self._detail = ""
        self._started_at = 0.0
        self._stream: TextIO | None = None
        self._interactive = False
        self._last_width = 0
        self._last_reported = -1
        self._entered = False
        self._closed = False

    def __enter__(self) -> ProgressBar:
        with _OUTPUT_LOCK:
            if self._entered:
                raise RuntimeError("progress bar cannot be entered more than once")
            self._entered = True
            self._started_at = time.monotonic()
            self._stream = sys.stderr
            isatty = getattr(self._stream, "isatty", None)
            self._interactive = bool(isatty and isatty())
            _ACTIVE_BARS.append(self)
            self._report_locked(force=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def set_detail(self, detail: str) -> None:
        """Update the currently running item without advancing the count."""

        with _OUTPUT_LOCK:
            self._require_open()
            self._detail = detail
            if self._interactive:
                self._render_locked()

    def advance(self, count: int = 1, *, detail: str | None = None) -> None:
        """Atomically mark one or more units complete."""

        if count < 0:
            raise ValueError("progress advance cannot be negative")
        with _OUTPUT_LOCK:
            self._require_open()
            next_completed = self.completed + count
            if next_completed > self.total:
                raise ValueError(
                    f"progress would exceed total: {next_completed} > {self.total}"
                )
            self.completed = next_completed
            if detail is not None:
                self._detail = detail
            self._report_locked()

    def close(self) -> None:
        with _OUTPUT_LOCK:
            if self._closed:
                return
            self._require_open()
            self._report_locked(force=self._last_reported != self.completed)
            if self._interactive:
                assert self._stream is not None
                self._stream.write("\n")
                self._stream.flush()
            _ACTIVE_BARS.remove(self)
            self._closed = True

    def _require_open(self) -> None:
        if not self._entered or self._closed:
            raise RuntimeError("progress bar is not active")

    def _report_locked(self, *, force: bool = False) -> None:
        if self._interactive:
            self._render_locked()
            return
        if not force and self._last_reported == self.completed:
            return
        assert self._stream is not None
        print(self._line(), file=self._stream, flush=True)
        self._last_reported = self.completed

    def _clear_locked(self) -> None:
        assert self._stream is not None
        self._stream.write("\r")
        self._stream.write(" " * self._last_width)
        self._stream.write("\r")
        self._stream.flush()

    def _render_locked(self) -> None:
        assert self._stream is not None
        line = self._line()
        self._stream.write("\r")
        self._stream.write(line)
        if len(line) < self._last_width:
            self._stream.write(" " * (self._last_width - len(line)))
        self._stream.flush()
        self._last_width = len(line)
        self._last_reported = self.completed

    def _line(self) -> str:
        ratio = self.completed / self.total if self.total else 1.0
        filled = round(self._BAR_WIDTH * ratio)
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        elapsed = max(0.0, time.monotonic() - self._started_at)
        eta = (
            elapsed / self.completed * (self.total - self.completed)
            if self.completed
            else None
        )
        timing = f"{_duration(elapsed)}<{_duration(eta)}"
        detail = f" | {self._detail}" if self._detail else ""
        return (
            f"{self.description} [{bar}] "
            f"{self.completed}/{self.total} {ratio:>4.0%} "
            f"{timing} {self.unit}{detail}"
        )


def configure_progress() -> None:
    """Enable INFO progress without affecting the JSON written to stdout."""

    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(handler, _DynamicStderrHandler) for handler in logger.handlers):
        handler = _DynamicStderrHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s INFO %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def info(message: str, *args: object) -> None:
    logging.getLogger(LOGGER_NAME).info(message, *args)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    rounded = max(0, round(seconds))
    minutes, seconds = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
