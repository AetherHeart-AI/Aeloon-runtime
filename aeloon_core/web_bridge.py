"""NDJSON transport between the Python runtime and the local Web UI.

The bridge deliberately contains no presentation policy.  It forwards live
``TurnEventProgress`` events unchanged inside a small transport envelope while
providing bounded, operator-safe snapshots for restoring UI state.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
import threading
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from loguru import logger
from pydantic import BaseModel, ValidationError

from aeloon_core.config import Config, load_config
from aeloon_core.operator_output import redact_sensitive_text as _redact_sensitive_text
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.turn_events import WEB_TOOL_RESULT_CHARS, TurnEventProgress

WEB_CONFIG_ENV = "AELOON_CORE_WEB_CONFIG_JSON"
WEB_SESSION_ENV = "AELOON_CORE_WEB_SESSION_ID"
WEB_LOG_LEVEL_ENV = "AELOON_CORE_WEB_LOG_LEVEL"
_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
_MAX_ERROR_CHARS = 1_000
_MAX_PENDING_LOG_EVENTS = 256
_TRACEBACK_LINE = re.compile(r"^Traceback \(most recent call last\):")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
RecordSink = Callable[[dict[str, Any]], Awaitable[None]]


class BridgeCommandError(ValueError):
    """A readable protocol error that is safe to return to the Web UI."""

    def __init__(self, message: str, *, code: str = "invalid_command") -> None:
        super().__init__(message)
        self.code = code


class NDJSONWriter:
    """Serialize bridge records to one flushed JSON object per stdout line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._lock = asyncio.Lock()

    async def __call__(self, record: dict[str, Any]) -> None:
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        async with self._lock:
            # A full stdout pipe can block.  Keep that backpressure away from the
            # agent loop while preserving record order with the writer lock.
            await asyncio.to_thread(self._write, line)

    def _write(self, line: str) -> None:
        self.stream.write(line + "\n")
        self.stream.flush()


class NDJSONStreamWriter:
    """Write NDJSON records to an asyncio byte stream with backpressure."""

    def __init__(self, stream: asyncio.StreamWriter) -> None:
        self.stream = stream
        self._lock = asyncio.Lock()

    async def __call__(self, record: dict[str, Any]) -> None:
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        async with self._lock:
            self.stream.write((line + "\n").encode("utf-8"))
            await self.stream.drain()


@dataclass(frozen=True)
class _QueuedPrompt:
    request_id: Any
    prompt: str
    session_id: str


class WebBridge:
    """Serve asynchronous commands without blocking the active agent turn."""

    def __init__(
        self,
        config: Config,
        *,
        orchestrator: Any | None = None,
        sink: RecordSink | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config.normalized()
        self._owns_orchestrator = orchestrator is None
        self.orchestrator = orchestrator or AeloonCoreOrchestrator(self.config)
        self.sink = sink or NDJSONWriter()
        self.session_id = session_id or self.orchestrator.sessions.new_session()

        self._prompt_queue: asyncio.Queue[_QueuedPrompt] = asyncio.Queue()
        self._prompt_runner: asyncio.Task[None] | None = None
        self._active_turn_task: asyncio.Task[Any] | None = None
        self._active_prompt: _QueuedPrompt | None = None
        self._closing = False
        self._shutdown_requested = False

    async def emit_ready(self) -> None:
        """Emit the initial bounded application snapshot."""

        await self.sink({"type": "ready", "payload": self.snapshot()})

    async def emit_event(self, event: str, payload: dict[str, Any]) -> None:
        """Forward one runtime event without changing its name or payload."""

        await self.sink({"type": "event", "event": event, "payload": payload})

    def snapshot(self, *, session_id: str | None = None) -> dict[str, Any]:
        """Return the current session context needed to hydrate the Web UI."""

        resolved_session_id = session_id or self.session_id
        history = self.orchestrator.sessions.history(resolved_session_id)
        return {
            "workspace": str(self.config.workspace),
            "model": self.config.agents.defaults.model_ref(),
            "session": resolved_session_id,
            # ``session_id`` is kept as a protocol alias because live runtime
            # events use that spelling for correlation.
            "session_id": resolved_session_id,
            "history": [_history_turn_view(record) for record in history],
        }

    async def dispatch(self, message: dict[str, Any]) -> None:
        """Validate and dispatch one decoded command object."""

        raw_request_id = message.get("request_id") if isinstance(message, dict) else None
        request_id = (
            raw_request_id
            if isinstance(raw_request_id, str) and raw_request_id.strip()
            else _bridge_request_id()
        )
        command: str | None = None
        try:
            if not isinstance(message, dict):
                raise BridgeCommandError("command must be a JSON object")
            if not isinstance(raw_request_id, str) or not raw_request_id.strip():
                raise BridgeCommandError("request_id is required")
            command = _command_name(message)
            payload = _command_payload(message, command=command)
            if command in {"prompt", "run_turn"}:
                await self._enqueue_prompt(request_id, payload)
                return

            result = await self._execute_control(command, payload)
            await self._respond_ok(request_id, command, result)
            if command == "shutdown":
                self._shutdown_requested = True
        except Exception as exc:
            await self._respond_error(request_id, command, exc)

    async def serve(self, input_stream: Any | None = None) -> None:
        """Read newline-delimited commands until EOF or ``shutdown``."""

        stream = input_stream or sys.stdin
        await self.emit_ready()
        try:
            while not self._shutdown_requested:
                line = await _readline(stream)
                if line == "":
                    break
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    await self._respond_error(
                        _bridge_request_id(),
                        None,
                        BridgeCommandError(
                            f"invalid JSON command at column {exc.colno}",
                            code="invalid_json",
                        ),
                    )
                    continue
                await self.dispatch(message)
        finally:
            await self.close()

    async def wait_for_idle(self) -> None:
        """Wait until every queued prompt has received a terminal response."""

        await self._prompt_queue.join()

    async def close(self) -> None:
        """Cancel bridge-owned prompt work and release the queue runner."""

        if self._closing:
            return
        self._closing = True

        active = self._active_turn_task
        if active is not None and not active.done():
            active.cancel()
        runner = self._prompt_runner
        if runner is not None and not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

        while True:
            try:
                self._prompt_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._prompt_queue.task_done()
        if self._owns_orchestrator:
            close = getattr(self.orchestrator, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def _enqueue_prompt(self, request_id: Any, payload: dict[str, Any]) -> None:
        prompt = _required_prompt(payload)
        if self._closing:
            raise BridgeCommandError("bridge is shutting down", code="bridge_closed")

        position = self._prompt_queue.qsize() + (1 if self._active_prompt else 0) + 1
        queued = _QueuedPrompt(
            request_id=request_id,
            prompt=prompt,
            # Capture the session at enqueue time so a later /new or /resume
            # cannot silently reroute an already-submitted prompt.
            session_id=self.session_id,
        )
        self._prompt_queue.put_nowait(queued)
        self._ensure_prompt_runner()
        await self.emit_event(
            "bridge.prompt.queued",
            {
                "request_id": request_id,
                "session_id": queued.session_id,
                "position": position,
                "queued": self._prompt_queue.qsize(),
            },
        )
        # Let the runner claim an immediately followed prompt before a
        # cancel_turn command is processed.
        await asyncio.sleep(0)

    def _ensure_prompt_runner(self) -> None:
        if self._prompt_runner is None or self._prompt_runner.done():
            self._prompt_runner = asyncio.create_task(self._run_prompt_queue())

    async def _run_prompt_queue(self) -> None:
        while True:
            queued = await self._prompt_queue.get()
            self._active_prompt = queued
            progress = TurnEventProgress(
                session_id=queued.session_id,
                emit=self.emit_event,
            )
            await self.emit_event(
                "bridge.prompt.started",
                {
                    "request_id": queued.request_id,
                    "session_id": queued.session_id,
                    "queued": self._prompt_queue.qsize(),
                },
            )
            self._active_turn_task = asyncio.create_task(
                self.orchestrator.run_turn(
                    queued.prompt,
                    session_id=queued.session_id,
                    on_progress=progress,
                )
            )
            try:
                result = await self._active_turn_task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                await self._respond_error(
                    queued.request_id,
                    "prompt",
                    BridgeCommandError("turn cancelled", code="turn_cancelled"),
                )
                await self.emit_event(
                    "bridge.turn.cancelled",
                    {
                        "request_id": queued.request_id,
                        "session_id": queued.session_id,
                        "turn_id": progress.turn_id,
                    },
                )
            except Exception as exc:
                await self._respond_error(queued.request_id, "prompt", exc)
            else:
                await self._respond_ok(
                    queued.request_id,
                    "prompt",
                    _turn_result_view(result),
                )
            finally:
                self._active_turn_task = None
                self._active_prompt = None
                self._prompt_queue.task_done()

    async def _execute_control(
        self,
        command: str,
        payload: dict[str, Any],
    ) -> Any:
        sessions = self.orchestrator.sessions

        if command == "refresh_snapshot":
            return self.snapshot()
        if command == "new_session":
            session_id = sessions.new_session()
            snapshot = self.snapshot(session_id=session_id)
            self.session_id = session_id
            return snapshot
        if command == "resume_session":
            session_id = _required_text(payload, "session_id", aliases=("session",))
            snapshot = self.snapshot(session_id=session_id)
            self.session_id = session_id
            return snapshot
        if command == "list_sessions":
            return [_session_summary_view(item) for item in sessions.list_sessions()]
        if command == "cancel_turn":
            active = self._active_turn_task
            active_prompt = self._active_prompt
            if active is None or active.done():
                return {"cancelled": False}
            active.cancel()
            return {
                "cancelled": True,
                "request_id": active_prompt.request_id if active_prompt else None,
            }
        if command == "shutdown":
            return {"status": "shutting_down"}
        raise BridgeCommandError(f"unknown command: {command}", code="unknown_command")

    async def _respond_ok(self, request_id: Any, command: str, result: Any) -> None:
        await self.sink(
            {
                "type": "response",
                "request_id": request_id,
                "command": command,
                "ok": True,
                "result": result,
            }
        )

    async def _respond_error(
        self,
        request_id: Any,
        command: str | None,
        exc: Exception,
    ) -> None:
        code = getattr(exc, "code", None) or _error_code(exc)
        await self.sink(
            {
                "type": "response",
                "request_id": request_id,
                "command": command,
                "ok": False,
                "error": {
                    "code": code,
                    "message": _safe_error_message(exc),
                },
            }
        )


def load_web_config(environ: Mapping[str, str] | None = None) -> Config:
    """Load the launcher's serialized config, falling back to normal config loading."""

    values = os.environ if environ is None else environ
    raw = (
        os.environ.pop(WEB_CONFIG_ENV, None)
        if environ is None
        else values.get(WEB_CONFIG_ENV)
    )
    if raw is None or not raw.strip():
        return load_config()
    try:
        return Config.model_validate_json(raw).normalized()
    except (ValidationError, ValueError, json.JSONDecodeError):
        raise BridgeCommandError(
            f"invalid {WEB_CONFIG_ENV}: expected a valid Config JSON object",
            code="invalid_config",
        ) from None


async def run_bridge(
    config: Config | None = None,
    *,
    input_stream: Any | None = None,
    output_stream: TextIO | None = None,
    session_id: str | None = None,
    sink: RecordSink | None = None,
    gateway_log_level: str | None = None,
) -> None:
    """Run one bridge process with injectable streams for tests and launchers."""

    # The Web UI consumes structured ``log.entry`` events.  The default Loguru
    # stderr sink can otherwise leak raw exception tracebacks into the normal UI
    # through the subprocess stderr watcher.
    logger.remove()
    resolved_config = config or load_web_config()
    os.environ.pop(WEB_CONFIG_ENV, None)
    bridge = WebBridge(
        resolved_config,
        sink=sink or NDJSONWriter(output_stream),
        session_id=session_id or os.environ.get(WEB_SESSION_ENV) or None,
    )
    log_level = (
        load_web_log_level()
        if gateway_log_level is None
        else load_web_log_level({WEB_LOG_LEVEL_ENV: gateway_log_level})
    )
    sink_id = _install_bridge_log_sink(bridge, level=log_level)
    try:
        await bridge.serve(input_stream)
    finally:
        logger.remove(sink_id)


def main() -> None:
    """Module entry point used by the local Web server subprocess."""

    try:
        asyncio.run(run_bridge())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        message = _safe_error_message(exc)
        record = {
            "type": "response",
            "request_id": "bridge-startup",
            "command": None,
            "ok": False,
            "error": {
                "code": getattr(exc, "code", None) or _error_code(exc),
                "message": message,
            },
        }
        sys.stdout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        sys.stdout.flush()
        sys.stderr.write(f"Aeloon Web bridge failed: {message}\n")
        sys.stderr.flush()
        raise SystemExit(1) from None


def load_web_log_level(environ: Mapping[str, str] | None = None) -> str:
    """Return a validated minimum level for structured gateway log events."""

    values = os.environ if environ is None else environ
    level = values.get(WEB_LOG_LEVEL_ENV, "INFO").strip().upper()
    return level if level in _LOG_LEVELS else "INFO"


def _command_name(message: dict[str, Any]) -> str:
    message_type = message.get("type")
    raw = message.get("command")
    if raw is None and isinstance(message_type, str) and message_type != "command":
        raw = message_type
    if not isinstance(raw, str) or not raw.strip():
        raise BridgeCommandError("command name is required")
    return raw.strip().lower().replace("-", "_")


def _bridge_request_id() -> str:
    return f"bridge-{uuid.uuid4().hex[:10]}"


def _install_bridge_log_sink(bridge: WebBridge, *, level: str = "INFO") -> int:
    """Route general runtime logs into the structured verbose event stream."""

    loop = asyncio.get_running_loop()
    pending: set[asyncio.Task[Any]] = set()
    capacity = threading.BoundedSemaphore(_MAX_PENDING_LOG_EVENTS)
    dropped_lock = threading.Lock()
    dropped = 0

    def note_dropped(count: int = 1) -> None:
        nonlocal dropped
        with dropped_lock:
            dropped += count

    def take_dropped() -> int:
        nonlocal dropped
        with dropped_lock:
            count = dropped
            dropped = 0
            return count

    def finish(task: asyncio.Task[Any]) -> None:
        pending.discard(task)
        capacity.release()
        _consume_task_exception(task)

    def sink(message: Any) -> None:
        if not capacity.acquire(blocking=False):
            note_dropped()
            return

        dropped_before = take_dropped()
        try:
            record = message.record
            detail: dict[str, Any] = {
                "logger": {
                    "name": record["name"],
                    "module": record["module"],
                    "function": record["function"],
                    "line": record["line"],
                }
            }
            if dropped_before:
                detail["dropped_before"] = dropped_before
            payload = {
                "level": record["level"].name,
                "message": _redact_sensitive_text(record["message"]),
                "source": str(record["name"]),
                "ts": record["time"].isoformat(),
                "detail": detail,
            }
        except BaseException:
            note_dropped(dropped_before)
            capacity.release()
            raise

        def schedule() -> None:
            if bridge._closing:
                capacity.release()
                return
            try:
                task = loop.create_task(bridge.emit_event("log.entry", payload))
                pending.add(task)
                task.add_done_callback(finish)
            except BaseException:
                capacity.release()
                raise

        try:
            loop.call_soon_threadsafe(schedule)
        except RuntimeError:
            # The event loop can close concurrently with a log emitted from a
            # background thread during shutdown.  The reserved slot must not
            # leak even though there is nowhere left to deliver the record.
            capacity.release()

    return logger.add(sink, level=level, backtrace=False, diagnose=False)


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


def _command_payload(message: dict[str, Any], *, command: str) -> dict[str, Any]:
    raw = message.get("payload", {})
    if command in {"prompt", "run_turn"} and isinstance(raw, str):
        return {"prompt": raw}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise BridgeCommandError("command payload must be a JSON object")

    # Accept top-level arguments as a compatibility convenience while the Web UI
    # always uses the explicit payload envelope.
    payload = dict(raw)
    for key, value in message.items():
        if key not in {"type", "command", "request_id", "payload"}:
            payload.setdefault(key, value)
    return payload


def _required_text(
    payload: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    for candidate in (name, *aliases):
        value = payload.get(candidate)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BridgeCommandError(f"{name} is required")


def _required_prompt(payload: Mapping[str, Any]) -> str:
    """Return non-blank prompt text without damaging pasted whitespace."""

    for name in ("prompt", "text"):
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise BridgeCommandError("prompt is required")


async def _readline(stream: Any) -> str:
    readline = getattr(stream, "readline", None)
    if readline is None:
        raise BridgeCommandError("input stream does not provide readline")
    if inspect.iscoroutinefunction(readline):
        value = await readline()
    else:
        value = await asyncio.to_thread(readline)
    if inspect.isawaitable(value):
        value = await value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        raise BridgeCommandError("input stream readline must return text")
    return value


def _history_turn_view(record: Any) -> dict[str, Any]:
    raw = _mapping(record)
    return {
        "turn_id": raw.get("turn_id"),
        "request_id": raw.get("request_id"),
        "created_at": raw.get("created_at"),
        "user_prompt": raw.get("user_prompt"),
        "final_content": raw.get("final_content"),
        "tools_used": list(raw.get("tools_used") or []),
        "blocks": [
            _bounded_web_block(block)
            for block in raw.get("blocks", [])
            if isinstance(block, dict)
        ],
        "usage": raw.get("usage") or {},
        "duration_ms": raw.get("duration_ms"),
    }


def _bounded_web_block(block: Mapping[str, Any]) -> dict[str, Any]:
    view = dict(block)
    if view.get("type") != "tool_call" or view.get("result") is None:
        return view
    result = str(view["result"])
    limit = WEB_TOOL_RESULT_CHARS
    if len(result) <= limit:
        return view
    marker = f"\n… {len(result) - limit} characters omitted …\n"
    available = limit - len(marker)
    head = available // 2
    tail = available - head
    view["result"] = f"{result[:head]}{marker}{result[-tail:]}"
    view["result_truncated"] = True
    return view


def _session_summary_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    return {
        "session_id": raw.get("session_id"),
        "title": raw.get("title"),
        "updated_at": raw.get("updated_at"),
        "turns": raw.get("turns", 0),
    }


def _turn_result_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    return {
        "session_id": raw.get("session_id"),
        "turn_id": raw.get("turn_id"),
        "status": raw.get("status"),
        "final_content": raw.get("final_content"),
        "tools_used": list(raw.get("tools_used") or []),
        "usage": raw.get("usage") or {},
    }


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    try:
        return dict(vars(value))
    except TypeError:
        return {}


def _safe_error_message(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        raw = str(exc.args[0])
    else:
        raw = str(exc).strip() or exc.__class__.__name__
    raw = _redact_sensitive_text(raw)
    lines: list[str] = []
    for line in raw.splitlines():
        if _TRACEBACK_LINE.match(line.strip()):
            break
        if line.strip():
            lines.append(line.strip())
    message = " ".join(lines) or exc.__class__.__name__
    message = _ANSI_ESCAPE.sub("", message)
    message = "".join(
        " " if character.isspace() else ""
        if unicodedata.category(character).startswith("C")
        else character
        for character in message
    )
    message = " ".join(message.split()) or exc.__class__.__name__
    if len(message) > _MAX_ERROR_CHARS:
        return message[: _MAX_ERROR_CHARS - 1] + "…"
    return message


def _error_code(exc: BaseException) -> str:
    name = exc.__class__.__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set | frozenset):
        return list(value)
    return str(value)


if __name__ == "__main__":
    main()
