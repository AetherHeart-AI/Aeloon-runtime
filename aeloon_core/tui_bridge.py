"""NDJSON transport between the Python runtime and the OpenTUI application.

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
from aeloon_core.turn_events import TurnEventProgress
from aeloon_core.worker_ui import WorkerUiQueryService

TUI_CONFIG_ENV = "AELOON_CORE_TUI_CONFIG_JSON"
TUI_SESSION_ENV = "AELOON_CORE_TUI_SESSION_ID"
TUI_LOG_LEVEL_ENV = "AELOON_CORE_TUI_LOG_LEVEL"
_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
_MAX_ERROR_CHARS = 1_000
_MAX_PENDING_LOG_EVENTS = 256
_TRACEBACK_LINE = re.compile(r"^Traceback \(most recent call last\):")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
RecordSink = Callable[[dict[str, Any]], Awaitable[None]]


class BridgeCommandError(ValueError):
    """A readable protocol error that is safe to return to the TUI."""

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


class TUIBridge:
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
        """Return the current session context needed to hydrate the TUI."""

        resolved_session_id = session_id or self.session_id
        history = self.orchestrator.sessions.history(resolved_session_id)
        workers = self.orchestrator.worker_control.list_workers(resolved_session_id)
        return {
            "workspace": str(self.config.workspace),
            "model": self.config.agents.defaults.model,
            "session": resolved_session_id,
            # ``session_id`` is kept as a protocol alias because live runtime
            # events use that spelling for correlation.
            "session_id": resolved_session_id,
            "history": [_history_turn_view(record) for record in history],
            "workers": [_worker_summary_view(worker) for worker in workers],
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

            result = await self._execute_control(command, payload, request_id=request_id)
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
                allow_worker_tool_output=True,
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
        *,
        request_id: str,
    ) -> Any:
        sessions = self.orchestrator.sessions
        workers = self.orchestrator.worker_control

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
        if command == "list_workers":
            session_id = _optional_text(payload, "session_id") or self.session_id
            return [
                _worker_summary_view(item)
                for item in workers.list_workers(session_id)
            ]
        if command == "inspect_worker":
            worker_id = _required_text(payload, "worker_id")
            return _inspect_worker_detail(workers, worker_id)
        if command == "discover_worker_types":
            return [
                _worker_type_view(item)
                for item in workers.discover_worker_types()
            ]
        if command == "spawn_worker":
            session_id = _optional_text(payload, "session_id") or self.session_id
            worker_type_id = _required_text(
                payload,
                "worker_type_id",
                aliases=("worker_type",),
            )
            objective = _required_text(payload, "objective")
            idempotency_key = _optional_text(payload, "idempotency_key")
            result = await workers.spawn_worker(
                base_session_id=session_id,
                worker_type_id=worker_type_id,
                objective=objective,
                idempotency_key=idempotency_key or f"tui:{uuid.uuid4().hex}",
                base_turn_id=_optional_text(payload, "base_turn_id"),
                progress=TurnEventProgress(
                    session_id=session_id,
                    emit=self.emit_event,
                    allow_worker_tool_output=True,
                ),
            )
            return _spawn_worker_view(result)
        if command == "cancel_worker":
            run_id = _required_text(payload, "run_id")
            return _worker_run_view(await workers.cancel_worker(run_id))
        if command == "resume_worker":
            run_id = _required_text(payload, "run_id")
            result = await workers.resume_worker(
                run_id,
                response=_required_text(payload, "response"),
                idempotency_key=f"tui:resume:{request_id}",
                base_session_id=self.session_id,
                progress=TurnEventProgress(
                    session_id=self.session_id,
                    emit=self.emit_event,
                    allow_worker_tool_output=True,
                ),
            )
            return _resume_worker_view(result)
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


# Backwards-compatible spelling for callers that prefer title-case initialism.
TuiBridge = TUIBridge


def load_tui_config(environ: Mapping[str, str] | None = None) -> Config:
    """Load the launcher's serialized config, falling back to normal config loading."""

    values = os.environ if environ is None else environ
    raw = (
        os.environ.pop(TUI_CONFIG_ENV, None)
        if environ is None
        else values.get(TUI_CONFIG_ENV)
    )
    if raw is None or not raw.strip():
        return load_config()
    try:
        return Config.model_validate_json(raw).normalized()
    except (ValidationError, ValueError, json.JSONDecodeError):
        raise BridgeCommandError(
            f"invalid {TUI_CONFIG_ENV}: expected a valid Config JSON object",
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

    # The TUI consumes structured ``log.entry`` events.  The default Loguru
    # stderr sink can otherwise leak raw exception tracebacks into the normal UI
    # through the subprocess stderr watcher.
    logger.remove()
    resolved_config = config or load_tui_config()
    os.environ.pop(TUI_CONFIG_ENV, None)
    bridge = TUIBridge(
        resolved_config,
        sink=sink or NDJSONWriter(output_stream),
        session_id=session_id or os.environ.get(TUI_SESSION_ENV) or None,
    )
    log_level = (
        load_tui_log_level()
        if gateway_log_level is None
        else load_tui_log_level({TUI_LOG_LEVEL_ENV: gateway_log_level})
    )
    sink_id = _install_bridge_log_sink(bridge, level=log_level)
    try:
        await bridge.serve(input_stream)
    finally:
        logger.remove(sink_id)


def main() -> None:
    """Module entry point used by the OpenTUI subprocess."""

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
        sys.stderr.write(f"Aeloon TUI bridge failed: {message}\n")
        sys.stderr.flush()
        raise SystemExit(1) from None


def load_tui_log_level(environ: Mapping[str, str] | None = None) -> str:
    """Return a validated minimum level for structured gateway log events."""

    values = os.environ if environ is None else environ
    level = values.get(TUI_LOG_LEVEL_ENV, "INFO").strip().upper()
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


def _install_bridge_log_sink(bridge: TUIBridge, *, level: str = "INFO") -> int:
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
                "source": "loguru",
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

    # Accept top-level arguments as a compatibility convenience while the TUI
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


def _optional_text(payload: Mapping[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BridgeCommandError(f"{name} must be a string")
    return value.strip() or None


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
        "created_at": raw.get("created_at"),
        "user_prompt": raw.get("user_prompt"),
        "final_content": raw.get("final_content"),
        "tools_used": list(raw.get("tools_used") or []),
        "usage": raw.get("usage") or {},
    }


def _session_summary_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    return {
        "session_id": raw.get("session_id"),
        "title": raw.get("title"),
        "updated_at": raw.get("updated_at"),
        "turns": raw.get("turns", 0),
    }


def _worker_summary_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    worker_id = str(raw.get("worker_id") or "")
    snapshot = _mapping(raw.get("snapshot"))
    worker_type_id = str(snapshot.get("id") or raw.get("worker_type_id") or "worker")
    latest = raw.get("latest_run")
    latest_view = _worker_run_view(latest) if isinstance(latest, dict) else None
    return {
        "worker_id": worker_id,
        "label": f"{worker_type_id}#{worker_id[:4]}",
        "worker_type_id": worker_type_id,
        "definition": _worker_definition_view(snapshot),
        "status": raw.get("status"),
        "created_at": raw.get("created_at"),
        "reusable": bool(raw.get("reusable", False)),
        "recommended_action": raw.get("recommended_action"),
        "run_count": raw.get("run_count", 0),
        "latest_run": latest_view,
    }


def _worker_detail_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    worker_id = str(raw.get("worker_id") or "")
    snapshot = _mapping(raw.get("snapshot"))
    worker_type_id = str(snapshot.get("id") or raw.get("worker_type_id") or "worker")
    runs = [
        _worker_run_view(run)
        for run in raw.get("runs", [])
        if isinstance(run, dict)
    ]
    return {
        "worker_id": worker_id,
        "label": f"{worker_type_id}#{worker_id[:4]}",
        "worker_type_id": worker_type_id,
        "definition": _worker_definition_view(snapshot),
        "status": raw.get("status"),
        "created_at": raw.get("created_at"),
        "runs": runs,
        # Worker progress is live and intentionally not persisted as a private
        # raw transcript.  The UI can state this limitation instead of implying
        # that a reconstructed timeline is complete.
        "timeline": [],
        "timeline_available": False,
    }


def _inspect_worker_detail(control: Any, worker_id: str) -> dict[str, Any]:
    """Build a TUI-only safe detail view without exposing Worker transcripts."""

    manager = getattr(control, "manager", None)
    journal = getattr(manager, "ui_journal", None)
    if manager is not None and journal is not None:
        return WorkerUiQueryService(manager=manager, journal=journal).inspect_worker(
            worker_id
        )

    detail = _worker_detail_view(control.inspect_worker(worker_id))
    if manager is None:
        return detail
    worker, runs = manager.inspect_worker(worker_id)
    detail["created_at"] = getattr(worker, "created_at", None)
    detail["runs"] = [_worker_record_view(run) for run in runs]
    return detail


def _worker_record_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    context = _mapping(raw.get("context"))
    result_envelope = _mapping(raw.get("result"))
    report = _mapping(result_envelope.get("report"))
    status = _enum_value(raw.get("status"))
    summary = report.get("summary")
    result = {
        "run_id": raw.get("run_id"),
        "worker_id": raw.get("worker_id"),
        "run_sequence": raw.get("run_sequence"),
        "status": status,
        "cancel_requested": bool(raw.get("cancel_requested_at")),
        "objective": context.get("objective"),
        "source_run_id": raw.get("source_run_id"),
        "created_at": raw.get("created_at"),
        "summary": summary,
        "duration_ms": result_envelope.get("duration_ms"),
        "tool_outcome": result_envelope.get("tool_outcome"),
        "usage": result_envelope.get("usage") or {},
    }
    waiting_request = _mapping(raw.get("waiting_request"))
    question = waiting_request.get("question") or _first_unresolved(report)
    if question:
        result["waiting_question"] = question
    if status == "failed" and summary:
        result["error_summary"] = summary
    return result


def _worker_run_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    context = _mapping(raw.get("context"))
    result_envelope = _mapping(raw.get("result"))
    report = _mapping(raw.get("report")) or _mapping(result_envelope.get("report"))
    status = _enum_value(raw.get("status"))
    summary = raw.get("summary") or report.get("summary")
    result = {
        "run_id": raw.get("run_id"),
        "worker_id": raw.get("worker_id"),
        "run_sequence": raw.get("run_sequence"),
        "created_at": raw.get("created_at"),
        "status": status,
        "cancel_requested": bool(raw.get("cancel_requested", False)),
        "action": raw.get("action"),
        "objective": (
            raw.get("objective")
            or raw.get("objective_preview")
            or context.get("objective")
        ),
        "source_run_id": raw.get("source_run_id"),
        "summary": summary,
        "duration_ms": raw.get("duration_ms") or result_envelope.get("duration_ms"),
        "tool_outcome": raw.get("tool_outcome") or result_envelope.get("tool_outcome"),
        "usage": raw.get("usage") or result_envelope.get("usage") or {},
    }
    waiting_request = _mapping(raw.get("waiting_request"))
    question = raw.get("waiting_question") or waiting_request.get("question")
    question = question or _first_unresolved(report)
    if question:
        result["waiting_question"] = question
    if status == "failed" and summary:
        result["error_summary"] = summary
    return result


def _resume_worker_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    run = raw.get("run") if isinstance(raw.get("run"), dict) else raw
    return {
        **_worker_run_view(run),
        "action": raw.get("action"),
        "source_status": _enum_value(raw.get("source_status")),
        "created": bool(raw.get("created", False)),
    }


def _worker_type_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    return {
        "id": raw.get("id"),
        "description": raw.get("description"),
        "source": _enum_value(raw.get("source")),
        "digest": raw.get("digest"),
    }


def _spawn_worker_view(item: Any) -> dict[str, Any]:
    raw = _mapping(item)
    snapshot = _mapping(raw.get("snapshot"))
    worker_id = raw.get("worker_id")
    run_id = raw.get("run_id")
    run_sequence = raw.get("run_sequence")
    created_at = raw.get("created_at")
    return {
        "worker_id": worker_id,
        "created_at": created_at,
        "created": bool(raw.get("created", False)),
        "worker_type_id": snapshot.get("id") or raw.get("worker_type_id"),
        "definition": _worker_definition_view(snapshot),
        "latest_run": {
            "worker_id": worker_id,
            "run_id": run_id,
            "run_sequence": run_sequence,
            "created_at": created_at,
            "status": _enum_value(raw.get("status")) or "queued",
        },
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


def _worker_definition_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": value.get("id"),
        "description": value.get("description"),
        "source": _enum_value(value.get("source")),
        "digest": value.get("digest"),
    }


def _first_unresolved(report: Mapping[str, Any]) -> Any:
    unresolved = report.get("unresolved")
    if isinstance(unresolved, list) and unresolved:
        return unresolved[0]
    return None


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


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
