"""Minimal aeloon-rpc v4 Runtime gateway.

The Runtime owns the v4 framing, handshake, workspace authorization, lifecycle,
and event replay guarantees needed by a desktop client.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import errno
import fcntl
import hashlib
import json
import mimetypes
import os
import platform
import re
import shutil
import socket
import stat
import struct
import sys
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from aeloon_runtime.blocking import run_blocking
from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.core.types import RunError
from aeloon_runtime.git_workspace import branch_create as git_branch_create
from aeloon_runtime.git_workspace import branches as git_branches
from aeloon_runtime.git_workspace import changes as git_changes
from aeloon_runtime.git_workspace import commit as git_commit
from aeloon_runtime.git_workspace import create_worktree as git_create_worktree
from aeloon_runtime.git_workspace import diff as git_diff
from aeloon_runtime.git_workspace import github_status as git_github_status
from aeloon_runtime.git_workspace import pr_create as git_pr_create
from aeloon_runtime.git_workspace import push as git_push
from aeloon_runtime.git_workspace import remove_worktree as git_remove_worktree
from aeloon_runtime.git_workspace import stage as git_stage
from aeloon_runtime.git_workspace import status as git_status
from aeloon_runtime.git_workspace import unstage as git_unstage
from aeloon_runtime.git_workspace import worktree_status as git_worktree_status
from aeloon_runtime.pairing import PairingState
from aeloon_runtime.pty_manager import PTYManager
from aeloon_runtime.rpc.protocol import RpcError
from aeloon_runtime.runtime.service import RuntimeService
from aeloon_runtime.runtime.types import RuntimeFailure
from aeloon_runtime.runtime_log import RuntimeLog
from aeloon_runtime.store import AsyncRuntimeStore, RuntimeStore
from aeloon_runtime.trace import TraceRecorder
from aeloon_runtime.version import RUNTIME_VERSION, runtime_commit

PROTOCOL = "aeloon-rpc"
RUNTIME_COMMIT = runtime_commit()
MAX_CLIENTS = 16
MAX_CLIENTS_PER_DEVICE = 4
MAX_PENDING_AUTH = 16
AUTH_HANDSHAKE_TIMEOUT_S = 10.0
EVENT_LIMIT = 10_000
EVENT_QUEUE_LIMIT = 1_000
_MANIFEST = json.loads(
    (Path(__file__).with_name("rpc") / "aeloon-rpc-v4.manifest.json").read_text(encoding="utf-8")
)
PROTOCOL_VERSION = "4.0.0"
SUPPORTED_PROTOCOLS: tuple[str, ...] = (PROTOCOL_VERSION,)
MAX_FRAME_BYTES = int(_MANIFEST["transport"]["max_frame_bytes"])
FILE_BYTES = int(_MANIFEST["transport"].get("file_bytes", 25 * 1024 * 1024))
IMAGE_BYTES = int(_MANIFEST["transport"].get("image_bytes", 10 * 1024 * 1024))
_ALL_METHODS = {
    **_MANIFEST["methods"],
    **_MANIFEST.get("plugin_methods", {}),
}
_METHOD_SCHEMAS = {
    name: Draft202012Validator(
        {
            "$schema": _MANIFEST["json_schema_draft"],
            "$defs": _MANIFEST["$defs"],
            "$ref": spec["params"]["$ref"],
        }
    )
    for name, spec in _ALL_METHODS.items()
}
_RESULT_SCHEMAS = {
    name: Draft202012Validator(
        {
            "$schema": _MANIFEST["json_schema_draft"],
            "$defs": _MANIFEST["$defs"],
            "$ref": spec["result"]["$ref"],
        }
    )
    for name, spec in _ALL_METHODS.items()
}
_EVENT_SCHEMAS = {
    name: Draft202012Validator(
        {
            "$schema": _MANIFEST["json_schema_draft"],
            "$defs": _MANIFEST["$defs"],
            "$ref": spec["payload"]["$ref"],
        }
    )
    for name, spec in _MANIFEST["events"].items()
}


def pack_frame(value: Any) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise RpcError("payload_too_large", "RPC frame exceeds 40 MiB")
    return struct.pack("!I", len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> Any:
    try:
        header = await reader.readexactly(4)
        (length,) = struct.unpack("!I", header)
        if length > MAX_FRAME_BYTES:
            raise RpcError("payload_too_large", "RPC frame exceeds 40 MiB")
        return json.loads((await reader.readexactly(length)).decode("utf-8"))
    except asyncio.IncompleteReadError:
        raise EOFError from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RpcError("invalid_argument", "RPC frame contains invalid JSON") from None


class RuntimeServer:
    def __init__(
        self,
        runtime: RuntimeService,
        socket_path: Path,
        workspace_roots: tuple[Path, ...],
        data_dir: Path,
        trace_recorder: TraceRecorder | None = None,
        preacquired_lock_fd: int | None = None,
        listen: tuple[str, int] | None = None,
        tls_context: Any | None = None,
        tls_certificate: Path | None = None,
        tls_key: Path | None = None,
        runtime_label: str | None = None,
        advertise_url: str | None = None,
    ) -> None:
        self.runtime = runtime
        # A WebSocket listener is additive: the Unix socket always exists, so a
        # single Runtime can serve both and the two transports can be compared
        # against identical state.
        self.listen = listen
        self.tls_context = tls_context
        self.tls_certificate = tls_certificate
        self.tls_key = tls_key
        self.advertise_url = advertise_url
        self.socket_path = socket_path.expanduser().resolve(strict=False)
        # Empty means no authorized roots. The CLI default of cwd is applied
        # in ``serve`` when the caller omits ``--workspace-root``; the
        # desktop launcher must pass an explicit list (possibly empty) so a
        # packaged Electron cwd of ``/`` cannot become a sandbox.
        self.workspace_roots = tuple(
            root.expanduser().resolve(strict=False) for root in workspace_roots
        )
        self.data_dir = data_dir.expanduser().resolve(strict=False)
        self.runtime_id = _load_runtime_identity(self.data_dir)
        self.runtime_label = (runtime_label or socket.gethostname()).strip() or socket.gethostname()
        self.trace = trace_recorder
        self.server: asyncio.AbstractServer | None = None
        self.connections: set[RuntimeConnection] = set()
        self.pending_connections: set[RuntimeConnection] = set()
        self.started_at = time.monotonic()
        self.server_instance_id = str(uuid.uuid4())
        self.current_seq = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=EVENT_LIMIT)
        self.stop_event = asyncio.Event()
        self._remove_listener = runtime.add_event_listener(self._on_runtime_event)
        # ``serve`` acquires this before composing RuntimeService so a
        # competing process cannot run repository/session cleanup while the
        # first owner is still being assembled. Direct unit composition keeps
        # the historical lazy acquisition in ``run``.
        self._lock_fd: int | None = preacquired_lock_fd
        self.log: RuntimeLog | None = None
        self.store = RuntimeStore(self.data_dir / "runtime.sqlite")
        self.async_store = AsyncRuntimeStore(self.store)
        self.pty = PTYManager(self._on_runtime_event)
        self.pairing = PairingState(self.data_dir)

    async def _store_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Run one projection operation on the dedicated SQLite worker."""

        return await getattr(self.async_store, method)(*args, **kwargs)

    async def _close_store(self) -> None:
        await self.async_store.close()
        self.store.close()

    async def run(self) -> None:
        try:
            if self._lock_fd is None:
                self._acquire_lock()
        except Exception:
            # The composition root opens the projection before `run()` so
            # direct unit callers can inspect it. Close that connection when
            # a concurrent Runtime loses the single-instance lock.
            await self._close_store()
            raise
        try:
            self.log = RuntimeLog(self.data_dir)
        except Exception:
            await self._close_store()
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise
        self.log.write(
            "started",
            pid=os.getpid(),
            server_instance_id=self.server_instance_id,
            socket=str(self.socket_path),
        )
        metadata_path = self.data_dir / "runtime.pid.json"
        try:
            # A crash between writing an attachment blob and committing its row
            # must not accumulate unreachable private files across Runtime starts.
            await self._store_call(
                "cleanup_orphan_attachments", self.data_dir / "attachments"
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_at": time.time(),
                        "server_instance_id": self.server_instance_id,
                        "socket": str(self.socket_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metadata_path.chmod(0o600)
            self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with contextlib.suppress(PermissionError):
                self.socket_path.parent.chmod(0o700)
            await self._remove_stale_socket()
            # Some container bind mounts (notably Docker Desktop's virtiofs)
            # reject chmod(2) on a Unix socket with EINVAL.  Create the
            # socket under a restrictive umask first, then retain the normal
            # chmod verification on filesystems that support it.
            previous_umask = os.umask(0o177)
            try:
                self.server = await asyncio.start_unix_server(
                    self._accept, path=str(self.socket_path)
                )
            finally:
                os.umask(previous_umask)
            try:
                self.socket_path.chmod(0o600)
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                    raise
            if stat.S_IMODE(self.socket_path.stat().st_mode) != 0o600:
                raise RuntimeError("Runtime socket must have mode 0600")
        except Exception:
            if self.log is not None:
                self.log.write("start_failed")
                self.log.close()
                self.log = None
            metadata_path.unlink(missing_ok=True)
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            await self._close_store()
            raise
        websocket_server = None
        try:
            if self.listen is not None:
                from aeloon_runtime.gateway_ws import require_listen_host, serve_websocket
                from aeloon_runtime.pairing import is_loopback_host

                host, port = self.listen
                if host in {"0.0.0.0", "::", "[::]"} and not self.advertise_url:
                    raise ValueError(
                        "Wildcard WebSocket listeners require --advertise-url wss://host:port"
                    )
                # Fail closed on a routable bind with zero paired devices *before*
                # opening the listener. TLS alone is not enough: anyone on the
                # network could otherwise walk enrollment.
                if not is_loopback_host(host) and not self.pairing.store.has_devices():
                    require_listen_host(host, tls_ready=False, paired=False)
                self.tls_context = self.pairing.prepare_listen(
                    host,
                    port,
                    certificate=self.tls_certificate,
                    key=self.tls_key,
                    advertise_url=self.advertise_url,
                )
                require_listen_host(
                    host,
                    tls_ready=self.tls_context is not None,
                    paired=self.pairing.store.has_devices(),
                )
                # A zero-device loopback Runtime needs a bootstrap path. Once
                # a device exists, or for any routable listener, enrollment is
                # explicit through devices.enroll and must not mint a new code
                # on every process restart.
                if is_loopback_host(host) and not self.pairing.store.has_devices():
                    _code, _expires_at, url = self.pairing.issue_enrollment()
                    print(url, flush=True)
                websocket_server = await serve_websocket(
                    self, host=host, port=port, tls=self.tls_context
                )
                if self.log is not None:
                    self.log.write(
                        "listening", host=host, port=port, tls=self.tls_context is not None
                    )
            await self.stop_event.wait()
        finally:
            if websocket_server is not None:
                websocket_server.close()
                with contextlib.suppress(Exception):
                    await websocket_server.wait_closed()
            if self.log is not None:
                self.log.write("stopped", server_instance_id=self.server_instance_id)
                self.log.close()
                self.log = None
            self.server.close()
            await self.server.wait_closed()
            for connection in tuple(self.connections):
                await connection.close()
            await self.pty.close_all()
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            metadata_path.unlink(missing_ok=True)
            self._remove_listener()
            await self.runtime.close()
            await self._close_store()
            if self.trace is not None:
                self.trace.close()
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None

    async def _remove_stale_socket(self) -> None:
        if not self.socket_path.exists():
            return
        if not stat.S_ISSOCK(self.socket_path.stat().st_mode):
            raise RuntimeError(f"Runtime socket path is not a Unix socket: {self.socket_path}")
        try:
            _reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            self.socket_path.unlink(missing_ok=True)
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        raise RuntimeError(f"Runtime socket is already in use: {self.socket_path}")

    def _acquire_lock(self) -> None:
        self._lock_fd = _open_runtime_lock(self.data_dir)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self.connections) >= MAX_CLIENTS:
            writer.close()
            await writer.wait_closed()
            return
        connection = RuntimeConnection(self, reader, writer, requires_auth=False)
        self.connections.add(connection)
        try:
            await connection.run()
        finally:
            self.connections.discard(connection)

    def activate_connection(self, connection: RuntimeConnection) -> None:
        if len(self.connections) >= MAX_CLIENTS:
            raise RpcError("busy", "Runtime connection limit reached")
        self.pending_connections.discard(connection)
        self.connections.add(connection)

    async def _on_runtime_event(self, event: Any) -> None:
        raw = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        # Runtime producers still expose a few legacy DTOs with camelCase
        # fields.  Normalize the complete public envelope here so callers do
        # not need per-event exceptions (payloads and envelope metadata share
        # the same snake_case contract).
        raw = _snake_case_value(raw)
        self.current_seq += 1
        if "session_id" in raw:
            raw["thread_id"] = raw.pop("session_id")
        raw["name"] = {
            "session.compacted": "thread.compacted",
            "session.navigated": "thread.navigated",
            "session.renamed": "thread.renamed",
            "cloud.account.updated": "plugin.cloud.account_updated",
        }.get(raw.get("name"), raw.get("name"))
        if raw.get("name") == "log.entry":
            # The producer is shared with the frozen v2 surface, which has to go
            # on emitting `core_*`. Rename on the way out so v4 never exposes
            # the old identity.
            payload = raw.get("payload")
            if isinstance(payload, dict):
                for legacy, current in (
                    ("core_version", "runtime_version"),
                    ("core_commit", "runtime_commit"),
                ):
                    if legacy in payload:
                        payload[current] = payload.pop(legacy)
        event_name = raw.get("name")
        validator = _EVENT_SCHEMAS.get(event_name) if isinstance(event_name, str) else None
        if validator is not None:
            try:
                validator.validate(raw.get("payload", {}))
            except ValidationError as exc:
                # Internal Runtime producers are the only event source. A
                # malformed event is therefore an implementation failure, not
                # a client error; keep it out of replay and surface a stable
                # internal error to trace diagnostics.
                raise RuntimeError(
                    f"Runtime produced an invalid {event_name} event: {exc.message}"
                ) from exc
        # Runtime DTOs are not allowed to choose the public cursor.  Always
        # write the gateway's global sequence last, even if a legacy producer
        # happened to include a stale ``seq`` field in its envelope.
        public = {**raw, "seq": self.current_seq}
        self.events.append(public)
        if self.trace is not None:
            self.trace.event(public)
        thread_id = public.get("thread_id")
        if (
            isinstance(thread_id, str)
            and await self._store_call("get_thread", thread_id) is not None
        ):
            operation_id = public.get("operation_id")
            if isinstance(operation_id, str) and operation_id:
                # ``RuntimeService.turn_start`` emits ``turn.created`` before
                # it returns the operation id.  The projection row therefore
                # cannot be created by the request handler yet.  Materialize
                # that durable turn from the event itself so a fast operation
                # (or a Runtime restart) cannot lose the first input/event.
                if public.get("name") == "turn.created":
                    event_payload = public.get("payload")
                    turn_value = (
                        event_payload.get("turn")
                        if isinstance(event_payload, Mapping)
                        else None
                    )
                    if isinstance(turn_value, Mapping):
                        status = str(turn_value.get("status") or "queued")
                        await self._store_call(
                            "ensure_turn",
                            thread_id,
                            operation_id,
                            status=status,
                        )
                        user_text = turn_value.get("user_text")
                        attachments = turn_value.get("attachments")
                        if isinstance(user_text, str) and isinstance(attachments, list):
                            await self._store_call(
                                "update_turn_input",
                                operation_id,
                                user_text=user_text,
                                attachments=[
                                    item for item in attachments if isinstance(item, dict)
                                ],
                            )
                    else:
                        await self._store_call("ensure_turn", thread_id, operation_id)
                else:
                    await self._store_call("ensure_turn", thread_id, operation_id)
            await self._store_call("project_event", thread_id, public)
        await asyncio.gather(
            *(connection.send_event(public) for connection in tuple(self.connections)),
            return_exceptions=True,
        )

    async def dispatch(
        self,
        method: str,
        params: Mapping[str, Any],
        connection: RuntimeConnection | None = None,
    ) -> Any:
        validator = _METHOD_SCHEMAS.get(method)
        if validator is not None:
            try:
                validator.validate(dict(params))
            except ValidationError as exc:
                path = ".".join(str(item) for item in exc.absolute_path)
                detail = f" ({path})" if path else ""
                raise RpcError(
                    "invalid_argument", f"Invalid parameters{detail}: {exc.message}"
                ) from None
        if method == "system.handshake":
            return await self._handshake(params, connection)
        if method == "system.health":
            return {
                "ok": True,
                "uptime_s": max(0, int(time.monotonic() - self.started_at)),
                "active_operations": self.runtime.active_operation_count,
            }
        if method == "system.capabilities":
            core_methods = [
                name for name in _MANIFEST["methods"]
                if not name.startswith(("plugins.", "plugin."))
            ]
            core_events = [
                name for name in _MANIFEST["events"] if not name.startswith("plugin.")
            ]
            cloud_methods = [
                name
                for name in _MANIFEST.get("plugin_methods", {})
                if name.startswith("plugin.cloud.")
            ]
            cloud_events = [
                name
                for name in _MANIFEST["events"]
                if name.startswith("plugin.cloud.")
            ]
            return {
                "capabilities": [
                    {
                        "id": "core",
                        "version": RUNTIME_VERSION,
                        "kind": "builtin",
                        "enabled": True,
                        "settings_schema": None,
                        "methods": core_methods,
                        "events": core_events,
                        "ui": {"group": "core", "order": 0},
                    },
                    {
                        "id": "cloud",
                        "version": RUNTIME_VERSION,
                        "kind": "builtin",
                        "enabled": True,
                        "settings_schema": None,
                        "methods": cloud_methods,
                        "events": cloud_events,
                        "ui": {"group": "providers", "order": 10},
                    },
                ]
            }
        if method == "system.snapshot":
            return await self._system_snapshot()
        if method in {"system.shutdown", "system.uninstall_inspect", "system.uninstall_prepare"}:
            if connection is not None and connection.requires_auth:
                raise RpcError("forbidden", "Remote WSS connections cannot control Runtime lifecycle")
        if method == "system.uninstall_inspect":
            return await self._uninstall_inspect()
        if method == "system.uninstall_prepare":
            inspection = await self._uninstall_inspect()
            if int(inspection["active_operations"]) > 0:
                raise RpcError("busy", "Runtime has active operations")
            dirty = [item for item in inspection["worktrees"] if item.get("dirty")]
            if dirty:
                raise RpcError("busy", "Commit or move changes from managed worktrees first")
            removed = [
                {
                    "project_path": str(item["project_path"]),
                    "workspace": str(item["workspace"]),
                }
                for item in inspection["worktrees"]
            ]
            response = {"prepared": True, "removed_worktrees": removed}
            self.validate_result("system.uninstall_prepare", response)
            deleted: list[dict[str, str]] = []
            for item in inspection["worktrees"]:
                record = {
                    "project_path": str(item["project_path"]),
                    "workspace": str(item["workspace"]),
                }
                if not item.get("exists"):
                    deleted.append(record)
                    continue
                try:
                    await run_blocking(
                        git_remove_worktree,
                        Path(str(item["project_path"])),
                        Path(str(item["workspace"])),
                    )
                    deleted.append(record)
                except (RuntimeError, OSError) as exc:
                    raise RpcError(
                        "invalid_state",
                        str(exc),
                        {"removed_worktrees": deleted},
                    ) from None
            return response
        if method == "system.shutdown":
            if self.runtime.active_operation_count > 0:
                raise RpcError("busy", "Runtime has active operations")
            await self._on_runtime_event(
                {
                    "name": "system.shutdown",
                    "payload": {"intentional": True, "reason": "requested"},
                }
            )
            asyncio.get_running_loop().call_later(0.05, self.stop_event.set)
            return {"accepted": True}
        if method == "events.subscribe":
            return self._subscribe(params)
        if method == "plugin.cloud.account_status":
            return await self.runtime.account_status(params)
        if method == "plugin.cloud.account_login":
            return await self.runtime.account_login(params)
        if method == "plugin.cloud.account_logout":
            return await self.runtime.account_logout(params)
        if method == "catalog.get":
            # The Runtime service keeps the complete resource catalog.  The
            # v4 wire result deliberately names all of those collections so
            # clients can render settings without a second legacy projection.
            catalog_params = dict(params)
            # Runtime internals still call the 1:1 session identity
            # ``session_id``.  Keep that name out of the public v4 surface;
            # the gateway is the sole translation point.
            if "thread_id" in catalog_params:
                catalog_params["session_id"] = catalog_params.pop("thread_id")
            return await self.runtime.catalog_get(catalog_params)
        if method == "provider.list":
            return await self.runtime.provider_list(params)
        if method == "provider.refresh":
            value = await self.runtime.provider_refresh(params)
            return {"provider": value.get("provider")}
        if method == "provider.add":
            provider_params = dict(params)
            config = provider_params.pop("config", None)
            if isinstance(config, Mapping):
                provider_params.update(config)
            value = await self.runtime.provider_add(provider_params)
            return {"provider": value.get("provider")}
        if method == "provider.remove":
            value = await self.runtime.provider_remove(params)
            return {"removed": bool(value.get("removed"))}
        if method == "settings.get":
            value = await self.runtime.settings_get(params)
            revision = value.get("revision")
            settings = {key: item for key, item in value.items() if key != "revision"}
            return {"settings": settings, "revision": revision}
        if method == "settings.update":
            value = await self.runtime.settings_update(params)
            revision = value.get("revision")
            settings = {key: item for key, item in value.items() if key != "revision"}
            return {"settings": settings, "revision": revision}
        if method == "tools.search_test":
            value = await self.runtime.tools_search_test(params)
            result: dict[str, Any] = {
                "results": value.get("results", []),
                "engine": value.get("provider", ""),
            }
            if value.get("message"):
                result["error"] = value["message"]
            return result
        # Declared in the manifest but not yet enabled. "Not enabled" and "no such
        # method" are different answers: the first says the capability exists and
        # may light up, the second says it never will.
        if (
            method.startswith("plugins.")
            or method.startswith("plugin.memory.")
            or method.startswith("plugin.knowledge.")
        ):
            raise RpcError("capability_unavailable", f"Capability is not enabled: {method}")
        if method in {"workspace.roots", "workspace.list"}:
            return await self._workspace_dispatch(method, params)
        if method.startswith("devices."):
            return await self._devices_dispatch(method, params, connection)
        if method.startswith("project."):
            return await self._project_dispatch(method, params, connection)
        if method.startswith("thread."):
            return await self._thread_dispatch(method, params)
        if method.startswith("turn."):
            return await self._turn_dispatch(method, params)
        if method.startswith("fs."):
            return await self._fs_dispatch(method, params)
        if method.startswith("attachment."):
            return await self._attachment_dispatch(method, params)
        if method.startswith("artifact."):
            return await self._artifact_dispatch(method, params)
        if method.startswith("terminal."):
            return await self._terminal_dispatch(method, params)
        if method.startswith("git."):
            try:
                return await self._git_dispatch(method, params)
            except ValueError as exc:
                raise RpcError("invalid_argument", str(exc)) from None
        raise RpcError("method_not_found", f"Unknown v4 method: {method}")

    def validate_result(self, method: str, value: Any) -> None:
        validator = _RESULT_SCHEMAS.get(method)
        if validator is None:
            return
        try:
            validator.validate(value)
        except ValidationError as exc:
            path = ".".join(str(item) for item in exc.absolute_path)
            detail = f" ({path})" if path else ""
            raise RpcError(
                "internal_error", f"Runtime produced an invalid {method} result{detail}"
            ) from None

    async def _system_snapshot(self) -> dict[str, Any]:
        projects = await self._store_call("list_projects")
        threads = await self._store_call("list_threads")
        return {
            "default_workspace": str(self.workspace_roots[0]) if self.workspace_roots else "",
            "projects": projects,
            "threads": threads,
        }

    async def _workspace_dispatch(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        if method == "workspace.roots":
            return {
                "roots": [
                    {
                        "id": _root_id(root),
                        "label": root.name or str(root),
                        "path": str(root),
                        "writable": os.access(root, os.W_OK),
                    }
                    for root in self.workspace_roots
                ]
            }
        root_id = self._required_string(params, "root_id")
        relative_path = params.get("relative_path")
        if not isinstance(relative_path, str):
            raise RpcError("invalid_argument", "relative_path is required")
        root = next((item for item in self.workspace_roots if _root_id(item) == root_id), None)
        if root is None:
            raise RpcError("invalid_argument", "Unknown workspace root")
        self._validate_relative(relative_path)
        target = (root / relative_path).resolve(strict=False)
        if not _is_within(target, root) or _contains_symlink(root / relative_path):
            raise RpcError("forbidden", "Path resolves outside authorized workspace root")
        if not target.exists() or not target.is_dir():
            raise RpcError("invalid_argument", "Workspace directory does not exist")
        entries = []
        for item in sorted(target.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower())):
            entries.append({"name": item.name, "kind": "directory" if item.is_dir() else "file"})
        return {"directory": relative_path or ".", "entries": entries}

    async def _uninstall_inspect(self) -> dict[str, Any]:
        estimated = 0
        for path in self.data_dir.rglob("*"):
            if path.is_file():
                with contextlib.suppress(OSError):
                    estimated += path.stat().st_size
        worktrees: list[dict[str, Any]] = []
        for thread in await self._store_call("list_threads"):
            if thread["kind"] != "worktree":
                continue
            project = await self._store_call("get_project", thread["project_id"])
            if project is None:
                continue
            try:
                state = await run_blocking(git_worktree_status, self._thread_workspace(thread))
            except (RuntimeError, OSError) as exc:
                raise RpcError("invalid_state", str(exc)) from None
            worktrees.append(
                {
                    "thread_id": thread["id"],
                    "title": thread["title"],
                    "project_path": project["path"],
                    "workspace": thread["workspace"],
                    "branch": thread["branch"],
                    **state,
                }
            )
        return {
            "data_paths": [str(self.data_dir)],
            "estimated_bytes": estimated,
            "active_operations": self.runtime.active_operation_count,
            "worktrees": worktrees,
        }

    async def _project_dispatch(
        self,
        method: str,
        params: Mapping[str, Any],
        connection: RuntimeConnection | None = None,
    ) -> dict[str, Any]:
        if method == "project.list":
            return {"projects": await self._store_call("list_projects")}
        if method == "project.add":
            root_id = self._required_string(params, "root_id")
            relative_path = params.get("relative_path")
            if not isinstance(relative_path, str):
                raise RpcError("invalid_argument", "relative_path is required")
            root = next((item for item in self.workspace_roots if _root_id(item) == root_id), None)
            if root is None:
                raise RpcError("invalid_argument", "Unknown workspace root")
            self._validate_relative(relative_path)
            path = self._allowed_path(str(root / relative_path), must_exist=True, directory=True)
            name = path.name or str(path)
            is_git = (path / ".git").exists()
            try:
                project = await self._store_call(
                    "add_project", name=name, path=str(path), is_git=is_git
                )
            except Exception as exc:
                if "UNIQUE constraint failed" not in str(exc):
                    raise
                project = next(
                    item
                    for item in await self._store_call("list_projects")
                    if item["path"] == str(path)
                )
            return {
                "project": project,
                "projects": await self._store_call("list_projects"),
            }
        project_id = self._required_string(params, "project_id")
        project = await self._store_call("get_project", project_id)
        if project is None:
            raise RpcError("invalid_argument", f"Project {project_id} not found")
        if method == "project.remove":
            # PTYs belong to the project through their thread. Close them before
            # the FK cascade removes the thread rows, otherwise the shell would
            # outlive its Runtime ownership and continue emitting orphan events.
            threads = await self._store_call("list_threads", project_id)
            for thread in threads:
                if thread["kind"] != "worktree":
                    continue
                try:
                    inspection = await run_blocking(
                        git_worktree_status, self._thread_workspace(thread)
                    )
                    if inspection["dirty"]:
                        raise RpcError("busy", "Refusing to remove a dirty managed worktree")
                except RpcError:
                    raise
                except (RuntimeError, OSError) as exc:
                    raise RpcError("invalid_state", str(exc)) from None
            for thread in threads:
                if self.runtime.session_active_operation_count(thread["id"]) > 0:
                    raise RpcError("busy", "Cannot remove a project with active operations")
            for thread in threads:
                await self.pty.close(thread["id"])
                if thread["kind"] == "worktree":
                    try:
                        await run_blocking(
                            git_remove_worktree,
                            Path(project["path"]),
                            self._thread_workspace(thread),
                        )
                    except (RuntimeError, OSError) as exc:
                        raise RpcError("invalid_state", str(exc)) from None
            # The SQLite FK cascade only removes the projection.  Remove the
            # corresponding Runtime session logs as well, otherwise a
            # project deletion leaves sessions that can still be discovered by
            # the provider-neutral Runtime service after the UI projection is
            # gone.
            for thread in threads:
                try:
                    await self.runtime.session_delete({"session_id": thread["id"]})
                except (RuntimeFailure, RunError) as exc:
                    if exc.code == "not_found":
                        continue
                    raise RpcError(exc.code, str(exc)) from None
            return {"removed": await self._store_call("remove_project", project_id)}
        if method == "project.refresh":
            refreshed = await self._store_call("refresh_project", project_id)
            if refreshed is None:
                raise RpcError("invalid_argument", f"Project {project_id} not found")
            return {
                "project": refreshed,
                "projects": await self._store_call("list_projects"),
            }
        raise RpcError("method_not_found", f"Unknown project method: {method}")

    async def _thread_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "thread.list":
            project_id = str(params["project_id"]) if params.get("project_id") else None
            filter_value = str(params.get("filter") or "active")
            archived = (
                True if filter_value == "archived" else False if filter_value == "active" else None
            )
            return {
                "threads": await self._store_call(
                    "list_threads", project_id, archived=archived
                )
            }
        if method == "thread.create":
            workspace = params.get("workspace")
            project_id = params.get("project_id")
            project_created = False
            project_persisted = bool(project_id)
            if project_id:
                project = await self._store_call("get_project", str(project_id))
                if project is None:
                    raise RpcError("invalid_argument", "project_id does not exist")
                workspace_path = self._allowed_path(
                    project["path"], must_exist=True, directory=True
                )
            else:
                workspace_path = self._allowed_path(workspace, must_exist=True, directory=True)
                project = next(
                    (
                        item
                        for item in await self._store_call("list_projects")
                        if item["path"] == str(workspace_path)
                    ),
                    None,
                )
                if project is None:
                    project = {
                        "id": str(uuid.uuid4()),
                        "name": workspace_path.name or str(workspace_path),
                        "path": str(workspace_path),
                        "is_git": (workspace_path / ".git").exists(),
                    }
                    project_created = True
                else:
                    project_persisted = True
            kind = str(params.get("kind") or "standard")
            if kind not in {"standard", "worktree"}:
                raise RpcError("invalid_argument", "kind must be standard or worktree")
            thread_id = str(uuid.uuid4())
            branch_value = str(params["branch"]) if params.get("branch") else None
            worktree_created = False
            session = None
            try:
                if kind == "worktree":
                    try:
                        workspace_path, branch_value = await run_blocking(
                            git_create_worktree,
                            Path(project["path"]),
                            self.data_dir / "worktrees",
                            str(project["id"]),
                            thread_id,
                            str(params.get("title") or "New thread"),
                            branch_value,
                        )
                        worktree_created = True
                    except (RuntimeError, ValueError) as exc:
                        raise RpcError("invalid_state", str(exc)) from None
                session = await self.runtime.create_session(
                    workspace=str(workspace_path),
                    session_id=thread_id,
                    title=str(params.get("title") or "").strip() or None,
                )
                if project_created:
                    project, thread = await self._store_call(
                        "create_project_and_thread",
                        project_name=str(project["name"]),
                        project_path=str(project["path"]),
                        project_is_git=bool(project["is_git"]),
                        project_id=str(project["id"]),
                        thread_title=session.title or "New thread",
                        thread_kind=kind,
                        thread_workspace=str(workspace_path),
                        thread_branch=branch_value,
                        thread_model_id=str(params.get("model_id") or ""),
                        thread_id=session.session_id,
                    )
                    project_persisted = True
                else:
                    thread = await self._store_call(
                        "create_thread",
                        project_id=project["id"],
                        title=session.title or "New thread",
                        kind=kind,
                        workspace=str(workspace_path),
                        branch=branch_value,
                        model_id=str(params.get("model_id") or ""),
                        thread_id=session.session_id,
                    )
            except Exception:
                if session is not None:
                    with contextlib.suppress(Exception):
                        await self.runtime.session_delete({"session_id": session.session_id})
                if worktree_created:
                    with contextlib.suppress(Exception):
                        await run_blocking(
                            git_remove_worktree, Path(project["path"]), workspace_path
                        )
                if project_created and project_persisted:
                    with contextlib.suppress(Exception):
                        await self._store_call("remove_project", project["id"])
                raise
            return {
                "thread": thread,
                "threads": await self._store_call("list_threads", project["id"]),
            }
        if method == "thread.reorder":
            project_id = self._required_string(params, "project_id")
            thread_ids = params.get("thread_ids")
            if not isinstance(thread_ids, list) or any(
                not isinstance(item, str) for item in thread_ids
            ):
                raise RpcError("invalid_argument", "thread_ids must be an array of strings")
            try:
                threads = await self._store_call("reorder_threads", project_id, thread_ids)
            except ValueError as exc:
                raise RpcError("invalid_argument", str(exc)) from None
            return {"threads": threads}
        thread_id = self._required_string(params, "thread_id")
        thread = await self._store_call("get_thread", thread_id)
        if thread is None:
            raise RpcError("thread_not_found", f"Thread {thread_id} not found")
        if method == "thread.get":
            snapshot = await self.runtime.get_session(thread_id)
            stored_turns = await self._store_call("list_turns", thread_id)
            return {
                "thread": thread,
                "stats": snapshot.stats,
                "history": list(snapshot.timeline),
                "turns": stored_turns or list(snapshot.timeline),
                "events": await self._store_call("list_events", thread_id),
            }
        if method == "thread.delete":
            if self.runtime.session_active_operation_count(thread_id) > 0:
                raise RpcError("busy", "Cannot delete a thread with active operations")
            if thread["kind"] == "worktree":
                project = await self._store_call("get_project", thread["project_id"])
                if project is not None:
                    try:
                        inspection = await run_blocking(
                            git_worktree_status, self._thread_workspace(thread)
                        )
                        if inspection["dirty"]:
                            raise RpcError(
                                "busy", "Refusing to delete a dirty managed worktree"
                            )
                        if inspection["exists"]:
                            await run_blocking(
                                git_remove_worktree,
                                Path(project["path"]),
                                self._thread_workspace(thread),
                            )
                    except RpcError:
                        raise
                    except (RuntimeError, OSError) as exc:
                        raise RpcError("invalid_state", str(exc)) from None
            await self.pty.close(thread_id)
            try:
                await self.runtime.session_delete({"session_id": thread_id})
            except (RuntimeFailure, RunError) as exc:
                if exc.code != "not_found":
                    raise RpcError(exc.code, str(exc)) from None
            return {"deleted": await self._store_call("delete_thread", thread_id)}
        if method == "thread.rename":
            title = self._required_string(params, "title")
            await self.runtime.session_rename({"session_id": thread_id, "title": title})
            return {
                "thread": await self._store_call("set_thread_title", thread_id, title)
            }
        if method == "thread.configure":
            patch = {key: params[key] for key in ("model_id", "thinking_level") if key in params}
            await self.runtime.session_configure({"session_id": thread_id, **patch})
            return {
                "thread": await self._store_call("configure_thread", thread_id, **patch)
            }
        if method == "thread.set_pinned":
            return {
                "thread": await self._store_call(
                    "set_thread_pinned", thread_id, bool(params.get("pinned"))
                )
            }
        if method == "thread.set_archived":
            return {
                "thread": await self._store_call(
                    "set_thread_archived", thread_id, bool(params.get("archived"))
                )
            }
        if method == "thread.set_read":
            updated = await self._store_call(
                "set_thread_read", thread_id, bool(params.get("read"))
            )
            return {"updated": updated is not None}
        if method == "thread.tree":
            value = await self.runtime.session_tree({"session_id": thread_id})
            return {"nodes": value.get("nodes", []), "current": value.get("leaf_id")}
        if method == "thread.context":
            snapshot = await self.runtime.get_session(thread_id)
            used = snapshot.stats.get("input_tokens") or snapshot.stats.get("tokens_used") or 0
            maximum = snapshot.stats.get("context_window")
            return {
                "used": used,
                "maximum": maximum,
                "ratio": (used / maximum if maximum else None),
            }
        if method == "thread.navigate":
            value = await self.runtime.session_navigate(
                {"session_id": thread_id, "target_id": params.get("node_id")}
            )
            return {
                "current": value.get("newLeafId"),
                "stats": {
                    "cancelled": value.get("cancelled", False),
                    "old_leaf_id": value.get("oldLeafId"),
                    "summary_entry_id": value.get("summaryEntryId"),
                },
            }
        if method == "thread.compact":
            value = await self.runtime.session_compact({"session_id": thread_id})
            return {"compacted": True, "stats": value}
        if method == "thread.next_turn":
            value = await self.runtime.session_next_turn(
                {
                    "session_id": thread_id,
                    "input": {"kind": "prompt", "text": "continue", "attachments": []},
                }
            )
            return {"turn_id": value.get("entry_id")}
        raise RpcError("method_not_found", f"Unknown thread method: {method}")

    async def _turn_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "turn.start":
            thread_id = self._required_string(params, "thread_id")
            thread = await self._store_call("get_thread", thread_id)
            if thread is None:
                raise RpcError("thread_not_found", f"Thread {thread_id} not found")
            attachment_ids = params.get("attachment_ids") or []
            if not isinstance(attachment_ids, list) or any(
                not isinstance(item, str) for item in attachment_ids
            ):
                raise RpcError("invalid_attachment", "attachment_ids must be strings")
            attachments: list[dict[str, Any]] = []
            for attachment_id in attachment_ids:
                record = await self._store_call("get_attachment", attachment_id)
                if record is None:
                    raise RpcError("invalid_attachment", f"Attachment {attachment_id} not found")
                mime_type = str(record.get("mime_type") or "application/octet-stream")
                attachments.append(
                    {
                        "id": attachment_id,
                        "type": "image" if mime_type.startswith("image/") else "file",
                        "display_name": str(record["name"]),
                        "mime_type": mime_type,
                        "size_bytes": int(record["size_bytes"]),
                        "source_path": str(record["storage_path"]),
                    }
                )
            value = await self.runtime.turn_start(
                {
                    "session_id": thread_id,
                    "input": {
                        "kind": "prompt",
                        "text": self._required_string(params, "text"),
                        "attachments": attachments,
                    },
                },
                attachment_roots=(self._thread_workspace(thread), self.data_dir / "attachments"),
            )
            stored_attachments = [
                {
                    key: item[key]
                    for key in ("id", "type", "display_name", "mime_type", "size_bytes")
                }
                for item in attachments
            ]
            await self._store_call("ensure_turn", thread_id, value["operation_id"])
            await self._store_call(
                "update_turn_input",
                value["operation_id"],
                user_text=self._required_string(params, "text"),
                attachments=stored_attachments,
            )
            return {"operation_id": value["operation_id"], "turn_id": value["operation_id"]}
        if method == "turn.cancel":
            return {
                "cancelling": bool(
                    (
                        await self.runtime.turn_cancel(
                            {"operation_id": self._required_string(params, "operation_id")}
                        )
                    ).get("cancelled")
                )
            }
        if method == "turn.steer":
            return {
                "accepted": bool(
                    (
                        await self.runtime.turn_steer(
                            {
                                "operation_id": self._required_string(params, "operation_id"),
                                "text": self._required_string(params, "text"),
                            }
                        )
                    ).get("accepted")
                )
            }
        if method == "turn.follow_up":
            value = await self.runtime.turn_follow_up(
                {
                    "thread_id": self._required_string(params, "thread_id"),
                    "text": self._required_string(params, "text"),
                }
            )
            return {"accepted": bool(value.get("accepted"))}
        raise RpcError("method_not_found", f"Unknown turn method: {method}")

    async def _fs_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "fs.roots":
            return {
                "roots": [
                    {
                        "path": str(root),
                        "label": root.name or str(root),
                        "writable": os.access(root, os.W_OK),
                    }
                    for root in self.workspace_roots
                ]
            }
        path = await self._thread_path(params)
        if method == "fs.list":
            if not path.is_dir():
                raise RpcError("invalid_argument", "Path must be a directory")
            # A directory can be authorized while one of its entries is a
            # symlink into an unrelated tree.  Do not leak metadata for that
            # target (and keep list/read semantics consistent) — every child
            # must resolve inside the same Runtime boundary.
            boundary: tuple[Path, ...]
            if params.get("thread_id"):
                thread_id = self._required_string(params, "thread_id")
                thread = await self._store_call("get_thread", thread_id)
                if thread is None:
                    raise RpcError("thread_not_found", f"Thread {thread_id} not found")
                boundary = (self._thread_workspace(thread),)
            else:
                boundary = self.workspace_roots
            entries = []
            for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                resolved_item = item.resolve(strict=False)
                if not any(_is_within(resolved_item, root) for root in boundary):
                    raise RpcError("forbidden", "Path resolves outside authorized workspace roots")
                if item.is_symlink():
                    raise RpcError("forbidden", "Symbolic links are not allowed in workspace paths")
                stat = item.stat()
                entries.append(
                    {
                        "name": item.name,
                        "kind": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else 0,
                        "mtime": stat.st_mtime,
                    }
                )
            return {"entries": entries}
        if method == "fs.read":
            if not path.is_file():
                raise RpcError("invalid_argument", "Path must be a file")
            requested_max = params.get("max_bytes")
            max_bytes = FILE_BYTES if requested_max is None else int(requested_max)
            if max_bytes < 0:
                raise RpcError("invalid_argument", "max_bytes must not be negative")
            max_bytes = min(max_bytes, FILE_BYTES)
            # Read one byte past the advertised limit so a very large file
            # cannot force the Runtime to materialize its entire contents just
            # to return a truncated preview.
            with path.open("rb") as handle:
                data = handle.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            return {
                "content": data[:max_bytes].decode("utf-8", errors="replace"),
                "truncated": truncated,
                "encoding": "utf-8",
            }
        raise RpcError("method_not_found", f"Unknown fs method: {method}")

    async def _attachment_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "attachment.upload":
            name = self._required_string(params, "name")
            if Path(name).name != name:
                raise RpcError("invalid_attachment", "Attachment name must be a file name")
            encoded = params.get("data_base64")
            if not isinstance(encoded, str):
                raise RpcError("invalid_attachment", "data_base64 is required")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                raise RpcError("invalid_attachment", "data_base64 is invalid") from None
            mime_type = str(params.get("mime_type") or "application/octet-stream")
            limit = IMAGE_BYTES if mime_type.startswith("image/") else FILE_BYTES
            if len(data) > limit:
                raise RpcError("payload_too_large", "Attachment exceeds the negotiated limit")
            value = await self._store_call(
                "add_attachment",
                name=name,
                mime_type=mime_type,
                data=data,
                root=self.data_dir / "attachments",
            )
            return {
                "attachment_id": value["id"],
                "name": value["name"],
                "mime_type": value["mime_type"],
                "size": value["size_bytes"],
            }
        attachment_id = self._required_string(params, "attachment_id")
        value = await self._store_call("get_attachment", attachment_id)
        if value is None:
            raise RpcError("invalid_attachment", "Attachment not found")
        storage_path = self._attachment_storage_path(value)
        if method == "attachment.delete":
            return {"deleted": await self._store_call("delete_attachment", attachment_id)}
        if method == "attachment.download":
            limit = (
                IMAGE_BYTES
                if str(value.get("mime_type") or "").startswith("image/")
                else FILE_BYTES
            )
            if storage_path.stat().st_size > limit:
                raise RpcError("payload_too_large", "Attachment exceeds the negotiated limit")
            data = await run_blocking(storage_path.read_bytes)
            if len(data) > limit:
                raise RpcError("payload_too_large", "Attachment exceeds the negotiated limit")
            return {
                "data_base64": base64.b64encode(data).decode("ascii"),
                "mime_type": value["mime_type"],
                "name": value["name"],
            }
        if method == "attachment.preview":
            path = storage_path
            if str(value["mime_type"] or "").startswith("text/"):
                def read_preview() -> bytes:
                    # The preview endpoint is intentionally bounded even when
                    # the attachment itself is close to the 25 MiB wire limit.
                    # Do not materialize the complete blob merely to truncate it
                    # after the read.
                    with path.open("rb") as handle:
                        return handle.read(20_000)

                preview = (await run_blocking(read_preview)).decode("utf-8", errors="replace")
                kind = "text"
            else:
                # Keep the wire result string-shaped for binary attachments;
                # callers can use ``kind`` to distinguish an empty preview
                # from a text payload.
                preview = ""
                kind = "binary"
            return {"kind": kind, "preview": preview}
        raise RpcError("method_not_found", f"Unknown attachment method: {method}")

    def _attachment_storage_path(self, value: Mapping[str, Any]) -> Path:
        raw = value.get("storage_path")
        if not isinstance(raw, str) or not raw:
            raise RpcError("invalid_attachment", "Attachment storage path is invalid")
        path = Path(raw).expanduser().resolve(strict=False)
        root = (self.data_dir / "attachments").resolve(strict=False)
        if not _is_within(path, root):
            raise RpcError("forbidden", "Attachment storage path is outside Runtime storage")
        if not path.is_file():
            raise RpcError("invalid_attachment", "Attachment content is missing")
        # Blobs are content-addressed.  Besides detecting interrupted writes,
        # verifying the digest prevents a guessed attachment id from serving
        # a different file after a local metadata/blob tampering incident.
        expected = path.stem
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise RpcError("invalid_attachment", "Attachment content address is invalid")
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            raise RpcError("invalid_attachment", "Attachment content is unavailable") from None
        if actual != expected or int(value.get("size_bytes") or -1) != path.stat().st_size:
            raise RpcError("invalid_attachment", "Attachment content failed integrity check")
        return path

    async def _artifact_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        thread_id = self._required_string(params, "thread_id")
        thread = await self._store_call("get_thread", thread_id)
        if thread is None:
            raise RpcError("thread_not_found", f"Thread {thread_id} not found")
        if method == "artifact.resolve":
            paths = params.get("paths") or []
            artifacts: list[dict[str, Any]] = []
            for raw in paths:
                if not isinstance(raw, str):
                    raise RpcError("invalid_argument", "paths must be strings")
                candidate = self._thread_relative_path(thread, raw, must_exist=False)
                artifacts.append(
                    {
                        "path": raw,
                        "kind": "file"
                        if candidate.is_file()
                        else "directory"
                        if candidate.is_dir()
                        else "unknown",
                        "exists": candidate.exists(),
                    }
                )
            return {"artifacts": artifacts}
        if method == "artifact.download":
            path = await self._thread_path(params)
            if not path.is_file():
                raise RpcError("invalid_argument", "Artifact is not a file")
            if path.stat().st_size > FILE_BYTES:
                raise RpcError("payload_too_large", "Artifact exceeds the file limit")
            data = await run_blocking(path.read_bytes)
            if len(data) > FILE_BYTES:
                raise RpcError("payload_too_large", "Artifact exceeds the file limit")
            return {
                "data_base64": base64.b64encode(data).decode("ascii"),
                "mime_type": _mime_type(path),
            }
        raise RpcError("method_not_found", f"Unknown artifact method: {method}")

    async def _terminal_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        thread_id = self._required_string(params, "thread_id")
        thread = await self._store_call("get_thread", thread_id)
        if thread is None:
            raise RpcError("thread_not_found", f"Thread {thread_id} not found")
        try:
            if method == "terminal.open":
                return await self.pty.open(
                    thread_id,
                    self._thread_workspace(thread),
                    int(params.get("columns") or 120),
                    int(params.get("rows") or 36),
                )
            if method == "terminal.input":
                return {
                    "accepted": await self.pty.input(
                        thread_id, self._required_string(params, "data")
                    )
                }
            if method == "terminal.resize":
                return {
                    "accepted": await self.pty.resize(
                        thread_id,
                        int(params.get("columns") or 0),
                        int(params.get("rows") or 0),
                    )
                }
            if method == "terminal.close":
                return {"closed": await self.pty.close(thread_id)}
        except (RuntimeError, ValueError, OSError) as exc:
            raise RpcError("invalid_state", str(exc)) from None
        raise RpcError("method_not_found", f"Unknown terminal method: {method}")

    async def _git_dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        thread_id = self._required_string(params, "thread_id")
        thread = await self._store_call("get_thread", thread_id)
        if thread is None:
            raise RpcError("thread_not_found", f"Thread {thread_id} not found")
        workspace = self._thread_workspace(thread)
        if method == "git.status":
            return await run_blocking(git_status, workspace)
        if method == "git.changes":
            return await run_blocking(git_changes, workspace)
        if method == "git.diff":
            scope = str(params.get("scope") or "changes")
            if scope not in {"changes", "staged"}:
                raise RpcError("invalid_argument", "scope must be changes or staged")
            path = params.get("path")
            if path is not None:
                self._validate_workspace_relative(workspace, str(path))
            return await run_blocking(git_diff, workspace, scope, str(path) if path else None)
        if method == "git.branches":
            return await run_blocking(git_branches, workspace)
        if method == "git.github_status":
            return await run_blocking(git_github_status, workspace)
        if method in {"git.stage", "git.unstage"}:
            paths = params.get("paths") or []
            if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
                raise RpcError("invalid_argument", "paths must be an array of strings")
            for path in paths:
                self._validate_workspace_relative(workspace, path)
            operation = git_stage if method == "git.stage" else git_unstage
            return await run_blocking(operation, workspace, paths)
        if method == "git.branch_create":
            return await run_blocking(
                git_branch_create, workspace, self._required_string(params, "branch")
            )
        if method == "git.commit":
            return await run_blocking(
                git_commit, workspace, self._required_string(params, "message")
            )
        if method == "git.push":
            return await run_blocking(
                git_push, workspace, str(params.get("remote") or "origin")
            )
        if method == "git.pr_create":
            return await run_blocking(
                git_pr_create,
                workspace,
                self._required_string(params, "title"),
                str(params.get("body") or ""),
            )
        raise RpcError("capability_unavailable", f"Git operation is not enabled: {method}")

    def _allowed_path(
        self, value: Any, *, must_exist: bool = False, directory: bool = False
    ) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise RpcError("invalid_argument", "path is required")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve(strict=False)
        if not any(_is_within(resolved, root) for root in self.workspace_roots):
            raise RpcError("forbidden", "Path is outside authorized workspace roots")
        if _contains_symlink(candidate):
            raise RpcError("forbidden", "Symbolic links are not allowed in workspace paths")
        if must_exist and not resolved.exists():
            raise RpcError("invalid_argument", "Path does not exist")
        if directory and resolved.exists() and not resolved.is_dir():
            raise RpcError("invalid_argument", "Path must be a directory")
        return resolved

    def _thread_workspace(self, thread: Mapping[str, Any]) -> Path:
        """Resolve a thread workspace, including Runtime-managed worktrees."""

        value = thread.get("workspace")
        if not isinstance(value, str) or not value:
            raise RpcError("invalid_state", "Thread workspace is invalid")
        candidate = Path(value).expanduser()
        if _contains_symlink(candidate):
            raise RpcError("forbidden", "Symbolic links are not allowed in thread workspaces")
        resolved = candidate.resolve(strict=False)
        managed_root = (self.data_dir / "worktrees").resolve(strict=False)
        if not any(_is_within(resolved, root) for root in (*self.workspace_roots, managed_root)):
            raise RpcError("forbidden", "Thread workspace is outside authorized roots")
        if not resolved.exists() or not resolved.is_dir():
            raise RpcError("invalid_state", "Thread workspace does not exist")
        return resolved

    async def _thread_path(self, params: Mapping[str, Any]) -> Path:
        if params.get("root"):
            root = self._allowed_path(params.get("root"), must_exist=True, directory=True)
            relative = params.get("path") or "."
            if not isinstance(relative, str):
                raise RpcError("invalid_argument", "path must be a string")
            self._validate_relative(relative)
            return self._allowed_path(str(root / relative), must_exist=True)
        thread_id = self._required_string(params, "thread_id")
        thread = await self._store_call("get_thread", thread_id)
        if thread is None:
            raise RpcError("thread_not_found", f"Thread {thread_id} not found")
        relative = params.get("path") or "."
        return self._thread_relative_path(thread, relative, must_exist=True)

    def _thread_relative_path(
        self,
        thread: Mapping[str, Any],
        relative: Any,
        *,
        must_exist: bool,
    ) -> Path:
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RpcError("forbidden", "Relative path escapes thread workspace")
        workspace = self._thread_workspace(thread)
        resolved = (workspace / relative).resolve(strict=False)
        if not _is_within(resolved, workspace):
            raise RpcError("forbidden", "Path resolves outside thread workspace")
        if _contains_symlink(workspace / relative):
            raise RpcError("forbidden", "Symbolic links are not allowed in workspace paths")
        if must_exist and not resolved.exists():
            raise RpcError("invalid_argument", "Path does not exist")
        return resolved

    @staticmethod
    def _validate_relative(value: str) -> None:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise RpcError("forbidden", "Relative path escapes thread workspace")

    @classmethod
    def _validate_workspace_relative(cls, workspace: Path, value: str) -> None:
        cls._validate_relative(value)
        resolved = (workspace / value).resolve(strict=False)
        if not _is_within(resolved, workspace):
            raise RpcError("forbidden", "Path resolves outside thread workspace")
        if _contains_symlink(workspace / value):
            raise RpcError("forbidden", "Symbolic links are not allowed in workspace paths")

    @staticmethod
    def _required_string(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RpcError("invalid_argument", f"{key} is required")
        return value.strip()

    async def _handshake(
        self,
        params: Mapping[str, Any],
        connection: RuntimeConnection | None = None,
    ) -> dict[str, Any]:
        offered = params.get("protocol")
        if not isinstance(offered, Mapping):
            raise RpcError(
                "protocol_incompatible",
                "Client did not provide a protocol range",
                {"runtime_min": SUPPORTED_PROTOCOLS[0], "runtime_max": SUPPORTED_PROTOCOLS[-1]},
            )
        minimum = str(offered.get("min") or "")
        maximum = str(offered.get("max") or "")
        if minimum != PROTOCOL_VERSION or maximum != PROTOCOL_VERSION:
            raise RpcError(
                "protocol_incompatible",
                f"Runtime requires exactly {PROTOCOL_VERSION}; client offered {minimum}..{maximum}",
                {
                    "client_min": minimum,
                    "client_max": maximum,
                    "runtime_min": PROTOCOL_VERSION,
                    "runtime_max": PROTOCOL_VERSION,
                },
            )
        result: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "runtime": {
                "id": self.runtime_id,
                "label": self.runtime_label,
                "version": RUNTIME_VERSION,
                "commit": RUNTIME_COMMIT,
            },
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.system().lower() or sys.platform,
                "architecture": platform.machine() or "unknown",
                "shell": os.environ.get("SHELL") or "unknown",
                "tools": {"git": bool(shutil.which("git")), "gh": bool(shutil.which("gh"))},
            },
            "limits": {
                "prompt_chars": 100_000,
                "attachments": 8,
                "image_bytes": IMAGE_BYTES,
                "file_bytes": FILE_BYTES,
                "request_bytes": MAX_FRAME_BYTES,
                "retained_events": EVENT_LIMIT,
            },
        }
        if connection is not None:
            device = await self._authenticate_handshake(params, connection)
            if device is not None:
                result["device"] = device
        return result

    async def _authenticate_handshake(
        self,
        params: Mapping[str, Any],
        connection: RuntimeConnection,
    ) -> dict[str, str] | None:
        requires_auth = connection.requires_auth and connection.device_id is None
        auth = params.get("auth")
        if not requires_auth and not isinstance(auth, Mapping):
            return None
        source = connection.auth_source or ("websocket" if connection.requires_auth else "unix")
        async with self.pairing.limiter.lock_for(source):
            return await self._authenticate_handshake_locked(params, connection, source, auth)

    async def _authenticate_handshake_locked(
        self,
        params: Mapping[str, Any],
        connection: RuntimeConnection,
        source: str,
        auth: Any,
    ) -> dict[str, str] | None:
        delay = self.pairing.limiter.delay_for(source)
        if delay > 0:
            await asyncio.sleep(delay)
        if not isinstance(auth, Mapping):
            await self._auth_failed(source, "missing")
            connection.close_after_response = True
            raise RpcError("unauthorized", "This connection requires device authentication")
        scheme = auth.get("kind")
        try:
            if scheme == "device_token":
                self._accept_device_token(auth, connection, source)
                return None
            raise RpcError("unauthorized", "Unsupported authentication scheme")
        except RpcError as exc:
            if exc.code == "unauthorized":
                await self._auth_failed(source, str(scheme or "unknown"))
            if connection.requires_auth:
                connection.close_after_response = True
            raise

    def _accept_device_token(
        self, auth: Mapping[str, Any], connection: RuntimeConnection, source: str
    ) -> None:
        token = auth.get("token")
        if not isinstance(token, str) or not token:
            raise RpcError("unauthorized", "Device token authentication requires a token")
        record = self.pairing.store.verify(token)
        if record is None:
            raise RpcError("unauthorized", "Device token is invalid")
        connection.device_id = str(record["id"])
        try:
            self._enforce_device_limit(connection.device_id)
            self.activate_connection(connection)
        except RpcError:
            connection.device_id = None
            raise
        self.pairing.store.touch(str(record["id"]))
        self.pairing.limiter.record_success(source)

    async def _auth_failed(self, source: str, scheme: str) -> None:
        self.pairing.limiter.record_failure(source)
        if self.log is not None:
            self.log.write("auth_failed", source=source, scheme=scheme)

    def _enforce_device_limit(self, device_id: str) -> None:
        count = sum(1 for item in self.connections if item.device_id == device_id)
        if count >= MAX_CLIENTS_PER_DEVICE:
            raise RpcError("busy", "This device already has the maximum number of connections")

    def _device_public(self, record: Mapping[str, Any]) -> dict[str, Any]:
        device_id = str(record.get("id") or "")
        return {
            "id": device_id,
            "name": str(record.get("name") or "device"),
            "platform": str(record.get("platform") or "unknown"),
            "paired_at": str(record.get("paired_at") or ""),
            "last_seen_at": record.get("last_seen_at"),
            "connected": any(item.device_id == device_id for item in self.connections),
        }

    async def _devices_dispatch(
        self,
        method: str,
        params: Mapping[str, Any],
        connection: RuntimeConnection | None,
    ) -> dict[str, Any]:
        if method == "devices.claim":
            if connection is None or not connection.requires_auth:
                raise RpcError("forbidden", "devices.claim is only available on WSS enrollment connections")
            code = self._required_string(params, "code")
            client = params.get("client")
            if not isinstance(client, Mapping):
                raise RpcError("invalid_argument", "client is required")
            if not isinstance(client.get("name"), str) or not str(client.get("name")).strip():
                raise RpcError("invalid_argument", "client.name is required")
            if not isinstance(client.get("version"), str) or not str(client.get("version")).strip():
                raise RpcError("invalid_argument", "client.version is required")
            if not self.pairing.enrollment.consume(code):
                connection.close_after_response = True
                raise RpcError("unauthorized", "Enrollment code is invalid or expired")
            record, token = self.pairing.store.issue(
                name=str(client["name"]),
                platform="remote",
            )
            connection.close_after_response = True
            self.pairing.limiter.record_success(connection.auth_source or "websocket")
            if self.log is not None:
                self.log.write("paired", device_id=record["id"], name=record["name"])
            return {"device_id": str(record["id"]), "token": token}
        if method == "devices.list":
            return {
                "devices": [self._device_public(item) for item in self.pairing.store.list_devices()]
            }
        if method == "devices.revoke":
            device_id = self._required_string(params, "device_id")
            revoked = self.pairing.store.revoke(device_id)
            for item in tuple(self.connections):
                if item.device_id != device_id:
                    continue
                if item is connection:
                    item.close_after_response = True
                    continue
                await item.close()
            if self.log is not None:
                self.log.write("revoked", device_id=device_id, revoked=revoked)
            return {
                "revoked": revoked,
                "devices": [
                    self._device_public(item) for item in self.pairing.store.list_devices()
                ],
            }
        if method == "devices.enroll":
            try:
                code, expires_at, url = self.pairing.issue_enrollment()
            except RuntimeError as exc:
                raise RpcError("invalid_state", str(exc)) from None
            return {"code": code, "expires_at": expires_at, "pairing_url": url}
        raise RpcError("method_not_found", f"Unknown v4 method: {method}")

    def _subscribe(self, params: Mapping[str, Any]) -> dict[str, Any]:
        after_seq = int(params.get("after_seq") or 0)
        previous_instance = params.get("server_instance_id")
        first_seq = self.events[0]["seq"] if self.events else self.current_seq + 1
        replay_complete = (
            previous_instance is None or previous_instance == self.server_instance_id
        ) and after_seq >= first_seq - 1
        thread_ids = {
            str(value) for value in params.get("thread_ids", []) if isinstance(value, str)
        }
        events = (
            [
                event
                for event in self.events
                if event["seq"] > after_seq
                and (
                    not thread_ids
                    or event.get("thread_id") is None
                    or event.get("thread_id") in thread_ids
                )
            ]
            if replay_complete
            else []
        )
        return {
            "server_instance_id": self.server_instance_id,
            "current_seq": self.current_seq,
            "replay_complete": replay_complete,
            "events": events,
            "cursor": {"server_instance_id": self.server_instance_id, "seq": self.current_seq},
        }


class RuntimeConnection:
    def __init__(
        self,
        server: RuntimeServer,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        requires_auth: bool = False,
        auth_source: str = "",
    ) -> None:
        self.server = server
        self.reader = reader
        self.writer = writer
        self.requires_auth = requires_auth
        self.auth_source = auth_source
        self.device_id: str | None = None
        self.close_after_response = False
        self.thread_ids: set[str] = set()
        self.subscribed = False
        self.handshaken = False
        self.write_lock = asyncio.Lock()
        self._writes: asyncio.PriorityQueue[
            tuple[int, int, dict[str, Any], asyncio.Future[None]]
        ] = asyncio.PriorityQueue()
        # Responses remain unblocked by event traffic, while event frames have
        # an explicit per-connection bound.  Counting event frames separately
        # avoids an unbounded mixed priority queue when a client stops reading
        # but Runtime producers continue to publish output.
        self._pending_event_frames = 0
        self._write_sequence = 0
        self._writer_task: asyncio.Task[None] | None = None
        self._request_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def run(self) -> None:
        self._writer_task = asyncio.create_task(self._writer_loop())
        try:
            while True:
                try:
                    read = read_frame(self.reader)
                    request = await (
                        asyncio.wait_for(read, AUTH_HANDSHAKE_TIMEOUT_S)
                        if self.requires_auth and not self.handshaken
                        else read
                    )
                except TimeoutError:
                    return
                except EOFError:
                    return
                except RpcError as exc:
                    # A frame-level failure has no trustworthy request id.  Send
                    # one JSON-RPC error with a null id, then close the stream so
                    # a malformed/oversized frame cannot be retried in place or
                    # leave trailing bytes to be interpreted as a new request.
                    with contextlib.suppress(Exception):
                        await self.send({"id": None, "error": exc.to_rpc()})
                    return
                if not self.handshaken:
                    # Authentication is deliberately serialized. Otherwise a
                    # burst of concurrent guesses can all observe the same
                    # enrollment state before the limiter or one-time consume
                    # path runs.
                    await self.handle(request)
                    if self._closed:
                        return
                else:
                    task = asyncio.create_task(self.handle(request))
                    self._request_tasks.add(task)
                    task.add_done_callback(self._request_done)
        finally:
            tasks = tuple(self._request_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if self._writer_task is not None:
                self._writer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._writer_task
                self._writer_task = None
            await self.close()

    def _request_done(self, task: asyncio.Task[None]) -> None:
        self._request_tasks.discard(task)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()

    async def handle(self, request: Any) -> None:
        request_id = request.get("id") if isinstance(request, Mapping) else None
        method = (
            str(request.get("method") or "<invalid>")
            if isinstance(request, Mapping)
            else "<invalid>"
        )
        params: Mapping[str, Any] = {}
        if self.server.trace is not None:
            raw_params = request.get("params") if isinstance(request, Mapping) else None
            self.server.trace.request(request_id, method, raw_params)
        try:
            if not isinstance(request, Mapping) or not isinstance(request.get("method"), str):
                raise RpcError("invalid_argument", "RPC request must contain a method")
            params = request.get("params", {})
            if not isinstance(params, Mapping):
                raise RpcError("invalid_argument", "RPC params must be an object")
            method = str(request["method"])
            if method != "system.handshake" and method != "devices.claim" and not self.handshaken:
                raise RpcError(
                    "protocol_incompatible", "Complete system.handshake before other methods"
                )
            result = await self.server.dispatch(method, params, self)
            if method == "system.handshake":
                self.handshaken = True
            self.server.validate_result(method, result)
            if method == "events.subscribe":
                self.thread_ids = {
                    str(value) for value in params.get("thread_ids", []) if isinstance(value, str)
                }
                self.subscribed = True
            if self.server.trace is not None:
                self.server.trace.response(request_id, method, result)
            await self.send({"id": request_id, "result": result})
            if self.close_after_response:
                await self.close()
        except RpcError as exc:
            if method in {"system.handshake", "devices.claim"} and self.requires_auth:
                self.close_after_response = True
            if self.server.trace is not None:
                self.server.trace.error(request_id, method, exc.code, str(exc))
            await self.send({"id": request_id, "error": exc.to_rpc()})
            if self.close_after_response:
                await self.close()
        except RuntimeFailure as exc:
            if self.server.trace is not None:
                self.server.trace.error(request_id, method, exc.code, str(exc))
            await self.send({"id": request_id, "error": RpcError(exc.code, str(exc)).to_rpc()})
        except ValueError as exc:
            if self.server.trace is not None:
                self.server.trace.error(request_id, method, "invalid_argument", str(exc))
            await self.send(
                {"id": request_id, "error": RpcError("invalid_argument", str(exc)).to_rpc()}
            )
        except Exception:
            if self.server.trace is not None:
                self.server.trace.error(
                    request_id,
                    method,
                    "internal_error",
                    "Aeloon Runtime could not complete the request",
                )
            await self.send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Aeloon Runtime could not complete the request",
                        "data": {"code": "internal_error"},
                    },
                }
            )

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self.subscribed or self._closed:
            return
        thread_id = event.get("thread_id")
        if self.thread_ids and thread_id is not None and thread_id not in self.thread_ids:
            return
        if self._pending_event_frames >= EVENT_QUEUE_LIMIT:
            await self.close()
            return
        self._pending_event_frames += 1
        try:
            future = await self._enqueue({"method": "event", "params": event}, event=True)
        except Exception:
            self._pending_event_frames = max(0, self._pending_event_frames - 1)
            raise
        # Event delivery is deliberately fire-and-forget.  The bounded queue
        # and overflow close protect the Runtime from a client that stops
        # reading, while this callback consumes a late writer exception so it
        # cannot become an unhandled Future warning.
        future.add_done_callback(self._event_frame_done)

    def _event_frame_done(self, future: asyncio.Future[None]) -> None:
        self._pending_event_frames = max(0, self._pending_event_frames - 1)
        _consume_future_exception(future)

    async def send(self, value: dict[str, Any], *, event: bool = False) -> None:
        future = await self._enqueue(value, event=event)
        await future

    async def _enqueue(self, value: dict[str, Any], *, event: bool) -> asyncio.Future[None]:
        if self._closed or self.writer.is_closing():
            raise ConnectionError("Runtime connection is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._write_sequence += 1
        await self._writes.put((1 if event else 0, self._write_sequence, value, future))
        return future

    async def _writer_loop(self) -> None:
        while True:
            _priority, _sequence, value, future = await self._writes.get()
            if future.cancelled():
                continue
            try:
                async with self.write_lock:
                    self.writer.write(pack_frame(value))
                    await self.writer.drain()
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                await self.close()
                return
            if not future.done():
                future.set_result(None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = tuple(task for task in self._request_tasks if task is not current)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        while True:
            try:
                _priority, _sequence, _value, future = self._writes.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not future.done():
                future.set_exception(ConnectionError("Runtime connection is closed"))
        self.writer.close()
        with contextlib.suppress(Exception):
            await self.writer.wait_closed()


def _consume_future_exception(future: asyncio.Future[None]) -> None:
    if future.cancelled():
        return
    with contextlib.suppress(Exception):
        future.exception()


def _load_runtime_identity(data_dir: Path) -> str:
    """Load or atomically create the v4 Runtime identity."""

    path = data_dir / "runtime-identity.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("id") if isinstance(payload, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    value = str(uuid.uuid4())
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".runtime-identity.", dir=data_dir)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"schemaVersion": 1, "id": value}, handle, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    return value


def _root_id(path: Path) -> str:
    return hashlib.sha256(
        str(path.expanduser().resolve(strict=False)).encode("utf-8")
    ).hexdigest()


def _open_runtime_lock(data_dir: Path) -> int:
    """Acquire the Runtime data lock before composing mutable services."""

    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_dir.chmod(0o700)
    lock = data_dir / ".runtime.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"Runtime data directory is already in use: {data_dir}") from None
    return fd


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _contains_symlink(path: Path) -> bool:
    """Return whether an existing component of ``path`` is a symlink.

    ``Path.resolve`` alone is not enough for the workspace policy: a link that
    happens to point back inside an authorized root would otherwise be treated
    as safe.  Walk the lexical path with ``lstat`` so links are rejected before
    any operation follows them.  Missing suffixes are fine for callers that
    validate a path before creating a file.
    """

    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                return True
        except FileNotFoundError:
            break
        except OSError:
            # Permission errors are handled by the subsequent realpath/stat
            # checks and must not turn into an accidental allow.
            return True
    return False


def _mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _snake_case_value(value: Any) -> Any:
    """Translate producer DTO keys at the public event boundary."""

    if isinstance(value, list):
        return [_snake_case_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        _CAMEL_BOUNDARY.sub(lambda match: f"_{match.group(1).lower()}", str(key)):
        _snake_case_value(item)
        for key, item in value.items()
    }


async def serve(
    *,
    socket_path: Path,
    data_dir: Path | None = None,
    config_path: Path | None = None,
    workspace_roots: tuple[Path, ...] | None = None,
    max_concurrent_operations: int = 4,
    record_trace: Path | None = None,
    listen: tuple[str, int] | None = None,
    tls_context: Any | None = None,
    tls_certificate: Path | None = None,
    tls_key: Path | None = None,
    runtime_label: str | None = None,
    advertise_url: str | None = None,
) -> None:
    resolved_data = (
        (data_dir or Path("~/.aeloon-runtime").expanduser()).expanduser().resolve(strict=False)
    )
    if (resolved_data / ".incomplete").exists() or (resolved_data / ".reset-incomplete").exists():
        raise RuntimeError(
            "Runtime reset state is incomplete; run the explicit v4 reset command before serving"
        )
    if (resolved_data / "devices.json").exists() or (resolved_data / "migration.complete").exists():
        raise RuntimeError(
            "Legacy Runtime data detected; run the explicit v4 reset command before serving"
        )
    # ``None`` is the standalone CLI default (launch directory). An explicit
    # empty tuple is the desktop "not yet authorized" state and must not
    # collapse to cwd, which is ``/`` for a packaged Electron app.
    roots = (Path.cwd(),) if workspace_roots is None else workspace_roots
    # Hold the single-instance lock before composing RuntimeService. Its
    # constructor creates session/attachment projections and performs orphan
    # cleanup, so locking only inside ``RuntimeServer.run`` leaves a startup
    # race between two standalone processes.
    lock_fd = _open_runtime_lock(resolved_data)
    runtime: RuntimeService | None = None
    trace: TraceRecorder | None = None
    server: RuntimeServer | None = None
    try:
        runtime = create_runtime_service(
            config_path=config_path or resolved_data / "config.json",
            data_dir=resolved_data,
            max_concurrent_operations=max_concurrent_operations,
        )
        trace = TraceRecorder(record_trace) if record_trace is not None else None
        server = RuntimeServer(
            runtime,
            socket_path,
            roots,
            resolved_data,
            trace_recorder=trace,
            preacquired_lock_fd=lock_fd,
            listen=listen,
            tls_context=tls_context,
            tls_certificate=tls_certificate,
            tls_key=tls_key,
            runtime_label=runtime_label,
            advertise_url=advertise_url,
        )
        lock_fd = None
        await server.run()
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if server is None:
            if trace is not None:
                trace.close()
            if runtime is not None:
                await runtime.close()
