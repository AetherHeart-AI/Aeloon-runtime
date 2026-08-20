"""aeloon-rpc-v2 adapter over the transport-free runtime service."""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeloon_runtime.rpc.protocol import (
    EVENTS,
    MAX_FRAME_BYTES,
    METHOD_REGISTRY,
    METHODS,
    PROTOCOL_NAME,
    RpcError,
)
from aeloon_runtime.runtime.service import (
    ATTACHMENT_LIMIT,
    FILE_LIMIT,
    IMAGE_LIMIT,
    PROMPT_LIMIT,
    RuntimeService,
)
from aeloon_runtime.runtime.session import SessionError
from aeloon_runtime.runtime.types import RuntimeEvent, RuntimeFailure, TurnInput
from aeloon_runtime.version import __version__, runtime_commit

EVENT_LIMIT = 5_000
RpcEventListener = Callable[[dict[str, Any]], Awaitable[None] | None]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AeloonRpcAdapter:
    """Own wire dispatch, public errors, event sequencing, and replay."""

    def __init__(self, runtime: RuntimeService) -> None:
        self.runtime = runtime
        self.server_instance_id = uuid.uuid4().hex
        self.started_at = _now()
        self.shutdown_requested = asyncio.Event()
        self.shutdown_signal = asyncio.Event()
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_LIMIT)
        self._seq = 0
        self._listeners: set[RpcEventListener] = set()
        self._remove_runtime_listener = runtime.add_event_listener(self._runtime_event)

    @property
    def config_path(self) -> Path:
        return self.runtime.config_path

    @property
    def data_dir(self) -> Path:
        return self.runtime.data_dir

    def add_event_listener(self, listener: RpcEventListener) -> Callable[[], None]:
        self._listeners.add(listener)

        def remove() -> None:
            self._listeners.discard(listener)

        return remove

    async def dispatch(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        attachment_roots: tuple[Path, ...] = (),
    ) -> Any:
        value = dict(params or {})
        spec = METHOD_REGISTRY.get(method)
        if spec is None:
            raise RpcError("method_not_found", f"Unknown RPC method: {method}")
        owner: Any = self
        handler_name = spec.handler
        if handler_name.startswith("runtime."):
            owner = self.runtime
            handler_name = handler_name.removeprefix("runtime.")
        handler: Callable[..., Awaitable[Any]] = getattr(owner, handler_name)
        try:
            if method == "turn.start":
                return await handler(
                    value,
                    attachment_roots=attachment_roots,
                )
            return await handler(value)
        except RpcError:
            raise
        except SessionError as exc:
            code = "session_not_found" if exc.code == "not_found" else "invalid_state"
            raise RpcError(code, self._sanitize(str(exc))) from None
        except RuntimeFailure as exc:
            allowed = {
                "busy",
                "invalid_state",
                "invalid_argument",
                "invalid_attachment",
                "attachment_processing_failed",
                "revision_conflict",
                "operation_not_found",
                "authentication_failed",
            }
            code = exc.code if exc.code in allowed else "internal_error"
            raise RpcError(code, str(exc)) from None
        except (KeyError, TypeError, ValueError) as exc:
            raise RpcError("invalid_argument", self._sanitize(str(exc))) from None
        except Exception:
            raise RpcError(
                "internal_error",
                "Aeloon Runtime could not complete the request",
            ) from None

    async def handshake(
        self,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        if params.get("protocol") != PROTOCOL_NAME:
            raise RpcError("protocol_incompatible", "Aeloon Runtime requires aeloon-rpc-v2")
        roots = params.get("attachment_roots") or []
        if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
            raise RpcError("invalid_argument", "attachment_roots must be a list of paths")
        return {
            "protocol": PROTOCOL_NAME,
            "core_version": __version__,
            "core_commit": runtime_commit(),
            "server_instance_id": self.server_instance_id,
            "methods": list(METHODS),
            "events": list(EVENTS),
            "attachment_roots": [
                str(Path(root).expanduser().resolve(strict=False)) for root in roots
            ],
            "config_path": str(self.config_path),
            "data_dir": str(self.data_dir),
            "limits": {
                "prompt_chars": PROMPT_LIMIT,
                "attachments": ATTACHMENT_LIMIT,
                "image_bytes": IMAGE_LIMIT,
                "file_bytes": FILE_LIMIT,
                "request_bytes": MAX_FRAME_BYTES,
                "retained_events": EVENT_LIMIT,
            },
        }

    async def _session_create(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workspace = self._required_string(params, "workspace")
        raw_title = params.get("title")
        title = str(raw_title).strip() if raw_title is not None else None
        session_id = self._required_string(params, "session_id")
        return (
            await self.runtime.create_session(
                workspace=workspace,
                session_id=session_id,
                title=title,
            )
        ).to_dict()

    async def _session_list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workspace = str(params["workspace"]) if params.get("workspace") else None
        sessions = await self.runtime.list_sessions(workspace=workspace)
        return {"sessions": [item.to_dict() for item in sessions]}

    async def _session_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session_id = self._required_string(params, "session_id")
        return (await self.runtime.get_session(session_id)).to_dict()

    async def _turn_start(
        self,
        params: Mapping[str, Any],
        *,
        attachment_roots: tuple[Path, ...] = (),
    ) -> dict[str, Any]:
        session_id = self._required_string(params, "session_id")
        turn_input = self._turn_input(params.get("input"))
        return (
            await self.runtime.start_turn(
                session_id=session_id,
                input=turn_input,
                attachment_roots=attachment_roots,
            )
        ).to_dict()

    async def _turn_cancel(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation_id = self._required_string(params, "operation_id")
        await self.runtime.cancel_turn(operation_id)
        return {"operation_id": operation_id, "cancelled": True}

    async def _turn_steer(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation_id = self._required_string(params, "operation_id")
        await self.runtime.steer_turn(operation_id, self._required_string(params, "text"))
        return {"operation_id": operation_id, "accepted": True}

    async def _turn_follow_up(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation_id = self._required_string(params, "operation_id")
        await self.runtime.follow_up_turn(
            operation_id,
            self._required_string(params, "text"),
        )
        return {"operation_id": operation_id, "accepted": True}

    async def health(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "stopping" if self.shutdown_requested.is_set() else "running",
            "pid": os.getpid(),
            "started_at": self.started_at,
            "active_operations": self.runtime.active_operation_count,
            "current_seq": self._seq,
        }

    async def shutdown(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        if not self.shutdown_requested.is_set():
            await self._publish(
                RuntimeEvent(
                    name="system.shutdown",
                    time=_now(),
                    workspace=None,
                    session_id=None,
                    operation_id=None,
                    payload={"intentional": True, "reason": "requested"},
                )
            )
        self.shutdown_requested.set()
        asyncio.get_running_loop().call_later(0.05, self.shutdown_signal.set)
        return {"status": "stopping"}

    def request_shutdown(self) -> None:
        self.shutdown_requested.set()
        self.shutdown_signal.set()

    async def events_subscribe(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session_ids = params.get("session_ids") or []
        after_seq = int(params.get("after_seq") or 0)
        invalid_session_ids = not isinstance(session_ids, list) or any(
            not isinstance(item, str) for item in session_ids
        )
        if invalid_session_ids:
            raise RpcError("invalid_argument", "session_ids must be a list")
        first_seq = self._events[0]["seq"] if self._events else self._seq + 1
        replay_complete = after_seq >= first_seq - 1
        replay = (
            [
                event
                for event in self._events
                if event["seq"] > after_seq
                and (event["session_id"] in session_ids or event["session_id"] is None)
            ]
            if replay_complete
            else []
        )
        return {
            "server_instance_id": self.server_instance_id,
            "current_seq": self._seq,
            "replay_complete": replay_complete,
            "events": replay,
            "cursor": {"server_instance_id": self.server_instance_id, "seq": self._seq},
        }

    async def close(self) -> None:
        self._remove_runtime_listener()
        await self.runtime.close()

    async def _runtime_event(self, event: RuntimeEvent) -> None:
        await self._publish(event)

    async def _publish(self, event: RuntimeEvent) -> None:
        self._seq += 1
        public = {"seq": self._seq, **event.to_dict()}
        self._events.append(public)
        for listener in tuple(self._listeners):
            try:
                result = listener(public)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue

    def _sanitize(self, message: str) -> str:
        value = message.replace(str(Path.home()), "~")
        secrets: list[str] = []
        for provider in self.runtime.config.providers.values():
            api_key = getattr(provider, "api_key", None)
            if api_key:
                secrets.append(api_key)
            for name, header_value in getattr(provider, "headers", {}).items():
                if name.lower() in {"authorization", "api-key", "x-api-key"}:
                    secrets.append(header_value)
        if self.runtime.config.tools.web.search.api_key:
            secrets.append(self.runtime.config.tools.web.search.api_key)
        for secret in secrets:
            if secret:
                value = value.replace(secret, "***")
        return value

    def _turn_input(self, raw: Any) -> TurnInput:
        if not isinstance(raw, Mapping):
            raise RpcError("invalid_argument", "turn.start.input must be an object")
        kind = str(raw.get("kind") or "prompt")
        if kind == "prompt":
            text = self._required_string(raw, "text")
            if len(text) > PROMPT_LIMIT:
                raise RpcError(
                    "invalid_argument",
                    f"Prompt must contain 1 to {PROMPT_LIMIT:,} characters",
                )
            attachments = raw.get("attachments") or []
            if not isinstance(attachments, list):
                raise RpcError("invalid_attachment", "attachments must be a list")
            return TurnInput(
                kind="prompt",
                text=text,
                attachments=tuple(
                    dict(item) if isinstance(item, Mapping) else {"invalid": item}
                    for item in attachments
                ),
            )
        if kind == "skill":
            return TurnInput(
                kind="skill",
                name=self._required_string(raw, "name"),
                additional_instructions=(
                    str(raw["additional_instructions"])
                    if raw.get("additional_instructions")
                    else None
                ),
            )
        if kind == "prompt_template":
            arguments = raw.get("arguments") or []
            if not isinstance(arguments, list) or any(
                not isinstance(item, str) for item in arguments
            ):
                raise RpcError("invalid_argument", "template arguments must be strings")
            return TurnInput(
                kind="prompt_template",
                name=self._required_string(raw, "name"),
                arguments=tuple(arguments),
            )
        raise RpcError(
            "invalid_argument",
            "input.kind must be prompt, skill, or prompt_template",
        )

    @staticmethod
    def _required_string(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RpcError("invalid_argument", f"{key} is required")
        return value.strip()


__all__ = ["AeloonRpcAdapter"]
