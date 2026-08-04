# ruff: noqa: E501
"""Application boundary shared by the CLI and Bridge v2 transport."""

from __future__ import annotations

import asyncio
import base64
import inspect
import shutil
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from aeloon_core.bridge.protocol import (
    CAPABILITIES,
    EVENTS,
    METHODS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BridgeError,
)
from aeloon_core.cloud import CloudAccountService, CloudError
from aeloon_core.config import (
    Config,
    LocalModelConfig,
    LocalProviderConfig,
    load_config,
    resolve_config_path,
    save_config,
)
from aeloon_core.harness import (
    ALL_TOOL_NAMES,
    DEFAULT_ACTIVE_TOOLS,
    AgentHarness,
    CompactionSettings,
    HarnessError,
    HarnessEvent,
    ImageContent,
    JsonlSessionRepository,
    Model,
    ResourceLoader,
    Session,
    SessionError,
    StreamOptions,
)
from aeloon_core.providers import (
    DEEPSEEK_PROVIDER_ID,
    UnifiedProviderRegistry,
    normalize_model_id,
    qualify_model_id,
    resolve_model_id,
    split_model_id,
    validate_provider_id,
)
from aeloon_core.version import __version__

PROMPT_LIMIT = 100_000
ATTACHMENT_LIMIT = 8
IMAGE_LIMIT = 10 * 1024 * 1024
FILE_LIMIT = 25 * 1024 * 1024
TOOL_OUTPUT_LIMIT = 20_000
EVENT_LIMIT = 5_000

EventListener = Callable[[dict[str, Any]], Awaitable[None] | None]
HarnessFactory = Callable[[Config, Session], AgentHarness | Awaitable[AgentHarness]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Operation:
    id: str
    session_id: str
    workspace: str
    kind: str
    input: dict[str, Any]
    created_at: str = field(default_factory=_now)
    status: str = "queued"
    blocks: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    task: asyncio.Task[None] | None = None
    harness: AgentHarness | None = None


@dataclass(slots=True)
class SessionRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operations: dict[str, Operation] = field(default_factory=dict)
    active: Operation | None = None


class CoreService:
    """Provider-neutral, transport-free application service for Core workflows."""

    def __init__(
        self,
        *,
        config_path: Path | str | None = None,
        data_dir: Path | str | None = None,
        max_concurrent_operations: int = 4,
        harness_factory: HarnessFactory | None = None,
        cloud_account_service: CloudAccountService | None = None,
    ) -> None:
        self.config_path = resolve_config_path(config_path).resolve(strict=False)
        self._data_dir_override = Path(data_dir).expanduser().resolve(strict=False) if data_dir is not None else None
        config = load_config(self.config_path)
        if self._data_dir_override is not None:
            config = config.model_copy(update={"data_dir": self._data_dir_override}).normalized()
        self.config = config
        self.data_dir = config.data_dir
        self.repository = JsonlSessionRepository(self.data_dir)
        self.attachment_dir = self.data_dir / "session-attachments"
        self.attachment_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.server_instance_id = uuid.uuid4().hex
        self.started_at = _now()
        self._revision = 1
        self._harness_factory = harness_factory
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_operations))
        self._runtimes: dict[str, SessionRuntime] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_LIMIT)
        self._seq = 0
        self._listeners: set[EventListener] = set()
        self._settings_lock = asyncio.Lock()
        self._owns_cloud_account = cloud_account_service is None
        self.cloud_account = cloud_account_service or CloudAccountService(
            config.cloud,
            data_dir=self.data_dir,
        )
        self.shutdown_requested = asyncio.Event()
        self.shutdown_signal = asyncio.Event()

    def add_event_listener(self, listener: EventListener) -> Callable[[], None]:
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
        routes: dict[str, Callable[..., Awaitable[Any]]] = {
            "system.handshake": self.handshake,
            "system.health": self.health,
            "system.shutdown": self.shutdown,
            "events.subscribe": self.events_subscribe,
            "session.create": self.session_create,
            "session.list": self.session_list,
            "session.get": self.session_get,
            "session.delete": self.session_delete,
            "session.rename": self.session_rename,
            "session.configure": self.session_configure,
            "session.tree": self.session_tree,
            "session.navigate": self.session_navigate,
            "session.compact": self.session_compact,
            "session.next_turn": self.session_next_turn,
            "turn.start": self.turn_start,
            "turn.cancel": self.turn_cancel,
            "turn.steer": self.turn_steer,
            "turn.follow_up": self.turn_follow_up,
            "catalog.get": self.catalog_get,
            "provider.list": self.provider_list,
            "provider.local.add": self.provider_local_add,
            "provider.local.remove": self.provider_local_remove,
            "settings.get": self.settings_get,
            "settings.update": self.settings_update,
            "cloud.account.status": self.cloud_account_status,
            "cloud.account.login": self.cloud_account_login,
            "cloud.account.logout": self.cloud_account_logout,
            "provider.cloud.status": self.cloud_account_status,
            "provider.cloud.login": self.cloud_account_login,
            "provider.cloud.logout": self.cloud_account_logout,
        }
        handler = routes.get(method)
        if handler is None:
            raise BridgeError("method_not_found", f"Unknown Bridge method: {method}")
        try:
            if method == "turn.start":
                return await handler(value, attachment_roots=attachment_roots)
            return await handler(value)
        except BridgeError:
            raise
        except SessionError as exc:
            code = "session_not_found" if exc.code == "not_found" else "invalid_state"
            raise BridgeError(code, self._sanitize(str(exc))) from None
        except HarnessError as exc:
            code = exc.code if exc.code in {"busy", "invalid_state", "invalid_argument"} else "internal_error"
            raise BridgeError(code, self._sanitize(str(exc))) from None
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeError("invalid_argument", self._sanitize(str(exc))) from None
        except Exception:
            raise BridgeError("internal_error", "Aeloon Core could not complete the request") from None

    async def handshake(self, params: Mapping[str, Any]) -> dict[str, Any]:
        versions = params.get("protocol_versions", [PROTOCOL_VERSION])
        if versions and PROTOCOL_VERSION not in versions:
            raise BridgeError("protocol_incompatible", "Aeloon Core requires Bridge protocol v2")
        roots = params.get("attachment_roots") or []
        if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
            raise BridgeError("invalid_argument", "attachment_roots must be a list of paths")
        return {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "core_version": __version__,
            "server_instance_id": self.server_instance_id,
            "capabilities": list(CAPABILITIES),
            "methods": list(METHODS),
            "events": list(EVENTS),
            "attachment_roots": [str(Path(root).expanduser().resolve(strict=False)) for root in roots],
            "config_path": str(self.config_path),
            "data_dir": str(self.data_dir),
            "limits": {
                "prompt_chars": PROMPT_LIMIT,
                "attachments": ATTACHMENT_LIMIT,
                "image_bytes": IMAGE_LIMIT,
                "file_bytes": FILE_LIMIT,
                "request_bytes": 1024 * 1024,
                "retained_events": EVENT_LIMIT,
            },
        }

    async def health(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        active = sum(1 for runtime in self._runtimes.values() if runtime.active is not None)
        return {
            "status": "stopping" if self.shutdown_requested.is_set() else "running",
            "pid": __import__("os").getpid(),
            "started_at": self.started_at,
            "active_operations": active,
            "current_seq": self._seq,
        }

    async def shutdown(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        if not self.shutdown_requested.is_set():
            await self._emit(
                "system.shutdown",
                None,
                None,
                {"intentional": True, "reason": "requested"},
            )
        self.shutdown_requested.set()
        asyncio.get_running_loop().call_later(0.05, self.shutdown_signal.set)
        return {"status": "stopping"}

    async def events_subscribe(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session_ids = params.get("session_ids") or []
        after_seq = int(params.get("after_seq") or 0)
        if not isinstance(session_ids, list) or any(not isinstance(item, str) for item in session_ids):
            raise BridgeError("invalid_argument", "session_ids must be a list")
        first_seq = self._events[0]["seq"] if self._events else self._seq + 1
        replay_complete = after_seq >= first_seq - 1
        replay = [
            event for event in self._events
            if event["seq"] > after_seq and (event["session_id"] in session_ids or event["session_id"] is None)
        ] if replay_complete else []
        return {
            "server_instance_id": self.server_instance_id,
            "current_seq": self._seq,
            "replay_complete": replay_complete,
            "events": replay,
            "cursor": {"server_instance_id": self.server_instance_id, "seq": self._seq},
        }

    async def session_create(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workspace = self._required_string(params, "workspace")
        session = await self.repository.create(
            cwd=workspace,
            metadata={"title": str(params.get("title") or "").strip() or None},
        )
        return await self._session_metadata(session)

    async def session_list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workspace = params.get("workspace")
        items = await self.repository.list(cwd=str(workspace) if workspace else None)
        sessions: list[dict[str, Any]] = []
        for item in items:
            session = await self.repository.open(item.id)
            sessions.append(
                {
                    "session_id": item.id,
                    "workspace": item.cwd,
                    "created_at": item.created_at,
                    "title": await session.get_name() or item.metadata.get("title"),
                    "schema_version": 3,
                }
            )
        return {"sessions": sessions}

    async def session_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        entries = await session.get_entries()
        branch = await session.get_branch()
        runtime = self._runtimes.get(session.id)
        active = []
        if runtime is not None:
            for operation in runtime.operations.values():
                if operation.status in {"queued", "active"}:
                    active.append(self._operation_dto(operation))
        overrides = self._session_overrides(entries)
        restored_model = (await session.build_context()).model
        model_id = str(
            overrides.get("model_id")
            or (restored_model[1] if restored_model is not None else None)
            or self.config.agent.model
        )
        context_window: int | None = None
        if model_id:
            try:
                context_window = (await self._model(model_id)).context_window
            except BridgeError:
                # A saved session remains readable when its provider or model is no
                # longer connected. Token totals are still useful without a limit.
                pass
        return {
            "metadata": await self._session_metadata(session),
            "state": {
                "leaf_id": await session.get_leaf_id(),
                "model_id": model_id,
                "thinking_level": overrides.get("thinking_level", self.config.agent.thinking_level),
                "active_tools": overrides.get("active_tools", list(DEFAULT_ACTIVE_TOOLS)),
            },
            "stats": await session.stats(context_window=context_window),
            "timeline": self._project_timeline(branch, active_ids={item["operation_id"] for item in active}),
            "active_operations": active,
        }

    async def session_delete(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        runtime = self._runtimes.get(session.id)
        if runtime and any(op.status in {"queued", "active"} for op in runtime.operations.values()):
            raise BridgeError("busy", "Cannot delete a session with active operations")
        await self.repository.delete(session.id)
        self._runtimes.pop(session.id, None)
        attachments = self.attachment_dir / session.id
        if attachments.exists():
            await asyncio.to_thread(shutil.rmtree, attachments)
        return {"session_id": session.id, "deleted": True}

    async def session_rename(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        title = str(params.get("title") or "").strip()
        await session.set_name(title or None)
        return {"session_id": session.id, "title": title or None}

    async def session_configure(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        runtime = self._runtime(session.id)
        patch: dict[str, Any] = {}
        if "model_id" in params:
            model_id = self._required_string(params, "model_id")
            await self._model(model_id)
            patch["model_id"] = model_id
        if "thinking_level" in params:
            level = self._thinking_level(params["thinking_level"])
            patch["thinking_level"] = level
        if "active_tools" in params:
            tools = params["active_tools"]
            if not isinstance(tools, list) or any(item not in ALL_TOOL_NAMES for item in tools):
                raise BridgeError("invalid_argument", "active_tools contains an unknown tool")
            patch["active_tools"] = list(dict.fromkeys(tools))
        if "steering_mode" in params:
            patch["steering_mode"] = self._queue_mode(params["steering_mode"])
        if "follow_up_mode" in params:
            patch["follow_up_mode"] = self._queue_mode(params["follow_up_mode"])
        async with runtime.lock:
            previous = self._session_overrides(await session.get_entries())
            merged = {**previous, **patch}
            await session.append_session_config(merged)
        return {"session_id": session.id, **merged}

    async def session_tree(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        entries = await session.get_entries()
        return {
            "session_id": session.id,
            "leaf_id": await session.get_leaf_id(),
            "nodes": [
                {
                    "id": str(entry["id"]),
                    "parent_id": entry.get("parentId"),
                    "type": str(entry.get("type")),
                    "time": str(entry.get("timestamp")),
                    "label": await session.get_label(str(entry["id"])),
                }
                for entry in entries
            ],
        }

    async def session_navigate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        target_id = self._required_string(params, "target_id")

        async def execute(harness: AgentHarness) -> Any:
            result = await harness.navigate_tree(
                target_id,
                summarize=bool(params.get("summarize", False)),
                custom_instructions=(str(params["custom_instructions"]) if params.get("custom_instructions") else None),
                replace_instructions=bool(params.get("replace_instructions", False)),
                label=(str(params["label"]) if params.get("label") else None),
            )
            await self._emit("session.navigated", session, None, result)
            return result

        return await self._session_operation(session, "navigate", execute)

    async def session_compact(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)

        async def execute(harness: AgentHarness) -> Any:
            result = await harness.compact(
                str(params["custom_instructions"]) if params.get("custom_instructions") else None
            )
            payload = {
                "summary": result.summary,
                "tokens_before": result.tokens_before,
                "first_kept_entry_id": result.first_kept_entry_id,
            }
            await self._emit("session.compacted", session, None, payload)
            return payload

        return await self._session_operation(session, "compact", execute)

    async def session_next_turn(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        input_value = self._prompt_input(params.get("input"))
        entry_id = await session.append_next_turn_input(input_value)
        return {"session_id": session.id, "entry_id": entry_id, "accepted": True}

    async def turn_start(
        self,
        params: Mapping[str, Any],
        *,
        attachment_roots: tuple[Path, ...] = (),
    ) -> dict[str, Any]:
        session = await self._session(params)
        input_value = self._turn_input(params.get("input"))
        if input_value["kind"] == "prompt":
            input_value["attachments"] = await self._copy_attachments(
                session.id,
                input_value.get("attachments") or [],
                attachment_roots,
            )
        runtime = self._runtime(session.id)
        operation = Operation(
            id=uuid.uuid4().hex,
            session_id=session.id,
            workspace=session.metadata.cwd,
            kind="turn",
            input=input_value,
        )
        overrides = self._session_overrides(await session.get_entries())
        await session.append_run_start(
            run_id=operation.id,
            input=self._public_input(input_value),
            model_id=str(overrides.get("model_id", self.config.agent.model)),
            thinking_level=str(overrides.get("thinking_level", self.config.agent.thinking_level)),
        )
        runtime.operations[operation.id] = operation
        queued = sum(1 for item in runtime.operations.values() if item.status == "queued")
        await self._emit("operation.queued", session, operation, {"kind": "turn", "queue_position": queued})
        await self._emit_queue(session, runtime)
        operation.task = asyncio.create_task(self._execute_turn(session, runtime, operation))
        return {"operation_id": operation.id, "turn_id": operation.id, "queue_position": queued}

    async def turn_cancel(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._operation(params)
        if operation.status == "queued" and operation.task is not None:
            operation.task.cancel()
        elif operation.status == "active" and operation.harness is not None:
            await operation.harness.abort()
        else:
            raise BridgeError("invalid_state", "Operation is no longer cancellable")
        return {"operation_id": operation.id, "cancelled": True}

    async def turn_steer(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._active_operation(params)
        text = self._required_string(params, "text")
        await operation.harness.steer(text)  # type: ignore[union-attr]
        return {"operation_id": operation.id, "accepted": True}

    async def turn_follow_up(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._active_operation(params)
        text = self._required_string(params, "text")
        await operation.harness.follow_up(text)  # type: ignore[union-attr]
        return {"operation_id": operation.id, "accepted": True}

    async def catalog_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workspace = self.config.workspace
        if params.get("session_id"):
            session = await self.repository.open(str(params["session_id"]))
            workspace = Path(session.metadata.cwd)
        loader = self._resource_loader(self.config.model_copy(update={"workspace": workspace}).normalized())
        resources = await asyncio.to_thread(loader.reload)
        models = await self._models()
        return {
            "providers": await self._provider_registry().providers(),
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "description": f"{model.context_window:,} token context",
                    "provider_id": model.provider,
                    "thinking_levels": ["off", "minimal", "low", "medium", "high", "max"],
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                }
                for model in models.values()
            ],
            "tools": [
                {"id": name, "name": name, "description": "Core-managed local tool"}
                for name in ALL_TOOL_NAMES
            ],
            "skills": [
                {"id": skill.name, "name": skill.name, "description": skill.description}
                for skill in resources.skills
            ],
            "prompt_templates": [
                {"id": item.name, "name": item.name, "description": item.description or ""}
                for item in resources.prompt_templates
            ],
        }

    async def settings_get(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        config = self.config
        return {
            "revision": self._revision,
            "config_path": str(self.config_path),
            "default_model_id": config.agent.model,
            "default_thinking_level": config.agent.thinking_level,
            "retry": config.agent.retry.model_dump(mode="json"),
            "compaction": config.agent.compaction.model_dump(mode="json"),
            "resources": {
                "roots": [str(root) for root in config.resources.roots],
                "load_skills": not config.resources.no_skills,
                "load_prompt_templates": not config.resources.no_prompt_templates,
                "load_context_files": not config.resources.no_context_files,
            },
            "tools": config.tools.model_dump(mode="json"),
            "deepseek": {
                "base_url": config.deepseek.base_url,
                "proxy": config.deepseek.proxy,
                "credential_configured": bool(config.deepseek.api_key and config.deepseek.api_key != "no-key"),
            },
            "local_providers": {
                provider_id: {
                    "name": provider.name,
                    "base_url": provider.base_url,
                    "proxy": provider.proxy,
                    "credential_configured": bool(
                        provider.api_key and provider.api_key != "no-key"
                    ),
                    "models": [model.model_dump(mode="json") for model in provider.models],
                }
                for provider_id, provider in config.local_providers.items()
            },
            "cloud": {
                "enabled": config.cloud.enabled,
                "base_url": config.cloud.base_url,
                "proxy": config.cloud.proxy,
                "device_name": config.cloud.device_name,
            },
        }

    async def settings_update(self, params: Mapping[str, Any]) -> dict[str, Any]:
        revision = params.get("revision")
        if not isinstance(revision, int):
            raise BridgeError("invalid_argument", "revision must be an integer")
        patch = params.get("patch") or {}
        actions = params.get("secret_actions") or []
        if not isinstance(patch, Mapping) or not isinstance(actions, list):
            raise BridgeError("invalid_argument", "patch and secret_actions are invalid")
        models = await self._models()
        valid_model_ids = [
            model.id
            for model in models.values()
            if model.provider != DEEPSEEK_PROVIDER_ID
            or self.config.deepseek.api_key != "no-key"
        ]
        valid_model_ids.extend(
            model_id for model_id in models if model_id not in valid_model_ids
        )
        if (
            "/" in self.config.agent.model
            and self.config.agent.model not in valid_model_ids
        ):
            valid_model_ids.append(self.config.agent.model)
        async with self._settings_lock:
            if revision != self._revision:
                raise BridgeError("revision_conflict", "Core settings changed; refresh and try again")
            raw = load_config(self.config_path).model_dump(mode="json")
            self._apply_settings_patch(raw, patch, valid_model_ids=valid_model_ids)
            for action in actions:
                if not isinstance(action, Mapping) or action.get("path") != "deepseek.api_key":
                    raise BridgeError("invalid_argument", "Unsupported secret action")
                if action.get("action") == "set":
                    value = str(action.get("value") or "")
                    if not value:
                        raise BridgeError("invalid_argument", "Secret set requires a value")
                    raw["deepseek"]["api_key"] = value
                elif action.get("action") == "clear":
                    raw["deepseek"]["api_key"] = "no-key"
                else:
                    raise BridgeError("invalid_argument", "Secret action must be set or clear")
            persisted_config = Config.model_validate(raw).normalized()
            await asyncio.to_thread(save_config, persisted_config, self.config_path)
            next_config = load_config(self.config_path)
            if self._data_dir_override is not None:
                next_config = next_config.model_copy(
                    update={"data_dir": self._data_dir_override}
                ).normalized()
            cloud_changed = next_config.cloud != self.config.cloud
            previous_cloud_base_url = self.config.cloud.base_url.rstrip("/")
            self.config = next_config
            if self._owns_cloud_account and cloud_changed:
                previous_cloud = self.cloud_account
                if next_config.cloud.base_url.rstrip("/") != previous_cloud_base_url:
                    previous_cloud.logout()
                self.cloud_account = CloudAccountService(
                    next_config.cloud,
                    data_dir=self.data_dir,
                )
                await previous_cloud.close()
            self._revision += 1
        await self._emit("settings.updated", None, None, {"revision": self._revision})
        return await self.settings_get({})

    async def close(self) -> None:
        tasks = [
            operation.task
            for runtime in self._runtimes.values()
            for operation in runtime.operations.values()
            if operation.task is not None and not operation.task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_cloud_account:
            await self.cloud_account.close()

    async def cloud_account_status(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return self.cloud_account.status()

    async def cloud_account_login(self, params: Mapping[str, Any]) -> dict[str, Any]:
        username = self._required_string(params, "username")
        password = self._required_string(params, "password")
        try:
            result = await self.cloud_account.login(username=username, password=password)
        except CloudError as exc:
            raise BridgeError("authentication_failed", self._sanitize(str(exc))) from None
        await self._emit("cloud.account.updated", None, None, result)
        await self._emit(
            "provider.updated",
            None,
            None,
            {"provider_id": "aeloon-cloud", "action": "login"},
        )
        return result

    async def cloud_account_logout(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        result = self.cloud_account.logout()
        await self._emit("cloud.account.updated", None, None, result)
        await self._emit(
            "provider.updated",
            None,
            None,
            {"provider_id": "aeloon-cloud", "action": "logout"},
        )
        return result

    async def provider_list(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return {"providers": await self._provider_registry().providers()}

    async def provider_local_add(self, params: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = validate_provider_id(self._required_string(params, "provider_id"))
        if provider_id in {DEEPSEEK_PROVIDER_ID, "aeloon-cloud"}:
            raise BridgeError("invalid_argument", f"Provider id is reserved: {provider_id}")
        base_url = self._http_base_url(self._required_string(params, "base_url"))
        name = str(params.get("name") or provider_id).strip() or provider_id
        api_key = str(params.get("api_key") or "no-key")
        proxy = str(params["proxy"]) if params.get("proxy") else None
        extra_headers = self._string_mapping(params.get("extra_headers"))
        raw_models = params.get("models")
        if raw_models is None:
            raw_models = await self._discover_local_models(
                base_url,
                api_key=api_key,
                proxy=proxy,
                extra_headers=extra_headers,
            )
        if not isinstance(raw_models, list) or not raw_models:
            raise BridgeError("invalid_argument", "models must be a non-empty list")
        models: list[LocalModelConfig] = []
        model_ids: set[str] = set()
        for raw_model in raw_models:
            value = {"id": raw_model} if isinstance(raw_model, str) else raw_model
            if not isinstance(value, Mapping):
                raise BridgeError("invalid_argument", "each model must be a string or object")
            parsed_model = LocalModelConfig.model_validate(dict(value))
            canonical = qualify_model_id(provider_id, parsed_model.id)
            if canonical in model_ids:
                raise BridgeError("invalid_argument", f"Duplicate model: {canonical}")
            model_ids.add(canonical)
            models.append(parsed_model.model_copy(update={"id": split_model_id(canonical)[1]}))
        provider = LocalProviderConfig(
            name=name,
            base_url=base_url,
            api_key=api_key,
            proxy=proxy,
            extra_headers=extra_headers,
            models=models,
        )
        revision = params.get("revision")
        async with self._settings_lock:
            if revision is not None and revision != self._revision:
                raise BridgeError("revision_conflict", "Core settings changed; refresh and try again")
            if provider_id in self.config.local_providers:
                raise BridgeError("invalid_argument", f"Provider already exists: {provider_id}")
            raw = load_config(self.config_path).model_dump(mode="json")
            raw["local_providers"][provider_id] = provider.model_dump(mode="json")
            await asyncio.to_thread(
                save_config, Config.model_validate(raw).normalized(), self.config_path
            )
            self._reload_config()
            self._revision += 1
        await self._emit(
            "provider.updated", None, None, {"provider_id": provider_id, "action": "added"}
        )
        await self._emit("settings.updated", None, None, {"revision": self._revision})
        return await self._provider_result(provider_id)

    async def provider_local_remove(self, params: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = validate_provider_id(self._required_string(params, "provider_id"))
        revision = params.get("revision")
        async with self._settings_lock:
            if revision is not None and revision != self._revision:
                raise BridgeError("revision_conflict", "Core settings changed; refresh and try again")
            if provider_id not in self.config.local_providers:
                raise BridgeError("invalid_argument", f"Unknown local provider: {provider_id}")
            raw = load_config(self.config_path).model_dump(mode="json")
            del raw["local_providers"][provider_id]
            fallback = self.config.agent.model
            try:
                if normalize_model_id(fallback).startswith(f"{provider_id}/"):
                    raw["agent"]["model"] = ""
            except ValueError:
                pass
            await asyncio.to_thread(
                save_config, Config.model_validate(raw).normalized(), self.config_path
            )
            self._reload_config()
            self._revision += 1
        await self._emit(
            "provider.updated", None, None, {"provider_id": provider_id, "action": "removed"}
        )
        await self._emit("settings.updated", None, None, {"revision": self._revision})
        return {"provider_id": provider_id, "removed": True, "revision": self._revision}

    async def _execute_turn(
        self, session: Session, runtime: SessionRuntime, operation: Operation
    ) -> None:
        started = time.monotonic()
        terminal_written = False
        try:
            async with runtime.lock, self._semaphore:
                operation.status = "active"
                runtime.active = operation
                await self._emit("operation.started", session, operation, {"kind": "turn"})
                await self._emit_queue(session, runtime)
                config_snapshot = self.config
                harness = await self._new_harness(config_snapshot, session)
                operation.harness = harness
                harness.subscribe(lambda event: self._harness_event(session, operation, event))
                pending = self._pending_next_turn(await session.get_entries())
                for _, item in pending:
                    await harness.next_turn(str(item.get("text") or ""))
                if pending:
                    await session.append_next_turn_consumed([entry_id for entry_id, _ in pending])
                result = await self._invoke_input(harness, operation.input)
                operation.usage = result.usage.to_dict()
                status = "cancelled" if result.stop_reason == "aborted" else "failed" if result.stop_reason == "error" else "completed"
                operation.status = status
                duration = round((time.monotonic() - started) * 1000)
                await session.append_run_end(
                    run_id=operation.id,
                    status=status,
                    duration_ms=duration,
                    error=self._sanitize(result.error_message or "") or None,
                )
                terminal_written = True
                name = f"operation.{status}"
                payload = {"kind": "turn", "duration_ms": duration}
                if status == "failed":
                    payload["error"] = self._sanitize(result.error_message or "Operation failed")
                await self._emit(name, session, operation, payload)
        except asyncio.CancelledError:
            operation.status = "cancelled"
            if not terminal_written:
                await session.append_run_end(run_id=operation.id, status="cancelled")
                await self._emit("operation.cancelled", session, operation, {"kind": "turn"})
        except Exception as exc:
            operation.status = "failed"
            if not terminal_written:
                safe = self._sanitize(str(exc))
                await session.append_run_end(run_id=operation.id, status="failed", error=safe)
                await self._emit("operation.failed", session, operation, {"kind": "turn", "error": safe})
        finally:
            if operation.harness is not None:
                await operation.harness.close()
            operation.harness = None
            if runtime.active is operation:
                runtime.active = None
            await self._emit_queue(session, runtime)

    async def _session_operation(
        self,
        session: Session,
        kind: str,
        execute: Callable[[AgentHarness], Awaitable[Any]],
    ) -> dict[str, Any]:
        runtime = self._runtime(session.id)
        operation = Operation(uuid.uuid4().hex, session.id, session.metadata.cwd, kind, {})
        runtime.operations[operation.id] = operation
        await self._emit("operation.queued", session, operation, {"kind": kind})
        async with runtime.lock, self._semaphore:
            operation.status = "active"
            runtime.active = operation
            await self._emit("operation.started", session, operation, {"kind": kind})
            harness = await self._new_harness(self.config, session)
            operation.harness = harness
            try:
                result = await execute(harness)
                operation.status = "completed"
                await self._emit("operation.completed", session, operation, {"kind": kind})
                return {"operation_id": operation.id, "result": result}
            except Exception as exc:
                operation.status = "failed"
                await self._emit("operation.failed", session, operation, {"kind": kind, "error": self._sanitize(str(exc))})
                raise
            finally:
                await harness.close()
                operation.harness = None
                runtime.active = None

    async def _new_harness(self, config: Config, session: Session) -> AgentHarness:
        effective = config.model_copy(update={"workspace": Path(session.metadata.cwd)}).normalized()
        overrides = self._session_overrides(await session.get_entries())
        if overrides.get("model_id") or overrides.get("thinking_level"):
            effective = effective.model_copy(
                update={
                    "agent": effective.agent.model_copy(
                        update={
                            "model": overrides.get("model_id", effective.agent.model),
                            "thinking_level": overrides.get("thinking_level", effective.agent.thinking_level),
                            "steering_mode": overrides.get("steering_mode", effective.agent.steering_mode),
                            "follow_up_mode": overrides.get("follow_up_mode", effective.agent.follow_up_mode),
                        }
                    )
                }
            )
        if self._harness_factory is not None:
            value = self._harness_factory(effective, session)
            return await value if inspect.isawaitable(value) else value
        registry = self._provider_registry(effective)
        model = await self._model(effective.agent.model, registry=registry)
        provider = registry.provider(model)
        active_tools = overrides.get("active_tools")
        return AgentHarness(
            provider=provider,
            model=model,
            cwd=str(effective.workspace),
            session=session,
            active_tool_names=active_tools or DEFAULT_ACTIVE_TOOLS,
            resource_loader=self._resource_loader(effective),
            thinking_level=effective.agent.thinking_level,
            stream_options=StreamOptions(
                timeout_ms=effective.agent.timeout_ms,
                max_tokens=effective.agent.max_tokens,
                temperature=effective.agent.temperature,
                thinking_level=effective.agent.thinking_level,
                max_retries=effective.agent.retry.max_retries if effective.agent.retry.enabled else 0,
                base_delay_ms=effective.agent.retry.base_delay_ms,
                max_retry_delay_ms=effective.agent.retry.max_retry_delay_ms,
            ),
            steering_mode=effective.agent.steering_mode,
            follow_up_mode=effective.agent.follow_up_mode,
            compaction=CompactionSettings(
                enabled=effective.agent.compaction.enabled,
                reserve_tokens=effective.agent.compaction.reserve_tokens,
                keep_recent_tokens=effective.agent.compaction.keep_recent_tokens,
            ),
            shell_path=effective.tools.shell_path,
            auto_resize_images=effective.tools.auto_resize_images,
        )

    async def _models(self) -> dict[str, Model]:
        return await self._provider_registry().models()

    async def _model(
        self, model_id: str, *, registry: UnifiedProviderRegistry | None = None
    ) -> Model:
        try:
            return await (registry or self._provider_registry()).model(model_id)
        except PermissionError as exc:
            raise BridgeError("authentication_failed", str(exc)) from None
        except RuntimeError as exc:
            raise BridgeError("invalid_state", str(exc)) from None
        except CloudError as exc:
            raise BridgeError("authentication_failed", self._sanitize(str(exc))) from None
        except KeyError:
            raise BridgeError("invalid_argument", f"Unknown model: {model_id}") from None

    def _provider_registry(self, config: Config | None = None) -> UnifiedProviderRegistry:
        return UnifiedProviderRegistry(config or self.config, self.cloud_account)

    async def _provider_result(self, provider_id: str) -> dict[str, Any]:
        providers = await self._provider_registry().providers()
        provider = next(item for item in providers if item["id"] == provider_id)
        return {"provider": provider, "revision": self._revision}

    async def _discover_local_models(
        self,
        base_url: str,
        *,
        api_key: str,
        proxy: str | None,
        extra_headers: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        headers = dict(extra_headers)
        if api_key and api_key != "no-key":
            headers.setdefault("authorization", f"Bearer {api_key}")
        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=15) as client:
                response = await client.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BridgeError(
                "invalid_argument",
                self._sanitize(f"Could not load models from the local API: {exc}"),
            ) from None
        if not isinstance(payload, Mapping):
            raise BridgeError("invalid_argument", "Local API /models returned an invalid payload")
        raw_models = payload.get("data", payload.get("models"))
        if not isinstance(raw_models, list):
            raise BridgeError("invalid_argument", "Local API /models returned no model list")
        models: list[dict[str, Any]] = []
        for raw in raw_models:
            if isinstance(raw, str):
                models.append({"id": raw})
                continue
            if not isinstance(raw, Mapping):
                continue
            model_id = str(
                raw.get("id")
                or raw.get("model")
                or raw.get("model_key")
                or raw.get("name")
                or ""
            ).strip()
            if not model_id:
                continue
            model: dict[str, Any] = {"id": model_id}
            display_name = raw.get("display_name") or raw.get("displayName")
            if display_name:
                model["name"] = str(display_name)
            for source, target in (
                ("context_window", "context_window"),
                ("contextWindow", "context_window"),
                ("max_tokens", "max_tokens"),
                ("maxTokens", "max_tokens"),
            ):
                if source in raw and target not in model:
                    model[target] = raw[source]
            model["reasoning"] = bool(
                raw.get("reasoning") or raw.get("supports_reasoning")
            )
            model["supports_image"] = bool(
                raw.get("supports_image") or raw.get("supportsImage")
            )
            models.append(model)
        if not models:
            raise BridgeError("invalid_argument", "Local API /models returned no usable models")
        return models

    def _reload_config(self) -> None:
        config = load_config(self.config_path)
        if self._data_dir_override is not None:
            config = config.model_copy(update={"data_dir": self._data_dir_override}).normalized()
        self.config = config

    @staticmethod
    def _http_base_url(value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BridgeError("invalid_argument", "base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise BridgeError(
                "invalid_argument",
                "base_url must not contain credentials, a query, or a fragment",
            )
        return value.rstrip("/")

    @staticmethod
    def _string_mapping(value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise BridgeError("invalid_argument", "extra_headers must be an object")
        result = {str(key): str(item) for key, item in value.items()}
        if any(not key.strip() for key in result):
            raise BridgeError("invalid_argument", "extra_headers contains an empty name")
        return result

    def _resource_loader(self, config: Config) -> ResourceLoader:
        return ResourceLoader(
            cwd=config.workspace,
            agent_dir=self.data_dir,
            additional_roots=tuple(config.resources.roots),
            no_skills=config.resources.no_skills,
            no_prompt_templates=config.resources.no_prompt_templates,
            no_context_files=config.resources.no_context_files,
        )

    async def _invoke_input(self, harness: AgentHarness, value: Mapping[str, Any]) -> Any:
        kind = value["kind"]
        if kind == "skill":
            return await harness.skill(str(value["name"]), value.get("additional_instructions"))
        if kind == "prompt_template":
            return await harness.prompt_template(str(value["name"]), tuple(value.get("arguments") or []))
        text = str(value.get("text") or "")
        images: list[ImageContent] = []
        supplements: list[str] = []
        for attachment in value.get("attachments") or []:
            if attachment["type"] == "assistant_selection":
                supplements.append(f"[Assistant selection]\n{attachment.get('text', '')}")
            elif attachment["type"] == "image":
                data = await asyncio.to_thread(Path(attachment["managed_path"]).read_bytes)
                images.append(ImageContent(base64.b64encode(data).decode(), attachment.get("mime_type") or "image/png"))
            elif attachment["type"] == "file":
                path = Path(attachment["managed_path"])
                try:
                    content = (await asyncio.to_thread(path.read_text, encoding="utf-8"))[:TOOL_OUTPUT_LIMIT]
                    supplements.append(f"[File: {attachment['name']}]\n{content}")
                except UnicodeError:
                    supplements.append(f"[Binary file attached: {attachment['name']}]")
        if supplements:
            text = "\n\n".join([text, *supplements])
        return await harness.prompt(text, images=tuple(images))

    async def _harness_event(self, session: Session, operation: Operation, event: HarnessEvent) -> None:
        data = event.data
        if event.type == "message_update":
            stream = data.get("assistantMessageEvent") or {}
            kind = stream.get("type")
            index = int(stream.get("contentIndex") or stream.get("toolCallIndex") or 0)
            if kind in {"text_delta", "thinking_delta"}:
                block_type = "text" if kind == "text_delta" else "thinking"
                block_id = f"{block_type}-{index}"
                block = next((item for item in operation.blocks if item["id"] == block_id), None)
                if block is None:
                    block = {"id": block_id, "type": block_type, "role": "narration" if block_type == "text" else None, "content": "", "status": "running"}
                    operation.blocks.append(block)
                    await self._emit("content.started", session, operation, {"block": self._clean_block(block)})
                delta = str(stream.get("delta") or "")
                block["content"] = f"{block.get('content', '')}{delta}"
                await self._emit("content.delta", session, operation, {"block_id": block_id, "delta": delta})
            elif kind == "toolcall_delta":
                block_id = str(stream.get("toolCallId") or f"tool-{index}")
                block = next((item for item in operation.blocks if item["id"] == block_id), None)
                if block is None:
                    block = {"id": block_id, "type": "tool_call", "name": str(stream.get("toolName") or "tool"), "arguments": {}, "status": "streaming"}
                    operation.blocks.append(block)
                    await self._emit("tool.started", session, operation, {"block": block})
                await self._emit("tool.updated", session, operation, {"block_id": block_id, "patch": {"status": "streaming"}})
        elif event.type == "message_end":
            message = data.get("message") or {}
            if message.get("role") == "assistant":
                self._merge_complete_message(operation, message)
                for block in operation.blocks:
                    name = "tool.updated" if block["type"] == "tool_call" else "content.completed"
                    await self._emit(name, session, operation, {"block_id": block["id"], "patch": self._clean_block(block)})
                usage = message.get("usage") or {}
                operation.usage = dict(usage) if isinstance(usage, Mapping) else {}
                context_window = (
                    operation.harness.model.context_window if operation.harness else None
                )
                await self._emit(
                    "usage.updated",
                    session,
                    operation,
                    {
                        "usage": operation.usage,
                        "stats": await session.stats(context_window=context_window),
                    },
                )
        elif event.type == "tool_execution_start":
            block_id = str(data.get("toolCallId") or "tool")
            block = next((item for item in operation.blocks if item["id"] == block_id), None)
            if block is None:
                block = {"id": block_id, "type": "tool_call", "name": str(data.get("toolName") or "tool"), "arguments": self._safe_mapping(data.get("args")), "status": "running"}
                operation.blocks.append(block)
                await self._emit("tool.started", session, operation, {"block": block})
            else:
                block.update({"arguments": self._safe_mapping(data.get("args")), "status": "running"})
                await self._emit("tool.updated", session, operation, {"block_id": block_id, "patch": self._clean_block(block)})
        elif event.type in {"tool_execution_update", "tool_execution_end"}:
            block_id = str(data.get("toolCallId") or "tool")
            raw = data.get("partialResult") if event.type.endswith("update") else data.get("result")
            result = self._tool_text(raw)
            patch = {"result": result, "status": "running" if event.type.endswith("update") else "failed" if data.get("isError") else "completed"}
            for block in operation.blocks:
                if block["id"] == block_id:
                    block.update(patch)
            await self._emit("tool.updated" if event.type.endswith("update") else "tool.completed", session, operation, {"block_id": block_id, "patch": patch})
        elif event.type == "queue_update":
            await self._emit("queue.updated", session, operation, {key: len(value) if isinstance(value, list) else value for key, value in data.items() if key.lower().endswith("count") or isinstance(value, list)})
        elif event.type == "auto_retry_start":
            await self._emit("retry.started", session, operation, self._safe_mapping(data))
        elif event.type == "auto_retry_end":
            await self._emit("retry.completed", session, operation, self._safe_mapping(data))
        elif event.type == "resources_update":
            await self._emit("resources.updated", session, operation, {"skills": len((data.get("resources") or {}).get("skills") or []), "prompt_templates": len((data.get("resources") or {}).get("promptTemplates") or [])})

    async def _emit(
        self,
        name: str,
        session: Session | None,
        operation: Operation | None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._seq += 1
        event = {
            "seq": self._seq,
            "time": _now(),
            "name": name,
            "workspace": session.metadata.cwd if session else None,
            "session_id": session.id if session else None,
            "operation_id": operation.id if operation else None,
            "payload": self._json_safe(dict(payload)),
        }
        self._events.append(event)
        for listener in tuple(self._listeners):
            try:
                result = listener(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue
        return event

    async def _emit_queue(self, session: Session, runtime: SessionRuntime) -> None:
        queued = [item.id for item in runtime.operations.values() if item.status == "queued"]
        await self._emit("queue.updated", session, runtime.active, {"queued_operation_ids": queued, "active_operation_id": runtime.active.id if runtime.active else None})

    async def _session(self, params: Mapping[str, Any]) -> Session:
        session_id = self._required_string(params, "session_id")
        return await self.repository.open(session_id)

    def _runtime(self, session_id: str) -> SessionRuntime:
        return self._runtimes.setdefault(session_id, SessionRuntime())

    def _operation(self, params: Mapping[str, Any]) -> Operation:
        operation_id = self._required_string(params, "operation_id")
        for runtime in self._runtimes.values():
            if operation_id in runtime.operations:
                return runtime.operations[operation_id]
        raise BridgeError("operation_not_found", f"Operation {operation_id} not found")

    def _active_operation(self, params: Mapping[str, Any]) -> Operation:
        operation = self._operation(params)
        if operation.status != "active" or operation.harness is None:
            raise BridgeError("invalid_state", "steer/follow_up requires the active turn")
        return operation

    async def _session_metadata(self, session: Session) -> dict[str, Any]:
        return {
            "session_id": session.id,
            "workspace": session.metadata.cwd,
            "created_at": session.metadata.created_at,
            "title": await session.get_name() or session.metadata.metadata.get("title"),
            "schema_version": 3,
        }

    def _turn_input(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise BridgeError("invalid_argument", "turn.start.input must be an object")
        kind = raw.get("kind")
        if kind == "prompt":
            return self._prompt_input(raw)
        if kind == "skill":
            return {"kind": "skill", "name": self._required_string(raw, "name"), "additional_instructions": str(raw.get("additional_instructions") or "") or None}
        if kind == "prompt_template":
            arguments = raw.get("arguments") or []
            if not isinstance(arguments, list) or any(not isinstance(item, str) for item in arguments):
                raise BridgeError("invalid_argument", "template arguments must be strings")
            return {"kind": "prompt_template", "name": self._required_string(raw, "name"), "arguments": arguments}
        raise BridgeError("invalid_argument", "input.kind must be prompt, skill, or prompt_template")

    def _prompt_input(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise BridgeError("invalid_argument", "input must be an object")
        text = str(raw.get("text") or "")
        if not text.strip() or len(text) > PROMPT_LIMIT:
            raise BridgeError("invalid_argument", f"Prompt must contain 1 to {PROMPT_LIMIT:,} characters")
        attachments = raw.get("attachments") or []
        if not isinstance(attachments, list) or len(attachments) > ATTACHMENT_LIMIT:
            raise BridgeError("invalid_attachment", f"At most {ATTACHMENT_LIMIT} attachments are allowed")
        return {"kind": "prompt", "text": text, "attachments": attachments}

    async def _copy_attachments(
        self,
        session_id: str,
        attachments: list[Any],
        roots: tuple[Path, ...],
    ) -> list[dict[str, Any]]:
        destination = self.attachment_dir / session_id
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved_roots = tuple(root.expanduser().resolve(strict=False) for root in roots)
        result: list[dict[str, Any]] = []
        for raw in attachments:
            if not isinstance(raw, Mapping):
                raise BridgeError("invalid_attachment", "Attachment must be an object")
            kind = str(raw.get("type") or "")
            if kind == "assistant_selection":
                text = str(raw.get("text") or "")[:PROMPT_LIMIT]
                result.append({"id": str(raw.get("id") or uuid.uuid4().hex), "type": kind, "name": str(raw.get("name") or "Assistant selection")[:255], "text": text})
                continue
            if kind not in {"image", "file"}:
                raise BridgeError("invalid_attachment", f"Unsupported attachment type: {kind}")
            source_raw = raw.get("path")
            if not isinstance(source_raw, str):
                raise BridgeError("invalid_attachment", "File attachment path is required")
            try:
                source = Path(source_raw).expanduser().resolve(strict=True)
            except OSError:
                raise BridgeError("invalid_attachment", "Attachment source does not exist") from None
            if not source.is_file() or not any(source.is_relative_to(root) for root in resolved_roots):
                raise BridgeError("invalid_attachment", "Attachment is outside the declared roots")
            size = source.stat().st_size
            maximum = IMAGE_LIMIT if kind == "image" else FILE_LIMIT
            if size > maximum:
                raise BridgeError("invalid_attachment", f"Attachment exceeds the {maximum // (1024 * 1024)} MiB limit")
            name = Path(str(raw.get("name") or source.name)).name[:255]
            target = destination / f"{uuid.uuid4().hex}{source.suffix[:20]}"
            await asyncio.to_thread(shutil.copy2, source, target)
            target.chmod(0o600)
            result.append({
                "id": str(raw.get("id") or uuid.uuid4().hex),
                "type": kind,
                "name": name,
                "mime_type": str(raw.get("mime_type") or "application/octet-stream"),
                "size_bytes": size,
                "managed_path": str(target),
            })
        return result

    def _public_input(self, value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        result["attachments"] = [
            {key: item[key] for key in ("id", "type", "name", "mime_type", "size_bytes", "text") if key in item}
            for item in value.get("attachments") or []
        ]
        return result

    def _operation_dto(self, operation: Operation) -> dict[str, Any]:
        return {
            "operation_id": operation.id,
            "kind": operation.kind,
            "status": operation.status,
            "input": self._public_input(operation.input),
            "blocks": [self._clean_block(block) for block in operation.blocks],
            "usage": operation.usage,
            "created_at": operation.created_at,
        }

    def _project_timeline(self, entries: list[dict[str, Any]], *, active_ids: set[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for entry in entries:
            kind = entry.get("type")
            if kind == "run_start":
                if current is not None:
                    current["status"] = "interrupted" if current["turn_id"] not in active_ids else "active"
                    result.append(current)
                current = {
                    "type": "turn",
                    "turn_id": str(entry.get("runId")),
                    "status": "active" if str(entry.get("runId")) in active_ids else "interrupted",
                    "input": self._public_input(entry.get("input") or {}),
                    "blocks": [],
                    "usage": {},
                    "model_id": str(entry.get("modelId") or self.config.agent.model),
                    "thinking_level": str(entry.get("thinkingLevel") or self.config.agent.thinking_level),
                    "created_at": str(entry.get("timestamp")),
                    "completed_at": None,
                    "duration_ms": None,
                    "final_content": None,
                    "error": None,
                }
            elif kind == "message" and current is not None:
                message = entry.get("message") or {}
                if message.get("role") == "assistant":
                    blocks = self._message_blocks(message)
                    current["blocks"].extend(blocks)
                    current["usage"] = message.get("usage") or {}
                    texts = [block.get("content", "") for block in blocks if block["type"] == "text"]
                    if texts:
                        current["final_content"] = "".join(texts)
                elif message.get("role") == "toolResult":
                    tool_id = str(message.get("toolCallId") or "")
                    for block in current["blocks"]:
                        if block["id"] == tool_id:
                            block.update({"status": "failed" if message.get("isError") else "completed", "result": self._content_text(message.get("content"))})
            elif kind == "run_end" and current is not None and entry.get("runId") == current["turn_id"]:
                current.update({
                    "status": str(entry.get("status") or "completed"),
                    "completed_at": str(entry.get("timestamp")),
                    "duration_ms": entry.get("durationMs"),
                    "error": self._sanitize(str(entry.get("error") or "")) or None,
                })
                result.append(current)
                current = None
            elif kind == "compaction":
                result.append({
                    "type": "compaction",
                    "id": str(entry.get("id")),
                    "created_at": str(entry.get("timestamp")),
                    "tokens_before": int(entry.get("tokensBefore") or 0),
                    "summary": str(entry.get("summary") or "")[:TOOL_OUTPUT_LIMIT],
                })
        if current is not None:
            result.append(current)
        return result

    def _message_blocks(self, message: Mapping[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        content = message.get("content") or []
        for index, raw in enumerate(content):
            if not isinstance(raw, Mapping):
                continue
            kind = raw.get("type")
            if kind == "text":
                result.append({"id": f"text-{len(result)}", "type": "text", "role": "final", "content": str(raw.get("text") or ""), "status": "completed"})
            elif kind == "thinking":
                result.append({"id": f"thinking-{len(result)}", "type": "thinking", "content": str(raw.get("thinking") or ""), "status": "completed"})
            elif kind == "toolCall":
                result.append({"id": str(raw.get("id") or f"tool-{index}"), "type": "tool_call", "name": str(raw.get("name") or "tool"), "arguments": self._safe_mapping(raw.get("arguments")), "status": "running"})
        return result

    def _merge_complete_message(self, operation: Operation, message: Mapping[str, Any]) -> None:
        complete = self._message_blocks(message)
        for block in complete:
            existing = next((item for item in operation.blocks if item["id"] == block["id"]), None)
            if existing is None:
                operation.blocks.append(block)
            else:
                existing.update(block)

    def _pending_next_turn(self, entries: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        pending: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if entry.get("type") == "next_turn_input":
                pending[str(entry["id"])] = dict(entry.get("input") or {})
            elif entry.get("type") == "next_turn_consumed":
                for entry_id in entry.get("entryIds") or []:
                    pending.pop(str(entry_id), None)
        return list(pending.items())

    def _session_overrides(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for entry in entries:
            if entry.get("type") == "session_config" and isinstance(entry.get("config"), Mapping):
                value = dict(entry["config"])
        return value

    def _apply_settings_patch(
        self,
        raw: dict[str, Any],
        patch: Mapping[str, Any],
        *,
        valid_model_ids: list[str],
    ) -> None:
        allowed = {"default_model_id", "default_thinking_level", "retry", "compaction", "resources", "tools", "deepseek", "cloud"}
        unknown = set(patch) - allowed
        if unknown:
            raise BridgeError("invalid_argument", f"Unknown settings fields: {', '.join(sorted(unknown))}")
        if "default_model_id" in patch:
            requested_model_id = str(patch["default_model_id"])
            try:
                model_id = resolve_model_id(requested_model_id, valid_model_ids)
            except KeyError:
                raise BridgeError(
                    "invalid_argument", f"Unknown model: {requested_model_id}"
                ) from None
            raw["agent"]["model"] = model_id
        if "default_thinking_level" in patch:
            raw["agent"]["thinking_level"] = self._thinking_level(patch["default_thinking_level"])
        for key in ("retry", "compaction"):
            if key in patch:
                if not isinstance(patch[key], Mapping):
                    raise BridgeError("invalid_argument", f"{key} must be an object")
                raw["agent"][key].update(patch[key])
        if "resources" in patch:
            value = patch["resources"]
            if not isinstance(value, Mapping):
                raise BridgeError("invalid_argument", "resources must be an object")
            raw["resources"].update({
                "roots": value.get("roots", raw["resources"]["roots"]),
                "no_skills": not bool(value.get("load_skills", not raw["resources"]["no_skills"])),
                "no_prompt_templates": not bool(value.get("load_prompt_templates", not raw["resources"]["no_prompt_templates"])),
                "no_context_files": not bool(value.get("load_context_files", not raw["resources"]["no_context_files"])),
            })
        for key in ("tools", "deepseek", "cloud"):
            if key in patch:
                if not isinstance(patch[key], Mapping):
                    raise BridgeError("invalid_argument", f"{key} must be an object")
                raw[key].update(patch[key])

    def _clean_block(self, block: Mapping[str, Any]) -> dict[str, Any]:
        return {key: self._json_safe(value) for key, value in block.items() if value is not None}

    def _tool_text(self, raw: Any) -> str:
        if not isinstance(raw, Mapping):
            return ""
        return self._content_text(raw.get("content"))[:TOOL_OUTPUT_LIMIT]

    def _content_text(self, raw: Any) -> str:
        if not isinstance(raw, list):
            return ""
        return "\n".join(str(item.get("text") or "") for item in raw if isinstance(item, Mapping))[:TOOL_OUTPUT_LIMIT]

    def _safe_mapping(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): self._json_safe(item) for key, item in value.items()}

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {str(key): self._json_safe(item) for key, item in value.items() if str(key).lower() not in {"api_key", "authorization", "systemprompt", "payload"}}
        if isinstance(value, list | tuple):
            return [self._json_safe(item) for item in value]
        return str(value)[:TOOL_OUTPUT_LIMIT]

    def _sanitize(self, message: str) -> str:
        value = message[:4_000]
        secrets = [
            self.config.deepseek.api_key,
            *(provider.api_key for provider in self.config.local_providers.values()),
        ]
        for secret in secrets:
            if secret and secret != "no-key":
                value = value.replace(secret, "***")
        return value

    @staticmethod
    def _required_string(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BridgeError("invalid_argument", f"{key} is required")
        return value.strip()

    @staticmethod
    def _thinking_level(value: Any) -> str:
        level = str(value)
        if level not in {"off", "minimal", "low", "medium", "high", "max"}:
            raise BridgeError("invalid_argument", "Invalid thinking level")
        return level

    @staticmethod
    def _queue_mode(value: Any) -> str:
        mode = str(value)
        if mode not in {"all", "one-at-a-time"}:
            raise BridgeError("invalid_argument", "Invalid queue mode")
        return mode


__all__ = ["CoreService"]
