# ruff: noqa: E501
"""Provider-neutral application runtime for sessions and agent operations."""

from __future__ import annotations

import asyncio
import base64
import inspect
import re
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeloon_core.config import (
    CloudProviderConfig,
    Config,
    CustomProviderConfig,
    DeepSeekProviderConfig,
    ProviderModelConfig,
    load_config,
    redact_sensitive_headers,
    resolve_config_path,
    save_config,
)
from aeloon_core.core import (
    ImageContent,
    InferenceError,
    Model,
    RunError,
    RunEvent,
    StreamOptions,
)
from aeloon_core.runtime.agent import SessionAgent, SessionAgentFactory
from aeloon_core.runtime.artifacts import (
    PRESENT_FILES_TOOL_NAME,
    artifacts_from_tool_result,
)
from aeloon_core.runtime.builtin_skills import provision_builtin_skills
from aeloon_core.runtime.coordinator import (
    Operation,
    OperationCoordinator,
    SessionRuntime,
)
from aeloon_core.runtime.input import TurnInputResolver
from aeloon_core.runtime.ports import AccountGateway, NullAccountGateway
from aeloon_core.runtime.projection import RuntimeProjection
from aeloon_core.runtime.providers import (
    ProviderManager,
    ProviderManagerFactory,
    resolve_model_id,
    split_model_id,
    validate_provider_id,
)
from aeloon_core.runtime.providers import (
    provider_manager_factory as default_provider_manager_factory,
)
from aeloon_core.runtime.rename import is_generic_session_title, rename_session
from aeloon_core.runtime.resources import ResourceLoader
from aeloon_core.runtime.session import JsonlSessionRepository, Session
from aeloon_core.runtime.tooling import RuntimeToolSet
from aeloon_core.runtime.types import (
    OperationSnapshot,
    RuntimeEvent,
    RuntimeEventListener,
    RuntimeFailure,
    SessionInfo,
    SessionSnapshot,
    TurnInput,
)

PROMPT_LIMIT = 100_000
ATTACHMENT_LIMIT = 8
IMAGE_LIMIT = 10 * 1024 * 1024
FILE_LIMIT = 25 * 1024 * 1024
TOOL_OUTPUT_LIMIT = 20_000
RuntimeFailure = RuntimeFailure


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RuntimeService:
    """Provider-neutral, transport-free application service for runtime workflows."""

    def __init__(
        self,
        *,
        config_path: Path | str | None = None,
        data_dir: Path | str | None = None,
        max_concurrent_operations: int = 4,
        agent_factory: SessionAgentFactory | None = None,
        provider_manager_factory: ProviderManagerFactory | None = None,
        account_gateway: AccountGateway | None = None,
    ) -> None:
        self.config_path = resolve_config_path(config_path).resolve(strict=False)
        self._data_dir_override = (
            Path(data_dir).expanduser().resolve(strict=False) if data_dir is not None else None
        )
        config = load_config(self.config_path)
        if self._data_dir_override is not None:
            config = config.model_copy(update={"data_dir": self._data_dir_override}).normalized()
        self.config = config
        self.data_dir = config.data_dir
        provision_builtin_skills(self.data_dir)
        self.repository = JsonlSessionRepository(self.data_dir)
        self.attachment_dir = self.data_dir / "session-attachments"
        self.attachment_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.started_at = _now()
        self._revision = 1
        self._agent_factory = agent_factory
        self._listeners: set[RuntimeEventListener] = set()
        self._settings_lock = asyncio.Lock()
        self.account = account_gateway or NullAccountGateway()
        self._provider_manager_factory = provider_manager_factory or (
            default_provider_manager_factory(account=self.account)
        )
        self.coordinator = OperationCoordinator(
            max_concurrent_operations=max_concurrent_operations,
            provider_manager_factory=self._provider_manager_factory,
        )
        self._semaphore = self.coordinator.semaphore
        self._runtimes = self.coordinator.runtimes
        self.input_resolver = TurnInputResolver(
            prompt_limit=PROMPT_LIMIT,
            attachment_limit=ATTACHMENT_LIMIT,
        )
        self.projection = RuntimeProjection(
            lambda: self.config,
            output_limit=TOOL_OUTPUT_LIMIT,
        )

    def add_event_listener(self, listener: RuntimeEventListener) -> Callable[[], None]:
        self._listeners.add(listener)

        def remove() -> None:
            self._listeners.discard(listener)

        return remove

    @property
    def active_operation_count(self) -> int:
        return self.coordinator.active_count

    async def create_session(self, *, workspace: str, title: str | None = None) -> SessionInfo:
        value = await self.session_create({"workspace": workspace, "title": title})
        return SessionInfo.from_dict(value)

    async def list_sessions(self, *, workspace: str | None = None) -> tuple[SessionInfo, ...]:
        value = await self.session_list({"workspace": workspace})
        return tuple(SessionInfo.from_dict(item) for item in value["sessions"])

    async def get_session(self, session_id: str) -> SessionSnapshot:
        value = await self.session_get({"session_id": session_id})
        return SessionSnapshot(
            metadata=dict(value["metadata"]),
            state=dict(value["state"]),
            stats=dict(value["stats"]),
            timeline=tuple(value["timeline"]),
            active_operations=tuple(value["active_operations"]),
        )

    async def start_turn(
        self,
        *,
        session_id: str,
        input: TurnInput,
        attachment_roots: tuple[Path, ...] = (),
    ) -> OperationSnapshot:
        value = await self.turn_start(
            {"session_id": session_id, "input": input.to_dict()},
            attachment_roots=attachment_roots,
        )
        return OperationSnapshot(
            operation_id=str(value["operation_id"]),
            turn_id=str(value["turn_id"]),
            queue_position=int(value["queue_position"]),
            skill_id=str(value["skill_id"]) if value.get("skill_id") else None,
        )

    async def cancel_turn(self, operation_id: str) -> None:
        await self.turn_cancel({"operation_id": operation_id})

    async def steer_turn(self, operation_id: str, text: str) -> None:
        await self.turn_steer({"operation_id": operation_id, "text": text})

    async def follow_up_turn(self, operation_id: str, text: str) -> None:
        await self.turn_follow_up({"operation_id": operation_id, "text": text})

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
            except RuntimeFailure:
                # A saved session remains readable when its provider or model is no
                # longer connected. Token totals are still useful without a limit.
                pass
        return {
            "metadata": await self._session_metadata(session),
            "state": {
                "leaf_id": await session.get_leaf_id(),
                "model_id": model_id,
                "thinking_level": overrides.get("thinking_level", self.config.agent.thinking_level),
                "active_tools": list(
                    self._tool_set(workspace=Path(session.metadata.cwd)).active_names(
                        None,
                        tuple(overrides["active_tools"]) if "active_tools" in overrides else None,
                    )
                ),
            },
            "stats": await session.stats(context_window=context_window),
            "timeline": self._project_timeline(
                branch, active_ids={item["operation_id"] for item in active}
            ),
            "active_operations": active,
        }

    async def session_delete(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)
        runtime = self._runtimes.get(session.id)
        if runtime and any(op.status in {"queued", "active"} for op in runtime.operations.values()):
            raise RuntimeFailure("busy", "Cannot delete a session with active operations")
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
        await self._emit(
            "session.renamed",
            session,
            None,
            {"title": title or None, "source": "manual"},
        )
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
            known_tools = self._tool_set(workspace=Path(session.metadata.cwd)).all_names
            if not isinstance(tools, list) or any(item not in known_tools for item in tools):
                raise RuntimeFailure("invalid_argument", "active_tools contains an unknown tool")
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

        async def execute(agent: SessionAgent) -> Any:
            result = await agent.navigate_tree(
                target_id,
                summarize=bool(params.get("summarize", False)),
                custom_instructions=(
                    str(params["custom_instructions"])
                    if params.get("custom_instructions")
                    else None
                ),
                replace_instructions=bool(params.get("replace_instructions", False)),
                label=(str(params["label"]) if params.get("label") else None),
            )
            await self._emit("session.navigated", session, None, result)
            return result

        return await self._session_operation(session, "navigate", execute)

    async def session_compact(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._session(params)

        async def execute(agent: SessionAgent) -> Any:
            result = await agent.compact(
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
        input_value = await self._resolve_skill_command(session, input_value)
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
        queued_payload: dict[str, Any] = {"kind": "turn", "queue_position": queued}
        if input_value.get("skill_id"):
            queued_payload["skill_id"] = input_value["skill_id"]
        await self._emit("operation.queued", session, operation, queued_payload)
        await self._emit_queue(session, runtime)
        operation.task = asyncio.create_task(self._execute_turn(session, runtime, operation))
        result = {
            "operation_id": operation.id,
            "turn_id": operation.id,
            "queue_position": queued,
        }
        if input_value.get("skill_id"):
            result["skill_id"] = input_value["skill_id"]
        return result

    async def turn_cancel(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._operation(params)
        if operation.status == "queued" and operation.task is not None:
            operation.task.cancel()
        elif operation.status == "active" and operation.agent is not None:
            await operation.agent.abort()
        else:
            raise RuntimeFailure("invalid_state", "Operation is no longer cancellable")
        return {"operation_id": operation.id, "cancelled": True}

    async def turn_steer(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._active_operation(params)
        text = self._required_string(params, "text")
        await operation.agent.steer(text)  # type: ignore[union-attr]
        return {"operation_id": operation.id, "accepted": True}

    async def turn_follow_up(self, params: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._active_operation(params)
        text = self._required_string(params, "text")
        await operation.agent.follow_up(text)  # type: ignore[union-attr]
        return {"operation_id": operation.id, "accepted": True}

    async def catalog_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        workspace = (
            Path(str(params["workspace"])).expanduser().resolve(strict=False)
            if params.get("workspace")
            else self.config.workspace
        )
        if params.get("session_id"):
            session = await self.repository.open(str(params["session_id"]))
            workspace = Path(session.metadata.cwd)
        loader = self._resource_loader(
            self.config.model_copy(update={"workspace": workspace}).normalized()
        )
        resources = await asyncio.to_thread(loader.reload)
        enabled_skill_ids = {skill.name for skill in resources.skills}
        selected_skill_ids = self._selected_skill_ids(loader)
        manager = self._provider_manager()
        try:
            models = await manager.models()
            providers = await manager.providers()
        finally:
            await manager.close()
        tool_set = self._tool_set(workspace=workspace)
        return {
            "providers": providers,
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "description": f"{model.context_window:,} token context",
                    "provider_id": model.provider,
                    "thinking_levels": ["off", "minimal", "low", "medium", "high", "max"],
                    "supports_image": "image" in model.input,
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                }
                for model in models.values()
            ],
            "tools": [
                {
                    "id": name,
                    "name": name,
                    "description": (
                        "Runtime-managed final deliverable tool"
                        if name == PRESENT_FILES_TOOL_NAME
                        else "Runtime-managed local tool"
                    ),
                }
                for name in sorted(tool_set.all_names)
            ],
            "skills": [
                {
                    "id": skill.name,
                    "name": skill.name,
                    "description": skill.description,
                    "command": f"/{skill.name}",
                    "source": skill.source,
                    "location": skill.file_path,
                    "selected": skill.name in selected_skill_ids,
                    "enabled": skill.name in enabled_skill_ids,
                    "explicit_invocation_enabled": skill.name in enabled_skill_ids,
                    "model_invocation_enabled": (
                        skill.name in enabled_skill_ids and not skill.disable_model_invocation
                    ),
                    "content_loading": "on_demand",
                }
                for skill in loader.available_skills
            ],
            "prompt_templates": [
                {"id": item.name, "name": item.name, "description": item.description or ""}
                for item in resources.prompt_templates
            ],
        }

    async def settings_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        config = self.config
        workspace = (
            Path(str(params["workspace"])).expanduser().resolve(strict=False)
            if params.get("workspace")
            else config.workspace
        )
        loader = self._resource_loader(
            config.model_copy(update={"workspace": workspace}).normalized()
        )
        await asyncio.to_thread(loader.reload)
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
                "enabled_skill_ids": sorted(self._selected_skill_ids(loader)),
                "load_prompt_templates": not config.resources.no_prompt_templates,
                "load_context_files": not config.resources.no_context_files,
            },
            "tools": config.tools.model_dump(mode="json"),
            "providers": {
                provider_id: self._public_provider_config(provider)
                for provider_id, provider in config.providers.items()
            },
        }

    async def settings_update(self, params: Mapping[str, Any]) -> dict[str, Any]:
        revision = params.get("revision")
        if not isinstance(revision, int):
            raise RuntimeFailure("invalid_argument", "revision must be an integer")
        patch = params.get("patch") or {}
        actions = params.get("secret_actions") or []
        if not isinstance(patch, Mapping) or not isinstance(actions, list):
            raise RuntimeFailure("invalid_argument", "patch and secret_actions are invalid")
        models = await self._models()
        valid_model_ids = list(models)
        if "/" in self.config.agent.model and self.config.agent.model not in valid_model_ids:
            valid_model_ids.append(self.config.agent.model)
        async with self._settings_lock:
            if revision != self._revision:
                raise RuntimeFailure(
                    "revision_conflict", "Core settings changed; refresh and try again"
                )
            raw = load_config(self.config_path).model_dump(mode="json")
            self._apply_settings_patch(raw, patch, valid_model_ids=valid_model_ids)
            for action in actions:
                if not isinstance(action, Mapping):
                    raise RuntimeFailure("invalid_argument", "Unsupported secret action")
                match = re.fullmatch(
                    r"providers\.([A-Za-z0-9][A-Za-z0-9._-]*)\.api_key",
                    str(action.get("path") or ""),
                )
                if match is None:
                    raise RuntimeFailure("invalid_argument", "Unsupported secret action")
                provider_id = match.group(1)
                provider = raw["providers"].get(provider_id)
                if not isinstance(provider, dict) or "api_key" not in provider:
                    raise RuntimeFailure("invalid_argument", "Provider does not support API keys")
                if action.get("action") == "set":
                    value = str(action.get("value") or "")
                    if not value:
                        raise RuntimeFailure("invalid_argument", "Secret set requires a value")
                    provider["api_key"] = value
                elif action.get("action") == "clear":
                    provider["api_key"] = None
                else:
                    raise RuntimeFailure("invalid_argument", "Secret action must be set or clear")
            persisted_config = Config.model_validate(raw).normalized()
            resource_patch = patch.get("resources")
            if isinstance(resource_patch, Mapping) and "enabled_skill_ids" in resource_patch:
                validation_config = persisted_config
                if params.get("workspace"):
                    validation_config = persisted_config.model_copy(
                        update={
                            "workspace": Path(str(params["workspace"]))
                            .expanduser()
                            .resolve(strict=False)
                        }
                    ).normalized()
                loader = self._resource_loader(validation_config)
                await asyncio.to_thread(loader.reload)
                known_skill_ids = {skill.name for skill in loader.available_skills}
                requested_skill_ids = set(persisted_config.resources.enabled_skills or ())
                unknown_skill_ids = requested_skill_ids - known_skill_ids
                if unknown_skill_ids:
                    raise RuntimeFailure(
                        "invalid_argument",
                        "Unknown skill ids: " + ", ".join(sorted(unknown_skill_ids)),
                    )
            await asyncio.to_thread(save_config, persisted_config, self.config_path)
            next_config = load_config(self.config_path)
            if self._data_dir_override is not None:
                next_config = next_config.model_copy(
                    update={"data_dir": self._data_dir_override}
                ).normalized()
            next_cloud = next_config.providers["aeloon-cloud"]
            current_cloud = self.config.providers["aeloon-cloud"]
            account_changed = next_cloud != current_cloud
            self.config = next_config
            if account_changed:
                assert isinstance(next_cloud, CloudProviderConfig)
                await self.account.configure(next_cloud)
            self._revision += 1
        await self._emit("settings.updated", None, None, {"revision": self._revision})
        return await self.settings_get(
            {"workspace": params["workspace"]} if params.get("workspace") else {}
        )

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
        await self.account.close()

    async def account_status(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return self.account.status()

    async def account_login(self, params: Mapping[str, Any]) -> dict[str, Any]:
        username = self._required_string(params, "username")
        password = self._required_string(params, "password")
        try:
            result = await self.account.login(username=username, password=password)
        except Exception as exc:
            raise RuntimeFailure("authentication_failed", self._sanitize(str(exc))) from None
        await self._emit("cloud.account.updated", None, None, result)
        await self._emit(
            "provider.updated",
            None,
            None,
            {"provider_id": "aeloon-cloud", "action": "login"},
        )
        return result

    async def account_logout(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        result = self.account.logout()
        await self._emit("cloud.account.updated", None, None, result)
        await self._emit(
            "provider.updated",
            None,
            None,
            {"provider_id": "aeloon-cloud", "action": "logout"},
        )
        return result

    async def provider_list(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        manager = self._provider_manager()
        try:
            return {"providers": await manager.providers()}
        finally:
            await manager.close()

    async def provider_refresh(self, params: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = validate_provider_id(self._required_string(params, "provider_id"))
        configured = self.config.providers.get(provider_id)
        if configured is None:
            raise RuntimeFailure("invalid_argument", f"Unknown Provider: {provider_id}")
        manager = self._provider_manager()
        try:
            discovered = await manager.discover_models(provider_id)
            resolved_endpoint = manager.provider_endpoint(provider_id)
        except PermissionError as exc:
            raise RuntimeFailure("authentication_failed", self._sanitize(str(exc))) from None
        except Exception as exc:
            raise RuntimeFailure(
                "invalid_argument",
                self._sanitize(f"Could not refresh Provider models: {exc}"),
            ) from None
        finally:
            await manager.close()

        if isinstance(configured, (CustomProviderConfig, DeepSeekProviderConfig)):
            models = [
                ProviderModelConfig(
                    id=split_model_id(model.id)[1],
                    name=model.name,
                    reasoning=model.reasoning,
                    supports_image="image" in model.input,
                    context_window=model.context_window,
                    max_tokens=model.max_tokens,
                    cost=model.cost,
                )
                for model in discovered
            ]
            revision = params.get("revision")
            async with self._settings_lock:
                if revision is not None and revision != self._revision:
                    raise RuntimeFailure(
                        "revision_conflict", "Core settings changed; refresh and try again"
                    )
                raw = load_config(self.config_path).model_dump(mode="json")
                raw["providers"][provider_id].update(
                    {
                        "endpoint": resolved_endpoint,
                        "models": [model.model_dump(mode="json") for model in models],
                    }
                )
                current_default = str(raw["agent"].get("model") or "")
                if current_default.startswith(f"{provider_id}/"):
                    refreshed_ids = {model.id for model in discovered}
                    if current_default not in refreshed_ids:
                        raw["agent"]["model"] = discovered[0].id if discovered else ""
                await asyncio.to_thread(
                    save_config, Config.model_validate(raw).normalized(), self.config_path
                )
                self._reload_config()
                self._revision += 1
            await self._emit("settings.updated", None, None, {"revision": self._revision})

        await self._emit(
            "provider.updated", None, None, {"provider_id": provider_id, "action": "refreshed"}
        )
        result = await self._provider_result(provider_id)
        result["provider"]["model_ids"] = [model.id for model in discovered]
        return result

    async def provider_add(self, params: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = validate_provider_id(self._required_string(params, "provider_id"))
        if provider_id in {"deepseek", "aeloon-cloud"}:
            raise RuntimeFailure("invalid_argument", f"Provider id is reserved: {provider_id}")
        endpoint = self._http_endpoint(self._required_string(params, "endpoint"))
        name = str(params.get("name") or provider_id).strip() or provider_id
        api_key = str(params["api_key"]) if params.get("api_key") else None
        proxy = str(params["proxy"]) if params.get("proxy") else None
        headers = self._string_mapping(params.get("headers"))
        provider = CustomProviderConfig(
            name=name,
            endpoint=endpoint,
            api_key=api_key,
            proxy=proxy,
            headers=headers,
        )
        candidate = self.config.model_copy(
            update={"providers": {**self.config.providers, provider_id: provider}}
        ).normalized()
        manager = self._provider_manager(candidate)
        try:
            discovered = await manager.discover_models(provider_id)
            resolved_endpoint = manager.provider_endpoint(provider_id)
        except Exception as exc:
            raise RuntimeFailure(
                "invalid_argument",
                self._sanitize(f"Could not discover Provider models: {exc}"),
            ) from None
        finally:
            await manager.close()
        models = [
            ProviderModelConfig(
                id=split_model_id(model.id)[1],
                name=model.name,
                reasoning=model.reasoning,
                supports_image="image" in model.input,
                context_window=model.context_window,
                max_tokens=model.max_tokens,
                cost=model.cost,
            )
            for model in discovered
        ]
        provider = provider.model_copy(
            update={"endpoint": resolved_endpoint, "models": models}
        )
        revision = params.get("revision")
        async with self._settings_lock:
            if revision is not None and revision != self._revision:
                raise RuntimeFailure(
                    "revision_conflict", "Core settings changed; refresh and try again"
                )
            if provider_id in self.config.providers:
                raise RuntimeFailure("invalid_argument", f"Provider already exists: {provider_id}")
            raw = load_config(self.config_path).model_dump(mode="json")
            raw["providers"][provider_id] = provider.model_dump(mode="json")
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

    async def provider_remove(self, params: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = validate_provider_id(self._required_string(params, "provider_id"))
        if provider_id in {"deepseek", "aeloon-cloud"}:
            raise RuntimeFailure("invalid_argument", f"Provider id is reserved: {provider_id}")
        revision = params.get("revision")
        async with self._settings_lock:
            if revision is not None and revision != self._revision:
                raise RuntimeFailure(
                    "revision_conflict", "Core settings changed; refresh and try again"
                )
            if provider_id not in self.config.providers:
                raise RuntimeFailure("invalid_argument", f"Unknown Provider: {provider_id}")
            raw = load_config(self.config_path).model_dump(mode="json")
            del raw["providers"][provider_id]
            fallback = self.config.agent.model
            try:
                if split_model_id(fallback)[0] == provider_id:
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
                config_snapshot = self.coordinator.snapshot(self.config)
                agent = await self._new_agent(config_snapshot, session)
                operation.agent = agent
                await agent.prepare()
                operation.model = agent.model
                agent.subscribe(lambda event: self._run_event(session, operation, event))
                pending = self._pending_next_turn(await session.get_entries())
                for _, item in pending:
                    await agent.next_turn(str(item.get("text") or ""))
                if pending:
                    await session.append_next_turn_consumed([entry_id for entry_id, _ in pending])
                result = await self._invoke_input(agent, operation.input, run_id=operation.id)
                operation.usage = result.usage.to_dict()
                status = (
                    "cancelled"
                    if result.stop_reason == "aborted"
                    else "failed"
                    if result.stop_reason == "error"
                    else "completed"
                )
                operation.status = status
                duration = round((time.monotonic() - started) * 1000)
                await session.append_run_end(
                    run_id=operation.id,
                    status=status,
                    duration_ms=duration,
                    error=self._sanitize(result.final_message.error_message or "") or None,
                )
                terminal_written = True
                name = f"operation.{status}"
                payload = {"kind": "turn", "duration_ms": duration}
                if status == "failed":
                    payload["error"] = self._sanitize(
                        result.final_message.error_message or "Operation failed"
                    )
                await self._emit(name, session, operation, payload)
                if status == "completed" and await self._should_auto_rename(session, operation):
                    try:
                        title = await rename_session(
                            session=session,
                            inference=agent.inference,  # type: ignore[arg-type]
                            model=agent.model,  # type: ignore[arg-type]
                            user_prompt=str(operation.input.get("text") or ""),
                            assistant_text=result.final_message.text,
                            stream_options=StreamOptions(
                                timeout_ms=config_snapshot.agent.timeout_ms,
                                max_retries=(
                                    config_snapshot.agent.retry.max_retries
                                    if config_snapshot.agent.retry.enabled
                                    else 0
                                ),
                                base_delay_ms=config_snapshot.agent.retry.base_delay_ms,
                                max_retry_delay_ms=config_snapshot.agent.retry.max_retry_delay_ms,
                            ),
                        )
                        if title:
                            await self._emit(
                                "session.renamed",
                                session,
                                operation,
                                {"title": title, "source": "automatic"},
                            )
                    except Exception:
                        # Naming is best-effort metadata and must never change a
                        # successfully completed user operation into a failure.
                        pass
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
                await self._emit(
                    "operation.failed", session, operation, {"kind": "turn", "error": safe}
                )
        finally:
            if operation.agent is not None:
                await operation.agent.close()
            operation.agent = None
            operation.model = None
            if runtime.active is operation:
                runtime.active = None
            await self._emit_queue(session, runtime)

    async def _session_operation(
        self,
        session: Session,
        kind: str,
        execute: Callable[[SessionAgent], Awaitable[Any]],
    ) -> dict[str, Any]:
        runtime = self._runtime(session.id)
        operation = Operation(uuid.uuid4().hex, session.id, session.metadata.cwd, kind, {})
        runtime.operations[operation.id] = operation
        await self._emit("operation.queued", session, operation, {"kind": kind})
        async with runtime.lock, self._semaphore:
            operation.status = "active"
            runtime.active = operation
            await self._emit("operation.started", session, operation, {"kind": kind})
            agent = await self._new_agent(self.config, session)
            operation.agent = agent
            try:
                result = await execute(agent)
                operation.status = "completed"
                await self._emit("operation.completed", session, operation, {"kind": kind})
                return {"operation_id": operation.id, "result": result}
            except Exception as exc:
                operation.status = "failed"
                await self._emit(
                    "operation.failed",
                    session,
                    operation,
                    {"kind": kind, "error": self._sanitize(str(exc))},
                )
                if isinstance(exc, RuntimeFailure):
                    raise
                if isinstance(exc, InferenceError | RunError):
                    raise RuntimeFailure(exc.code, str(exc), cause=exc) from None
                raise
            finally:
                await agent.close()
                operation.agent = None
                runtime.active = None

    async def _new_agent(self, config: Config, session: Session) -> SessionAgent:
        effective = config.model_copy(update={"workspace": Path(session.metadata.cwd)}).normalized()
        overrides = self._session_overrides(await session.get_entries())
        if overrides.get("model_id") or overrides.get("thinking_level"):
            effective = effective.model_copy(
                update={
                    "agent": effective.agent.model_copy(
                        update={
                            "model": overrides.get("model_id", effective.agent.model),
                            "thinking_level": overrides.get(
                                "thinking_level", effective.agent.thinking_level
                            ),
                            "steering_mode": overrides.get(
                                "steering_mode", effective.agent.steering_mode
                            ),
                            "follow_up_mode": overrides.get(
                                "follow_up_mode", effective.agent.follow_up_mode
                            ),
                        }
                    )
                }
            )
        manager = self._provider_manager(effective)
        if self._agent_factory is not None:
            try:
                value = self._agent_factory(effective, session, manager)
                return await value if inspect.isawaitable(value) else value
            except Exception:
                await manager.close()
                raise
        active_tools = overrides.get("active_tools")
        return SessionAgent(
            config=effective,
            session=session,
            provider_manager=manager,
            active_tool_names=(tuple(active_tools) if active_tools is not None else None),
        )

    async def _should_auto_rename(self, session: Session, operation: Operation) -> bool:
        if operation.input.get("kind") != "prompt":
            return False
        entries = await session.get_entries()
        run_ids = [
            str(entry.get("runId") or "") for entry in entries if entry.get("type") == "run_start"
        ]
        if run_ids != [operation.id]:
            return False
        title = await session.get_name() or session.metadata.metadata.get("title")
        return is_generic_session_title(str(title or ""))

    async def _models(self) -> dict[str, Model]:
        manager = self._provider_manager()
        try:
            return await manager.models()
        finally:
            await manager.close()

    async def _model(self, model_id: str, *, manager: ProviderManager | None = None) -> Model:
        owns_manager = manager is None
        selected = manager or self._provider_manager()
        try:
            return await selected.model(model_id)
        except PermissionError as exc:
            raise RuntimeFailure("authentication_failed", str(exc)) from None
        except RuntimeError as exc:
            raise RuntimeFailure("invalid_state", str(exc)) from None
        except KeyError:
            raise RuntimeFailure("invalid_argument", f"Unknown model: {model_id}") from None
        finally:
            if owns_manager:
                await selected.close()

    def _provider_manager(self, config: Config | None = None) -> ProviderManager:
        return self.coordinator.provider_manager(config or self.config)

    async def _provider_result(self, provider_id: str) -> dict[str, Any]:
        manager = self._provider_manager()
        try:
            providers = await manager.providers()
            provider = next(item for item in providers if item["id"] == provider_id)
            return {"provider": provider, "revision": self._revision}
        finally:
            await manager.close()

    def _reload_config(self) -> None:
        config = load_config(self.config_path)
        if self._data_dir_override is not None:
            config = config.model_copy(update={"data_dir": self._data_dir_override}).normalized()
        self.config = config

    @staticmethod
    def _http_endpoint(value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeFailure("invalid_argument", "endpoint must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeFailure(
                "invalid_argument",
                "endpoint must not contain credentials, a query, or a fragment",
            )
        return value.rstrip("/")

    @staticmethod
    def _string_mapping(value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise RuntimeFailure("invalid_argument", "headers must be an object")
        result = {str(key): str(item) for key, item in value.items()}
        if any(not key.strip() for key in result):
            raise RuntimeFailure("invalid_argument", "headers contains an empty name")
        return result

    def _resource_loader(self, config: Config) -> ResourceLoader:
        return ResourceLoader(
            cwd=config.workspace,
            agent_dir=self.data_dir,
            additional_roots=tuple(config.resources.roots),
            enabled_skills=(
                tuple(config.resources.enabled_skills)
                if config.resources.enabled_skills is not None
                else None
            ),
            no_skills=config.resources.no_skills,
            no_prompt_templates=config.resources.no_prompt_templates,
            no_context_files=config.resources.no_context_files,
        )

    @staticmethod
    def _selected_skill_ids(loader: ResourceLoader) -> set[str]:
        available = {skill.name for skill in loader.available_skills}
        if loader.enabled_skills is None:
            return available
        return available.intersection(loader.enabled_skills)

    async def _resolve_skill_command(
        self,
        session: Session,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = await self.input_resolver.resolve_slash_skill(
            session=session,
            value=value,
            config=self.config,
            resource_loader=self._resource_loader,
        )
        return prepared.input

    async def _invoke_input(
        self,
        agent: SessionAgent,
        value: Mapping[str, Any],
        *,
        run_id: str,
    ) -> Any:
        kind = value["kind"]
        if kind == "skill":
            return await agent.skill(
                str(value["name"]),
                value.get("additional_instructions"),
                run_id=run_id,
            )
        if kind == "prompt_template":
            return await agent.prompt_template(
                str(value["name"]),
                tuple(value.get("arguments") or []),
                run_id=run_id,
            )
        text = str(value.get("text") or "")
        images: list[ImageContent] = []
        supplements: list[str] = []
        for attachment in value.get("attachments") or []:
            if attachment["type"] == "assistant_selection":
                supplements.append(f"[Assistant selection]\n{attachment.get('text', '')}")
            elif attachment["type"] == "image":
                data = await asyncio.to_thread(Path(attachment["managed_path"]).read_bytes)
                images.append(
                    ImageContent(
                        base64.b64encode(data).decode(), attachment.get("mime_type") or "image/png"
                    )
                )
            elif attachment["type"] == "file":
                path = Path(attachment["managed_path"])
                try:
                    content = (await asyncio.to_thread(path.read_text, encoding="utf-8"))[
                        :TOOL_OUTPUT_LIMIT
                    ]
                    supplements.append(f"[File: {attachment['name']}]\n{content}")
                except UnicodeError:
                    supplements.append(f"[Binary file attached: {attachment['name']}]")
        if value.get("skill_id"):
            instructions = str(value.get("_skill_instructions") or "")
            if supplements:
                instructions = "\n\n".join(item for item in (instructions, *supplements) if item)
            return await agent.skill(
                str(value["skill_id"]),
                instructions or None,
                images=tuple(images),
                run_id=run_id,
            )
        if supplements:
            text = "\n\n".join([text, *supplements])
        return await agent.prompt(text, images=tuple(images), run_id=run_id)

    async def _run_event(
        self,
        session: Session,
        operation: Operation,
        event: RunEvent,
    ) -> None:
        data = event.data
        if event.type == "message_update":
            stream = data.get("assistantMessageEvent") or {}
            attempt = int(stream.get("attempt") or 0)
            kind = stream.get("type")
            index = int(stream.get("contentIndex") or stream.get("toolCallIndex") or 0)
            if kind in {"text_delta", "thinking_delta"}:
                block_type = "text" if kind == "text_delta" else "thinking"
                block_id = self._attempt_block_id(f"{block_type}-{index}", attempt)
                block = next((item for item in operation.blocks if item["id"] == block_id), None)
                if block is None:
                    block = {
                        "id": block_id,
                        "type": block_type,
                        "role": "narration" if block_type == "text" else None,
                        "content": "",
                        "status": "running",
                        "_attempt": attempt,
                    }
                    operation.blocks.append(block)
                    await self._emit(
                        "content.started", session, operation, {"block": self._clean_block(block)}
                    )
                delta = str(stream.get("delta") or "")
                block["content"] = f"{block.get('content', '')}{delta}"
                await self._emit(
                    "content.delta", session, operation, {"block_id": block_id, "delta": delta}
                )
            elif kind == "toolcall_delta":
                source_id = str(stream.get("toolCallId") or f"tool-{index}")
                block_id = self._attempt_block_id(source_id, attempt)
                block = next((item for item in operation.blocks if item["id"] == block_id), None)
                if block is None:
                    block = {
                        "id": block_id,
                        "type": "tool_call",
                        "name": str(stream.get("toolName") or "tool"),
                        "arguments": {},
                        "status": "streaming",
                        "_attempt": attempt,
                        "_source_id": source_id,
                    }
                    operation.blocks.append(block)
                    await self._emit(
                        "tool.started",
                        session,
                        operation,
                        {"block": self._clean_block(block)},
                    )
                await self._emit(
                    "tool.updated",
                    session,
                    operation,
                    {"block_id": block_id, "patch": {"status": "streaming"}},
                )
        elif event.type == "message_end":
            message = data.get("message") or {}
            if message.get("role") == "assistant":
                attempt = int(data.get("attempt") or 0)
                self._merge_complete_message(operation, message, attempt=attempt)
                attempt_blocks = [
                    block
                    for block in operation.blocks
                    if int(block.get("_attempt") or 0) == attempt
                ]
                failed = message.get("stopReason") in {"error", "aborted"}
                for block in attempt_blocks:
                    if failed:
                        block["status"] = "failed"
                    name = "tool.updated" if block["type"] == "tool_call" else "content.completed"
                    await self._emit(
                        name,
                        session,
                        operation,
                        {"block_id": block["id"], "patch": self._clean_block(block)},
                    )
                usage = message.get("usage") or {}
                operation.usage = dict(usage) if isinstance(usage, Mapping) else {}
                context_window = operation.model.context_window if operation.model else None
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
            source_id = str(data.get("toolCallId") or "tool")
            block = self._find_tool_block(operation, source_id)
            if block is None:
                block_id = source_id
                block = {
                    "id": block_id,
                    "type": "tool_call",
                    "name": str(data.get("toolName") or "tool"),
                    "arguments": self._safe_mapping(data.get("args")),
                    "status": "running",
                    "_attempt": self._latest_attempt(operation),
                    "_source_id": source_id,
                }
                operation.blocks.append(block)
                await self._emit(
                    "tool.started",
                    session,
                    operation,
                    {"block": self._clean_block(block)},
                )
            else:
                block_id = str(block["id"])
                block.update(
                    {"arguments": self._safe_mapping(data.get("args")), "status": "running"}
                )
                await self._emit(
                    "tool.updated",
                    session,
                    operation,
                    {"block_id": block_id, "patch": self._clean_block(block)},
                )
        elif event.type in {"tool_execution_update", "tool_execution_end"}:
            source_id = str(data.get("toolCallId") or "tool")
            block = self._find_tool_block(operation, source_id)
            block_id = str(block["id"]) if block is not None else source_id
            raw = data.get("partialResult") if event.type.endswith("update") else data.get("result")
            result = self._tool_text(raw)
            patch = {
                "result": result,
                "status": "running"
                if event.type.endswith("update")
                else "failed"
                if data.get("isError")
                else "completed",
            }
            typed_artifacts = (
                artifacts_from_tool_result(raw)
                if event.type == "tool_execution_end"
                and data.get("toolName") == PRESENT_FILES_TOOL_NAME
                and not data.get("isError")
                else []
            )
            artifacts = [artifact.to_dict() for artifact in typed_artifacts]
            if artifacts:
                patch["artifacts"] = artifacts
            if block is not None:
                block.update(patch)
            if artifacts:
                await session.append_artifact_delivery(
                    run_id=operation.id,
                    tool_call_id=source_id,
                    artifacts=artifacts,
                )
            await self._emit(
                "tool.updated" if event.type.endswith("update") else "tool.completed",
                session,
                operation,
                {"block_id": block_id, "patch": patch},
            )
        elif event.type == "queue_update":
            await self._emit(
                "queue.updated",
                session,
                operation,
                {
                    key: len(value) if isinstance(value, list) else value
                    for key, value in data.items()
                    if key.lower().endswith("count") or isinstance(value, list)
                },
            )
        elif event.type == "auto_retry_start":
            await self._emit("retry.started", session, operation, self._safe_mapping(data))
        elif event.type == "auto_retry_end":
            await self._emit("retry.completed", session, operation, self._safe_mapping(data))
        elif event.type == "context_compacted":
            await self._emit(
                "session.compacted",
                session,
                operation,
                self._safe_mapping(data),
            )
        elif event.type == "compaction_end" and data.get("error"):
            await self._emit(
                "session.compacted",
                session,
                operation,
                {**self._safe_mapping(data), "succeeded": False},
            )

    async def _emit(
        self,
        name: str,
        session: Session | None,
        operation: Operation | None,
        payload: Mapping[str, Any],
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            time=_now(),
            name=name,
            workspace=session.metadata.cwd if session else None,
            session_id=session.id if session else None,
            operation_id=operation.id if operation else None,
            payload=self._json_safe(dict(payload)),
        )
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
        await self._emit(
            "queue.updated",
            session,
            runtime.active,
            {
                "queued_operation_ids": queued,
                "active_operation_id": runtime.active.id if runtime.active else None,
            },
        )

    async def _session(self, params: Mapping[str, Any]) -> Session:
        session_id = self._required_string(params, "session_id")
        return await self.repository.open(session_id)

    def _runtime(self, session_id: str) -> SessionRuntime:
        return self.coordinator.runtime(session_id)

    def _operation(self, params: Mapping[str, Any]) -> Operation:
        operation_id = self._required_string(params, "operation_id")
        for runtime in self._runtimes.values():
            if operation_id in runtime.operations:
                return runtime.operations[operation_id]
        raise RuntimeFailure("operation_not_found", f"Operation {operation_id} not found")

    def _active_operation(self, params: Mapping[str, Any]) -> Operation:
        operation = self._operation(params)
        if operation.status != "active" or operation.agent is None:
            raise RuntimeFailure("invalid_state", "steer/follow_up requires the active turn")
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
        return self.input_resolver.parse(raw)

    def _prompt_input(self, raw: Any) -> dict[str, Any]:
        return self.input_resolver.prompt(raw)

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
                raise RuntimeFailure("invalid_attachment", "Attachment must be an object")
            kind = str(raw.get("type") or "")
            if kind == "assistant_selection":
                text = str(raw.get("text") or "")[:PROMPT_LIMIT]
                result.append(
                    {
                        "id": str(raw.get("id") or uuid.uuid4().hex),
                        "type": kind,
                        "name": str(raw.get("name") or "Assistant selection")[:255],
                        "text": text,
                    }
                )
                continue
            if kind not in {"image", "file"}:
                raise RuntimeFailure("invalid_attachment", f"Unsupported attachment type: {kind}")
            source_raw = raw.get("path")
            if not isinstance(source_raw, str):
                raise RuntimeFailure("invalid_attachment", "File attachment path is required")
            try:
                source = Path(source_raw).expanduser().resolve(strict=True)
            except OSError:
                raise RuntimeFailure(
                    "invalid_attachment", "Attachment source does not exist"
                ) from None
            if not source.is_file() or not any(
                source.is_relative_to(root) for root in resolved_roots
            ):
                raise RuntimeFailure(
                    "invalid_attachment", "Attachment is outside the declared roots"
                )
            size = source.stat().st_size
            maximum = IMAGE_LIMIT if kind == "image" else FILE_LIMIT
            if size > maximum:
                raise RuntimeFailure(
                    "invalid_attachment",
                    f"Attachment exceeds the {maximum // (1024 * 1024)} MiB limit",
                )
            name = Path(str(raw.get("name") or source.name)).name[:255]
            target = destination / f"{uuid.uuid4().hex}{source.suffix[:20]}"
            await asyncio.to_thread(shutil.copy2, source, target)
            target.chmod(0o600)
            result.append(
                {
                    "id": str(raw.get("id") or uuid.uuid4().hex),
                    "type": kind,
                    "name": name,
                    "mime_type": str(raw.get("mime_type") or "application/octet-stream"),
                    "size_bytes": size,
                    "managed_path": str(target),
                }
            )
        return result

    def _public_input(self, value: Mapping[str, Any]) -> dict[str, Any]:
        result = {key: item for key, item in value.items() if not key.startswith("_")}
        result["attachments"] = [
            {
                key: item[key]
                for key in ("id", "type", "name", "mime_type", "size_bytes", "text")
                if key in item
            }
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

    def _project_timeline(
        self, entries: list[dict[str, Any]], *, active_ids: set[str]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for entry in entries:
            kind = entry.get("type")
            if kind == "run_start":
                if current is not None:
                    current["status"] = (
                        "interrupted" if current["turn_id"] not in active_ids else "active"
                    )
                    result.append(current)
                current = {
                    "type": "turn",
                    "turn_id": str(entry.get("runId")),
                    "status": "active" if str(entry.get("runId")) in active_ids else "interrupted",
                    "input": self._public_input(entry.get("input") or {}),
                    "blocks": [],
                    "usage": {},
                    "model_id": str(entry.get("modelId") or self.config.agent.model),
                    "thinking_level": str(
                        entry.get("thinkingLevel") or self.config.agent.thinking_level
                    ),
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
                    texts = [
                        block.get("content", "") for block in blocks if block["type"] == "text"
                    ]
                    if texts:
                        current["final_content"] = "".join(texts)
                elif message.get("role") == "toolResult":
                    tool_id = str(message.get("toolCallId") or "")
                    for block in current["blocks"]:
                        if block["id"] == tool_id:
                            block.update(
                                {
                                    "status": "failed" if message.get("isError") else "completed",
                                    "result": self._content_text(message.get("content")),
                                }
                            )
            elif kind == "artifact_delivery" and current is not None:
                if str(entry.get("runId") or "") != current["turn_id"]:
                    continue
                tool_id = str(entry.get("toolCallId") or "")
                artifacts = entry.get("artifacts")
                if not isinstance(artifacts, list):
                    continue
                for block in current["blocks"]:
                    if block["id"] == tool_id:
                        block["artifacts"] = artifacts
            elif (
                kind == "run_end"
                and current is not None
                and entry.get("runId") == current["turn_id"]
            ):
                current.update(
                    {
                        "status": str(entry.get("status") or "completed"),
                        "completed_at": str(entry.get("timestamp")),
                        "duration_ms": entry.get("durationMs"),
                        "error": self._sanitize(str(entry.get("error") or "")) or None,
                    }
                )
                result.append(current)
                current = None
            elif kind == "compaction":
                result.append(
                    {
                        "type": "compaction",
                        "id": str(entry.get("id")),
                        "created_at": str(entry.get("timestamp")),
                        "tokens_before": int(entry.get("tokensBefore") or 0),
                        "summary": str(entry.get("summary") or "")[:TOOL_OUTPUT_LIMIT],
                    }
                )
        if current is not None:
            result.append(current)
        return result

    def _message_blocks(
        self,
        message: Mapping[str, Any],
        *,
        attempt: int = 0,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        type_indices = {"text": 0, "thinking": 0}
        content = message.get("content") or []
        for index, raw in enumerate(content):
            if not isinstance(raw, Mapping):
                continue
            kind = raw.get("type")
            if kind == "text":
                source_id = f"text-{type_indices['text']}"
                type_indices["text"] += 1
                result.append(
                    {
                        "id": self._attempt_block_id(source_id, attempt),
                        "type": "text",
                        "role": "final",
                        "content": str(raw.get("text") or ""),
                        "status": "completed",
                        "_attempt": attempt,
                    }
                )
            elif kind == "thinking":
                source_id = f"thinking-{type_indices['thinking']}"
                type_indices["thinking"] += 1
                result.append(
                    {
                        "id": self._attempt_block_id(source_id, attempt),
                        "type": "thinking",
                        "content": str(raw.get("thinking") or ""),
                        "status": "completed",
                        "_attempt": attempt,
                    }
                )
            elif kind == "toolCall":
                source_id = str(raw.get("id") or f"tool-{index}")
                result.append(
                    {
                        "id": self._attempt_block_id(source_id, attempt),
                        "type": "tool_call",
                        "name": str(raw.get("name") or "tool"),
                        "arguments": self._safe_mapping(raw.get("arguments")),
                        "status": "running",
                        "_attempt": attempt,
                        "_source_id": source_id,
                    }
                )
        error_message = str(message.get("errorMessage") or "").strip()
        if message.get("stopReason") in {"error", "aborted"} and error_message:
            result.append(
                {
                    "id": self._attempt_block_id("error-0", attempt),
                    "type": "text",
                    "role": "narration",
                    "content": error_message,
                    "status": "failed",
                    "_attempt": attempt,
                }
            )
        return result

    def _merge_complete_message(
        self,
        operation: Operation,
        message: Mapping[str, Any],
        *,
        attempt: int = 0,
    ) -> None:
        complete = self._message_blocks(message, attempt=attempt)
        for block in complete:
            existing = next((item for item in operation.blocks if item["id"] == block["id"]), None)
            if existing is None:
                operation.blocks.append(block)
            else:
                existing.update(block)

    def _find_tool_block(self, operation: Operation, source_id: str) -> dict[str, Any] | None:
        return next(
            (
                block
                for block in reversed(operation.blocks)
                if block.get("type") == "tool_call"
                and (block.get("_source_id") == source_id or block.get("id") == source_id)
            ),
            None,
        )

    def _latest_attempt(self, operation: Operation) -> int:
        return max((int(block.get("_attempt") or 0) for block in operation.blocks), default=0)

    def _attempt_block_id(self, source_id: str, attempt: int) -> str:
        return source_id if attempt <= 0 else f"attempt-{attempt}-{source_id}"

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
        allowed = {
            "default_model_id",
            "default_thinking_level",
            "retry",
            "compaction",
            "resources",
            "tools",
            "providers",
        }
        unknown = set(patch) - allowed
        if unknown:
            raise RuntimeFailure(
                "invalid_argument", f"Unknown settings fields: {', '.join(sorted(unknown))}"
            )
        if "default_model_id" in patch:
            requested_model_id = str(patch["default_model_id"])
            try:
                model_id = resolve_model_id(requested_model_id, valid_model_ids)
            except KeyError:
                raise RuntimeFailure(
                    "invalid_argument", f"Unknown model: {requested_model_id}"
                ) from None
            raw["agent"]["model"] = model_id
        if "default_thinking_level" in patch:
            raw["agent"]["thinking_level"] = self._thinking_level(patch["default_thinking_level"])
        for key in ("retry", "compaction"):
            if key in patch:
                if not isinstance(patch[key], Mapping):
                    raise RuntimeFailure("invalid_argument", f"{key} must be an object")
                raw["agent"][key].update(patch[key])
        if "resources" in patch:
            value = patch["resources"]
            if not isinstance(value, Mapping):
                raise RuntimeFailure("invalid_argument", "resources must be an object")
            enabled_skill_ids = value.get(
                "enabled_skill_ids",
                raw["resources"].get("enabled_skills"),
            )
            if enabled_skill_ids is not None and (
                not isinstance(enabled_skill_ids, list)
                or any(not isinstance(item, str) or not item.strip() for item in enabled_skill_ids)
            ):
                raise RuntimeFailure(
                    "invalid_argument",
                    "resources.enabled_skill_ids must be a list of skill ids or null",
                )
            raw["resources"].update(
                {
                    "roots": value.get("roots", raw["resources"]["roots"]),
                    "enabled_skills": (
                        None
                        if enabled_skill_ids is None
                        else list(dict.fromkeys(item.strip() for item in enabled_skill_ids))
                    ),
                    "no_skills": not bool(
                        value.get("load_skills", not raw["resources"]["no_skills"])
                    ),
                    "no_prompt_templates": not bool(
                        value.get(
                            "load_prompt_templates", not raw["resources"]["no_prompt_templates"]
                        )
                    ),
                    "no_context_files": not bool(
                        value.get("load_context_files", not raw["resources"]["no_context_files"])
                    ),
                }
            )
        if "providers" in patch:
            providers = patch["providers"]
            if not isinstance(providers, Mapping):
                raise RuntimeFailure("invalid_argument", "providers must be an object")
            for provider_id, provider_patch in providers.items():
                if provider_id not in raw["providers"]:
                    raise RuntimeFailure(
                        "invalid_argument",
                        f"Unknown Provider: {provider_id}",
                    )
                if not isinstance(provider_patch, Mapping):
                    raise RuntimeFailure(
                        "invalid_argument",
                        f"providers.{provider_id} must be an object",
                    )
                if "api_key" in provider_patch:
                    raise RuntimeFailure(
                        "invalid_argument",
                        "API keys must be changed through secret_actions",
                    )
                if (
                    "driver" in provider_patch
                    and provider_patch["driver"] != raw["providers"][provider_id]["driver"]
                ):
                    raise RuntimeFailure(
                        "invalid_argument",
                        f"Provider driver cannot be changed: {provider_id}",
                    )
                if "endpoint" in provider_patch:
                    self._http_endpoint(str(provider_patch["endpoint"]))
                raw["providers"][provider_id].update(provider_patch)
        for key in ("tools",):
            if key in patch:
                if not isinstance(patch[key], Mapping):
                    raise RuntimeFailure("invalid_argument", f"{key} must be an object")
                raw[key].update(patch[key])

    def _clean_block(self, block: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: self._json_safe(value)
            for key, value in block.items()
            if value is not None and not key.startswith("_")
        }

    def _tool_text(self, raw: Any) -> str:
        if not isinstance(raw, Mapping):
            return ""
        return self._content_text(raw.get("content"))[:TOOL_OUTPUT_LIMIT]

    def _content_text(self, raw: Any) -> str:
        if not isinstance(raw, list):
            return ""
        return "\n".join(str(item.get("text") or "") for item in raw if isinstance(item, Mapping))[
            :TOOL_OUTPUT_LIMIT
        ]

    def _safe_mapping(self, value: Any) -> dict[str, Any]:
        return self.projection.safe_mapping(value)

    def _json_safe(self, value: Any) -> Any:
        return self.projection.json_safe(value)

    def _sanitize(self, message: str) -> str:
        return self.projection.sanitize(message)

    def _public_provider_config(self, provider: Any) -> dict[str, Any]:
        value = provider.model_dump(mode="json", exclude={"api_key"})
        value["credential_configured"] = bool(getattr(provider, "api_key", None))
        headers = value.get("headers")
        if isinstance(headers, dict):
            redact_sensitive_headers(headers)
        if provider.driver == "cloud":
            value["credential_configured"] = bool(self.account.status().get("authenticated"))
        return value

    def _tool_set(
        self,
        *,
        workspace: Path | None = None,
    ) -> RuntimeToolSet:
        return RuntimeToolSet(
            workspace or self.config.workspace,
            shell_path=self.config.tools.shell_path,
            auto_resize_images=self.config.tools.auto_resize_images,
        )

    @staticmethod
    def _required_string(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeFailure("invalid_argument", f"{key} is required")
        return value.strip()

    @staticmethod
    def _thinking_level(value: Any) -> str:
        level = str(value)
        if level not in {"off", "minimal", "low", "medium", "high", "max"}:
            raise RuntimeFailure("invalid_argument", "Invalid thinking level")
        return level

    @staticmethod
    def _queue_mode(value: Any) -> str:
        mode = str(value)
        if mode not in {"all", "one-at-a-time"}:
            raise RuntimeFailure("invalid_argument", "Invalid queue mode")
        return mode


__all__ = ["RuntimeService"]
