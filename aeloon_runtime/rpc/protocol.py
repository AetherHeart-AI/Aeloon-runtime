"""Strict aeloon-rpc-v2 protocol registry and public errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from aeloon_runtime.rpc import models as wire

PROTOCOL_NAME = "aeloon-rpc-v2"
PROTOCOL_VERSION = 2
MAX_FRAME_BYTES = 12 * 1024 * 1024

ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class MethodSpec(Generic[ParamsT, ResultT]):
    name: str
    handler: str
    params: type[ParamsT] | Any
    result: type[ResultT] | Any


@dataclass(frozen=True, slots=True)
class EventSpec(Generic[PayloadT]):
    name: str
    payload: type[PayloadT] | Any


METHOD_SPECS = (
    MethodSpec("system.handshake", "handshake", wire.HandshakeParams, wire.HandshakeResult),
    MethodSpec("system.health", "health", wire.EmptyParams, wire.HealthResult),
    MethodSpec("system.shutdown", "shutdown", wire.EmptyParams, wire.ShutdownResult),
    MethodSpec(
        "events.subscribe",
        "events_subscribe",
        wire.EventsSubscribeParams,
        wire.EventsSubscribeResult,
    ),
    MethodSpec("session.create", "_session_create", wire.SessionCreateParams, wire.SessionMetadata),
    MethodSpec("session.list", "_session_list", wire.SessionListParams, wire.SessionListResult),
    MethodSpec("session.get", "_session_get", wire.SessionIdParams, wire.SessionSnapshot),
    MethodSpec(
        "session.delete", "runtime.session_delete", wire.SessionIdParams, wire.SessionDeleteResult
    ),
    MethodSpec(
        "session.rename",
        "runtime.session_rename",
        wire.SessionRenameParams,
        wire.SessionRenameResult,
    ),
    MethodSpec(
        "session.configure",
        "runtime.session_configure",
        wire.SessionConfigureParams,
        wire.SessionConfigureResult,
    ),
    MethodSpec(
        "session.tree", "runtime.session_tree", wire.SessionIdParams, wire.SessionTreeResult
    ),
    MethodSpec(
        "session.navigate",
        "runtime.session_navigate",
        wire.SessionNavigateParams,
        wire.SessionNavigateResult,
    ),
    MethodSpec(
        "session.compact",
        "runtime.session_compact",
        wire.SessionCompactParams,
        wire.SessionCompactResult,
    ),
    MethodSpec(
        "session.next_turn",
        "runtime.session_next_turn",
        wire.SessionNextTurnParams,
        wire.SessionNextTurnResult,
    ),
    MethodSpec("turn.start", "_turn_start", wire.TurnStartParams, wire.TurnStartResult),
    MethodSpec("turn.cancel", "_turn_cancel", wire.OperationIdParams, wire.TurnCancelResult),
    MethodSpec("turn.steer", "_turn_steer", wire.TurnTextParams, wire.TurnAcceptedResult),
    MethodSpec("turn.follow_up", "_turn_follow_up", wire.TurnTextParams, wire.TurnAcceptedResult),
    MethodSpec("catalog.get", "runtime.catalog_get", wire.CatalogParams, wire.CatalogResult),
    MethodSpec("provider.list", "runtime.provider_list", wire.EmptyParams, wire.ProviderListResult),
    MethodSpec(
        "provider.refresh",
        "runtime.provider_refresh",
        wire.ProviderIdParams,
        wire.ProviderMutationResult,
    ),
    MethodSpec(
        "provider.add", "runtime.provider_add", wire.ProviderAddParams, wire.ProviderMutationResult
    ),
    MethodSpec(
        "provider.remove",
        "runtime.provider_remove",
        wire.ProviderIdParams,
        wire.ProviderRemoveResult,
    ),
    MethodSpec("settings.get", "runtime.settings_get", wire.SettingsGetParams, wire.SettingsResult),
    MethodSpec(
        "settings.update", "runtime.settings_update", wire.SettingsUpdateParams, wire.SettingsResult
    ),
    MethodSpec(
        "tools.search.test",
        "runtime.tools_search_test",
        wire.ToolsSearchTestParams,
        wire.ToolsSearchTestResult,
    ),
    MethodSpec(
        "cloud.account.status", "runtime.account_status", wire.EmptyParams, wire.CloudStatusResult
    ),
    MethodSpec(
        "cloud.account.login",
        "runtime.account_login",
        wire.CloudLoginParams,
        wire.CloudStatusResult,
    ),
    MethodSpec(
        "cloud.account.logout", "runtime.account_logout", wire.EmptyParams, wire.CloudStatusResult
    ),
)

EVENT_SPECS = (
    EventSpec("operation.queued", wire.OperationPayload),
    EventSpec("operation.started", wire.OperationPayload),
    EventSpec("operation.cancelling", wire.OperationPayload),
    EventSpec("operation.completed", wire.OperationPayload),
    EventSpec("operation.failed", wire.OperationPayload),
    EventSpec("operation.cancelled", wire.OperationPayload),
    EventSpec("content.started", wire.ContentStartedPayload),
    EventSpec("content.delta", wire.ContentDeltaPayload),
    EventSpec("content.updated", wire.BlockPatchPayload),
    EventSpec("content.completed", wire.BlockPatchPayload),
    EventSpec("tool.started", wire.ContentStartedPayload),
    EventSpec("tool.updated", wire.BlockPatchPayload),
    EventSpec("tool.completed", wire.BlockPatchPayload),
    EventSpec("usage.updated", wire.UsagePayload),
    EventSpec("queue.updated", wire.QueuePayload),
    EventSpec("retry.started", wire.JsonObject),
    EventSpec("retry.completed", wire.JsonObject),
    EventSpec("session.compacted", wire.JsonObject),
    EventSpec("session.navigated", wire.JsonObject),
    EventSpec("session.renamed", wire.SessionRenamedPayload),
    EventSpec("settings.updated", wire.RevisionPayload),
    EventSpec("cloud.account.updated", wire.CloudStatusResult),
    EventSpec("provider.updated", wire.ProviderUpdatedPayload),
    EventSpec("log.entry", wire.LogPayload),
    EventSpec("system.shutdown", wire.ShutdownPayload),
)

METHOD_REGISTRY = {spec.name: spec for spec in METHOD_SPECS}
EVENT_REGISTRY = {spec.name: spec for spec in EVENT_SPECS}
METHODS = tuple(METHOD_REGISTRY)
EVENTS = tuple(EVENT_REGISTRY)

RPC_CODES = {
    "protocol_incompatible": -32010,
    "invalid_argument": -32602,
    "session_not_found": -32020,
    "operation_not_found": -32021,
    "busy": -32022,
    "invalid_state": -32023,
    "invalid_attachment": -32024,
    "attachment_processing_failed": -32028,
    "revision_conflict": -32025,
    "authentication_failed": -32027,
    "internal_error": -32603,
    "method_not_found": -32601,
    "thread_not_found": -32020,
    "unauthorized": -32011,
    "forbidden": -32012,
    "capability_unavailable": -32013,
    "payload_too_large": -32014,
}


class RpcError(RuntimeError):
    """Sanitized failure crossing the Core process boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code if code in RPC_CODES else "internal_error"

    def to_rpc(self) -> dict[str, Any]:
        return {
            "code": RPC_CODES[self.code],
            "message": str(self),
            "data": {"code": self.code},
        }


__all__ = [
    "EVENTS",
    "EVENT_REGISTRY",
    "EVENT_SPECS",
    "MAX_FRAME_BYTES",
    "METHODS",
    "METHOD_REGISTRY",
    "METHOD_SPECS",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "RpcError",
]
