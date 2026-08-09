"""Strict aeloon-rpc-v2 protocol metadata and errors."""

from __future__ import annotations

from typing import Any

PROTOCOL_NAME = "aeloon-rpc-v2"
PROTOCOL_VERSION = 2
MAX_FRAME_BYTES = 12 * 1024 * 1024

METHODS = (
    "system.handshake",
    "system.health",
    "system.shutdown",
    "events.subscribe",
    "session.create",
    "session.list",
    "session.get",
    "session.delete",
    "session.rename",
    "session.configure",
    "session.tree",
    "session.navigate",
    "session.compact",
    "session.next_turn",
    "turn.start",
    "turn.cancel",
    "turn.steer",
    "turn.follow_up",
    "catalog.get",
    "provider.list",
    "provider.refresh",
    "provider.add",
    "provider.remove",
    "settings.get",
    "settings.update",
    "cloud.account.status",
    "cloud.account.login",
    "cloud.account.logout",
)

EVENTS = (
    "operation.queued",
    "operation.started",
    "operation.completed",
    "operation.failed",
    "operation.cancelled",
    "content.started",
    "content.delta",
    "content.updated",
    "content.completed",
    "tool.started",
    "tool.updated",
    "tool.completed",
    "usage.updated",
    "queue.updated",
    "retry.started",
    "retry.completed",
    "session.compacted",
    "session.navigated",
    "session.renamed",
    "settings.updated",
    "cloud.account.updated",
    "provider.updated",
    "log.entry",
    "system.shutdown",
)

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
    "MAX_FRAME_BYTES",
    "METHODS",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "RpcError",
]
