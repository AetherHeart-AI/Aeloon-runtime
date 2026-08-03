"""Stable Bridge v2 protocol metadata and errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_NAME = "aeloon-core-bridge"
PROTOCOL_VERSION = 2
METHODS = (
    "system.handshake", "system.health", "system.shutdown", "events.subscribe",
    "session.create", "session.list", "session.get", "session.delete", "session.rename",
    "session.configure", "session.tree", "session.navigate", "session.compact",
    "session.next_turn", "turn.start", "turn.cancel", "turn.steer", "turn.follow_up",
    "catalog.get", "provider.list", "provider.local.add", "provider.local.remove",
    "settings.get", "settings.update",
    "cloud.account.status", "cloud.account.login", "cloud.account.logout",
    "provider.cloud.status", "provider.cloud.login", "provider.cloud.logout",
)
EVENTS = (
    "operation.queued", "operation.started", "operation.completed", "operation.failed",
    "operation.cancelled", "content.started", "content.delta", "content.updated",
    "content.completed", "tool.started", "tool.updated", "tool.completed",
    "usage.updated", "queue.updated", "retry.started", "retry.completed",
    "resources.updated", "session.compacted", "session.navigated", "settings.updated",
    "cloud.account.updated", "provider.updated", "log.entry", "system.shutdown",
)
CAPABILITIES = (
    "daemon", "sessions", "turn-queue", "ordered-events", "event-replay",
    "session-snapshots", "attachments", "revisioned-settings", "dynamic-catalog",
    "cloud-account", "unified-providers",
)

RPC_CODES = {
    "protocol_incompatible": -32010,
    "invalid_argument": -32602,
    "session_not_found": -32020,
    "operation_not_found": -32021,
    "busy": -32022,
    "invalid_state": -32023,
    "invalid_attachment": -32024,
    "revision_conflict": -32025,
    "daemon_config_conflict": -32026,
    "authentication_failed": -32027,
    "internal_error": -32603,
    "method_not_found": -32601,
}


class BridgeError(RuntimeError):
    """Sanitized public Bridge failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code if code in RPC_CODES else "internal_error"

    def to_rpc(self) -> dict[str, Any]:
        return {
            "code": RPC_CODES[self.code],
            "message": str(self),
            "data": {"code": self.code},
        }


def load_schema() -> dict[str, Any]:
    path = Path(__file__).with_name("bridge-protocol-v2.json")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CAPABILITIES", "EVENTS", "METHODS", "PROTOCOL_NAME", "PROTOCOL_VERSION",
    "BridgeError", "load_schema",
]
