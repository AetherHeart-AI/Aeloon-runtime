"""Trusted browser-runtime-v1 contracts owned by Aeloon Core."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_NAME = "browser-runtime-v1"
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_SOCKET_PATH_BYTES = 4_096


class BrowserRuntimeConfigurationError(ValueError):
    """Raised when the Electron-owned Browser Runtime endpoint is invalid."""


@dataclass(frozen=True, slots=True)
class BrowserRuntimeEndpoint:
    """Process-wide route to the trusted Electron Browser Runtime."""

    socket_path: Path

    @classmethod
    def create(cls, socket_path: Path | str) -> BrowserRuntimeEndpoint:
        raw_path = os.fspath(socket_path)
        if not raw_path or "\x00" in raw_path:
            raise BrowserRuntimeConfigurationError("Browser Runtime socket path is invalid")
        if len(os.fsencode(raw_path)) > MAX_SOCKET_PATH_BYTES:
            raise BrowserRuntimeConfigurationError("Browser Runtime socket path is too long")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise BrowserRuntimeConfigurationError("Browser Runtime socket path must be absolute")
        return cls(socket_path=path.resolve(strict=False))


@dataclass(frozen=True, slots=True)
class BrowserContext:
    """Operation-local routing state derived only from the Core session."""

    endpoint: BrowserRuntimeEndpoint
    session_id: str
    operation_id: str
    workspace: Path

    @classmethod
    def create(
        cls,
        *,
        endpoint: BrowserRuntimeEndpoint,
        session_id: str,
        operation_id: str,
        workspace: Path | str,
    ) -> BrowserContext:
        if not session_id:
            raise BrowserRuntimeConfigurationError("Browser Runtime session id is required")
        if not operation_id:
            raise BrowserRuntimeConfigurationError("Browser Runtime operation id is required")
        return cls(
            endpoint=endpoint,
            session_id=session_id,
            operation_id=operation_id,
            workspace=Path(workspace).expanduser().resolve(strict=False),
        )

    def request_params(self, *, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "workspace_root": str(self.workspace),
            "tool": tool,
            "arguments": dict(arguments),
        }


__all__ = [
    "BrowserContext",
    "BrowserRuntimeConfigurationError",
    "BrowserRuntimeEndpoint",
    "MAX_FRAME_BYTES",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
]
