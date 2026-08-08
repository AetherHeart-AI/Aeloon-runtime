"""Length-framed client for the trusted Electron Browser Runtime."""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from collections.abc import Mapping
from typing import Any

from aeloon_core.browser.protocol import MAX_FRAME_BYTES, BrowserContext

CONNECT_TIMEOUT_SECONDS = 5.0


class BrowserRuntimeError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        envelope: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.envelope = dict(envelope or {})


class BrowserRuntimeUnavailable(BrowserRuntimeError):
    """Stable failure used when Electron's Browser Runtime is disconnected."""

    def __init__(self, message: str = "Browser Runtime is unavailable") -> None:
        super().__init__("unavailable", message)


def pack_frame(value: Any) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise BrowserRuntimeError("malformed", "Browser Runtime request exceeds 12 MiB")
    return struct.pack("!I", len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader, timeout: float) -> Any:
    try:
        header = await asyncio.wait_for(reader.readexactly(4), timeout)
        (length,) = struct.unpack("!I", header)
        if length > MAX_FRAME_BYTES:
            raise BrowserRuntimeError("malformed", "Browser Runtime frame exceeds 12 MiB")
        payload = await asyncio.wait_for(reader.readexactly(length), timeout)
        return json.loads(payload)
    except TimeoutError:
        raise BrowserRuntimeError("timeout", "Browser Runtime request timed out") from None
    except asyncio.IncompleteReadError:
        raise BrowserRuntimeUnavailable("Browser Runtime connection closed") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BrowserRuntimeError("malformed", "Browser Runtime returned invalid JSON") from None


async def execute_browser_tool(
    context: BrowserContext,
    *,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    timeout_ms: int,
) -> Any:
    """Execute one Core-owned browser tool over a short-lived trusted connection."""

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(context.endpoint.socket_path)),
            min(CONNECT_TIMEOUT_SECONDS, timeout_ms / 1000),
        )
    except TimeoutError:
        raise BrowserRuntimeUnavailable("Browser Runtime connection timed out") from None
    except OSError as exc:
        raise BrowserRuntimeUnavailable(f"Browser Runtime is unavailable: {exc}") from None

    request = {
        "id": call_id,
        "method": "execute",
        "params": context.request_params(tool=name, arguments=arguments),
    }
    try:
        writer.write(pack_frame(request))
        await writer.drain()
        value = await read_frame(reader, timeout_ms / 1000)
        if not isinstance(value, Mapping) or value.get("id") != call_id:
            raise BrowserRuntimeError("malformed", "Browser Runtime returned a mismatched response")
        if value.get("error") is not None:
            raw_error = value.get("error")
            error = raw_error if isinstance(raw_error, Mapping) else {}
            envelope = error.get("data") if isinstance(error.get("data"), Mapping) else None
            raise BrowserRuntimeError(
                "remote",
                str(error.get("message") or "Browser Runtime rejected the request"),
                envelope=envelope,
            )
        if "result" not in value:
            raise BrowserRuntimeError("malformed", "Browser Runtime response is incomplete")
        return value["result"]
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


__all__ = [
    "BrowserRuntimeError",
    "BrowserRuntimeUnavailable",
    "execute_browser_tool",
    "pack_frame",
    "read_frame",
]
