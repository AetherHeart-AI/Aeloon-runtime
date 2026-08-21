"""Private Runtime R3 benchmark helpers.

These APIs wrap a real ``RuntimeServer`` for tests and the remote harness.
They are not RPC methods, capabilities, or CLI entry points.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
import struct
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import websockets

from aeloon_runtime.runtime_server import pack_frame

WS_SEND_CHUNK_BYTES = 1024 * 1024
WS_WRITE_LIMIT_BYTES = 4 * 1024 * 1024

USER_TURN_BYTES = 1024
ASSISTANT_TURN_BYTES = 8192
THREAD_GET_TURNS = 128
ATTACHMENT_BYTES = 25 * 1024 * 1024
THROUGHPUT_RATES = (250, 500, 1000, 1500, 2000, 3000, 4000)
EVENT_PAYLOAD_BYTES = 1024


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round((q / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "n": len(samples),
        "p50_ms": (
            percentile(samples, 50) * 1000
            if samples and samples[0] < 100
            else percentile(samples, 50)
        ),
        "p95_ms": (
            percentile(samples, 95) * 1000
            if samples and samples[0] < 100
            else percentile(samples, 95)
        ),
    }


def _timed_ms(samples_s: list[float]) -> dict[str, float]:
    return {
        "n": len(samples_s),
        "p50_ms": percentile(samples_s, 50) * 1000,
        "p95_ms": percentile(samples_s, 95) * 1000,
    }


class BenchClient:
    """Length-prefixed aeloon-rpc client over a WebSocket."""

    def __init__(self) -> None:
        self._next_id = 1
        self._ws: Any = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._pump_task: asyncio.Task[None] | None = None
        self.events: list[dict[str, Any]] = []

    async def connect_wss(self, url: str, *, ssl_ctx: ssl.SSLContext) -> None:
        self._ws = await websockets.connect(
            url,
            ssl=ssl_ctx,
            max_size=None,
            open_timeout=10,
            ping_interval=15,
            ping_timeout=15,
            compression=None,
            write_limit=WS_WRITE_LIMIT_BYTES,
        )
        self._pump_task = asyncio.create_task(self._pump())

    async def close(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
            self._pump_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("benchmark client closed"))
        self._pending.clear()

    async def _pump(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if isinstance(message, str) or len(message) < 4:
                    continue
                (size,) = struct.unpack("!I", message[:4])
                frame = json.loads(message[4 : 4 + size])
                if frame.get("method") == "event" and isinstance(frame.get("params"), dict):
                    self.events.append(frame["params"])
                    continue
                request_id = frame.get("id")
                future = self._pending.get(request_id) if isinstance(request_id, str) else None
                if future is None or future.done():
                    continue
                if frame.get("error"):
                    future.set_exception(RuntimeError(frame["error"]))
                else:
                    future.set_result(frame.get("result"))
        except Exception:
            return

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = f"bench-{self._next_id}"
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self._send_frame(pack_frame({"id": request_id, "method": method, "params": params or {}}))
        try:
            return await asyncio.wait_for(future, timeout=120)
        finally:
            self._pending.pop(request_id, None)

    async def _send_frame(self, payload: bytes) -> None:
        # Split one RPC frame across WebSocket messages so heartbeats can still
        # run while a 25 MiB attachment is in flight over a slow tunnel.
        offset = 0
        while offset < len(payload):
            await self._ws.send(payload[offset : offset + WS_SEND_CHUNK_BYTES])
            offset += WS_SEND_CHUNK_BYTES

    async def handshake(self, token: str) -> Any:
        return await self.request(
            "system.handshake",
            {
                "protocol": {"min": "4.0.0", "max": "4.0.0"},
                "client": {"name": "r3-bench", "version": "0", "platform": "test"},
                "auth": {"kind": "device_token", "token": token},
            },
        )

    async def wait_event(
        self,
        name: str,
        *,
        timeout_s: float = 5,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        after: int = 0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for event in self.events[after:]:
                if event.get("name") != name:
                    continue
                if predicate is None or predicate(event):
                    return event
            await asyncio.sleep(0.005)
        raise TimeoutError(f"Timed out waiting for {name}")


async def wait_for_unix(socket_path: Path, timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            try:
                reader, writer = await asyncio.open_unix_connection(str(socket_path))
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                pass
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Runtime socket did not become ready: {socket_path}")


def seed_completed_turns(
    store: Any,
    thread_id: str,
    *,
    count: int = THREAD_GET_TURNS,
    user_bytes: int = USER_TURN_BYTES,
    assistant_bytes: int = ASSISTANT_TURN_BYTES,
) -> None:
    user_text = "u" * user_bytes
    assistant = "a" * assistant_bytes
    for index in range(count):
        core_turn_id = f"r3-turn-{index}"
        store.create_turn(
            thread_id=thread_id,
            core_turn_id=core_turn_id,
            user_text=user_text,
            status="completed",
        )
        with store.transaction() as db:
            db.execute(
                "UPDATE turns SET status = ?, blocks_json = ? WHERE core_turn_id = ?",
                (
                    "completed",
                    json.dumps(
                        [{"id": f"block-{index}", "type": "text", "content": assistant}],
                        ensure_ascii=False,
                    ),
                    core_turn_id,
                ),
            )


async def measure_samples(
    warmup: int,
    count: int,
    operation: Callable[[], Awaitable[Any]],
) -> list[float]:
    for _ in range(warmup):
        await operation()
    samples: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        await operation()
        samples.append(time.perf_counter() - started)
    return samples


__all__ = [
    "ATTACHMENT_BYTES",
    "ASSISTANT_TURN_BYTES",
    "BenchClient",
    "EVENT_PAYLOAD_BYTES",
    "THREAD_GET_TURNS",
    "THROUGHPUT_RATES",
    "USER_TURN_BYTES",
    "WS_SEND_CHUNK_BYTES",
    "measure_samples",
    "percentile",
    "seed_completed_turns",
    "summarize",
    "wait_for_unix",
]
