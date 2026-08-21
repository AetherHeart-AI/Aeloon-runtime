#!/usr/bin/env python3
"""R3 Runtime benchmark harness.

In-process mode starts a real RuntimeServer and drives it over loopback WSS.
Client mode connects to an already-running Runtime (typically through an SSH
tunnel) and talks to the private control socket for inject/seed only.

Private generators stay off the RPC manifest, capabilities, and product CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import platform
import ssl
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from aeloon_runtime.bench_support import (
    ASSISTANT_TURN_BYTES,
    ATTACHMENT_BYTES,
    EVENT_PAYLOAD_BYTES,
    THREAD_GET_TURNS,
    THROUGHPUT_RATES,
    USER_TURN_BYTES,
    BenchClient,
    measure_samples,
    percentile,
    seed_completed_turns,
    wait_for_unix,
)
from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.rpc.protocol import PROTOCOL_VERSION
from aeloon_runtime.runtime_server import RuntimeServer, pack_frame

InjectAtRate = Callable[[int, float, int], Awaitable[int]]
SeedTurns = Callable[[str], Awaitable[None]]


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ssl_client() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def pairing_code(pairing: str) -> str:
    query = parse_qs(urlparse(pairing).query)
    code = (query.get("code") or [""])[0]
    if not code:
        raise ValueError("pairing URL is missing a claim code")
    return code


def unique_attachment_blobs(count: int, size: int = ATTACHMENT_BYTES) -> list[bytes]:
    return [os.urandom(size) for _ in range(count)]


async def _unix_rpc(socket_path: Path, method: str, params: dict[str, Any] | None = None) -> Any:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        async def once(payload: dict[str, Any]) -> dict[str, Any]:
            writer.write(pack_frame(payload))
            await writer.drain()
            header = await reader.readexactly(4)
            import struct

            (size,) = struct.unpack("!I", header)
            return json.loads(await reader.readexactly(size))

        handshake = await once(
            {
                "id": "hs",
                "method": "system.handshake",
                "params": {
                    "protocol": {"min": PROTOCOL_VERSION, "max": PROTOCOL_VERSION},
                    "client": {"name": "r3-bench", "version": "0", "platform": sys.platform},
                },
            }
        )
        if "error" in handshake:
            raise RuntimeError(handshake["error"])
        result = await once({"id": "req", "method": method, "params": params or {}})
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("result")
    finally:
        writer.close()
        await writer.wait_closed()


async def _claim_token(url: str, code: str) -> str:
    import websockets

    ssl_ctx = _ssl_client()
    async with websockets.connect(
        url,
        ssl=ssl_ctx,
        max_size=None,
        compression=None,
        write_limit=4 * 1024 * 1024,
    ) as connection:
        await connection.send(
            pack_frame(
                {
                    "id": "claim",
                    "method": "devices.claim",
                    "params": {"code": code, "client": {"name": "r3-bench", "version": "0"}},
                }
            )
        )
        import struct

        raw = await connection.recv()
        (size,) = struct.unpack("!I", raw[:4])
        frame = json.loads(raw[4 : 4 + size])
        return str(frame["result"]["token"])


async def control_request(
    host: str,
    port: int,
    request: dict[str, Any],
    *,
    timeout_s: float = 120,
) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write((json.dumps(request) + "\n").encode("utf-8"))
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
        payload = json.loads(raw.decode("utf-8") or "{}")
        if payload.get("ok") is False:
            raise RuntimeError(str(payload.get("error") or "control request failed"))
        return payload
    finally:
        writer.close()
        await writer.wait_closed()


def _write_report(report: dict[str, Any], json_out: Path | None) -> str:
    text = json.dumps(report, indent=2, sort_keys=True)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n", encoding="utf-8")
    return text


def build_report(
    *,
    path: str,
    host_info: dict[str, Any],
    pty_samples: list[float],
    thread_samples: list[float],
    encoded_sizes: list[int],
    attach_samples: list[float],
    first_samples: list[float],
    throughput: dict[str, Any],
    methods: set[str],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "path": path,
        "runtime_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "host": host_info,
        "pty_echo": {
            "warmup": 5,
            "n": 50,
            "p50_ms": percentile(pty_samples, 50) * 1000,
            "p95_ms": percentile(pty_samples, 95) * 1000,
            "budget_p95_ms": 250,
        },
        "thread_get": {
            "warmup": 3,
            "n": 30,
            "turns": THREAD_GET_TURNS,
            "user_bytes": USER_TURN_BYTES,
            "assistant_bytes": ASSISTANT_TURN_BYTES,
            "p50_ms": percentile(thread_samples, 50) * 1000,
            "p95_ms": percentile(thread_samples, 95) * 1000,
            "encoded_bytes_max": max(encoded_sizes) if encoded_sizes else 0,
            "budget_p95_ms": 1000,
            "budget_encoded_bytes": 5 * 1024 * 1024,
        },
        "attachment_25mib": {
            "warmup": 1,
            "n": 10,
            "p50_ms": percentile(attach_samples, 50) * 1000,
            "p95_ms": percentile(attach_samples, 95) * 1000,
            "budget_p95_ms": 8000,
            "includes_base64": True,
            "unique_payloads": True,
        },
        "first_available": {
            "warmup": 5,
            "n": 30,
            "p50_ms": percentile(first_samples, 50) * 1000,
            "p95_ms": percentile(first_samples, 95) * 1000,
            "budget_p95_ms": 2000,
        },
        "event_throughput": throughput,
        "private_methods_leaked": "inject_benchmark_events" in methods,
    }


async def measure_against(
    *,
    url: str,
    token: str,
    inject_at_rate: InjectAtRate,
    seed_turns: SeedTurns,
    methods: set[str],
    host_info: dict[str, Any],
    path: str,
    json_out: Path | None = None,
) -> dict[str, Any]:
    ssl_ctx = _ssl_client()

    async def fresh_session() -> BenchClient:
        client = BenchClient()
        await client.connect_wss(url, ssl_ctx=ssl_ctx)
        await client.handshake(token)
        await client.request("events.subscribe", {"thread_ids": []})
        return client

    first_samples: list[float] = []
    for index in range(35):
        elapsed: float | None = None
        last_error: Exception | None = None
        for _attempt in range(10):
            client = BenchClient()
            started = time.perf_counter()
            try:
                await client.connect_wss(url, ssl_ctx=ssl_ctx)
                await client.handshake(token)
                await client.request("events.subscribe", {"thread_ids": []})
                await client.request("system.capabilities", {})
                await client.request("system.snapshot", {})
                elapsed = time.perf_counter() - started
                await client.close()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                await client.close()
                await asyncio.sleep(0.05)
        if elapsed is None:
            raise RuntimeError(last_error)
        if index >= 5:
            first_samples.append(elapsed)

    client = await fresh_session()
    roots = await client.request("workspace.roots", {})
    root_id = roots["roots"][0]["id"]
    project = await client.request(
        "project.add", {"root_id": root_id, "relative_path": "."}
    )
    thread = await client.request(
        "thread.create",
        {"project_id": project["project"]["id"], "kind": "standard", "title": "r3-bench"},
    )
    thread_id = thread["thread"]["id"]
    await seed_turns(thread_id)

    encoded_sizes: list[int] = []

    async def thread_get() -> None:
        result = await client.request("thread.get", {"thread_id": thread_id})
        encoded_sizes.append(len(json.dumps(result, ensure_ascii=False).encode("utf-8")))

    thread_samples = await measure_samples(3, 30, thread_get)

    await client.request("terminal.open", {"thread_id": thread_id})

    async def pty_echo() -> None:
        marker = uuid.uuid4().hex
        before = len(client.events)
        await client.request(
            "terminal.input", {"thread_id": thread_id, "data": f"echo {marker}\n"}
        )
        await client.wait_event(
            "terminal.output",
            timeout_s=15,
            after=before,
            predicate=lambda event: marker in str(
                (event.get("payload") or {}).get("data") or ""
            ),
        )

    pty_samples = await measure_samples(5, 50, pty_echo)

    blobs = unique_attachment_blobs(11)
    blob_index = 0

    async def upload() -> None:
        nonlocal blob_index
        payload = base64.b64encode(blobs[blob_index]).decode("ascii")
        blob_index += 1
        await client.request(
            "attachment.upload",
            {
                "name": f"r3-{blob_index}.bin",
                "mime_type": "application/octet-stream",
                "data_base64": payload,
            },
        )

    attach_samples = await measure_samples(1, 10, upload)

    throughput: dict[str, Any] = {"rates": {}}
    stable = 0
    for rate in THROUGHPUT_RATES:
        received_rates: list[float] = []
        overflows = 0
        for _repeat in range(3):
            before_events = len(client.events)
            started = time.perf_counter()
            expected = await inject_at_rate(rate, 10, EVENT_PAYLOAD_BYTES)
            deadline = time.perf_counter() + 2
            while time.perf_counter() < deadline:
                received = len(client.events) - before_events
                if received >= expected:
                    break
                await asyncio.sleep(0.01)
            elapsed = max(1e-6, time.perf_counter() - started)
            received = len(client.events) - before_events
            received_rates.append(received / elapsed)
            overflowed = False
            try:
                logs = await client.request("diagnostics.logs", {"limit": 50})
                overflowed = any(
                    item["event"] == "event_overflow_closed" for item in logs["entries"]
                )
            except Exception:
                overflowed = True
                await client.close()
                client = await fresh_session()
            if overflowed or received < expected:
                overflows += 1
        delivered = overflows == 0
        if delivered:
            stable = rate
        throughput["rates"][str(rate)] = {
            "delivered": delivered,
            "overflows": overflows,
            "receive_p50": percentile(received_rates, 50),
            "receive_p95": percentile(received_rates, 95),
        }
        if not delivered and rate > 1000:
            break
    throughput["highest_stable_events_s"] = stable
    throughput["required_1000_delivered"] = (
        throughput["rates"].get("1000", {}).get("delivered") is True
    )

    await client.close()
    report = build_report(
        path=path,
        host_info=host_info,
        pty_samples=pty_samples,
        thread_samples=thread_samples,
        encoded_sizes=encoded_sizes,
        attach_samples=attach_samples,
        first_samples=first_samples,
        throughput=throughput,
        methods=methods,
    )
    _write_report(report, json_out)
    print("R3_BENCH_JSON_WRITTEN", flush=True)
    return report


async def run_suite(
    *,
    data_dir: Path,
    workspace: Path,
    socket_path: Path,
    host: str,
    port: int,
    event_limit: int | None = None,
    event_queue_limit: int | None = None,
    event_queue_bytes: int | None = None,
    json_out: Path | None = None,
    path: str = "loopback",
) -> dict[str, Any]:
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(
        runtime,
        socket_path,
        (workspace,),
        data_dir,
        listen=(host, port),
        event_limit=event_limit,
        event_queue_limit=event_queue_limit,
        event_queue_bytes=event_queue_bytes,
    )
    task = asyncio.create_task(server.run())
    try:
        await wait_for_unix(socket_path)
        enrollment = await _unix_rpc(socket_path, "devices.enroll", {})
        url = f"wss://{host}:{port}"
        token = await _claim_token(url, enrollment["code"])

        async def inject_at_rate(rate: int, duration_s: float, payload_bytes: int) -> int:
            return await server.inject_benchmark_events_at_rate(
                rate, duration_s, payload_bytes=payload_bytes
            )

        async def seed(thread_id: str) -> None:
            seed_completed_turns(server.store, thread_id)

        capabilities = await _unix_rpc(socket_path, "system.capabilities", {})
        methods = {
            name
            for item in capabilities["capabilities"]
            for name in item.get("methods", [])
        }
        return await measure_against(
            url=url,
            token=token,
            inject_at_rate=inject_at_rate,
            seed_turns=seed,
            methods=methods,
            host_info={
                "hostname": platform.node(),
                "platform": platform.system().lower(),
                "machine": platform.machine(),
                "client_hostname": platform.node(),
            },
            path=path,
            json_out=json_out,
        )
    finally:
        server.stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=8)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
        await runtime.close()


async def run_client_suite(
    *,
    endpoint: str,
    pairing: str,
    control_host: str,
    control_port: int,
    json_out: Path | None = None,
    path: str = "ssh-tunnel",
) -> dict[str, Any]:
    token = await _claim_token(endpoint, pairing_code(pairing))
    status = await control_request(control_host, control_port, {"op": "status"})
    methods = set(status.get("methods") or [])

    async def inject_at_rate(rate: int, duration_s: float, payload_bytes: int) -> int:
        result = await control_request(
            control_host,
            control_port,
            {
                "op": "inject_at_rate",
                "rate": rate,
                "duration_s": duration_s,
                "payload_bytes": payload_bytes,
            },
            timeout_s=30,
        )
        return int(result.get("expected") or 0)

    async def seed(thread_id: str) -> None:
        await control_request(
            control_host,
            control_port,
            {"op": "seed_turns", "thread_id": thread_id},
        )

    return await measure_against(
        url=endpoint,
        token=token,
        inject_at_rate=inject_at_rate,
        seed_turns=seed,
        methods=methods,
        host_info={
            "hostname": str(status.get("hostname") or ""),
            "platform": str(status.get("platform") or ""),
            "machine": str(status.get("machine") or ""),
            "client_hostname": platform.node(),
        },
        path=path,
        json_out=json_out,
    )


def budgets_hold(report: dict[str, Any]) -> bool:
    return (
        report["pty_echo"]["p95_ms"] <= report["pty_echo"]["budget_p95_ms"]
        and report["thread_get"]["p95_ms"] <= report["thread_get"]["budget_p95_ms"]
        and report["thread_get"]["encoded_bytes_max"]
        <= report["thread_get"]["budget_encoded_bytes"]
        and report["attachment_25mib"]["p95_ms"] <= report["attachment_25mib"]["budget_p95_ms"]
        and report["first_available"]["p95_ms"] <= report["first_available"]["budget_p95_ms"]
        and report["event_throughput"]["required_1000_delivered"] is True
        and report["private_methods_leaked"] is False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--unix", type=Path)
    parser.add_argument("--listen", default="127.0.0.1:7420")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--pairing")
    parser.add_argument("--control")
    parser.add_argument("--path")
    args = parser.parse_args()
    if args.endpoint:
        if not args.pairing or not args.control:
            parser.error("client mode requires --pairing and --control host:port")
        control_host, control_port = args.control.rsplit(":", 1)
        report = asyncio.run(
            run_client_suite(
                endpoint=args.endpoint,
                pairing=args.pairing,
                control_host=control_host,
                control_port=int(control_port),
                json_out=args.json_out,
                path=args.path or "ssh-tunnel",
            )
        )
    else:
        root = (
            Path(tempfile.mkdtemp(prefix="aeloon-r3-bench-"))
            if args.data_dir is None
            else args.data_dir
        )
        workspace = args.workspace or (root / "workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        socket_path = args.unix or (root / "runtime.sock")
        host, port_text = args.listen.rsplit(":", 1)
        report = asyncio.run(
            run_suite(
                data_dir=root / "data" if args.data_dir is None else root,
                workspace=workspace,
                socket_path=socket_path,
                host=host,
                port=int(port_text),
                json_out=args.json_out,
                path=args.path or "loopback",
            )
        )
    report["runtime_commit"] = _git_commit(Path(__file__).resolve().parents[1])
    print(_write_report(report, args.json_out), flush=True)
    return 0 if budgets_hold(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
