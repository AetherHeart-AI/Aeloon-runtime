#!/usr/bin/env python3
"""Test-only RuntimeServer with optional reduced event bounds and a control socket.

This is not a product CLI entry point. Playwright and the R3 harness use it to
inject smaller retention/queue limits and private events without exposing those
controls on the RPC surface.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from aeloon_runtime.bench_support import (
    ASSISTANT_TURN_BYTES,
    THREAD_GET_TURNS,
    USER_TURN_BYTES,
    seed_completed_turns,
)
from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.runtime_log import read_runtime_logs
from aeloon_runtime.runtime_server import RuntimeServer

MAX_LINK_PROBE_BYTES = 16 * 1024 * 1024
LINK_DOWNLOAD_PAYLOAD = os.urandom(MAX_LINK_PROBE_BYTES)


async def _handle_control(
    server: RuntimeServer,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        raw = await reader.readline()
        request = json.loads(raw.decode("utf-8") or "{}")
        if request.get("op") == "link_download":
            await _send_link_download(writer, request)
            return
        if request.get("op") == "link_upload":
            result = await _receive_link_upload(reader, request)
        else:
            result = await _dispatch_control(server, request)
        writer.write((json.dumps(result) + "\n").encode("utf-8"))
        await writer.drain()
    except Exception as exc:
        with contextlib.suppress(Exception):
            writer.write((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))
            await writer.drain()
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def _receive_link_upload(
    reader: asyncio.StreamReader,
    request: dict[str, Any],
) -> dict[str, Any]:
    byte_count = request.get("byte_count")
    expected_digest = request.get("sha256")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or not 1 <= byte_count <= MAX_LINK_PROBE_BYTES
    ):
        raise ValueError("link_upload byte_count is outside the test-only limit")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("link_upload sha256 is required")
    digest = hashlib.sha256()
    remaining = byte_count
    while remaining:
        chunk = await reader.readexactly(min(256 * 1024, remaining))
        digest.update(chunk)
        remaining -= len(chunk)
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("link_upload payload digest did not match")
    return {"ok": True, "bytes": byte_count, "sha256": actual_digest}


async def _send_link_download(
    writer: asyncio.StreamWriter,
    request: dict[str, Any],
) -> None:
    byte_count = request.get("byte_count")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or not 1 <= byte_count <= MAX_LINK_PROBE_BYTES
    ):
        raise ValueError("link_download byte_count is outside the test-only limit")
    payload = LINK_DOWNLOAD_PAYLOAD[:byte_count]
    digest = hashlib.sha256(payload).hexdigest()
    writer.write(
        (
            json.dumps({"ok": True, "bytes": byte_count, "sha256": digest})
            + "\n"
        ).encode("utf-8")
    )
    writer.write(payload)
    await writer.drain()


async def _dispatch_control(server: RuntimeServer, request: dict[str, Any]) -> dict[str, Any]:
    op = request.get("op")
    if op == "inject":
        count = int(request.get("count") or 0)
        payload_bytes = int(request.get("payload_bytes") or 1024)
        await server.inject_benchmark_events(count, payload_bytes=payload_bytes)
        return {
            "ok": True,
            "current_seq": server.current_seq,
            "server_instance_id": server.server_instance_id,
        }
    if op == "inject_at_rate":
        rate = int(request.get("rate") or 0)
        duration_s = float(request.get("duration_s") or 0)
        payload_bytes = int(request.get("payload_bytes") or 1024)
        expected = await server.inject_benchmark_events_at_rate(
            rate, duration_s, payload_bytes=payload_bytes
        )
        return {
            "ok": True,
            "expected": expected,
            "current_seq": server.current_seq,
            "server_instance_id": server.server_instance_id,
        }
    if op == "seed_turns":
        thread_id = str(request.get("thread_id") or "")
        if not thread_id:
            return {"ok": False, "error": "thread_id is required"}
        seed_completed_turns(
            server.store,
            thread_id,
            count=int(request.get("count") or THREAD_GET_TURNS),
            user_bytes=int(request.get("user_bytes") or USER_TURN_BYTES),
            assistant_bytes=int(request.get("assistant_bytes") or ASSISTANT_TURN_BYTES),
        )
        return {"ok": True, "thread_id": thread_id}
    if op == "ensure_project":
        roots = await server.dispatch("workspace.roots", {})
        root_id = roots["roots"][0]["id"]
        try:
            await server.dispatch(
                "project.add", {"root_id": root_id, "relative_path": "."}
            )
        except Exception:
            pass
        return {"ok": True, "root_id": root_id}
    if op == "verify_attachment":
        attachment_id = str(request.get("attachment_id") or "")
        expected_size = int(request.get("size") or -1)
        expected_digest = str(request.get("sha256") or "")
        value = server.store.get_attachment(attachment_id)
        if value is None:
            raise ValueError("attachment does not exist")
        path = server._attachment_storage_path(value)

        def inspect() -> tuple[int, str]:
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                while chunk := handle.read(256 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            return size, digest.hexdigest()

        actual_size, actual_digest = await asyncio.to_thread(inspect)
        if actual_size != expected_size or actual_digest != expected_digest:
            raise ValueError("stored attachment failed the benchmark integrity check")
        return {"ok": True, "size": actual_size, "sha256": actual_digest}
    if op == "logs":
        payload = read_runtime_logs(
            server.data_dir, limit=int(request.get("limit") or 200)
        )
        return {"ok": True, **payload}
    if op == "status":
        capabilities = await server.dispatch("system.capabilities", {})
        methods = {
            name
            for item in capabilities["capabilities"]
            for name in item.get("methods", [])
        }
        return {
            "ok": True,
            "current_seq": server.current_seq,
            "server_instance_id": server.server_instance_id,
            "event_limit": server.event_limit,
            "event_queue_limit": server.event_queue_limit,
            "hostname": platform.node(),
            "platform": platform.system().lower(),
            "machine": platform.machine(),
            "methods": sorted(methods),
        }
    return {"ok": False, "error": f"unknown op {op}"}


async def _control_loop(
    server: RuntimeServer,
    unix_path: Path,
    tcp_listen: tuple[str, int] | None,
) -> None:
    unix_path.parent.mkdir(parents=True, exist_ok=True)
    unix_path.unlink(missing_ok=True)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _handle_control(server, reader, writer)

    unix = await asyncio.start_unix_server(handler, path=str(unix_path))
    print(f"CONTROL {unix_path}", flush=True)
    servers = [unix]
    if tcp_listen is not None:
        tcp = await asyncio.start_server(handler, tcp_listen[0], tcp_listen[1])
        servers.append(tcp)
        print(f"CONTROL_TCP {tcp_listen[0]}:{tcp_listen[1]}", flush=True)
    try:
        await asyncio.gather(*(item.serve_forever() for item in servers))
    finally:
        for item in servers:
            item.close()
            await item.wait_closed()


async def run(args: argparse.Namespace) -> None:
    host, port_text = args.listen.rsplit(":", 1)
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime_service(
        config_path=args.data_dir / "config.json",
        data_dir=args.data_dir,
    )
    server = RuntimeServer(
        runtime,
        args.unix,
        (args.workspace,),
        args.data_dir,
        listen=(host, int(port_text)),
        event_limit=args.event_limit,
        event_queue_limit=args.event_queue_limit,
        event_queue_bytes=args.event_queue_bytes,
    )
    tcp_listen = None
    if args.control_tcp:
        tcp_host, tcp_port = args.control_tcp.rsplit(":", 1)
        tcp_listen = (tcp_host, int(tcp_port))
    control_task = asyncio.create_task(_control_loop(server, args.control_unix, tcp_listen))
    try:
        print(f"LISTEN {host}:{port_text}", flush=True)
        await server.run()
    finally:
        control_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await control_task
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--unix", type=Path, required=True)
    parser.add_argument("--listen", required=True)
    parser.add_argument("--control-unix", type=Path, required=True)
    parser.add_argument("--control-tcp")
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--event-queue-limit", type=int)
    parser.add_argument("--event-queue-bytes", type=int)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
