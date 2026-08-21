from __future__ import annotations

import asyncio
import hashlib
import importlib.util
from pathlib import Path

import pytest

from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.runtime_server import RuntimeServer


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owned_cleanup_requires_env_marker_and_harness_cmdline() -> None:
    cleanup = _load_tool("r3_owned_cleanup.py")
    prefix = "/root/aeloon-remote-test/runs/demo"
    assert cleanup.is_owned(
        b"AELOON_R3_OWNED=/root/aeloon-remote-test/runs/demo/case\0HOME=/root\0",
        b"python\0tools/r3_test_server.py\0--data-dir\0/root/aeloon-remote-test/runs/demo/case\0",
        prefix,
    )
    assert not cleanup.is_owned(
        b"HOME=/root\0",
        b"python\0tools/r3_test_server.py\0",
        prefix,
    )
    assert not cleanup.is_owned(
        b"AELOON_R3_OWNED=/root/aeloon-remote-test/runs/demo\0",
        b"sshd: root@notty\0",
        prefix,
    )
    assert not cleanup.is_owned(
        b"AELOON_R3_OWNED=/tmp/other\0",
        b"python\0tools/r3_test_server.py\0",
        prefix,
    )


def test_unique_attachment_blobs_and_pairing_code() -> None:
    bench = _load_tool("r3_runtime_bench.py")
    blobs = bench.unique_attachment_blobs(3, size=64)
    assert len(blobs) == 3
    assert len({item for item in blobs}) == 3
    assert all(len(item) == 64 for item in blobs)
    assert (
        bench.pairing_code(
            "aeloon://pair?v=2&endpoint=wss:%2F%2F127.0.0.1:7431&code=ABC123"
        )
        == "ABC123"
    )


def test_remote_attachment_budget_tracks_independent_link_baseline() -> None:
    bench = _load_tool("r3_runtime_bench.py")
    common = {
        "path": "ssh-tunnel",
        "host_info": {},
        "pty_samples": [0.1],
        "thread_samples": [0.2],
        "encoded_sizes": [1024],
        "first_samples": [0.3],
        "throughput": {"required_1000_delivered": True},
        "methods": set(),
        "link_samples": [8.0, 8.0, 8.0],
    }
    passing = bench.build_report(attach_samples=[32.0], **common)
    assert passing["attachment_25mib"]["budget_model"] == "bandwidth-relative"
    assert passing["attachment_25mib"]["projected_from_link_p95_ms"] == pytest.approx(
        33_333.335, abs=0.01
    )
    assert passing["attachment_25mib"]["budget_p95_ms"] == pytest.approx(
        41_666.66875, abs=0.01
    )
    assert bench.budgets_hold(passing)

    failing = bench.build_report(attach_samples=[45.0], **common)
    assert not bench.budgets_hold(failing)


@pytest.mark.asyncio
async def test_link_upload_probe_validates_size_and_digest() -> None:
    server_mod = _load_tool("r3_test_server.py")
    payload = b"raw-link-probe" * 1024
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    result = await server_mod._receive_link_upload(
        reader,
        {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    assert result == {
        "ok": True,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.mark.asyncio
async def test_control_dispatch_seeds_turns_and_exposes_status(tmp_path: Path) -> None:
    server_mod = _load_tool("r3_test_server.py")
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(runtime, tmp_path / "runtime.sock", (tmp_path,), data_dir)
    try:
        roots = await server.dispatch("workspace.roots", {})
        project = await server.dispatch(
            "project.add",
            {"root_id": roots["roots"][0]["id"], "relative_path": "."},
        )
        created = await server.dispatch(
            "thread.create",
            {"project_id": project["project"]["id"], "kind": "standard"},
        )
        seeded = await server_mod._dispatch_control(
            server,
            {
                "op": "seed_turns",
                "thread_id": created["thread"]["id"],
                "count": 2,
                "user_bytes": 4,
                "assistant_bytes": 8,
            },
        )
        assert seeded["ok"] is True
        snapshot = await server.dispatch("thread.get", {"thread_id": created["thread"]["id"]})
        assert len(snapshot["turns"]) == 2
        status = await server_mod._dispatch_control(server, {"op": "status"})
        assert status["ok"] is True
        assert "diagnostics.logs" in status["methods"]
        assert "inject_benchmark_events" not in status["methods"]
        await server.inject_benchmark_events(1, payload_bytes=32)
        injected = await server_mod._dispatch_control(
            server, {"op": "inject", "count": 2, "payload_bytes": 32}
        )
        assert injected["current_seq"] == 3
        logs = await server_mod._dispatch_control(server, {"op": "logs", "limit": 20})
        assert logs["ok"] is True
        assert "entries" in logs
        attachment = server.store.add_attachment(
            name="verified.bin",
            mime_type="application/octet-stream",
            data=b"verified attachment",
            root=data_dir / "attachments",
        )
        verified = await server_mod._dispatch_control(
            server,
            {
                "op": "verify_attachment",
                "attachment_id": attachment["id"],
                "size": len(b"verified attachment"),
                "sha256": hashlib.sha256(b"verified attachment").hexdigest(),
            },
        )
        assert verified["ok"] is True
    finally:
        await runtime.close()
        server.store.close()
