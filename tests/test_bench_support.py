from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeloon_runtime.bench_support import seed_completed_turns
from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.runtime_server import RuntimeServer


@pytest.mark.asyncio
async def test_private_benchmark_generator_is_not_an_rpc_method(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeServer(
        runtime,
        tmp_path / "runtime.sock",
        (tmp_path,),
        data_dir,
        event_limit=32,
        event_queue_limit=8,
        event_queue_bytes=64 * 1024,
    )
    try:
        await server.inject_benchmark_events(3, payload_bytes=128)
        assert server.current_seq == 3
        capabilities = await server.dispatch("system.capabilities", {})
        methods = {
            name
            for item in capabilities["capabilities"]
            for name in item.get("methods", [])
        }
        assert "diagnostics.logs" in methods
        assert "inject_benchmark_events" not in methods
        assert "inject_benchmark_events_at_rate" not in methods
        manifest = json.loads(
            Path("aeloon_runtime/rpc/aeloon-rpc-v4.manifest.json").read_text(encoding="utf-8")
        )
        assert "inject_benchmark_events" not in manifest["methods"]
        assert "diagnostics.logs" in manifest["methods"]
        project = await server.dispatch(
            "project.add",
            {
                "root_id": __import__(
                    "aeloon_runtime.runtime_server", fromlist=["_root_id"]
                )._root_id(tmp_path),
                "relative_path": ".",
            },
        )
        created = await server.dispatch(
            "thread.create", {"project_id": project["project"]["id"], "kind": "standard"}
        )
        seed_completed_turns(
            server.store,
            created["thread"]["id"],
            count=2,
            user_bytes=8,
            assistant_bytes=16,
        )
        snapshot = await server.dispatch("thread.get", {"thread_id": created["thread"]["id"]})
        assert len(snapshot["turns"]) == 2
    finally:
        await runtime.close()
        server.store.close()
