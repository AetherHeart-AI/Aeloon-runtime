from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from aeloon_core.browser import BROWSER_TOOL_NAMES
from aeloon_core.config import Config, save_config
from aeloon_core.rpc import AeloonRpcAdapter, RpcError
from aeloon_core.rpc.server import AeloonRpcServer, rpc_request
from aeloon_core.runtime import RuntimeService
from aeloon_core.runtime.providers import DEEPSEEK_MODELS, BaseProvider, ProviderManager


class EmptyProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(provider_id="deepseek", name="Test", endpoint="scripted://local")

    async def models(self):
        return dict(DEEPSEEK_MODELS)

    def stream(self, _model, _context, _options):
        raise AssertionError("Inference is not expected in RPC boundary tests")


def runtime_service(tmp_path: Path, *, browser: bool = False) -> RuntimeService:
    config_path = tmp_path / "config.json"
    save_config(
        Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            agent={"model": "deepseek/deepseek-v4-flash"},
        ),
        config_path,
    )
    return RuntimeService(
        config_path=config_path,
        browser_runtime_socket=(tmp_path / "browser.sock" if browser else None),
        provider_manager_factory=lambda config: ProviderManager(
            config,
            driver_factories={
                "deepseek": lambda _provider_id, _configured, _account: EmptyProvider()
            },
        ),
    )


@pytest.mark.asyncio
async def test_rpc_v2_rejects_every_legacy_handshake(tmp_path: Path) -> None:
    adapter = AeloonRpcAdapter(runtime_service(tmp_path))
    try:
        with pytest.raises(RpcError) as incompatible:
            await adapter.dispatch("system.handshake", {"protocol_versions": [3]})
        assert incompatible.value.code == "protocol_incompatible"

        with pytest.raises(RpcError) as legacy:
            await adapter.dispatch("system.handshake", {"protocol": "aeloon-rpc-v1"})
        assert legacy.value.code == "protocol_incompatible"

        handshake = await adapter.dispatch(
            "system.handshake",
            {"protocol": "aeloon-rpc-v2", "attachment_roots": [str(tmp_path)]},
        )
        assert handshake["protocol"] == "aeloon-rpc-v2"
        assert handshake["core_version"] == "0.0.12"
        assert len(handshake["core_commit"]) == 40
        assert "protocol_version" not in handshake
        assert "capabilities" not in handshake
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_ui_thread_id_is_the_core_session_id(tmp_path: Path) -> None:
    adapter = AeloonRpcAdapter(runtime_service(tmp_path))
    thread_id = "5d66dd9e-72d5-435c-a454-216817d1945c"
    try:
        created = await adapter.dispatch(
            "session.create",
            {"session_id": thread_id, "workspace": str(tmp_path), "title": "Desktop"},
        )
        assert created["session_id"] == thread_id
        fetched = await adapter.dispatch("session.get", {"session_id": thread_id})
        assert fetched["metadata"]["session_id"] == thread_id
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_browser_tools_are_process_scoped_and_always_in_catalogue(tmp_path: Path) -> None:
    adapter = AeloonRpcAdapter(runtime_service(tmp_path, browser=True))
    try:
        handshake = await adapter.dispatch("system.handshake", {"protocol": "aeloon-rpc-v2"})
        assert handshake["browser_runtime"] is True
        catalog = await adapter.dispatch("catalog.get", {"workspace": str(tmp_path)})
        names = {item["name"] for item in catalog["tools"]}
        assert set(BROWSER_TOOL_NAMES).issubset(names)
        assert len(BROWSER_TOOL_NAMES) == 22
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_length_framed_rpc_round_trip_and_shutdown(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aeloon-rpc-test-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    server = AeloonRpcServer(AeloonRpcAdapter(runtime_service(tmp_path)), socket_path)
    task = asyncio.create_task(server.run())
    try:
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.01)
        result = await rpc_request(
            socket_path,
            "system.handshake",
            {"protocol": "aeloon-rpc-v2"},
        )
        assert result["protocol"] == "aeloon-rpc-v2"
        await rpc_request(socket_path, "system.shutdown")
        await asyncio.wait_for(task, 2)
    finally:
        if not task.done():
            server.adapter.request_shutdown()
            await task
        socket_path.unlink(missing_ok=True)
