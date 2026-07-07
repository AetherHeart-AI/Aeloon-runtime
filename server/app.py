"""Aiohttp application factory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from aiohttp import web
from loguru import logger

from aeloon_core.config import Config
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from server.rpc import RpcServer

ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = ROOT / "web" / "dist"


def create_app(config: Config) -> web.Application:
    """Create the local web app."""

    orchestrator = AeloonCoreOrchestrator(config)
    rpc = RpcServer(orchestrator)
    app = web.Application()
    app["rpc"] = rpc
    app.router.add_get("/ws", rpc.websocket_handler)
    app.router.add_get("/health", _health)

    if WEB_DIST.exists():
        assets = WEB_DIST / "assets"
        if assets.exists():
            app.router.add_static("/assets", assets)
        app.router.add_get("/", _index)
        app.router.add_get("/{tail:.*}", _index)
    else:
        app.router.add_get("/", _missing_frontend)

    app.on_startup.append(_install_log_sink)
    return app


async def _health(request: web.Request) -> web.Response:
    rpc: RpcServer = request.app["rpc"]
    return web.json_response({"ok": True, "tool_count": len(rpc.orchestrator.registry)})


async def _index(request: web.Request) -> web.FileResponse:
    del request
    return web.FileResponse(WEB_DIST / "index.html")


async def _missing_frontend(request: web.Request) -> web.Response:
    del request
    return web.Response(
        text="Aeloon Core server is running. Build web/ first to serve the UI.",
        content_type="text/plain",
    )


async def _install_log_sink(app: web.Application) -> None:
    loop = asyncio.get_running_loop()
    rpc: RpcServer = app["rpc"]

    def sink(message: Any) -> None:
        record = message.record
        exception = record.get("exception")
        payload = {
            "level": record["level"].name,
            "message": record["message"],
            "source": "loguru",
            "ts": record["time"].isoformat(),
            "detail": _json_safe(
                {
                    "logger": {
                        "name": record["name"],
                        "module": record["module"],
                        "function": record["function"],
                        "line": record["line"],
                        "file": {
                            "name": record["file"].name,
                            "path": record["file"].path,
                        },
                        "process": {
                            "id": record["process"].id,
                            "name": record["process"].name,
                        },
                        "thread": {
                            "id": record["thread"].id,
                            "name": record["thread"].name,
                        },
                        "elapsed": str(record["elapsed"]),
                        "extra": record["extra"],
                        "exception": str(exception) if exception else None,
                    }
                }
            ),
        }
        loop.create_task(rpc.broadcast("log.entry", payload))

    logger.add(sink, level="INFO")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
