"""JSON-RPC-ish WebSocket request handling."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import WSMsgType, web

from aeloon_core.orchestrator import AeloonCoreOrchestrator
from server.bridge import WebUITurnProgress, encode_event


class RpcServer:
    """Handle WebSocket RPC requests and broadcast runtime events."""

    def __init__(self, orchestrator: AeloonCoreOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.websockets: set[web.WebSocketResponse] = set()
        self.running_tasks: dict[str, asyncio.Task[Any]] = {}

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.websockets.add(ws)
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await self._handle_text(ws, message.data)
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            self.websockets.discard(ws)
        return ws

    async def emit_to(self, ws: web.WebSocketResponse, event: str, payload: dict[str, Any]) -> None:
        if ws.closed:
            return
        await ws.send_str(encode_event(event, payload))

    async def broadcast(self, event: str, payload: dict[str, Any]) -> None:
        if not self.websockets:
            return
        await asyncio.gather(
            *(self.emit_to(ws, event, payload) for ws in list(self.websockets)),
            return_exceptions=True,
        )

    async def _handle_text(self, ws: web.WebSocketResponse, raw: str) -> None:
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(ws, None, "invalid_json", "Message is not valid JSON")
            return

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            await self._send_error(ws, request_id, "invalid_params", "params must be an object")
            return

        try:
            result = await self._dispatch(ws, str(method), params)
        except asyncio.CancelledError:
            await self._send_error(ws, request_id, "cancelled", "Request was cancelled")
            return
        except Exception as exc:
            await self._send_error(ws, request_id, "server_error", str(exc))
            return
        await self._send_result(ws, request_id, result)

    async def _dispatch(
        self,
        ws: web.WebSocketResponse,
        method: str,
        params: dict[str, Any],
    ) -> Any:
        if method == "debug.health":
            return {"ok": True, "tool_count": len(self.orchestrator.registry)}
        if method == "session.new":
            return {"session_id": self.orchestrator.sessions.new_session()}
        if method == "session.list":
            return {
                "sessions": [
                    summary.__dict__ for summary in self.orchestrator.sessions.list_sessions()
                ]
            }
        if method == "session.resume":
            session_id = str(params.get("session_id") or "")
            return {
                "session_id": session_id,
                "history": self.orchestrator.sessions.history(session_id),
            }
        if method == "session.delete":
            session_id = str(params.get("session_id") or "")
            return {"deleted": self.orchestrator.sessions.delete_session(session_id)}
        if method == "chat.history":
            session_id = str(params.get("session_id") or "")
            return {
                "session_id": session_id,
                "history": self.orchestrator.sessions.history(session_id),
            }
        if method == "chat.abort":
            session_id = str(params.get("session_id") or "")
            task = self.running_tasks.get(session_id)
            if task is None or task.done():
                return {"aborted": False}
            task.cancel()
            return {"aborted": True}
        if method == "chat.send":
            return await self._chat_send(ws, params)
        raise ValueError(f"Unknown method: {method}")

    async def _chat_send(
        self,
        ws: web.WebSocketResponse,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = str(params.get("message") or params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("chat.send requires params.message")
        session_id = str(params.get("session_id") or self.orchestrator.sessions.new_session())

        async def emit(event: str, payload: dict[str, Any]) -> None:
            await self.emit_to(ws, event, payload)

        progress = WebUITurnProgress(session_id=session_id, emit=emit)
        task = asyncio.create_task(
            self.orchestrator.run_turn(prompt, session_id=session_id, on_progress=progress)
        )
        self.running_tasks[session_id] = task
        try:
            result = await task
        finally:
            self.running_tasks.pop(session_id, None)
        return {
            "session_id": result.session_id,
            "final": result.final_content,
            "tools_used": result.tools_used,
            "blocks": result.blocks,
        }

    async def _send_result(
        self,
        ws: web.WebSocketResponse,
        request_id: Any,
        result: Any,
    ) -> None:
        await ws.send_str(
            json.dumps(
                {"type": "response", "id": request_id, "result": result},
                ensure_ascii=False,
            )
        )

    async def _send_error(
        self,
        ws: web.WebSocketResponse,
        request_id: Any,
        code: str,
        message: str,
    ) -> None:
        await ws.send_str(
            json.dumps(
                {
                    "type": "response",
                    "id": request_id,
                    "error": {"code": code, "message": message},
                },
                ensure_ascii=False,
            )
        )
