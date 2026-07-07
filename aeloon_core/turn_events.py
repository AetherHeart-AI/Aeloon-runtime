"""Bridge kernel progress callbacks into terminal event streams."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from aeloon_core.providers.base import ToolCallRequest
from aeloon_core.task_graph import TaskNode

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
LOG_TEXT_PREVIEW_CHARS = 240


class TurnEventProgress:
    """Accumulate a turn and emit structured progress events."""

    def __init__(self, *, session_id: str, emit: Emit) -> None:
        self.session_id = session_id
        self.turn_id = uuid.uuid4().hex[:12]
        self.emit = emit
        self.blocks: list[dict[str, Any]] = []
        self._text_block_id: str | None = None
        self._reasoning_block_id: str | None = None
        self._started = False
        self._turn_started_at: str | None = None

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        kind = "tool_hint" if tool_hint else "status"
        await self._append_reasoning_line(text, kind=kind)
        detail = {
            "event": "chat.status",
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "text": _text_summary(text),
            "kind": kind,
        }
        await self.emit(
            "chat.status",
            {
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "text": text,
                "kind": "tool_hint" if tool_hint else "status",
                "ts": _now(),
            },
        )
        await self._emit_log(
            level="INFO",
            source="kernel.status",
            message=_preview_text(text),
            detail=detail,
        )

    async def on_turn_start(self) -> None:
        self._started = True
        self._turn_started_at = _now()
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "ts": self._turn_started_at,
        }
        await self.emit("chat.turn.start", payload)
        await self._emit_log(
            level="INFO",
            source="chat.turn.start",
            message=f"turn {self.turn_id} started",
            detail={"event": "chat.turn.start", **payload},
        )

    async def on_llm_delta(self, delta: str) -> None:
        block = await self._ensure_text_block()
        block["content"] = str(block.get("content") or "") + delta
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "block_id": block["id"],
            "delta": delta,
            "content_length": len(str(block.get("content") or "")),
            "ts": _now(),
        }
        await self.emit("chat.block.delta", payload)

    async def on_llm_reasoning_delta(self, delta: str) -> None:
        block = await self._ensure_reasoning_block()
        block["content"] = str(block.get("content") or "") + delta
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "block_id": block["id"],
            "delta": delta,
            "content_length": len(str(block.get("content") or "")),
            "ts": _now(),
        }
        await self.emit("chat.block.delta", payload)

    async def on_llm_response(self, response: Any) -> None:
        reasoning = str(getattr(response, "reasoning_content", None) or "").strip()
        if reasoning:
            block = await self._ensure_reasoning_block()
            if reasoning not in str(block.get("content") or ""):
                separator = "\n" if block.get("content") else ""
                block["content"] = f"{block.get('content') or ''}{separator}{reasoning}"
                await self.emit(
                    "chat.block.update",
                    {
                        "session_id": self.session_id,
                        "turn_id": self.turn_id,
                        "block_id": block["id"],
                        "patch": {"content": block["content"]},
                        "ts": _now(),
                    },
                )
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "finish_reason": getattr(response, "finish_reason", None),
            "usage": getattr(response, "usage", {}),
            "ts": _now(),
        }
        assistant_block = self._find_block(self._text_block_id) if self._text_block_id else None
        reasoning_block = (
            self._find_block(self._reasoning_block_id) if self._reasoning_block_id else None
        )
        assistant_content = str((assistant_block or {}).get("content") or "")
        reasoning_content = str((reasoning_block or {}).get("content") or "")
        await self.emit("chat.llm.response", payload)
        await self._emit_log(
            level="INFO",
            source="llm.response",
            message=f"LLM stream complete finish={payload['finish_reason']}",
            detail={
                "event": "chat.llm.response",
                **payload,
                "content": _text_summary(getattr(response, "content", None)),
                "reasoning_content": _text_summary(getattr(response, "reasoning_content", None)),
                "assistant_block_content": _text_summary(assistant_content),
                "reasoning_block_content": _text_summary(reasoning_content),
                "thinking_blocks": _text_summary(getattr(response, "thinking_blocks", None)),
                "tool_calls": [
                    _tool_call_detail(tool_call)
                    for tool_call in getattr(response, "tool_calls", []) or []
                ],
            },
        )

    async def on_tool_calls(self, tool_calls: list[ToolCallRequest]) -> None:
        for tool_call in tool_calls:
            tool_detail = _tool_call_detail(tool_call)
            summary = f"Call {tool_call.name}"
            await self._append_reasoning_line(
                summary,
                kind="tool_call",
                data={
                    "call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "summary": summary,
                },
            )
            block = {
                "id": tool_call.id,
                "type": "tool_call",
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "running",
                "result": None,
                "created_at": _now(),
            }
            self.blocks.append(block)
            payload = {
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "block": block,
                "ts": _now(),
            }
            await self.emit("chat.block.add", payload)
            await self._emit_log(
                level="INFO",
                source="tool.call",
                message=f"tool call {tool_call.name}",
                detail={
                    "event": "chat.block.add",
                    **payload,
                    "tool_call": tool_detail,
                },
            )

    async def on_tool_result(self, node: TaskNode) -> None:
        result = str(node.result or "")
        status = "error" if result.startswith("Error") else "done"
        block = self._find_block(node.call_id)
        if block is None:
            block = {
                "id": node.call_id,
                "type": "tool_call",
                "name": node.tool_name,
                "arguments": node.arguments,
                "created_at": _now(),
            }
            self.blocks.append(block)
        block["status"] = status
        block["result"] = result
        block["completed_at"] = _now()
        duration_ms = _duration_ms(block.get("created_at"), block["completed_at"])
        await self._append_reasoning_line(
            f"{node.tool_name} returned {len(result)} characters",
            kind="tool_result",
            data={
                "call_id": node.call_id,
                "tool_name": node.tool_name,
                "arguments": node.arguments,
                "status": status,
                "result_length": len(result),
                "duration_ms": duration_ms,
                "summary": f"{node.tool_name} returned {len(result)} characters",
            },
        )
        ui_patch = {
            "status": block["status"],
            "result": block["result"],
            "completed_at": block["completed_at"],
        }
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "block_id": node.call_id,
            "patch": ui_patch,
            "ts": _now(),
        }
        await self.emit("chat.block.update", payload)
        log_patch = {
            "status": block["status"],
            "result": _text_summary(block["result"]),
            "completed_at": block["completed_at"],
            "duration_ms": duration_ms,
        }
        await self._emit_log(
            level="ERROR" if block["status"] == "error" else "INFO",
            source="tool.result",
            message=f"tool {node.tool_name} -> {block['status']}",
            detail={
                "event": "chat.block.update",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "block_id": node.call_id,
                "patch": log_patch,
                "ts": payload["ts"],
                "task_node": _task_node_detail(node),
                "block": _block_log_detail(block),
            },
        )

    async def on_final(self, content: str, **kwargs: Any) -> None:
        del kwargs
        if self._reasoning_block_id:
            reasoning_block = self._find_block(self._reasoning_block_id)
            if reasoning_block is not None:
                reasoning_block["status"] = "done"
                reasoning_block["completed_at"] = _now()
                await self.emit(
                    "chat.block.update",
                    {
                        "session_id": self.session_id,
                        "turn_id": self.turn_id,
                        "block_id": reasoning_block["id"],
                        "patch": {
                            "status": "done",
                            "completed_at": reasoning_block["completed_at"],
                        },
                        "ts": _now(),
                    },
                )
                await self._emit_log(
                    level="INFO",
                    source="reasoning.done",
                    message=f"reasoning block {reasoning_block['id']} completed",
                    detail={
                        "event": "chat.block.update",
                        "session_id": self.session_id,
                        "turn_id": self.turn_id,
                        "block": _block_log_detail(reasoning_block),
                    },
                )
        block = None
        if self._text_block_id:
            block = self._find_block(self._text_block_id)
        if block is None or not str(block.get("content") or "").strip():
            block = await self._ensure_text_block()
            block["content"] = content
            await self.emit(
                "chat.block.update",
                {
                    "session_id": self.session_id,
                    "turn_id": self.turn_id,
                    "block_id": block["id"],
                    "patch": {"content": content},
                    "ts": _now(),
                },
            )
        completed_at = _now()
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "final": content,
            "blocks": self.blocks,
            "ts": completed_at,
        }
        await self.emit("chat.turn.end", payload)
        await self._emit_log(
            level="INFO",
            source="chat.turn.end",
            message=f"turn {self.turn_id} ended",
            detail={
                "event": "chat.turn.end",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "final": _text_summary(content),
                "blocks": [_block_log_detail(item) for item in self.blocks],
                "duration_ms": _duration_ms(self._turn_started_at, completed_at),
                "ts": completed_at,
            },
        )

    async def _ensure_text_block(self) -> dict[str, Any]:
        if self._text_block_id:
            block = self._find_block(self._text_block_id)
            if block is not None:
                return block
        block = {
            "id": f"text-{uuid.uuid4().hex[:10]}",
            "type": "text",
            "role": "assistant",
            "content": "",
            "created_at": _now(),
        }
        self._text_block_id = block["id"]
        self.blocks.append(block)
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "block": block,
            "ts": _now(),
        }
        await self.emit("chat.block.add", payload)
        await self._emit_log(
            level="DEBUG",
            source="chat.block.add",
            message=f"text block {block['id']} added",
            detail={
                "event": "chat.block.add",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "block": _block_log_detail(block),
                "ts": payload["ts"],
            },
        )
        return block

    async def _append_reasoning_line(
        self,
        text: str,
        *,
        kind: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        clean = text.strip()
        if not clean:
            return
        block = await self._ensure_reasoning_block()
        entry = _json_safe({"text": clean, **(data or {})}) if data else {"text": clean}
        rendered = json.dumps(entry, ensure_ascii=False) if data else clean
        line = f"{_now()} [{kind}] {rendered}"
        current = str(block.get("content") or "")
        block["content"] = f"{current}\n{line}" if current else line
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "block_id": block["id"],
            "patch": {"content": block["content"]},
            "ts": _now(),
        }
        await self.emit("chat.block.update", payload)
        await self._emit_log(
            level="DEBUG",
            source="reasoning.update",
            message=f"reasoning {kind}: {clean}",
            detail={
                "event": "chat.block.update",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "block_id": block["id"],
                "patch": {"content": _text_summary(block["content"])},
                "ts": payload["ts"],
                "line": _text_summary(line),
                "kind": kind,
                "entry": entry,
            },
        )

    async def _ensure_reasoning_block(self) -> dict[str, Any]:
        if self._reasoning_block_id:
            block = self._find_block(self._reasoning_block_id)
            if block is not None:
                return block
        block = {
            "id": f"reasoning-{uuid.uuid4().hex[:10]}",
            "type": "reasoning",
            "role": "assistant",
            "content": "",
            "status": "running",
            "created_at": _now(),
        }
        self._reasoning_block_id = block["id"]
        self.blocks.append(block)
        payload = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "block": block,
            "ts": _now(),
        }
        await self.emit("chat.block.add", payload)
        await self._emit_log(
            level="DEBUG",
            source="chat.block.add",
            message=f"reasoning block {block['id']} added",
            detail={
                "event": "chat.block.add",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "block": _block_log_detail(block),
                "ts": payload["ts"],
            },
        )
        return block

    async def _emit_log(
        self,
        *,
        level: str,
        source: str,
        message: str,
        detail: dict[str, Any],
    ) -> None:
        await self.emit(
            "log.entry",
            {
                "level": level,
                "message": message,
                "source": source,
                "session_id": self.session_id,
                "ts": _now(),
                "detail": _json_safe(detail),
            },
        )

    def _find_block(self, block_id: str) -> dict[str, Any] | None:
        for block in self.blocks:
            if block.get("id") == block_id:
                return block
        return None


def _tool_call_detail(tool_call: ToolCallRequest) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "provider_specific_fields": tool_call.provider_specific_fields,
        "function_provider_specific_fields": tool_call.function_provider_specific_fields,
        "openai_tool_call": tool_call.to_openai_tool_call(),
    }


def _task_node_detail(node: TaskNode) -> dict[str, Any]:
    return {
        "index": node.index,
        "call_id": node.call_id,
        "tool_name": node.tool_name,
        "arguments": node.arguments,
        "mode": node.mode,
        "resources": [
            {"kind": resource.kind, "key": resource.key, "access": resource.access}
            for resource in node.resources
        ],
        "deps": sorted(node.deps),
        "dependents": sorted(node.dependents),
        "state": str(node.state),
        "result": _text_summary(node.result),
        "error": node.error,
    }


def _block_log_detail(block: dict[str, Any]) -> dict[str, Any]:
    detail = {
        "id": block.get("id"),
        "type": block.get("type"),
        "role": block.get("role"),
        "name": block.get("name"),
        "status": block.get("status"),
        "arguments": block.get("arguments"),
        "created_at": block.get("created_at"),
        "completed_at": block.get("completed_at"),
    }
    if "content" in block:
        detail["content"] = _text_summary(block.get("content"))
    if "result" in block:
        detail["result"] = _text_summary(block.get("result"))
    completed_at = block.get("completed_at")
    created_at = block.get("created_at")
    if isinstance(created_at, str) and isinstance(completed_at, str):
        detail["duration_ms"] = _duration_ms(created_at, completed_at)
    return {key: value for key, value in detail.items() if value is not None}


def _text_summary(value: Any, *, limit: int = LOG_TEXT_PREVIEW_CHARS) -> dict[str, Any]:
    text = "" if value is None else str(value)
    return {
        "preview": _preview_text(text, limit=limit),
        "length": len(text),
        "truncated": len(text) > limit,
    }


def _preview_text(value: Any, *, limit: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit]


def _duration_ms(start: Any, end: Any) -> int | None:
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        start_at = datetime.fromisoformat(start)
        end_at = datetime.fromisoformat(end)
    except ValueError:
        return None
    return max(0, round((end_at - start_at).total_seconds() * 1000))


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _now() -> str:
    return datetime.now(UTC).isoformat()
