"""Bridge kernel progress callbacks into terminal event streams."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from aeloon_core.providers.base import ToolCallRequest
from aeloon_core.task_graph import TaskNode
from aeloon_core.transitions import accumulate_usage

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
        self._reasoning_raw_open = False
        self._started = False
        self._turn_started_at: str | None = None
        self.usage: dict[str, int] = {}
        self.usage_by_node_kind: dict[str, dict[str, int]] = {}
        self.usage_by_component: dict[str, dict[str, int]] = {}

    def _payload(self, **extra: Any) -> dict[str, Any]:
        """Build an event payload stamped with the session and turn ids."""

        return {"session_id": self.session_id, "turn_id": self.turn_id, **extra}

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        kind = "tool_hint" if tool_hint else "status"
        if not tool_hint and not _is_internal_status(text):
            await self._append_reasoning_line(text, kind="thought")
        detail = self._payload(event="chat.status", text=_text_summary(text), kind=kind)
        await self.emit("chat.status", self._payload(text=text, kind=kind, ts=_now()))
        await self._emit_log(
            level="INFO",
            source="kernel.status",
            message=_preview_text(text),
            detail=detail,
        )

    async def on_turn_start(self) -> None:
        self._started = True
        self._turn_started_at = _now()
        payload = self._payload(ts=self._turn_started_at)
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
        payload = self._payload(
            block_id=block["id"],
            delta=delta,
            content_length=len(str(block.get("content") or "")),
            ts=_now(),
        )
        await self.emit("chat.block.delta", payload)

    async def on_llm_reasoning_delta(self, delta: str) -> None:
        block = await self._ensure_reasoning_block()
        current = str(block.get("content") or "")
        needs_separator = (
            current and not current.endswith("\n") and not self._reasoning_raw_open
        )
        separator = "\n" if needs_separator else ""
        block["content"] = f"{current}{separator}{delta}"
        self._reasoning_raw_open = True
        payload = self._payload(
            block_id=block["id"],
            delta=delta,
            content_length=len(str(block.get("content") or "")),
            ts=_now(),
        )
        await self.emit("chat.block.delta", payload)

    async def on_llm_response(
        self,
        response: Any,
        *,
        component: str = "worker",
    ) -> None:
        reasoning = str(getattr(response, "reasoning_content", None) or "").strip()
        if reasoning:
            block = await self._ensure_reasoning_block()
            if reasoning not in str(block.get("content") or ""):
                current = str(block.get("content") or "")
                separator = "\n" if current and not current.endswith("\n") else ""
                block["content"] = f"{current}{separator}{reasoning}"
                self._reasoning_raw_open = True
                await self.emit(
                    "chat.block.update",
                    self._payload(
                        block_id=block["id"],
                        patch={"content": block["content"]},
                        ts=_now(),
                    ),
                )
        call_usage = getattr(response, "usage", {})
        if isinstance(call_usage, dict):
            self._record_usage(
                call_usage,
                node_kind="domain",
                component=component,
            )
        payload = self._payload(
            finish_reason=getattr(response, "finish_reason", None),
            component=component,
            usage=call_usage if isinstance(call_usage, dict) else {},
            call_usage=call_usage if isinstance(call_usage, dict) else {},
            aggregate_usage=dict(self.usage),
            aggregate_by_component={
                name: dict(values) for name, values in self.usage_by_component.items()
            },
            ts=_now(),
        )
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

    async def on_usage(
        self,
        usage: dict[str, Any],
        *,
        node_kind: str,
        component: str | None = None,
    ) -> None:
        """Record non-domain provider usage and publish the turn aggregate."""

        resolved_component = component or (
            "minimal_context" if node_kind == "context_processing" else node_kind
        )
        self._record_usage(
            usage,
            node_kind=node_kind,
            component=resolved_component,
        )
        await self.emit(
            "chat.usage",
            self._payload(
                usage=dict(self.usage),
                by_node_kind={
                    kind: dict(values) for kind, values in self.usage_by_node_kind.items()
                },
                by_component={
                    name: dict(values) for name, values in self.usage_by_component.items()
                },
                node_kind=node_kind,
                component=resolved_component,
                ts=_now(),
            ),
        )

    async def on_guard_decision(self, resolution: Any) -> None:
        """Account for a TemporaryGuard call without exposing response text."""

        usage = getattr(resolution, "usage", {})
        if isinstance(usage, dict) and usage:
            await self.on_usage(
                usage,
                node_kind="harness",
                component="temporary_guard",
            )

    async def on_profile_pinned(self, profile: dict[str, Any]) -> None:
        """Expose immutable turn provenance before any profile routing."""

        await self.emit(
            "chat.profile.pinned",
            self._payload(profile=_json_safe(profile), ts=_now()),
        )

    async def on_profile_route(
        self,
        agent_id: str,
        *,
        source: str,
        fallback_used: bool,
    ) -> None:
        await self.emit(
            "chat.profile.route",
            self._payload(
                agent_id=agent_id,
                source=source,
                fallback_used=fallback_used,
                ts=_now(),
            ),
        )
        await self._append_reasoning_line(
            f"Profile selected role {agent_id}",
            kind="profile_route",
            data={
                "agent_id": agent_id,
                "source": source,
                "fallback_used": fallback_used,
            },
        )

    async def on_profile_handoff(
        self,
        from_agent_id: str,
        recommended_agent_id: str | None,
        summary: str,
        *,
        handoff_count: int,
        handoff_limit: int,
    ) -> None:
        await self.emit(
            "chat.profile.handoff",
            self._payload(
                from_agent_id=from_agent_id,
                recommended_agent_id=recommended_agent_id,
                summary=summary,
                handoff_count=handoff_count,
                handoff_limit=handoff_limit,
                ts=_now(),
            ),
        )

    async def on_profile_completion(
        self,
        agent_id: str | None,
        final_content: str,
    ) -> None:
        await self.emit(
            "chat.profile.completion",
            self._payload(
                agent_id=agent_id,
                final=_text_summary(final_content),
                ts=_now(),
            ),
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
            payload = self._payload(block=block, ts=_now())
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
        payload = self._payload(block_id=node.call_id, patch=ui_patch, ts=_now())
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
            detail=self._payload(
                event="chat.block.update",
                block_id=node.call_id,
                patch=log_patch,
                ts=payload["ts"],
                task_node=_task_node_detail(node),
                block=_block_log_detail(block),
            ),
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
                    self._payload(
                        block_id=reasoning_block["id"],
                        patch={
                            "status": "done",
                            "completed_at": reasoning_block["completed_at"],
                        },
                        ts=_now(),
                    ),
                )
                await self._emit_log(
                    level="INFO",
                    source="reasoning.done",
                    message=f"reasoning block {reasoning_block['id']} completed",
                    detail=self._payload(
                        event="chat.block.update",
                        block=_block_log_detail(reasoning_block),
                    ),
                )
        block = None
        if self._text_block_id:
            block = self._find_block(self._text_block_id)
        if block is None or not str(block.get("content") or "").strip():
            block = await self._ensure_text_block()
            block["content"] = content
            await self.emit(
                "chat.block.update",
                self._payload(block_id=block["id"], patch={"content": content}, ts=_now()),
            )
        completed_at = _now()
        duration_ms = _duration_ms(self._turn_started_at, completed_at)
        payload = self._payload(
            final=content,
            blocks=self.blocks,
            duration_ms=duration_ms,
            ts=completed_at,
        )
        await self.emit("chat.turn.end", payload)
        await self._emit_log(
            level="INFO",
            source="chat.turn.end",
            message=f"turn {self.turn_id} ended",
            detail=self._payload(
                event="chat.turn.end",
                final=_text_summary(content),
                blocks=[_block_log_detail(item) for item in self.blocks],
                duration_ms=duration_ms,
                ts=completed_at,
            ),
        )

    async def _ensure_block(
        self,
        existing_id: str | None,
        block_type: str,
        *,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the current block of this type, creating and announcing a new one."""

        if existing_id:
            block = self._find_block(existing_id)
            if block is not None:
                return block
        block = {
            "id": f"{block_type}-{uuid.uuid4().hex[:10]}",
            "type": block_type,
            "role": "assistant",
            "content": "",
            **(extra_fields or {}),
            "created_at": _now(),
        }
        self.blocks.append(block)
        payload = self._payload(block=block, ts=_now())
        await self.emit("chat.block.add", payload)
        await self._emit_log(
            level="DEBUG",
            source="chat.block.add",
            message=f"{block_type} block {block['id']} added",
            detail=self._payload(
                event="chat.block.add",
                block=_block_log_detail(block),
                ts=payload["ts"],
            ),
        )
        return block

    async def _ensure_text_block(self) -> dict[str, Any]:
        block = await self._ensure_block(self._text_block_id, "text")
        self._text_block_id = block["id"]
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
        separator = "\n" if current else ""
        block["content"] = f"{current}{separator}{line}"
        self._reasoning_raw_open = False
        payload = self._payload(
            block_id=block["id"], patch={"content": block["content"]}, ts=_now()
        )
        await self.emit("chat.block.update", payload)
        await self._emit_log(
            level="DEBUG",
            source="reasoning.update",
            message=f"reasoning {kind}: {clean}",
            detail=self._payload(
                event="chat.block.update",
                block_id=block["id"],
                patch={"content": _text_summary(block["content"])},
                ts=payload["ts"],
                line=_text_summary(line),
                kind=kind,
                entry=entry,
            ),
        )

    async def _ensure_reasoning_block(self) -> dict[str, Any]:
        block = await self._ensure_block(
            self._reasoning_block_id, "reasoning", extra_fields={"status": "running"}
        )
        self._reasoning_block_id = block["id"]
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

    def _record_usage(
        self,
        usage: dict[str, Any],
        *,
        node_kind: str,
        component: str,
    ) -> None:
        accumulate_usage(self.usage, usage)
        bucket = self.usage_by_node_kind.setdefault(node_kind, {})
        accumulate_usage(bucket, usage)
        component_bucket = self.usage_by_component.setdefault(component, {})
        accumulate_usage(component_bucket, usage)


def _tool_call_detail(tool_call: ToolCallRequest) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "openai_tool_call": tool_call.to_openai_tool_call(),
    }


def _task_node_detail(node: TaskNode) -> dict[str, Any]:
    return {
        "index": node.index,
        "call_id": node.call_id,
        "tool_name": node.tool_name,
        "arguments": node.arguments,
        "mode": node.mode,
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


def _is_internal_status(text: str) -> bool:
    clean = text.strip()
    return bool(re.fullmatch(r"Thinking(?: \(step \d+\))?\.\.\.", clean))


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
