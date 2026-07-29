"""Bridge kernel progress callbacks into structured UI event streams."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from aeloon_core.harness.execution.events import (
    ToolCallView,
    ToolExecutionRecord,
    ToolExecutionState,
    tool_result_failed,
)
from aeloon_core.harness.execution.transitions import accumulate_usage

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
LOG_TEXT_PREVIEW_CHARS = 240
WEB_TOOL_RESULT_CHARS = 16_000
_WORKER_ACTIVITY_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


class TurnEventProgress:
    """Accumulate a turn and emit structured progress events."""

    def __init__(
        self,
        *,
        session_id: str,
        emit: Emit,
    ) -> None:
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
        self.duration_ms: int | None = None
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
        block = await self._ensure_text_block(role="narration")
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
        component: str = "model",
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
                node_kind="model",
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

        normalized_node_kind = {
            "context_processing": "runtime",
            "domain": "model",
            "harness": "runtime",
        }.get(node_kind, node_kind)
        resolved_component = component or (
            "context_view"
            if node_kind == "context_processing"
            else normalized_node_kind
        )
        self._record_usage(
            usage,
            node_kind=normalized_node_kind,
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
                node_kind=normalized_node_kind,
                component=resolved_component,
                ts=_now(),
            ),
        )

    async def on_worker_lifecycle(
        self,
        *,
        event: str,
        worker_id: str,
        run_id: str,
        worker_type_id: str,
        status: str,
        run_sequence: int = 1,
        duration_ms: int | None = None,
        summary: str | None = None,
        usage: dict[str, Any] | None = None,
        objective: str | None = None,
        template_id: str | None = None,
        node_id: str | None = None,
        expert_id: str | None = None,
        runner_id: str | None = None,
        stage_id: str | None = None,
    ) -> None:
        """Publish bounded lifecycle data for an ephemeral Harness agent."""

        safe_summary = (
            _safe_worker_activity_text(summary, limit=1_000) if summary else ""
        )
        safe_usage = (
            {
                str(key): max(0, int(value))
                for key, value in (usage or {}).items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            if usage
            else {}
        )
        safe_objective = (
            _safe_worker_activity_text(objective, limit=500) if objective else ""
        )
        if safe_usage:
            component = (
                f"expert:{expert_id}:{stage_id}"
                if expert_id and stage_id
                else (
                    f"template:{template_id}:{node_id}"
                    if template_id and node_id
                    else f"worker:{worker_type_id}"
                )
            )
            self._record_usage(
                safe_usage,
                node_kind="worker",
                component=component,
            )
        await self.emit(
            "chat.worker.lifecycle",
            self._payload(
                phase=event,
                worker_id=worker_id,
                run_id=run_id,
                run_sequence=max(1, int(run_sequence)),
                worker_type_id=worker_type_id,
                status=status,
                ephemeral=True,
                duration_ms=duration_ms,
                **({"summary": safe_summary} if safe_summary else {}),
                **({"usage": safe_usage} if safe_usage else {}),
                **({"objective": safe_objective} if safe_objective else {}),
                **({"template_id": template_id} if template_id else {}),
                **({"node_id": node_id} if node_id else {}),
                **({"expert_id": expert_id} if expert_id else {}),
                **({"runner_id": runner_id} if runner_id else {}),
                **({"stage_id": stage_id} if stage_id else {}),
                ts=_now(),
            ),
        )

    async def on_tool_calls(
        self,
        tool_calls: list[ToolCallView],
        *,
        record_reasoning: bool = True,
    ) -> None:
        for tool_call in tool_calls:
            tool_detail = _tool_call_detail(tool_call)
            summary = f"Call {tool_call.name}"
            if record_reasoning:
                reasoning_data = {
                    "call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "summary": summary,
                }
                await self._append_reasoning_line(
                    summary,
                    kind="tool_call",
                    data=reasoning_data,
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

    async def on_tool_result(
        self,
        node: ToolExecutionRecord,
        *,
        record_reasoning: bool = True,
    ) -> None:
        raw_result = node.result if node.result is not None else node.error
        result = "" if raw_result is None else str(raw_result)
        failed = (
            node.state == ToolExecutionState.FAILED
            or node.error is not None
            or tool_result_failed(result)
        )
        status = "error" if failed else "done"
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
        block["duration_ms"] = duration_ms
        if record_reasoning:
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
        bounded_result = _bounded_web_tool_result(block["result"])
        ui_patch = {
            "name": node.tool_name,
            "arguments": node.arguments,
            "status": block["status"],
            "result": bounded_result,
            "result_truncated": len(block["result"]) > WEB_TOOL_RESULT_CHARS,
            "completed_at": block["completed_at"],
            "duration_ms": duration_ms,
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
        current_content = str((block or {}).get("content") or "")
        if block is None or not current_content.strip():
            block = await self._ensure_text_block(role="final")
            block["content"] = content
            block["role"] = "final"
            await self.emit(
                "chat.block.update",
                self._payload(
                    block_id=block["id"],
                    patch={"content": content, "role": "final"},
                    ts=_now(),
                ),
            )
        elif current_content.strip() != content.strip():
            # Streaming text may be process narration from an earlier Master model
            # round. Preserve it, but project the final answer as a distinct canonical
            # block for both the live UI and persisted session record.
            block = await self._ensure_block(
                None,
                "text",
                extra_fields={"role": "final"},
            )
            self._text_block_id = block["id"]
            block["content"] = content
            await self.emit(
                "chat.block.update",
                self._payload(
                    block_id=block["id"],
                    patch={"content": content, "role": "final"},
                    ts=_now(),
                ),
            )
        else:
            block["role"] = "final"
            await self.emit(
                "chat.block.update",
                self._payload(
                    block_id=block["id"],
                    patch={"role": "final"},
                    ts=_now(),
                ),
            )
        completed_at = _now()
        duration_ms = _duration_ms(self._turn_started_at, completed_at)
        self.duration_ms = duration_ms
        payload = self._payload(
            final=content,
            blocks=[_web_block_view(block) for block in self.blocks],
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

    async def _ensure_text_block(self, *, role: str) -> dict[str, Any]:
        block = await self._ensure_block(
            self._text_block_id,
            "text",
            extra_fields={"role": role},
        )
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
        line = clean
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


def _tool_call_detail(tool_call: ToolCallView) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def _bounded_web_tool_result(value: Any) -> str:
    text = str(value or "")
    if len(text) <= WEB_TOOL_RESULT_CHARS:
        return text
    omitted = len(text) - WEB_TOOL_RESULT_CHARS
    marker = f"\n… {omitted} characters omitted …\n"
    available = WEB_TOOL_RESULT_CHARS - len(marker)
    head = available // 2
    tail = available - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def _web_block_view(block: dict[str, Any]) -> dict[str, Any]:
    view = dict(block)
    if view.get("type") == "tool_call" and view.get("result") is not None:
        result = str(view["result"])
        view["result"] = _bounded_web_tool_result(result)
        view["result_truncated"] = len(result) > WEB_TOOL_RESULT_CHARS
    return view


def _safe_worker_activity_text(value: Any, *, limit: int = 100) -> str:
    text = _WORKER_ACTIVITY_ANSI.sub("", str(value or ""))
    text = "".join(
        " " if char.isspace() else char
        for char in text
        if char.isprintable() or char.isspace()
    )
    return " ".join(text.split())[:limit]


def _task_node_detail(node: ToolExecutionRecord) -> dict[str, Any]:
    return {
        "index": node.index,
        "call_id": node.call_id,
        "tool_name": node.tool_name,
        "arguments": node.arguments,
        "mode": node.mode,
        "deps": sorted(getattr(node, "deps", ())),
        "dependents": sorted(getattr(node, "dependents", ())),
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
