"""Privacy-preserving bridge from private Worker activity to Base UI events."""

from __future__ import annotations

import inspect
import re
import unicodedata
from time import perf_counter
from typing import Any

from loguru import logger

from aeloon_core.loop_guard import tool_result_failed
from aeloon_core.providers.base import ToolCallRequest
from aeloon_core.task_graph import TaskNode

_HIDDEN_CONTROL_TOOLS = {
    "complete_task",
    "delegate_tasks",
    "handoff",
    "request_handoff",
}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
_ACTIVITY_PHASES = {
    "analyzing",
    "planning",
    "drafting",
    "using_tool",
    "processing",
    "working_step",
    "finalizing",
    "delegating",
    "handoff",
    "branch_running",
    "branch_done",
    "synthesizing",
}


class WorkerProgress:
    """Forward sanitized activity while swallowing all Worker-authored text."""

    def __init__(
        self,
        *,
        parent: Any,
        worker_id: str,
        run_id: str,
        profile_id: str,
    ) -> None:
        self.parent = parent
        self.worker_id = worker_id
        self.run_id = run_id
        self.profile_id = profile_id
        self.label = f"{profile_id}#{worker_id[:8]}"
        self._tool_started: dict[str, float] = {}
        self._active_tools: dict[str, dict[str, str]] = {}
        self._current_steps: dict[str, tuple[str, int, int]] = {}
        self._role_ids: dict[str, str] = {}
        self._last_activity: dict[str, tuple[Any, ...]] = {}
        self._activity_revisions: dict[str, int] = {}

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        del text, tool_hint

    async def on_llm_delta(self, delta: str) -> None:
        del delta
        await self._emit_activity("drafting", label=self.label)

    async def on_llm_reasoning_delta(self, delta: str) -> None:
        del delta

    async def on_llm_response(self, response: Any, *, component: str = "worker") -> None:
        del component
        content = getattr(response, "content", None)
        tool_calls = getattr(response, "tool_calls", None)
        if content and not tool_calls:
            await self._emit_activity("drafting", label=self.label)

    async def on_usage(
        self,
        usage: dict[str, Any],
        *,
        node_kind: str,
        component: str | None = None,
    ) -> None:
        del usage, node_kind, component

    async def on_guard_decision(self, resolution: Any) -> None:
        del resolution

    async def on_final(self, content: str, **kwargs: Any) -> None:
        del content, kwargs
        await self._emit_activity("finalizing", label=self.label)

    async def on_agent_activity(
        self,
        *,
        phase: str,
        role_id: str | None = None,
        subagent_label: str | None = None,
    ) -> None:
        label = self._scope_label(subagent_label)
        safe_role = _safe_identifier(role_id)
        if safe_role is not None:
            self._role_ids[label] = safe_role
        await self._emit_activity(phase, label=label, role_id=safe_role)

    async def on_profile_route(
        self,
        agent_id: str,
        *,
        source: str,
        fallback_used: bool,
    ) -> None:
        del source, fallback_used
        role_id = _safe_identifier(agent_id)
        if role_id is not None:
            self._role_ids[self.label] = role_id
        await self._emit_activity("analyzing", label=self.label, role_id=role_id)

    async def on_profile_handoff(
        self,
        from_agent_id: str,
        recommended_agent_id: str | None,
        summary: str,
        **kwargs: Any,
    ) -> None:
        del from_agent_id, summary, kwargs
        role_id = _safe_identifier(recommended_agent_id)
        await self._emit_activity("handoff", label=self.label, role_id=role_id)

    async def on_profile_completion(
        self,
        agent_id: str | None,
        final_content: str,
    ) -> None:
        del final_content
        await self._emit_activity(
            "finalizing",
            label=self.label,
            role_id=_safe_identifier(agent_id),
        )

    async def on_profile_delegate_branch_start(
        self,
        branch_id: str,
        label: str,
        agent_id: str,
        task: str,
    ) -> None:
        del branch_id, task
        await self._emit_activity(
            "branch_running",
            label=self._scope_label(label),
            role_id=_safe_identifier(agent_id),
        )

    async def on_profile_delegate_branch_complete(
        self,
        branch_id: str,
        label: str,
        agent_id: str,
        *,
        status: str,
        summary: str,
        duration_ms: int,
        tools_used: list[str],
    ) -> None:
        del branch_id, status, summary, duration_ms, tools_used
        await self._emit_activity(
            "branch_done",
            label=self._scope_label(label),
            role_id=_safe_identifier(agent_id),
        )

    async def on_profile_delegate_join(
        self,
        source_agent_id: str,
        **kwargs: Any,
    ) -> None:
        del kwargs
        await self._emit_activity(
            "synthesizing",
            label=self.label,
            role_id=_safe_identifier(source_agent_id),
        )

    async def on_tool_calls(
        self,
        tool_calls: list[ToolCallRequest],
        *,
        subagent_label: str | None = None,
        record_reasoning: bool = False,
    ) -> None:
        del record_reasoning
        label = self._scope_label(subagent_label)
        now = perf_counter()
        visible_tools: list[str] = []
        control_tools: set[str] = set()
        for tool_call in tool_calls:
            if tool_call.name in _HIDDEN_CONTROL_TOOLS:
                control_tools.add(tool_call.name)
                continue
            self._tool_started[tool_call.id] = now
            safe_name = _safe_identifier(tool_call.name) or "tool"
            self._active_tools.setdefault(label, {})[tool_call.id] = safe_name
            visible_tools.append(safe_name)
        if visible_tools:
            await self._emit_activity(
                "using_tool",
                label=label,
                tool_names=tuple(sorted(set(visible_tools))),
            )
        elif "delegate_tasks" in control_tools:
            await self._emit_activity("delegating", label=label)
        elif control_tools & {"handoff", "request_handoff"}:
            await self._emit_activity("handoff", label=label)
        elif "complete_task" in control_tools:
            await self._emit_activity("finalizing", label=label)

    async def on_tool_result(
        self,
        node: TaskNode,
        *,
        subagent_label: str | None = None,
        record_reasoning: bool = False,
    ) -> None:
        del record_reasoning
        if node.tool_name in _HIDDEN_CONTROL_TOOLS:
            return
        label = self._scope_label(subagent_label)
        started = self._tool_started.pop(node.call_id, None)
        duration_ms = (
            max(0, int((perf_counter() - started) * 1_000))
            if started is not None
            else None
        )
        tool_name, status, metrics = _safe_tool_projection(node)
        await self._call_parent(
            "on_worker_tool_result",
            worker_id=self.worker_id,
            run_id=self.run_id,
            profile_id=self.profile_id,
            label=label,
            tool_name=tool_name,
            status=status,
            metrics=metrics,
            duration_ms=duration_ms,
        )
        active = self._active_tools.get(label)
        if active is not None:
            active.pop(node.call_id, None)
            if not active:
                self._active_tools.pop(label, None)
        if node.tool_name == "todowrite" and status == "done":
            current_step = _safe_current_todo(node)
            if current_step is not None:
                self._current_steps[label] = current_step
                await self._emit_activity("working_step", label=label)
                return
            self._current_steps.pop(label, None)
            total = metrics.get("item_count")
            completed = metrics.get("todo_completed")
            if isinstance(total, int) and total > 0 and completed == total:
                await self._emit_activity("finalizing", label=label)
                return
        if label not in self._active_tools:
            await self._emit_activity("processing", label=label)

    async def on_loop_guard_decision(
        self,
        decision: Any,
        *,
        event: str,
        source: str,
        fallback_used: bool = False,
        budget_grant: int | None = None,
    ) -> None:
        await self._call_parent(
            "on_worker_guard_decision",
            worker_id=self.worker_id,
            run_id=self.run_id,
            profile_id=self.profile_id,
            label=self.label,
            decision=decision,
            event=event,
            source=source,
            fallback_used=fallback_used,
            budget_grant=budget_grant,
        )
        await self._emit_activity("planning", label=self.label)

    def _scope_label(self, subagent_label: str | None) -> str:
        safe = _safe_identifier(subagent_label)
        return f"{self.label}/{safe}" if safe is not None else self.label

    async def _emit_activity(
        self,
        phase: str,
        *,
        label: str,
        role_id: str | None = None,
        tool_names: tuple[str, ...] = (),
    ) -> None:
        if phase not in _ACTIVITY_PHASES:
            return
        current_step = self._current_steps.get(label)
        detail_source = "host"
        if current_step is not None and phase not in {
            "finalizing",
            "delegating",
            "handoff",
            "branch_done",
            "synthesizing",
        }:
            phase = "working_step"
            tool_names = ()
            detail_source = "worker_declared"
        resolved_role = role_id or self._role_ids.get(label)
        step_text = current_step[0] if current_step is not None else None
        completed = current_step[1] if current_step is not None else None
        total = current_step[2] if current_step is not None else None
        fingerprint = (
            phase,
            None if detail_source == "worker_declared" else resolved_role,
            tool_names,
            step_text,
            completed,
            total,
            detail_source,
        )
        if self._last_activity.get(label) == fingerprint:
            return
        self._last_activity[label] = fingerprint
        revision = self._activity_revisions.get(label, 0) + 1
        self._activity_revisions[label] = revision
        await self._call_parent(
            "on_worker_activity",
            worker_id=self.worker_id,
            run_id=self.run_id,
            profile_id=self.profile_id,
            label=label,
            revision=revision,
            phase=phase,
            role_id=resolved_role,
            tool_names=tool_names,
            current_step=step_text,
            todo_completed=completed,
            todo_total=total,
            detail_source=detail_source,
        )

    async def _call_parent(self, name: str, **kwargs: Any) -> None:
        hook = getattr(self.parent, name, None)
        if hook is None:
            return
        try:
            result = hook(**kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Ignoring Worker progress observer failure: {}", exc)


def _safe_tool_projection(node: TaskNode) -> tuple[str, str, dict[str, Any]]:
    """Cross the Worker/Base boundary with only a strict display allowlist."""

    arguments = node.arguments if isinstance(node.arguments, dict) else {}
    result = str(node.result or "")
    status = "error" if tool_result_failed(node.result) else "done"
    state = str(node.state)
    if state == "failed":
        status = "error"
    elif state == "cancelled":
        status = "cancelled"
    metrics: dict[str, Any] = {
        "result_chars": len(result),
        "result_lines": len(result.splitlines()),
    }
    if node.tool_name == "write":
        metrics["input_chars"] = len(str(arguments.get("content") or ""))
    elif node.tool_name == "edit":
        metrics["old_chars"] = len(str(arguments.get("old_text") or ""))
        metrics["new_chars"] = len(str(arguments.get("new_text") or ""))
    elif node.tool_name == "exec":
        match = re.search(r"(?:Exit code|exit code|exit)\D*(-?\d+)", result)
        if match is not None:
            metrics["exit_code"] = int(match.group(1))
    elif node.tool_name == "todowrite":
        todos = arguments.get("todos")
        if isinstance(todos, list):
            metrics["item_count"] = len(todos)
            metrics["todo_completed"] = sum(
                isinstance(item, dict) and item.get("status") == "completed"
                for item in todos
            )
    elif node.tool_name in {"glob", "grep", "websearch"}:
        metrics["item_count"] = len([line for line in result.splitlines() if line.strip()])
    return node.tool_name, status, metrics


def _safe_current_todo(node: TaskNode) -> tuple[str, int, int] | None:
    arguments = node.arguments if isinstance(node.arguments, dict) else {}
    todos = arguments.get("todos")
    if not isinstance(todos, list):
        return None
    active = [
        item
        for item in todos
        if isinstance(item, dict) and item.get("status") == "in_progress"
    ]
    if len(active) != 1:
        return None
    content = _safe_display_text(active[0].get("content"))
    if not content:
        return None
    completed = sum(
        isinstance(item, dict) and item.get("status") == "completed" for item in todos
    )
    return content, completed, len(todos)


def _safe_display_text(value: Any, *, limit: int = 100) -> str:
    text = _ANSI_ESCAPE.sub("", str(value or ""))
    text = "".join(
        " "
        if char.isspace()
        else ""
        if unicodedata.category(char).startswith("C")
        else char
        for char in text
    )
    return " ".join(text.split())[:limit]


def _safe_identifier(value: Any) -> str | None:
    text = str(value or "")
    return text if _SAFE_IDENTIFIER.fullmatch(text) else None
