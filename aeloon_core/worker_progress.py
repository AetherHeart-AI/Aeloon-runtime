"""Privacy-preserving bridge from private Worker activity to Base UI events."""

from __future__ import annotations

import asyncio
import inspect
import re
import shlex
import unicodedata
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from loguru import logger

from aeloon_core.loop_guard import tool_result_failed
from aeloon_core.operator_output import sanitize_operator_output
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
_MAX_PENDING_JOURNAL_CALLS = 64
_MAX_TOOL_RESULT_PREVIEW_CHARS = 4_000
_MAX_TOOL_FAILURE_PREVIEW_CHARS = 400
_SAFE_FAILURE_PREVIEW_TOOLS = {
    "glob",
    "grep",
    "read",
    "skill",
    "str_replace",
    "todowrite",
    "webfetch",
    "websearch",
    "write",
}
_DISPLAYABLE_COMMANDS = {
    "bun",
    "find",
    "git",
    "ls",
    "npm",
    "pnpm",
    "pytest",
    "python",
    "python3",
    "ruff",
    "tsc",
    "uv",
    "yarn",
}
_SHELL_OPERATORS = {"&", "&&", ";", "|", "||"}


@dataclass(frozen=True, slots=True)
class _BufferedJournalCall:
    name: str
    kwargs: dict[str, Any]
    priority: int


class WorkerProgress:
    """Hide model-authored text and project bounded activity for operators."""

    def __init__(
        self,
        *,
        parent: Any,
        worker_id: str,
        run_id: str,
        profile_id: str,
        run_sequence: int = 1,
        journal: Any | None = None,
    ) -> None:
        self.parent = parent
        self.journal = journal
        self.allow_tool_output = bool(
            getattr(parent, "allow_worker_tool_output", False)
        )
        self.worker_id = worker_id
        self.run_id = run_id
        self.run_sequence = max(1, int(run_sequence))
        self.profile_id = profile_id
        self.label = f"{profile_id}#{worker_id[:8]}"
        self._tool_started: dict[str, float] = {}
        self._active_tools: dict[str, dict[str, str]] = {}
        self._current_steps: dict[str, tuple[str, int, int]] = {}
        self._role_ids: dict[str, str] = {}
        self._last_activity: dict[str, tuple[Any, ...]] = {}
        self._activity_revisions: dict[str, int] = {}
        self._journal_calls: deque[_BufferedJournalCall] = deque()
        self._journal_task: asyncio.Task[None] | None = None
        self._journal_dropped_calls = 0

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

    async def on_guard_resolution(self, resolution: Any) -> None:
        self._call_journal(
            "record_guard",
            run_id=self.run_id,
            resolution=resolution,
        )
        await self._call_parent(
            "on_worker_guard_resolution",
            worker_id=self.worker_id,
            run_id=self.run_id,
            run_sequence=self.run_sequence,
            profile_id=self.profile_id,
            label=self.label,
            resolution=resolution,
        )
        await self._emit_activity("planning", label=self.label)

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
            max(0, int((perf_counter() - started) * 1_000)) if started is not None else None
        )
        tool_name, status, metrics = _safe_tool_projection(
            node,
            include_result_preview=self.allow_tool_output,
        )
        self._call_journal(
            "record_tool",
            run_id=self.run_id,
            tool_name=tool_name,
            status=status,
            metrics=metrics,
            duration_ms=duration_ms,
        )
        await self._call_parent(
            "on_worker_tool_result",
            worker_id=self.worker_id,
            run_id=self.run_id,
            run_sequence=self.run_sequence,
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
        self._call_journal(
            "record_activity",
            run_id=self.run_id,
            phase=phase,
            role_id=resolved_role,
            tool_names=tool_names,
            current_step=step_text,
            todo_completed=completed,
            todo_total=total,
            detail_source=detail_source,
        )
        await self._call_parent(
            "on_worker_activity",
            worker_id=self.worker_id,
            run_id=self.run_id,
            run_sequence=self.run_sequence,
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
            if inspect.iscoroutinefunction(hook):
                result = hook(**kwargs)
            else:
                result = await asyncio.to_thread(hook, **kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Ignoring Worker progress observer failure: {}", exc)

    @property
    def pending_journal_calls(self) -> int:
        return len(self._journal_calls)

    @property
    def dropped_journal_calls(self) -> int:
        return self._journal_dropped_calls

    async def flush_journal(self) -> None:
        """Drain optional observability work outside the Worker execution path."""

        while self._journal_task is not None:
            await asyncio.shield(self._journal_task)
        flush = getattr(self.journal, "flush", None)
        if flush is None:
            return
        try:
            if inspect.iscoroutinefunction(flush):
                result = flush()
            else:
                result = await asyncio.to_thread(flush)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Ignoring Worker UI journal flush failure: {}", exc)

    def _call_journal(self, name: str, **kwargs: Any) -> None:
        hook = getattr(self.journal, name, None)
        if hook is None:
            return
        if getattr(self.journal, "writes_are_buffered", False):
            try:
                hook(**kwargs)
            except Exception as exc:
                logger.warning("Ignoring Worker UI journal failure: {}", exc)
            return

        call = _BufferedJournalCall(
            name=name,
            kwargs=kwargs,
            priority=_journal_call_priority(name, kwargs),
        )
        if len(self._journal_calls) >= _MAX_PENDING_JOURNAL_CALLS:
            if name == "record_activity":
                self._remove_latest_activity_call()
            if len(self._journal_calls) >= _MAX_PENDING_JOURNAL_CALLS:
                victim = next(
                    (
                        index
                        for index, pending in enumerate(self._journal_calls)
                        if pending.priority < call.priority
                    ),
                    None,
                )
                if victim is None:
                    victim = next(
                        (
                            index
                            for index, pending in enumerate(self._journal_calls)
                            if pending.priority == call.priority and call.priority > 0
                        ),
                        None,
                    )
                    if victim is None:
                        self._journal_dropped_calls += 1
                        return
                del self._journal_calls[victim]
                self._journal_dropped_calls += 1
        self._journal_calls.append(call)
        if self._journal_task is None:
            self._journal_task = asyncio.create_task(self._drain_journal_calls())

    def _remove_latest_activity_call(self) -> None:
        for index in range(len(self._journal_calls) - 1, -1, -1):
            if self._journal_calls[index].name == "record_activity":
                del self._journal_calls[index]
                self._journal_dropped_calls += 1
                return

    async def _drain_journal_calls(self) -> None:
        try:
            while self._journal_calls:
                call = self._journal_calls.popleft()
                hook = getattr(self.journal, call.name, None)
                if hook is None:
                    continue
                try:
                    if inspect.iscoroutinefunction(hook):
                        result = hook(**call.kwargs)
                    else:
                        result = await asyncio.to_thread(hook, **call.kwargs)
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    # UI observability cannot become part of Worker correctness.
                    logger.warning("Ignoring Worker UI journal failure: {}", exc)
        finally:
            self._journal_task = None


def _journal_call_priority(name: str, kwargs: dict[str, Any]) -> int:
    if name == "record_activity":
        return 0
    if name == "record_guard":
        return 2
    if name == "record_tool" and kwargs.get("status") != "done":
        return 2
    return 1


def _safe_tool_projection(
    node: TaskNode,
    *,
    include_result_preview: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Project Worker activity through a strict observer allowlist."""

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
    if include_result_preview and node.tool_name == "exec":
        result_preview = _safe_tool_result_preview(result)
        if result_preview:
            metrics["result_preview"] = result_preview
    elif (
        include_result_preview
        and status != "done"
        and node.tool_name in _SAFE_FAILURE_PREVIEW_TOOLS
    ):
        first_line = next((line for line in result.splitlines() if line.strip()), "")
        result_preview = _safe_tool_result_preview(
            first_line,
            limit=_MAX_TOOL_FAILURE_PREVIEW_CHARS,
        )
        if result_preview:
            metrics["result_preview"] = result_preview
    if node.tool_name == "write":
        content = arguments.get("content")
        content_text = content if isinstance(content, str) else ""
        metrics["input_chars"] = len(content_text)
        try:
            metrics["input_bytes"] = len(content_text.encode("utf-8"))
        except UnicodeEncodeError:
            pass
        expected_offset = arguments.get("expected_offset")
        if isinstance(expected_offset, int) and not isinstance(expected_offset, bool):
            metrics["expected_offset"] = expected_offset
    elif node.tool_name == "str_replace":
        metrics["old_chars"] = len(str(arguments.get("old_str") or ""))
        metrics["new_chars"] = len(str(arguments.get("new_str") or ""))
        if arguments.get("replace_all") is True:
            metrics["replace_all"] = True
    elif node.tool_name == "exec":
        command = _safe_command_summary(arguments.get("command"))
        if command:
            metrics["command"] = command
        match = re.search(r"(?:Exit code|exit code|exit)\D*(-?\d+)", result)
        if match is not None:
            metrics["exit_code"] = int(match.group(1))
    elif node.tool_name == "todowrite":
        todos = arguments.get("todos")
        if isinstance(todos, list):
            metrics["item_count"] = len(todos)
            metrics["todo_completed"] = sum(
                isinstance(item, dict) and item.get("status") == "completed" for item in todos
            )
    elif node.tool_name in {"glob", "grep", "websearch"}:
        metrics["item_count"] = len([line for line in result.splitlines() if line.strip()])
    return node.tool_name, status, metrics


def _safe_command_summary(value: Any, *, limit: int = 160) -> str:
    """Return a bounded allowlisted operator hint without shell payloads."""

    text = " ".join(str(value or "").splitlines()[0].split()) if value else ""
    if not text:
        return ""
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    while tokens and "=" in tokens[0] and not tokens[0].startswith(("./", "/")):
        tokens.pop(0)
    if not tokens:
        return ""
    executable = tokens[0].rsplit("/", 1)[-1]
    if executable not in _DISPLAYABLE_COMMANDS:
        return executable[:limit]
    visible = [executable]
    redact_next = False
    for token in tokens[1:]:
        if redact_next:
            redact_next = False
            continue
        if token in _SHELL_OPERATORS:
            visible.append("…")
            break
        if token in {"-c", "-e", "--eval"}:
            visible.extend((token, "…"))
            break
        if re.search(r"(?i)(token|password|passwd|secret|api[-_]?key)", token):
            visible.append("[redacted]")
            redact_next = "=" not in token
            continue
        if re.match(r"(?i)https?://", token):
            visible.append("[url]")
            continue
        visible.append(token[:80])
        if len(" ".join(visible)) >= limit:
            break
    summary = " ".join(visible)
    return summary if len(summary) <= limit else f"{summary[: limit - 1]}…"


def _safe_tool_result_preview(
    value: Any,
    *,
    limit: int = _MAX_TOOL_RESULT_PREVIEW_CHARS,
) -> str:
    """Sanitize and bound tool output for privileged local operator display."""

    return sanitize_operator_output(value, limit=limit)


def _safe_current_todo(node: TaskNode) -> tuple[str, int, int] | None:
    arguments = node.arguments if isinstance(node.arguments, dict) else {}
    todos = arguments.get("todos")
    if not isinstance(todos, list):
        return None
    active = [
        item for item in todos if isinstance(item, dict) and item.get("status") == "in_progress"
    ]
    if len(active) != 1:
        return None
    content = _safe_display_text(active[0].get("content"))
    if not content:
        return None
    completed = sum(isinstance(item, dict) and item.get("status") == "completed" for item in todos)
    return content, completed, len(todos)


def _safe_display_text(value: Any, *, limit: int = 100) -> str:
    text = _ANSI_ESCAPE.sub("", str(value or ""))
    text = "".join(
        " " if char.isspace() else "" if unicodedata.category(char).startswith("C") else char
        for char in text
    )
    return " ".join(text.split())[:limit]


def _safe_identifier(value: Any) -> str | None:
    text = str(value or "")
    return text if _SAFE_IDENTIFIER.fullmatch(text) else None
