"""Reusable LLM-to-tool iteration kernel."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon_core.loop_guard import (
    AgentLoopGuard,
    LoopGuardAction,
    LoopGuardDecision,
    rejected_arguments_summary,
)
from aeloon_core.providers.base import LLMProvider, ToolCallRequest
from aeloon_core.task_graph import TaskNode, TaskState, build_task_graph
from aeloon_core.utils.helpers import build_assistant_message

if TYPE_CHECKING:
    from aeloon_core.providers.base import LLMResponse
    from aeloon_core.tools.registry import ToolRegistry


def _default_strip_think(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None


def _suffix_prefix_len(text: str, prefix: str) -> int:
    max_len = min(len(text), len(prefix) - 1)
    for size in range(max_len, 0, -1):
        if prefix.startswith(text[-size:]):
            return size
    return 0


class _ThinkTagDeltaFilter:
    """Hide streamed <think>...</think> spans while preserving visible deltas."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        visible: list[str] = []
        while self._buffer:
            if self._inside:
                end = self._buffer.find(self._CLOSE)
                if end < 0:
                    keep = _suffix_prefix_len(self._buffer, self._CLOSE)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(visible)
                self._buffer = self._buffer[end + len(self._CLOSE) :]
                self._inside = False
                continue

            start = self._buffer.find(self._OPEN)
            if start < 0:
                keep = _suffix_prefix_len(self._buffer, self._OPEN)
                emit = self._buffer[:-keep] if keep else self._buffer
                if emit:
                    visible.append(emit)
                self._buffer = self._buffer[-keep:] if keep else ""
                return "".join(visible)

            if start > 0:
                visible.append(self._buffer[:start])
            self._buffer = self._buffer[start + len(self._OPEN) :]
            self._inside = True
        return "".join(visible)

    def flush(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        return tail


def _provider_supports_streaming(provider: LLMProvider) -> bool:
    chat_stream = getattr(type(provider), "chat_stream", None)
    return chat_stream is not None and chat_stream is not LLMProvider.chat_stream


def _default_tool_hint(tool_calls: list[ToolCallRequest]) -> str:
    def _fmt(tool_call: ToolCallRequest) -> str:
        args = tool_call.arguments or {}
        val = next(iter(args.values()), None) if isinstance(args, dict) else None
        if not isinstance(val, str):
            return tool_call.name
        if len(val) > 40:
            return f'{tool_call.name}("{val[:40]}...")'
        return f'{tool_call.name}("{val}")'

    return ", ".join(_fmt(tool_call) for tool_call in tool_calls)


def _default_add_assistant_message(
    messages: list[dict],
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> list[dict]:
    messages.append(
        build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    return messages


def _default_add_tool_result(
    messages: list[dict],
    tool_call_id: str,
    tool_name: str,
    result: str,
) -> list[dict]:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
    )
    return messages


async def _execute_tool_batch(
    *,
    tool_calls: list[ToolCallRequest],
    tools: ToolRegistry,
) -> list[TaskNode]:
    nodes = build_task_graph(tool_calls, tools)
    pending = {node.index: node for node in nodes}
    running: dict[int, asyncio.Task[str]] = {}

    async def _execute_node(node: TaskNode) -> str:
        return await tools.execute(node.tool_name, node.arguments)

    try:
        while pending or running:
            ready = [node for node in pending.values() if not node.deps]
            for node in ready:
                node.state = TaskState.RUNNING
                running[node.index] = asyncio.create_task(_execute_node(node))
                pending.pop(node.index)

            if not running:
                raise RuntimeError("deadlock detected in tool task graph")

            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            finished_indexes = [index for index, task in running.items() if task in done]

            for index in finished_indexes:
                task = running.pop(index)
                node = nodes[index]
                try:
                    node.result = await task
                    node.state = TaskState.DONE
                except asyncio.CancelledError:
                    node.state = TaskState.CANCELLED
                    raise
                except Exception as exc:
                    node.state = TaskState.FAILED
                    node.error = str(exc)
                    node.result = f"Error executing {node.tool_name}: {exc}"

                for dependent_index in node.dependents:
                    nodes[dependent_index].deps.discard(index)
    except asyncio.CancelledError:
        for task in running.values():
            task.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        raise
    return nodes


_PROVIDER_TOOL_CALL_ARGS_MAX_CHARS = 4096
_PROVIDER_TOOL_ARG_STRING_MAX_CHARS = 1024


def _shrink_oversized_tool_arguments(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    for key, value in parsed.items():
        if isinstance(value, str) and len(value) > _PROVIDER_TOOL_ARG_STRING_MAX_CHARS:
            omitted = len(value) - _PROVIDER_TOOL_ARG_STRING_MAX_CHARS
            parsed[key] = (
                value[:_PROVIDER_TOOL_ARG_STRING_MAX_CHARS] + f"... [truncated {omitted} chars]"
            )
    shrunk = json.dumps(parsed, ensure_ascii=False)
    if len(shrunk) <= _PROVIDER_TOOL_CALL_ARGS_MAX_CHARS:
        return shrunk
    return json.dumps(
        {"_compacted_tool_arguments": True, "keys": list(parsed.keys())[:20]},
        ensure_ascii=False,
    )


def _shrink_answered_tool_args_for_provider(messages: list[dict]) -> list[dict]:
    answered = {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str)
    }
    if not answered:
        return messages

    out: list[dict] = []
    changed = False
    for message in messages:
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not isinstance(calls, list):
            out.append(message)
            continue
        new_calls: list[Any] = []
        message_changed = False
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if (
                call.get("id") in answered
                and isinstance(arguments, str)
                and len(arguments) > _PROVIDER_TOOL_CALL_ARGS_MAX_CHARS
            ):
                shrunk = _shrink_oversized_tool_arguments(arguments)
                if shrunk != arguments:
                    new_calls.append({**call, "function": {**function, "arguments": shrunk}})
                    message_changed = True
                    continue
            new_calls.append(call)
        out.append({**message, "tool_calls": new_calls} if message_changed else message)
        changed = changed or message_changed
    return out if changed else messages


async def run_agent_kernel(
    *,
    provider: LLMProvider,
    model: str,
    tools: ToolRegistry,
    messages: list[dict],
    max_iterations: int = 25,
    max_auto_continue_iterations: int = 25,
    max_finalization_iterations: int = 2,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    prepare_model_input: Callable[
        [list[dict], list[dict], list[dict]], Awaitable[list[dict]]
    ]
    | None = None,
    add_assistant_message: Callable[..., list[dict]] | None = None,
    add_tool_result: Callable[[list[dict], str, str, str], list[dict]] | None = None,
    strip_think: Callable[[str | None], str | None] | None = None,
    tool_hint: Callable[[list[ToolCallRequest]], str] | None = None,
) -> tuple[str | None, list[str], list[dict]]:
    """Execute a reusable tool-augmented LLM loop."""

    add_assistant = add_assistant_message or _default_add_assistant_message
    add_tool = add_tool_result or _default_add_tool_result
    _strip = strip_think or _default_strip_think
    _tool_hint = tool_hint or _default_tool_hint

    guard = AgentLoopGuard(
        max_iterations=max_iterations,
        max_auto_continue_iterations=max_auto_continue_iterations,
        max_finalization_iterations=max_finalization_iterations,
    )
    iteration = 0
    finalization_iteration = 0
    finalizing = False
    finalization_message: dict[str, str] | None = None
    final_content = None
    tools_used: list[str] = []

    async def _emit_progress(text: str, *, tool_hint: bool = False) -> None:
        if on_progress is not None:
            await on_progress(text, tool_hint=tool_hint)

    async def _emit_hook(name: str, *args: Any, **kwargs: Any) -> None:
        if on_progress is None:
            return
        hook = getattr(on_progress, name, None)
        if hook is None:
            return
        result = hook(*args, **kwargs)
        if inspect.isawaitable(result):
            await result

    await _emit_hook("on_turn_start")

    async def _switch_to_finalization(
        prompt_message: dict[str, str] | None = None,
        *,
        reason: str = "iteration budgets exhausted",
    ) -> bool:
        nonlocal finalization_message, finalizing
        if finalizing or guard.finalization_budget <= 0:
            return False
        finalizing = True
        finalization_message = prompt_message or guard.finalization_prompt_message()
        logger.info(
            "Entering finalization because {}; base={}, auto_continue={}, finalization={}",
            reason,
            max_iterations,
            max_auto_continue_iterations,
            max_finalization_iterations,
        )
        await _emit_progress(f"{reason}; asking for a text-only wrap-up with tools disabled.")
        return True

    async def _finish_with_guard_decision(
        decision: LoopGuardDecision,
        *,
        add_message: bool = True,
    ) -> None:
        nonlocal final_content, messages
        final_content = decision.final_content or decision.reason
        if decision.action == LoopGuardAction.STOP_OFF_TRACK:
            logger.warning("Stopping off-track agent loop: {}", decision.reason)
        if add_message:
            messages = add_assistant(messages, final_content)
        await _emit_hook("on_final", final_content, messages=messages)

    async def _apply_budget_decision(decision: LoopGuardDecision) -> bool:
        if decision.action == LoopGuardAction.EXTEND_BUDGET:
            if decision.progress_message:
                await _emit_progress(decision.progress_message)
            return False
        if decision.action == LoopGuardAction.FINALIZE:
            await _switch_to_finalization(
                decision.prompt_message,
                reason=decision.reason or "iteration budgets exhausted",
            )
            return False
        if decision.action == LoopGuardAction.FINAL_RESPONSE:
            await _finish_with_guard_decision(decision, add_message=False)
            return True
        return False

    async def _return_to_model_with_guard_context(decision: LoopGuardDecision) -> bool:
        """Append guard recovery context before the next model call."""

        nonlocal messages
        if decision.prompt_message:
            messages.append(decision.prompt_message)
        if decision.progress_message:
            await _emit_progress(decision.progress_message)
        return await _grant_more_or_finalize()

    async def _grant_more_or_finalize() -> bool:
        """When the budget is spent, grant an auto-continue or enter finalization."""
        if iteration < guard.iteration_limit:
            return False
        return await _apply_budget_decision(guard.handle_iteration_budget_reached())

    async def _do_llm_call(
        current_messages: list[dict],
        current_tool_defs: list[dict],
    ) -> LLMResponse:
        provider_messages = _shrink_answered_tool_args_for_provider(current_messages)
        delta_hook = getattr(on_progress, "on_llm_delta", None) if on_progress else None
        reasoning_delta_hook = (
            getattr(on_progress, "on_llm_reasoning_delta", None) if on_progress else None
        )
        if (
            delta_hook is not None or reasoning_delta_hook is not None
        ) and _provider_supports_streaming(provider):
            think_filter = _ThinkTagDeltaFilter()

            async def _on_delta(delta: str) -> None:
                if delta_hook is None:
                    return
                visible = think_filter.feed(delta)
                if not visible:
                    return
                result = delta_hook(visible)
                if inspect.isawaitable(result):
                    await result

            async def _on_reasoning_delta(delta: str) -> None:
                if reasoning_delta_hook is None or not delta:
                    return
                result = reasoning_delta_hook(delta)
                if inspect.isawaitable(result):
                    await result

            response = await provider.chat_stream_with_retry(
                messages=provider_messages,
                tools=current_tool_defs,
                model=model,
                on_delta=_on_delta if delta_hook is not None else None,
                on_reasoning_delta=(
                    _on_reasoning_delta if reasoning_delta_hook is not None else None
                ),
            )
            tail = think_filter.flush() if delta_hook is not None else ""
            if tail and delta_hook is not None:
                result = delta_hook(tail)
                if inspect.isawaitable(result):
                    await result
            return response

        return await provider.chat_with_retry(
            messages=provider_messages,
            tools=current_tool_defs,
            model=model,
        )

    while True:
        if finalizing:
            if finalization_iteration >= guard.finalization_budget:
                break
            finalization_iteration += 1
            tool_defs: list[dict] = []
            await _emit_progress(
                "Wrapping up..."
                if finalization_iteration == 1
                else f"Wrapping up (attempt {finalization_iteration})..."
            )
        else:
            if iteration >= guard.iteration_limit:
                if await _apply_budget_decision(guard.handle_iteration_budget_reached()):
                    break
                continue
            iteration += 1
            tool_defs = tools.get_definitions()
            await _emit_progress(
                "Thinking..." if iteration == 1 else f"Thinking (step {iteration})..."
            )

        additional_messages = (
            [finalization_message or guard.finalization_prompt_message()]
            if finalizing
            else []
        )
        if prepare_model_input is not None:
            messages = await prepare_model_input(messages, tool_defs, additional_messages)
        call_messages = [*messages, *additional_messages]

        response = await _do_llm_call(call_messages, tool_defs)
        await _emit_hook("on_llm_response", response)

        if response.tool_calls:
            if finalizing:
                logger.warning(
                    "Ignoring {} tool call(s) attempted during finalization",
                    len(response.tool_calls),
                )
                clean = _strip(response.content)
                if clean is not None:
                    messages = add_assistant(
                        messages,
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    final_content = clean
                    await _emit_hook("on_final", clean, messages=messages)
                    break
                await _finish_with_guard_decision(
                    guard.handle_finalization_tool_call_violation()
                )
                break

            thought = _strip(response.content)
            if thought:
                await _emit_progress(thought)
            malformed_result = guard.handle_malformed_tool_calls(response.tool_calls)
            tool_calls = malformed_result.executable_calls
            malformed_ids = {bad_call.id for bad_call in malformed_result.malformed_calls}
            tool_call_dicts = []
            for tool_call in response.tool_calls:
                call_dict = tool_call.to_openai_tool_call()
                if tool_call.id in malformed_ids:
                    call_dict["function"]["arguments"] = rejected_arguments_summary(tool_call)
                tool_call_dicts.append(call_dict)
            messages = add_assistant(
                messages,
                response.content,
                tool_calls=tool_call_dicts,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            if malformed_result.malformed_calls:
                for tool_result in malformed_result.tool_results:
                    messages = add_tool(
                        messages,
                        tool_result.call_id,
                        tool_result.tool_name,
                        tool_result.content,
                    )
                if malformed_result.decision.action == LoopGuardAction.STOP_OFF_TRACK:
                    await _finish_with_guard_decision(malformed_result.decision)
                    break
                if malformed_result.decision.action == LoopGuardAction.RETURN_TO_MODEL:
                    if await _return_to_model_with_guard_context(malformed_result.decision):
                        break
                    continue

            duplicate_result = guard.handle_duplicate_tool_calls(messages, tool_calls)
            for duplicate in duplicate_result.duplicate_calls:
                logger.warning(
                    "Skipping duplicate tool_call '{}' with identical arguments",
                    duplicate.name,
                )
            for tool_result in duplicate_result.tool_results:
                messages = add_tool(
                    messages,
                    tool_result.call_id,
                    tool_result.tool_name,
                    tool_result.content,
                )
            if duplicate_result.decision.action == LoopGuardAction.STOP_OFF_TRACK:
                await _finish_with_guard_decision(duplicate_result.decision)
                break
            if duplicate_result.decision.action == LoopGuardAction.RETURN_TO_MODEL:
                if await _return_to_model_with_guard_context(duplicate_result.decision):
                    break
                continue

            tool_calls = duplicate_result.executable_calls
            hint = _strip(_tool_hint(tool_calls))
            if hint:
                await _emit_progress(hint, tool_hint=True)
            await _emit_hook("on_tool_calls", tool_calls)

            for tool_call in tool_calls:
                args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                logger.info("Tool call: {}({})", tool_call.name, args_str[:200])

            executed_nodes = await _execute_tool_batch(
                tool_calls=tool_calls,
                tools=tools,
            )
            for node in sorted(executed_nodes, key=lambda item: item.index):
                tools_used.append(node.tool_name)
                messages = add_tool(
                    messages,
                    node.call_id,
                    node.tool_name,
                    node.result or f"Error executing {node.tool_name}: unknown failure",
                )
                await _emit_hook("on_tool_result", node)
            tool_result_decision = guard.handle_tool_results(executed_nodes)
            if tool_result_decision.action == LoopGuardAction.STOP_OFF_TRACK:
                await _finish_with_guard_decision(tool_result_decision)
                break
            if tool_result_decision.action == LoopGuardAction.RETURN_TO_MODEL:
                if await _return_to_model_with_guard_context(tool_result_decision):
                    break
                continue
            if await _grant_more_or_finalize():
                break
            continue

        clean = _strip(response.content)
        logger.debug(
            "LLM final response - content={!r}, reasoning={!r}, finish={}",
            (response.content or "")[:200],
            (response.reasoning_content or "")[:200],
            response.finish_reason,
        )
        if response.finish_reason == "error":
            logger.error("LLM returned error: {}", (clean or "")[:200])
            final_content = clean or "Sorry, I encountered an error calling the AI model."
            await _emit_hook("on_final", final_content, messages=messages)
            break

        if clean is None:
            empty_decision = guard.handle_empty_or_exhausted_response(
                finish_reason=response.finish_reason,
                finalizing=finalizing,
                finalization_iteration=finalization_iteration,
            )
            if empty_decision.action == LoopGuardAction.CONTINUE:
                continue
            if empty_decision.action == LoopGuardAction.FINALIZE:
                if empty_decision.progress_message:
                    await _emit_progress(empty_decision.progress_message)
                if await _switch_to_finalization(
                    empty_decision.prompt_message,
                    reason=(
                        empty_decision.reason
                        or "output budget exhausted without visible answer"
                    ),
                ):
                    continue
            if empty_decision.action == LoopGuardAction.FINAL_RESPONSE:
                await _finish_with_guard_decision(empty_decision)
                break
            break

        messages = add_assistant(
            messages,
            clean,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        final_content = clean
        await _emit_hook("on_final", clean, messages=messages)
        break

    if final_content is None and finalizing and finalization_iteration >= guard.finalization_budget:
        logger.warning(
            "Finalization exhausted without visible output; finalization={}",
            max_finalization_iterations,
        )
        final_content = guard.finalization_exhausted_message()
        messages = add_assistant(messages, final_content)
        await _emit_hook("on_final", final_content, messages=messages)

    if final_content is None and iteration >= guard.iteration_limit:
        logger.warning(
            "Iteration budget exhausted: base={}, auto_continue={}, finalization={}",
            max_iterations,
            max_auto_continue_iterations,
            max_finalization_iterations,
        )
        final_content = guard.final_message_for_exhausted_loop()
        await _emit_hook("on_final", final_content, messages=messages)

    return final_content, tools_used, messages
