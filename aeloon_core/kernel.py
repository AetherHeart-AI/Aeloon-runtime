"""Reusable LLM-to-tool iteration kernel."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon_core.middleware import AgentMiddleware
from aeloon_core.providers.base import LLMProvider, ToolCallRequest
from aeloon_core.task_graph import TaskNode, TaskState, build_task_graph
from aeloon_core.utils.helpers import build_assistant_message
from aeloon_core.utils.tool_history import (
    collect_tool_call_fingerprints,
    duplicate_tool_result,
    tool_call_fingerprint,
)

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
    if "chat_stream_with_retry" in getattr(provider, "__dict__", {}):
        return True
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


def _partition_duplicate_tool_calls(
    messages: list[dict],
    tool_calls: list[ToolCallRequest],
) -> tuple[list[ToolCallRequest], list[ToolCallRequest]]:
    history_end = len(messages)
    while history_end > 0 and messages[history_end - 1].get("role") == "tool":
        history_end -= 1
    if history_end > 0 and messages[history_end - 1].get("role") == "assistant":
        history_end -= 1
    seen = collect_tool_call_fingerprints(messages[:history_end])
    executable: list[ToolCallRequest] = []
    duplicates: list[ToolCallRequest] = []
    batch_seen: set[str] = set()
    for tool_call in tool_calls:
        fingerprint = tool_call_fingerprint(tool_call.name, tool_call.arguments)
        if fingerprint in seen or fingerprint in batch_seen:
            duplicates.append(tool_call)
            continue
        batch_seen.add(fingerprint)
        executable.append(tool_call)
    return executable, duplicates


async def _call_llm_with_middlewares(
    *,
    middlewares: list[AgentMiddleware],
    messages: list[dict],
    tool_defs: list[dict],
    call_llm: Callable[[list[dict], list[dict]], Awaitable[LLMResponse]],
) -> LLMResponse:
    async def _invoke(index: int, msgs: list[dict], defs: list[dict]) -> LLMResponse:
        if index >= len(middlewares):
            return await call_llm(msgs, defs)
        middleware = middlewares[index]

        async def _next(next_messages: list[dict], next_defs: list[dict]) -> LLMResponse:
            return await _invoke(index + 1, next_messages, next_defs)

        return await middleware.around_llm(msgs, defs, _next)

    return await _invoke(0, messages, tool_defs)


async def _call_tool_with_middlewares(
    *,
    middlewares: list[AgentMiddleware],
    name: str,
    args: dict | list | None,
    execute: Callable[[], Awaitable[str]],
) -> str:
    async def _invoke(index: int) -> str:
        if index >= len(middlewares):
            return await execute()
        middleware = middlewares[index]

        async def _next() -> str:
            return await _invoke(index + 1)

        return await middleware.around_tool(name, args, _next)

    return await _invoke(0)


async def _execute_tool_batch(
    *,
    tool_calls: list[ToolCallRequest],
    tools: ToolRegistry,
    middlewares: list[AgentMiddleware],
) -> list[TaskNode]:
    nodes = build_task_graph(tool_calls, tools)
    pending = {node.index: node for node in nodes}
    running: dict[int, asyncio.Task[str]] = {}

    async def _execute_node(node: TaskNode) -> str:
        async def _do_tool_call() -> str:
            return await tools.execute(node.tool_name, node.arguments)

        return await _call_tool_with_middlewares(
            middlewares=middlewares,
            name=node.tool_name,
            args=node.arguments,
            execute=_do_tool_call,
        )

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


def _split_malformed_tool_calls(
    tool_calls: list[ToolCallRequest],
) -> tuple[list[ToolCallRequest], list[ToolCallRequest]]:
    valid: list[ToolCallRequest] = []
    invalid: list[ToolCallRequest] = []
    for tool_call in tool_calls:
        if isinstance(tool_call.arguments, dict):
            valid.append(tool_call)
        else:
            invalid.append(tool_call)
    return valid, invalid


def _format_malformed_arguments_error(tool_call: ToolCallRequest) -> str:
    try:
        raw = json.dumps(tool_call.arguments, ensure_ascii=False, default=str)
    except Exception:
        raw = repr(tool_call.arguments)
    if len(raw) > 500:
        raw = raw[:500] + "..."
    return (
        f"Error: arguments for tool '{tool_call.name}' must be a JSON object, "
        f"but received {type(tool_call.arguments).__name__}: {raw}. Retry with a "
        "single JSON object whose keys match the tool schema."
    )


def _rejected_arguments_summary(tool_call: ToolCallRequest) -> str:
    return json.dumps(
        {
            "_rejected_malformed_arguments": True,
            "original_type": type(tool_call.arguments).__name__,
        },
        ensure_ascii=False,
    )


_PROVIDER_TOOL_CALL_ARGS_MAX_CHARS = 4096
_PROVIDER_TOOL_ARG_STRING_MAX_CHARS = 1024
_MAX_UNPRODUCTIVE_TOOL_ROUNDS = 2
_MAX_ITERATIONS_FINALIZATION_PROMPT = """CRITICAL - MAXIMUM ITERATIONS REACHED

The normal tool-call iteration budget for this task has been reached.
Tools are disabled for this finalization pass.
Respond with text only.

STRICT REQUIREMENTS:
1. Do NOT make any tool calls.
2. Provide a concise text response summarizing work done so far.
3. Clearly state any remaining work that could not be completed.
4. Recommend the next best action.

Any attempt to use tools is a critical violation. Respond with text ONLY."""
_VISIBLE_ANSWER_FINALIZATION_PROMPT = """VISIBLE ANSWER REQUIRED

The previous model response exhausted its output token budget without producing visible answer text.
Tools are disabled for this recovery pass. Respond with concise visible text only.

STRICT REQUIREMENTS:
1. Do NOT make any tool calls.
2. Do NOT continue hidden reasoning.
3. Provide the best answer possible from the context already available.
4. If the task is incomplete, clearly state what remains and what should happen next.

Respond with text ONLY."""
_OUTPUT_EXHAUSTED_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}


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


def _finalization_prompt_message(max_iterations: int) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"{_MAX_ITERATIONS_FINALIZATION_PROMPT}\n\n"
            f"Configured normal iteration budget: {max_iterations}."
        ),
    }


def _visible_answer_prompt_message() -> dict[str, str]:
    return {"role": "user", "content": _VISIBLE_ANSWER_FINALIZATION_PROMPT}


def _tool_result_failed(result: str | None) -> bool:
    text = (result or "").lstrip().lower()
    return text.startswith("error") or text.startswith("skipped duplicate call")


def _off_track_message(reason: str) -> str:
    return (
        "I stopped the agent loop because it appears to be off track: "
        f"{reason}. I did not continue automatically to avoid spending more "
        "iterations on a loop. Please review the last tool result or provide "
        "narrower instructions."
    )


def _finalization_exhausted_message(finalization_budget: int) -> str:
    return (
        "I stopped because the model repeatedly exhausted its output budget "
        "without producing a visible final answer. No final artifact was produced. "
        f"The text-only recovery budget was {finalization_budget} attempt(s). "
        "Increase max_tokens or ask for a smaller first step, then retry."
    )


async def run_agent_kernel(
    *,
    provider: LLMProvider,
    model: str,
    tools: ToolRegistry,
    messages: list[dict],
    max_iterations: int = 25,
    max_auto_continue_iterations: int = 25,
    max_finalization_iterations: int = 2,
    middlewares: list[AgentMiddleware] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
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
    _middlewares = middlewares or []

    base_budget = max(0, max_iterations)
    auto_continue_budget = max(0, max_auto_continue_iterations)
    finalization_budget = max(0, max_finalization_iterations)
    iteration_limit = base_budget
    auto_continue_remaining = auto_continue_budget
    iteration = 0
    finalization_iteration = 0
    finalizing = False
    finalization_message: dict[str, str] | None = None
    final_content = None
    tools_used: list[str] = []
    empty_stop_retries = 0
    max_empty_stop_retries = 1
    unproductive_tool_rounds = 0

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

    async def _extend_iteration_budget() -> bool:
        nonlocal auto_continue_remaining, iteration_limit
        if auto_continue_remaining <= 0:
            return False
        grant_size = base_budget if base_budget > 0 else 1
        grant = min(grant_size, auto_continue_remaining)
        iteration_limit += grant
        auto_continue_remaining -= grant
        logger.info(
            "Iteration budget reached; automatically continuing with {} more iteration(s), "
            "{} auto-continue iteration(s) remaining",
            grant,
            auto_continue_remaining,
        )
        await _emit_progress(
            f"Iteration budget reached; automatically continuing with {grant} more step(s)."
        )
        return True

    async def _switch_to_finalization(
        prompt_message: dict[str, str] | None = None,
        *,
        reason: str = "iteration budgets exhausted",
    ) -> bool:
        nonlocal finalization_message, finalizing
        if finalizing or finalization_budget <= 0:
            return False
        finalizing = True
        finalization_message = prompt_message or _finalization_prompt_message(max_iterations)
        logger.info(
            "Entering finalization because {}; base={}, auto_continue={}, finalization={}",
            reason,
            max_iterations,
            max_auto_continue_iterations,
            max_finalization_iterations,
        )
        await _emit_progress(f"{reason}; asking for a text-only wrap-up with tools disabled.")
        return True

    def _record_productive_tool_round() -> None:
        nonlocal unproductive_tool_rounds
        unproductive_tool_rounds = 0

    def _record_unproductive_tool_round(reason: str, *, immediate: bool = False) -> str | None:
        nonlocal unproductive_tool_rounds
        unproductive_tool_rounds += 1
        if immediate or unproductive_tool_rounds >= _MAX_UNPRODUCTIVE_TOOL_ROUNDS:
            return reason
        logger.info(
            "Unproductive tool round ({}/{}): {}",
            unproductive_tool_rounds,
            _MAX_UNPRODUCTIVE_TOOL_ROUNDS,
            reason,
        )
        return None

    async def _stop_off_track(reason: str) -> None:
        nonlocal final_content, messages
        final_content = _off_track_message(reason)
        logger.warning("Stopping off-track agent loop: {}", reason)
        messages = add_assistant(messages, final_content)
        await _emit_hook("on_final", final_content, messages=messages)

    while True:
        if finalizing:
            if finalization_iteration >= finalization_budget:
                break
            finalization_iteration += 1
            tool_defs: list[dict] = []
            call_messages = [
                *messages,
                finalization_message or _finalization_prompt_message(max_iterations),
            ]
            await _emit_progress(
                "Wrapping up..."
                if finalization_iteration == 1
                else f"Wrapping up (attempt {finalization_iteration})..."
            )
        else:
            if iteration >= iteration_limit:
                if await _extend_iteration_budget():
                    continue
                if await _switch_to_finalization():
                    continue
                break
            iteration += 1
            tool_defs = tools.get_definitions()
            call_messages = messages
            await _emit_progress(
                "Thinking..." if iteration == 1 else f"Thinking (step {iteration})..."
            )

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

        response = await _call_llm_with_middlewares(
            middlewares=_middlewares,
            messages=call_messages,
            tool_defs=tool_defs,
            call_llm=_do_llm_call,
        )
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
                await _stop_off_track(
                    "the model attempted to call tools after tools were disabled for finalization"
                )
                break

            thought = _strip(response.content)
            if thought:
                await _emit_progress(thought)
            tool_calls, malformed_calls = _split_malformed_tool_calls(response.tool_calls)
            malformed_ids = {bad_call.id for bad_call in malformed_calls}
            tool_call_dicts = []
            for tool_call in response.tool_calls:
                call_dict = tool_call.to_openai_tool_call()
                if tool_call.id in malformed_ids:
                    call_dict["function"]["arguments"] = _rejected_arguments_summary(tool_call)
                tool_call_dicts.append(call_dict)
            messages = add_assistant(
                messages,
                response.content,
                tool_calls=tool_call_dicts,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            if malformed_calls:
                for bad_call in malformed_calls:
                    messages = add_tool(
                        messages,
                        bad_call.id,
                        bad_call.name,
                        _format_malformed_arguments_error(bad_call),
                    )
                if not tool_calls:
                    reason = _record_unproductive_tool_round(
                        "the model only supplied malformed tool arguments"
                    )
                    if reason is not None:
                        await _stop_off_track(reason)
                        break
                    if iteration >= iteration_limit:
                        if await _extend_iteration_budget():
                            continue
                        await _switch_to_finalization()
                    continue

            executable_calls, duplicate_calls = _partition_duplicate_tool_calls(
                messages,
                tool_calls,
            )
            for duplicate in duplicate_calls:
                logger.warning(
                    "Skipping duplicate tool_call '{}' with identical arguments",
                    duplicate.name,
                )
                messages = add_tool(
                    messages,
                    duplicate.id,
                    duplicate.name,
                    duplicate_tool_result(duplicate.name),
                )
            if duplicate_calls and not executable_calls:
                reason = _record_unproductive_tool_round(
                    "the model repeated tool calls that already ran with identical arguments",
                    immediate=True,
                )
                if reason is not None:
                    await _stop_off_track(reason)
                    break
                if iteration >= iteration_limit:
                    if await _extend_iteration_budget():
                        continue
                    await _switch_to_finalization()
                continue

            tool_calls = executable_calls
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
                middlewares=_middlewares,
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
            if executed_nodes and all(_tool_result_failed(node.result) for node in executed_nodes):
                reason = _record_unproductive_tool_round(
                    "all tool calls in the latest round failed or returned errors"
                )
                if reason is not None:
                    await _stop_off_track(reason)
                    break
            else:
                _record_productive_tool_round()
            if iteration >= iteration_limit:
                if await _extend_iteration_budget():
                    continue
                await _switch_to_finalization()
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
            if finalizing and response.finish_reason in _OUTPUT_EXHAUSTED_FINISH_REASONS:
                logger.warning(
                    "Finalization output budget exhausted (attempt {}/{})",
                    finalization_iteration,
                    finalization_budget,
                )
                if finalization_iteration >= finalization_budget:
                    break
                continue

            if (
                response.finish_reason in _OUTPUT_EXHAUSTED_FINISH_REASONS
                and not finalizing
                and finalization_budget > 0
            ):
                logger.warning(
                    "LLM exhausted output budget without visible answer; entering finalization"
                )
                await _emit_progress(
                    "The model used its output budget without a visible answer; "
                    "asking for a concise text-only answer."
                )
                if await _switch_to_finalization(
                    _visible_answer_prompt_message(),
                    reason="output budget exhausted without visible answer",
                ):
                    continue

            logger.warning(
                "LLM returned empty stop response (attempt {}/{})",
                empty_stop_retries + 1,
                max_empty_stop_retries + 1,
            )
            if empty_stop_retries < max_empty_stop_retries:
                empty_stop_retries += 1
                continue
            final_content = "Sorry, the AI model returned an empty response. Please try again."
            messages = add_assistant(messages, final_content)
            await _emit_hook("on_final", final_content, messages=messages)
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

    if final_content is None and finalizing and finalization_iteration >= finalization_budget:
        logger.warning(
            "Finalization exhausted without visible output; finalization={}",
            max_finalization_iterations,
        )
        final_content = _finalization_exhausted_message(max_finalization_iterations)
        messages = add_assistant(messages, final_content)
        await _emit_hook("on_final", final_content, messages=messages)

    if final_content is None and iteration >= iteration_limit:
        logger.warning(
            "Iteration budget exhausted: base={}, auto_continue={}, finalization={}",
            max_iterations,
            max_auto_continue_iterations,
            max_finalization_iterations,
        )
        if finalization_budget > 0:
            final_content = (
                f"I reached the maximum number of tool call iterations ({max_iterations}) "
                f"plus the automatic continuation budget ({max_auto_continue_iterations}) "
                f"and could not produce a final text response within the finalization budget "
                f"({max_finalization_iterations}). Try breaking the task into smaller steps."
            )
        else:
            final_content = (
                f"I reached the maximum number of tool call iterations ({max_iterations}) "
                f"plus the automatic continuation budget ({max_auto_continue_iterations}) "
                "without completing the task. Try breaking the task into smaller steps."
            )
        await _emit_hook("on_final", final_content, messages=messages)

    return final_content, tools_used, messages
