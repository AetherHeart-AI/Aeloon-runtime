"""Deterministic, offline fault-injection fixtures for UASM experiments."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from aeloon_core.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.transitions import accumulate_usage


class ValueArgs(BaseModel):
    value: str


class CommandArgs(BaseModel):
    command: str


class EchoTool(Tool):
    """A deterministic successful tool used beside injected failures."""

    name = "echo"
    description = "Echo a value."
    args_model = ValueArgs

    async def execute(self, value: str) -> str:
        return f"echo:{value}"


class FailNTimesTool(Tool):
    """Return an error N times, then succeed."""

    name = "fault"
    description = "A deterministic fail-N-then-succeed tool."
    args_model = ValueArgs

    def __init__(self, failures: int) -> None:
        self.failures_remaining = max(0, failures)

    async def execute(self, value: str) -> str:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            return f"Error: injected failure for {value}"
        return f"recovered:{value}"


class TimeoutNTimesTool(Tool):
    """Return the exec-timeout signature N times, then succeed."""

    name = "exec"
    description = "A deterministic timeout-N-then-succeed exec tool."
    args_model = CommandArgs

    def __init__(self, failures: int) -> None:
        self.failures_remaining = max(0, failures)

    async def execute(self, command: str) -> str:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            return "Error: Command timed out after 3 seconds"
        return f"completed:{command}"


class ScriptedFaultProvider(LLMProvider):
    """Serve fixed domain responses and separate guard actions without I/O."""

    def __init__(
        self,
        domain_responses: list[LLMResponse],
        *,
        guard_actions: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.domain_responses = copy.deepcopy(domain_responses)
        self.guard_actions = list(guard_actions or ["continue"])
        self.domain_calls = 0
        self.guard_calls = 0
        self.domain_usage: dict[str, int] = {}
        self.harness_usage: dict[str, int] = {}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if response_format == {"type": "json_object"}:
            self.guard_calls += 1
            action = self.guard_actions.pop(0) if self.guard_actions else "continue"
            response = LLMResponse(
                content=json.dumps({"action": action}),
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            )
            accumulate_usage(self.harness_usage, response.usage)
            return response

        self.domain_calls += 1
        if not self.domain_responses:
            raise AssertionError("No scripted domain response left")
        response = self.domain_responses.pop(0)
        accumulate_usage(self.domain_usage, response.usage)
        return response


@dataclass(frozen=True)
class FaultScenario:
    """A reproducible provider/tool script plus benchmark metadata."""

    name: str
    provider: ScriptedFaultProvider
    tools: ToolRegistry
    has_tool_error: bool = True
    success_text: str = "scenario complete"


def build_fault_scenario(name: str) -> FaultScenario:
    """Build a fresh named scenario for one isolated experiment run."""

    builders = {
        "fail_n_then_succeed": _fail_n_then_succeed,
        "timeout_n_then_succeed": _timeout_n_then_succeed,
        "duplicate_loop": _duplicate_loop,
        "malformed_loop": _malformed_loop,
        "mixed_batch": _mixed_batch,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(f"unknown fault scenario: {name}") from exc


def scenario_names() -> tuple[str, ...]:
    return (
        "fail_n_then_succeed",
        "timeout_n_then_succeed",
        "duplicate_loop",
        "malformed_loop",
        "mixed_batch",
    )


def _fail_n_then_succeed() -> FaultScenario:
    responses = [
        _tool_response(f"call-{index}", "fault", {"value": f"attempt-{index}"})
        for index in range(1, 4)
    ]
    responses.append(_final_response())
    return FaultScenario(
        name="fail_n_then_succeed",
        provider=ScriptedFaultProvider(responses),
        tools=_registry(FailNTimesTool(2)),
    )


def _timeout_n_then_succeed() -> FaultScenario:
    responses = [
        _tool_response(f"call-{index}", "exec", {"command": f"attempt-{index}"})
        for index in range(1, 4)
    ]
    responses.append(_final_response())
    return FaultScenario(
        name="timeout_n_then_succeed",
        provider=ScriptedFaultProvider(responses),
        tools=_registry(TimeoutNTimesTool(2)),
    )


def _duplicate_loop() -> FaultScenario:
    responses = [
        _tool_response(f"call-{index}", "echo", {"value": "same"})
        for index in range(1, 4)
    ]
    responses.append(_final_response())
    return FaultScenario(
        name="duplicate_loop",
        provider=ScriptedFaultProvider(responses, guard_actions=["continue"]),
        tools=_registry(EchoTool()),
    )


def _malformed_loop() -> FaultScenario:
    responses = [
        _tool_response(f"call-{index}", "echo", ["not-an-object"])
        for index in range(1, 3)
    ]
    responses.append(_final_response())
    return FaultScenario(
        name="malformed_loop",
        provider=ScriptedFaultProvider(responses, guard_actions=["continue"]),
        tools=_registry(EchoTool()),
    )


def _mixed_batch() -> FaultScenario:
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="fail-1", name="fault", arguments={"value": "one"}),
                ToolCallRequest(id="echo-1", name="echo", arguments={"value": "one"}),
            ],
            finish_reason="tool_calls",
            usage=_domain_usage(),
        ),
        _final_response(),
    ]
    return FaultScenario(
        name="mixed_batch",
        provider=ScriptedFaultProvider(responses),
        tools=_registry(FailNTimesTool(1), EchoTool()),
    )


def _tool_response(call_id: str, name: str, arguments: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
        finish_reason="tool_calls",
        usage=_domain_usage(),
    )


def _final_response() -> LLMResponse:
    return LLMResponse(content="scenario complete", usage=_domain_usage())


def _domain_usage() -> dict[str, int]:
    return {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry
