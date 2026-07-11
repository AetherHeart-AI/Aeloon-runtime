from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from aeloon_core.context_compaction import COMPACTION_MARKER
from aeloon_core.minimal_context import (
    LAZY_TOOL_RESULT_MARKER,
    MinimalContextProcessor,
)


class FakeState:
    def __init__(self, *, active_tools: list[str]) -> None:
        self.active_tools = active_tools
        self.lazy_values: dict[str, Any] = {}

    def store_lazy(self, value: Any, *, prefix: str = "context") -> str:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        reference = f"lazy://{prefix}/{digest}"
        self.lazy_values[reference] = value
        return reference


def tool_definition(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": name}}


def assistant_tool_call(call_id: str, name: str = "read") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def tool_result(call_id: str, content: str, name: str = "read") -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def test_minimal_processor_keeps_prefix_latest_checkpoint_and_recent_turns() -> None:
    first_checkpoint = {
        "role": "system",
        "content": f"{COMPACTION_MARKER}\nfirst checkpoint",
    }
    latest_checkpoint = {
        "role": "system",
        "content": f"{COMPACTION_MARKER}\nlatest checkpoint",
    }
    messages = [
        {"role": "system", "content": "runtime rules"},
        {"role": "system", "content": "skill guidance"},
        first_checkpoint,
        {"role": "user", "content": "old turn"},
        assistant_tool_call("old-call"),
        tool_result("old-call", "old result"),
        latest_checkpoint,
        {"role": "user", "content": "recent turn one"},
        assistant_tool_call("recent-call"),
        tool_result("recent-call", "recent result"),
        {"role": "assistant", "content": "recent answer one"},
        {"role": "user", "content": "recent turn two"},
        {"role": "assistant", "content": "recent answer two"},
    ]
    original = copy.deepcopy(messages)

    result = MinimalContextProcessor(preserve_recent_turns=2).process(
        state=FakeState(active_tools=["read"]),
        messages=messages,
        tools=[tool_definition("read")],
    )

    contents = [str(message.get("content")) for message in result.messages]
    assert result.messages[:2] == messages[:2]
    assert latest_checkpoint in result.messages
    assert first_checkpoint not in result.messages
    assert "old turn" not in contents
    assert "recent turn one" in contents
    assert "recent turn two" in contents
    assert assistant_tool_call("recent-call") in result.messages
    assert tool_result("recent-call", "recent result") in result.messages
    assert messages == original


def test_minimal_processor_closes_tool_call_result_pairs_at_selection_boundary() -> None:
    messages = [
        {"role": "user", "content": "old turn"},
        assistant_tool_call("boundary-call"),
        {"role": "user", "content": "recent turn"},
        tool_result("boundary-call", "paired result"),
        {"role": "assistant", "content": "done"},
    ]

    result = MinimalContextProcessor(preserve_recent_turns=1).process(
        state=FakeState(active_tools=["read"]),
        messages=messages,
        tools=[tool_definition("read")],
    )

    assert assistant_tool_call("boundary-call") in result.messages
    assert tool_result("boundary-call", "paired result") in result.messages


def test_minimal_processor_lazily_stores_large_tool_results_and_filters_tools() -> None:
    state = FakeState(active_tools=["read"])
    large_result = "IMPORTANT-START " + ("large result " * 20) + "IMPORTANT-END"
    messages = [
        {"role": "user", "content": "read it"},
        assistant_tool_call("large-call"),
        tool_result("large-call", large_result),
    ]
    original = copy.deepcopy(messages)

    result = MinimalContextProcessor(max_tool_result_chars=48).process(
        state=state,
        messages=messages,
        tools=[tool_definition("read"), tool_definition("write")],
        additional_messages=[{"role": "user", "content": "finalize now"}],
    )

    assert [tool["function"]["name"] for tool in result.tools] == ["read"]
    assert len(result.lazy_references) == 1
    reference = result.lazy_references[0]
    assert state.lazy_values[reference] == large_result
    placeholder = result.messages[-2]["content"]
    assert LAZY_TOOL_RESULT_MARKER in placeholder
    assert reference in placeholder
    assert "IMPORTANT-START" in placeholder
    assert "IMPORTANT-END" in placeholder
    assert result.messages[-1] == {"role": "user", "content": "finalize now"}
    assert messages == original


@pytest.mark.parametrize(
    ("preserve_recent_turns", "max_tool_result_chars"),
    [(0, 10), (1, 0)],
)
def test_minimal_processor_rejects_non_positive_bounds(
    preserve_recent_turns: int,
    max_tool_result_chars: int,
) -> None:
    with pytest.raises(ValueError):
        MinimalContextProcessor(
            preserve_recent_turns=preserve_recent_turns,
            max_tool_result_chars=max_tool_result_chars,
        )
