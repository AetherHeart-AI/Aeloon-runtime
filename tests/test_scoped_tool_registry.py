from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from aeloon_core.tools.base import Tool
from aeloon_core.tools.registry import ScopedToolRegistry, ToolRegistry


class ValueArgs(BaseModel):
    value: str


class RecordingTool(Tool):
    name = "record"
    description = "Record one value."
    args_model = ValueArgs

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"{self.name}:{kwargs['value']}"


def definition_names(registry: ScopedToolRegistry) -> list[str]:
    return [definition["function"]["name"] for definition in registry.get_definitions()]


def test_scoped_registry_hides_disallowed_and_missing_tool_schemas() -> None:
    host = ToolRegistry()
    allowed = RecordingTool("allowed")
    denied = RecordingTool("denied")
    host.register(allowed)
    host.register(denied)

    scoped = ScopedToolRegistry(host, {"allowed", "missing"})

    assert definition_names(scoped) == ["allowed"]
    assert scoped.get("allowed") is allowed
    assert scoped.get("denied") is None
    assert scoped.get("missing") is None


@pytest.mark.asyncio
async def test_scoped_registry_executes_an_allowed_host_tool() -> None:
    host = ToolRegistry()
    allowed = RecordingTool("allowed")
    host.register(allowed)
    scoped = ScopedToolRegistry(host, {"allowed"})

    result = await scoped.execute("allowed", {"value": "one"})

    assert result == "allowed:one"
    assert allowed.calls == [{"value": "one"}]


@pytest.mark.asyncio
async def test_scoped_registry_rejects_unauthorized_tool_without_side_effects() -> None:
    host = ToolRegistry()
    allowed = RecordingTool("allowed")
    denied = RecordingTool("denied")
    host.register(allowed)
    host.register(denied)
    scoped = ScopedToolRegistry(host, {"allowed"})

    result = await scoped.execute("denied", {"value": "must-not-run"})

    assert result == "Error: Tool 'denied' not found. Available: allowed"
    assert denied.calls == []


@pytest.mark.asyncio
async def test_scoped_registry_does_not_gain_a_tool_missing_at_creation() -> None:
    host = ToolRegistry()
    allowed = RecordingTool("allowed")
    host.register(allowed)
    scoped = ScopedToolRegistry(host, {"allowed", "late"})

    late = RecordingTool("late")
    host.register(late)

    assert not hasattr(scoped, "register")
    assert definition_names(scoped) == ["allowed"]
    assert scoped.get("late") is None
    assert await scoped.execute("late", {"value": "must-not-run"}) == (
        "Error: Tool 'late' not found. Available: allowed"
    )
    assert late.calls == []
