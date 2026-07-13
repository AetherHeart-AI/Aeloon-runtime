from __future__ import annotations

from typing import cast

import pytest
from pydantic import BaseModel

from aeloon_core.base_scheduler_tools import build_base_scheduler_tools
from aeloon_core.tools.base import FunctionTool
from aeloon_core.worker_control import WorkerControlService


def test_scheduler_tools_declare_polling_and_queries_read_only() -> None:
    tools = build_base_scheduler_tools(
        control=cast(WorkerControlService, object()),
        base_session_id="session",
        base_turn_id="turn",
    )

    modes = {}
    for name in (
        "discover_profiles",
        "list_workers",
        "inspect_worker",
        "await_workers",
        "spawn_worker",
        "send_worker",
        "resume_worker",
        "cancel_worker",
        "archive_worker",
    ):
        tool = tools.get(name)
        assert tool is not None
        modes[name] = tool.concurrency_mode

    assert modes == {
        "discover_profiles": "read_only",
        "list_workers": "read_only",
        "inspect_worker": "read_only",
        "await_workers": "read_only",
        "spawn_worker": "mutating",
        "send_worker": "mutating",
        "resume_worker": "mutating",
        "cancel_worker": "mutating",
        "archive_worker": "mutating",
    }


def test_function_tool_rejects_an_unknown_concurrency_mode() -> None:
    async def handler() -> str:
        return "ok"

    with pytest.raises(ValueError, match="invalid concurrency mode"):
        FunctionTool(
            name="bad",
            description="bad",
            args_model=BaseModel,
            handler=handler,
            concurrency_mode="invalid",  # type: ignore[arg-type]
        )
