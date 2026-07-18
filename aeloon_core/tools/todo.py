"""Tiny persistent todo tool."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from aeloon_core.tools.base import Tool


class TodoItem(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed"]
    id: str | None = None


class TodoArgs(BaseModel):
    todos: list[TodoItem] = Field(description="Full todo list to persist.")


class TodoWriteTool(Tool):
    """Write the current task todo list to disk."""

    name = "todowrite"
    concurrency_mode = "mutating"
    description = (
        "Persist a concise todo list for the current task. Send the full list each time, "
        "with statuses pending, in_progress, or completed."
    )
    args_model = TodoArgs

    def __init__(self, *, data_dir: Path, run_id: str) -> None:
        self.data_dir = data_dir
        self.run_id = run_id

    async def execute(self, todos: list[dict[str, Any]]) -> str:
        cleaned: list[dict[str, str]] = []
        for index, item in enumerate(todos, start=1):
            content = str(item.get("content") or "").strip()
            status = str(item.get("status") or "").strip()
            if not content:
                return f"Error: todos[{index - 1}].content is required"
            if status not in {"pending", "in_progress", "completed"}:
                return f"Error: todos[{index - 1}].status is invalid"
            cleaned.append(
                {
                    "id": str(item.get("id") or f"todo-{index}"),
                    "content": content,
                    "status": status,
                }
            )
        path = self.data_dir / "todos" / f"{self.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "todos": cleaned,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        done = sum(1 for item in cleaned if item["status"] == "completed")
        return f"Todo list updated: {done}/{len(cleaned)} completed ({path})."
