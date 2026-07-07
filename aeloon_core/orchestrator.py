"""Thin orchestration layer around the standalone kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aeloon_core.config import Config
from aeloon_core.context import append_user_message
from aeloon_core.kernel import run_agent_kernel
from aeloon_core.providers.factory import create_provider
from aeloon_core.session import SessionStore
from aeloon_core.tools.factory import register_core_tools
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.tools.todo import TodoWriteTool


@dataclass
class TurnResult:
    """Result of one orchestrated agent turn."""

    session_id: str
    final_content: str | None
    tools_used: list[str]
    messages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]


class ConsoleProgress:
    """Console progress consumer used by the CLI."""

    def __init__(self) -> None:
        self._streaming = False

    async def __call__(self, text: str, *, tool_hint: bool = False) -> None:
        prefix = "tools" if tool_hint else "status"
        if self._streaming:
            print()
            self._streaming = False
        print(f"[{prefix}] {text}")

    async def on_llm_delta(self, delta: str) -> None:
        print(delta, end="", flush=True)
        self._streaming = True

    async def on_tool_calls(self, tool_calls: list[Any]) -> None:
        if self._streaming:
            print()
            self._streaming = False
        names = ", ".join(call.name for call in tool_calls)
        print(f"[tool calls] {names}")

    async def on_tool_result(self, node: Any) -> None:
        result = str(node.result or "")
        preview = result[:500] + ("..." if len(result) > 500 else "")
        print(f"[tool result] {node.tool_name}: {preview}")

    async def on_final(self, content: str, **kwargs: Any) -> None:
        del kwargs
        if self._streaming:
            print()
            self._streaming = False
        print(f"\n[final]\n{content}")


class AeloonCoreOrchestrator:
    """Build messages, run the kernel, and persist turns."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.provider = create_provider(config)
        self.registry = ToolRegistry()
        self.todo_tool: TodoWriteTool = register_core_tools(self.registry, config)
        self.sessions = SessionStore(data_dir=config.data_dir, workspace=config.workspace)

    async def run_turn(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        on_progress: Any | None = None,
    ) -> TurnResult:
        """Run one prompt through the agent loop."""

        actual_session_id = session_id or self.sessions.new_session()
        self.todo_tool.set_session_id(actual_session_id)
        messages = append_user_message(self.sessions.load_messages(actual_session_id), prompt)
        final_content, tools_used, messages = await run_agent_kernel(
            provider=self.provider,
            model=self.config.agents.defaults.model,
            tools=self.registry,
            messages=messages,
            max_iterations=self.config.agents.defaults.max_iterations,
            on_progress=on_progress,
        )
        blocks = list(getattr(on_progress, "blocks", []) or [])
        self.sessions.append_turn(
            session_id=actual_session_id,
            user_prompt=prompt,
            final_content=final_content,
            tools_used=tools_used,
            messages=messages,
            blocks=blocks,
        )
        return TurnResult(
            session_id=actual_session_id,
            final_content=final_content,
            tools_used=tools_used,
            messages=messages,
            blocks=blocks,
        )
