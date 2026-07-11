"""Profile-specific domain and control nodes."""

from __future__ import annotations

from aeloon_core.agents import BaseAgent, WorkerAgent
from aeloon_core.profile_runtime import (
    CompleteArguments,
    HandoffArguments,
    parse_control_arguments,
)
from aeloon_core.state import AgentNode, LightweightState, PendingHandoff, RunStatus
from aeloon_core.transitions import NodeKind


class ProfileDomainAgent(WorkerAgent):
    """One independent worker instance bound to a declared profile role."""

    def __init__(self, runtime, agent_id: str) -> None:
        super().__init__(runtime)
        self.agent_id = agent_id

    async def run(self, state: LightweightState) -> LightweightState:
        if state.active_agent_id != self.agent_id:
            return await self.runtime.finish(
                state,
                content="The profile runtime selected an inconsistent role.",
                status=RunStatus.FAILED,
                reason="profile role dispatch mismatch",
                add_message=True,
            )
        return await super().run(state)

    def component_for(self, state: LightweightState) -> str:
        del state
        return f"domain:{self.agent_id}"


class ControlAgent(BaseAgent):
    """Execute the closed, side-effect-free profile control protocol."""

    node = AgentNode.CONTROL
    node_kind = NodeKind.HARNESS

    async def run(self, state: LightweightState) -> LightweightState:
        response = state.pending_response
        tool_call = state.pending_control_call
        profile = self.runtime.profile
        if response is None or tool_call is None or profile is None:
            self._clear_pending(state)
            state.metadata.phase = AgentNode.MASTER
            self.runtime.describe_step(
                {"route": "master", "reason": "no pending profile control call"}
            )
            return state

        state.messages = self.runtime.add_assistant_message(
            state.messages,
            response.content,
            tool_calls=[tool_call.to_openai_tool_call()],
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        try:
            arguments = parse_control_arguments(
                tool_call.name,
                tool_call.arguments,
                declared_agent_ids={agent.id for agent in profile.agents},
            )
            if (
                isinstance(arguments, HandoffArguments)
                and arguments.recommended_agent == state.active_agent_id
            ):
                raise ValueError("handoff recommendation must name a different role")
        except ValueError as exc:
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call.id,
                tool_call.name,
                f"Error: Invalid profile control call: {exc}",
            )
            self._clear_pending(state)
            return await self.runtime.profile_protocol_error(
                state,
                reason="invalid control arguments",
            )

        if isinstance(arguments, CompleteArguments):
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call.id,
                tool_call.name,
                "Task completion accepted.",
            )
            self._clear_pending(state)
            state.messages = self.runtime.add_assistant_message(
                state.messages,
                arguments.final_content,
            )
            self.runtime.describe_step(
                {"action": "complete", "agent_id": state.active_agent_id}
            )
            await self.runtime.emit_hook(
                "on_profile_completion",
                state.active_agent_id,
                arguments.final_content,
            )
            return await self.runtime.finish(
                state,
                content=arguments.final_content,
                status=RunStatus.COMPLETED,
                reason="profile role completed",
                add_message=False,
            )

        limit = self.runtime.profile_handoff_limit()
        if state.handoff_count >= limit:
            content = (
                f"The profile handoff budget was exhausted at {state.handoff_count}/{limit}; "
                "the active role attempted another handoff instead of completing the task."
            )
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call.id,
                tool_call.name,
                f"Error: Handoff budget exhausted ({state.handoff_count}/{limit}).",
            )
            self._clear_pending(state)
            self.runtime.describe_step(
                {
                    "action": "terminate",
                    "source": "handoff_budget",
                    "handoff_count": state.handoff_count,
                    "handoff_limit": limit,
                }
            )
            return await self.runtime.finish(
                state,
                content=content,
                status=RunStatus.TERMINATED_BY_RULE,
                reason="profile handoff budget exhausted",
                add_message=False,
            )

        source_agent = state.active_agent_id or profile.default_agent_id
        state.handoff_count += 1
        state.pending_handoff = PendingHandoff(
            from_agent_id=source_agent,
            summary=arguments.summary,
            recommended_agent_id=arguments.recommended_agent,
        )
        state.messages = self.runtime.add_tool_result(
            state.messages,
            tool_call.id,
            tool_call.name,
            "Handoff accepted; the profile master will select the next role.",
        )
        self._clear_pending(state)
        state.metadata.phase = AgentNode.MASTER
        self.runtime.describe_step(
            {
                "action": "handoff",
                "from_agent_id": source_agent,
                "recommended_agent_id": arguments.recommended_agent,
                "handoff_count": state.handoff_count,
                "handoff_limit": limit,
            }
        )
        await self.runtime.emit_hook(
            "on_profile_handoff",
            source_agent,
            arguments.recommended_agent,
            arguments.summary,
            handoff_count=state.handoff_count,
            handoff_limit=limit,
        )
        return state

    @staticmethod
    def _clear_pending(state: LightweightState) -> None:
        state.pending_response = None
        state.pending_control_call = None
        state.pending_tool_calls = []
