"""Profile-specific domain and control nodes."""

from __future__ import annotations

from time import perf_counter

from aeloon_core.agents import BaseAgent, WorkerAgent
from aeloon_core.loop_guard import LoopGuardAction, LoopGuardDecision
from aeloon_core.profile_delegation import (
    joined_tool_result,
    prepare_delegation,
    run_parallel_delegation,
)
from aeloon_core.profile_runtime import (
    MAX_DELEGATION_ROUNDS,
    CompleteArguments,
    DelegateArguments,
    HandoffArguments,
    delegation_fingerprint,
    parse_control_arguments,
)
from aeloon_core.state import AgentNode, LightweightState, PendingHandoff, RunStatus
from aeloon_core.transitions import NodeKind, accumulate_usage


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

        if isinstance(arguments, DelegateArguments):
            return await self._run_delegation(state, tool_call.id, tool_call.name, arguments)

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

    async def _run_delegation(
        self,
        state: LightweightState,
        tool_call_id: str,
        tool_name: str,
        arguments: DelegateArguments,
    ) -> LightweightState:
        if state.delegation_count >= MAX_DELEGATION_ROUNDS:
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call_id,
                tool_name,
                "Error: Parallel delegation budget exhausted "
                f"({state.delegation_count}/{MAX_DELEGATION_ROUNDS}).",
            )
            self._clear_pending(state)
            return await self.runtime.profile_protocol_error(
                state,
                reason="parallel delegation budget exhausted",
            )

        round_number = state.delegation_count + 1
        fingerprint = delegation_fingerprint(arguments)
        if (
            fingerprint == state.last_delegation_fingerprint
            and state.last_delegation_succeeded
        ):
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call_id,
                tool_name,
                "Error: This successful parallel delegation was already completed; "
                "change the task set or synthesize the existing reports.",
            )
            self._clear_pending(state)
            state.resume_agent_id = state.active_agent_id
            decision = LoopGuardDecision(
                LoopGuardAction.RETURN_TO_MODEL,
                reason="duplicate successful parallel delegation was rejected",
                prompt_message={
                    "role": "system",
                    "content": (
                        "Do not repeat the completed delegation. Use its existing reports, "
                        "change the task set for a material evidence gap, or complete the task."
                    ),
                },
            )
            await self.runtime.emit_guard_decision(
                decision,
                event="duplicate_delegation",
            )
            return await self.runtime.return_to_model(state, decision)
        try:
            branches = prepare_delegation(
                self.runtime,
                arguments,
                round_number=round_number,
            )
        except (KeyError, ValueError) as exc:
            state.messages = self.runtime.add_tool_result(
                state.messages,
                tool_call_id,
                tool_name,
                f"Error: Invalid parallel delegation: {exc}",
            )
            self._clear_pending(state)
            return await self.runtime.profile_protocol_error(
                state,
                reason="invalid parallel delegation",
            )

        source_agent = state.active_agent_id or self.runtime.profile.default_agent_id
        state.delegation_count = round_number
        started_at = perf_counter()
        results = await run_parallel_delegation(self.runtime, state, branches)
        aggregate_usage: dict[str, int] = {}
        for result in results:
            usage = state.token_ledger.merge(
                result.ledger,
                component_prefix=(
                    f"subagent:{result.branch.branch_id}:{result.branch.label}"
                ),
            )
            accumulate_usage(aggregate_usage, usage)
            state.tools_used.extend(result.tools_used)
            component_prefix = f"subagent:{result.branch.branch_id}:{result.branch.label}"
            for component, component_usage in result.ledger.by_component.items():
                if component_usage:
                    node_kind = _subagent_component_node_kind(component)
                    await self.runtime.emit_hook(
                        "on_usage",
                        component_usage,
                        node_kind=node_kind.value,
                        component=f"{component_prefix}:{component}",
                    )

        state.messages = self.runtime.add_tool_result(
            state.messages,
            tool_call_id,
            tool_name,
            joined_tool_result(results),
        )
        self._clear_pending(state)
        state.resume_agent_id = source_agent
        state.metadata.phase = AgentNode.MASTER
        succeeded = sum(result.succeeded for result in results)
        state.last_delegation_fingerprint = fingerprint
        state.last_delegation_succeeded = succeeded == len(results)
        duration_ms = max(0, int((perf_counter() - started_at) * 1_000))
        self.runtime.describe_step(
            {
                "action": "delegate",
                "source_agent_id": source_agent,
                "delegation_round": round_number,
                "branches": len(results),
                "succeeded": succeeded,
            },
            usage=aggregate_usage,
        )
        await self.runtime.emit_hook(
            "on_profile_delegate_join",
            source_agent,
            delegation_round=round_number,
            branch_count=len(results),
            succeeded=succeeded,
            duration_ms=duration_ms,
        )
        return state

    @staticmethod
    def _clear_pending(state: LightweightState) -> None:
        state.pending_response = None
        state.pending_control_call = None
        state.pending_tool_calls = []


def _subagent_component_node_kind(component: str) -> NodeKind:
    if component == "minimal_context":
        return NodeKind.CONTEXT_PROCESSING
    if component in {"profile_master", "temporary_guard", "control"}:
        return NodeKind.HARNESS
    return NodeKind.DOMAIN
