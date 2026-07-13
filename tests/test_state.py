from __future__ import annotations

from aeloon_core.state import (
    AgentNode,
    LazyValue,
    LightweightState,
    ProfileRef,
    RunStatus,
    StateMetadata,
)
from aeloon_core.transitions import NodeKind, TransitionRecord


def test_lightweight_state_defaults_to_a_detached_model_context() -> None:
    messages = [{"role": "user", "content": "hello"}]

    state = LightweightState(messages=messages)

    assert state.minimal_context == messages
    assert state.minimal_context is not messages
    assert state.metadata.phase == AgentNode.MASTER
    assert state.metadata.status == RunStatus.RUNNING


def test_state_metadata_requires_visible_content_for_terminal_status() -> None:
    metadata = StateMetadata()

    metadata.finish(
        status=RunStatus.TERMINATED_BY_GUARD,
        final_content="Iteration budget exhausted.",
        reason="budget",
    )

    assert metadata.is_terminal
    assert metadata.phase == AgentNode.DONE
    assert metadata.final_content == "Iteration budget exhausted."
    assert metadata.termination_reason == "budget"


def test_lazy_values_are_content_addressed_and_deduplicated() -> None:
    state = LightweightState(messages=[])
    value = {"result": "large output", "count": 2}

    first = state.store_lazy(value, prefix="tool-result")
    second = state.store_lazy({"count": 2, "result": "large output"}, prefix="tool-result")

    assert first == second
    assert first.startswith("lazy://tool-result/")
    assert len(state.lazy_refs) == 1
    assert state.resolve_lazy(first) == value


def test_resolving_deferred_value_does_not_change_state_digest() -> None:
    state = LightweightState(messages=[])
    lazy = LazyValue.deferred("lazy://fixture/stable", lambda: "loaded later")
    state.register_lazy(lazy)
    before = state.stable_digest()

    assert state.resolve_lazy(lazy.ref) == "loaded later"

    assert state.stable_digest() == before


def test_state_digest_is_stable_and_excludes_transition_history() -> None:
    left = LightweightState(
        messages=[{"role": "user", "content": {"b": 2, "a": 1}}],
        permissions={"write": False, "read": True},
        active_tools=["read"],
    )
    right = LightweightState(
        messages=[{"content": {"a": 1, "b": 2}, "role": "user"}],
        permissions={"read": True, "write": False},
        active_tools=["read"],
    )

    assert left.stable_digest() == right.stable_digest()

    before_transition = left.stable_digest()
    left.append_transition(
        TransitionRecord(
            sequence=1,
            iteration=0,
            node="master",
            node_kind=NodeKind.DOMAIN,
            before_digest=before_transition,
            after_digest=before_transition,
        )
    )
    assert left.stable_digest() == before_transition

    left.messages.append({"role": "assistant", "content": "changed"})
    assert left.stable_digest() != right.stable_digest()


def test_profile_state_is_omitted_until_a_turn_pins_an_artifact() -> None:
    state = LightweightState(messages=[{"role": "user", "content": "work"}])
    no_profile_digest = state.digest()

    state.profile_ref = ProfileRef(
        profile_id="coding-team",
        revision=1,
        artifact_id="artifact-1",
        generation=2,
    )
    pinned_digest = state.digest()
    state.active_agent_id = "implementer"

    assert pinned_digest != no_profile_digest
    assert state.digest() != pinned_digest


def test_from_messages_sets_the_single_iteration_limit() -> None:
    state = LightweightState.from_messages(
        [{"role": "user", "content": "work"}],
        max_iterations=7,
    )

    assert state.metadata.iteration_limit == 7
    assert not hasattr(state, "guard_state")
