"""Tests for the Ultra Master prompt contract."""

from aeloon_core.harness.agent.prompt import (
    MASTER_SYSTEM_MARKER,
    MASTER_USER_REQUEST_MARKER,
    master_system_prompt,
)


def test_prompt_describes_ultra_master_and_ephemeral_experts() -> None:
    prompt = master_system_prompt(
        expert_descriptors=[
            {
                "id": "builtin:coding",
                "kind": "expert",
                "description": "Build and review",
                "runner": "builtin.coding",
            }
        ],
        plain_skill_ids=[],
        mode="normal",
        mcp_server_ids=["github"],
        capability_names=["filesystem", "shell"],
    )

    assert prompt.startswith(MASTER_SYSTEM_MARKER)
    assert "full-capability Master" in prompt
    assert "`expert_run`" in prompt
    assert "Experts cannot call other experts" in prompt
    assert "There is no generic DAG" in prompt
    assert "builtin:coding" in prompt
    assert "Normal mode is active" in prompt
    assert "github" in prompt
    assert "workflow_execute" not in prompt
    assert "run_workflow" not in prompt
    assert "resume" in prompt


def test_prompt_lists_only_allowlisted_plain_skills() -> None:
    prompt = master_system_prompt(
        expert_descriptors=[],
        plain_skill_ids=["workspace:conventions"],
        mode="expert",
        mcp_server_ids=[],
        capability_names=["filesystem"],
    )

    assert "workspace:conventions" in prompt
    assert "Expert mode is active" in prompt
    assert "limited to the explicit scopes" in prompt


def test_user_request_marker_is_stable() -> None:
    assert MASTER_USER_REQUEST_MARKER.endswith(":\n")
