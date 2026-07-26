from aeloon_core.master_prompt import (
    MASTER_SYSTEM_MARKER,
    MASTER_USER_REQUEST_MARKER,
    master_system_prompt,
)


def test_prompt_describes_one_ephemeral_harness_path() -> None:
    prompt = master_system_prompt(
        worker_types=[
            {
                "id": "builder",
                "description": "Build",
                "source": "builtin:builder.md",
                "digest": "a" * 64,
            }
        ]
    )

    assert prompt.startswith(MASTER_SYSTEM_MARKER)
    assert "All child-agent work is ephemeral" in prompt
    assert "run_workflow" in prompt
    assert "workflow_execute" in prompt
    assert "Workflow Template candidates" in prompt
    assert "asyncio.gather" in prompt
    assert "plain text" in prompt
    assert "Each Worker segment has at most 25 model requests" in prompt
    assert "At most 4 continuations (5 total segments)" in prompt
    assert "you—not the Worker—must decide" in prompt
    assert "durable WorkerSessions" in prompt
    assert "create_flow" not in prompt
    assert "resume_worker" not in prompt
    assert "finish_turn" not in prompt


def test_prompt_can_disable_template_tools() -> None:
    prompt = master_system_prompt(
        worker_types=[],
        workflow_templates_enabled=False,
    )

    assert "workflow_execute" not in prompt
    assert "Workflow Template candidates" not in prompt
    assert "`run_workflow`." in prompt


def test_user_request_marker_is_stable() -> None:
    assert MASTER_USER_REQUEST_MARKER.endswith(":\n")
