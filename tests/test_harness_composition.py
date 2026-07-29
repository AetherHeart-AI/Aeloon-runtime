"""Integration tests for Ultra and Expert Harness composition."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.compaction import SlidingWindow
from pydantic_ai_harness.context import RepoContext
from pydantic_ai_harness.planning import Planning

from aeloon_core.config import Config
from aeloon_core.harness.capabilities import history_capability, master_capabilities


def test_history_policy_is_owned_by_harness_sliding_window(tmp_path: Path) -> None:
    config = Config(
        workspace=tmp_path,
        agents={
            "defaults": {
                "context_window_tokens": 100_000,
                "context_compaction": {
                    "trigger_ratio": 0.8,
                    "preserve_recent_tokens": 20_000,
                },
            }
        },
    )

    capability = history_capability(config)

    assert isinstance(capability, SlidingWindow)
    assert capability.max_tokens == 80_000
    assert capability.keep_tokens == 20_000


def test_master_is_an_ultra_worker_with_direct_workspace_capabilities(
    tmp_path: Path,
) -> None:
    capabilities = master_capabilities(
        Config(
            mode="normal",
            workspace=tmp_path,
            tools={"master_capabilities": []},
        ).normalized()
    )

    assert any(isinstance(item, FileSystem) for item in capabilities)
    assert any(isinstance(item, Shell) for item in capabilities)
    assert any(isinstance(item, RepoContext) for item in capabilities)
    assert any(isinstance(item, Planning) for item in capabilities)
    assert any(isinstance(item, SlidingWindow) for item in capabilities)


def test_master_capabilities_do_not_include_dynamic_workflow(tmp_path: Path) -> None:
    names = {type(item).__name__ for item in master_capabilities(Config(workspace=tmp_path))}

    assert "DynamicWorkflow" not in names
    assert "SubAgents" not in names


def test_expert_mode_restricts_master_host_capabilities(tmp_path: Path) -> None:
    capabilities = master_capabilities(
        Config(
            mode="expert",
            workspace=tmp_path,
            tools={"master_capabilities": ["repo_context", "planning"]},
        ).normalized()
    )

    assert not any(isinstance(item, FileSystem) for item in capabilities)
    assert not any(isinstance(item, Shell) for item in capabilities)
    assert any(isinstance(item, RepoContext) for item in capabilities)
    assert any(isinstance(item, Planning) for item in capabilities)
