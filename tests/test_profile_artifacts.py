from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aeloon_core.profile_artifacts import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    ArtifactLifecycleError,
    CompatibilityPolicy,
    ProfileArtifactStore,
    ProfileCompilationError,
)

PROFILE = """---
schema_version: 1
id: coding-team
revision: 1
description: Coding team
default_agent: implementer
max_handoffs: 3
agents:
  - id: planner
    description: Plan the work
    tools: [read]
  - id: implementer
    description: Implement the work
    tools: [read, write]
---

## Shared
Shared constraints.

## Master
Route to the best role.

## Agent: planner
Produce a plan.

## Agent: implementer
Implement and verify.
"""


def compile_sync(store: ProfileArtifactStore, source: str = PROFILE) -> dict[str, Any]:
    return asyncio.run(store.compile(source))


def store_for(tmp_path: Path) -> ProfileArtifactStore:
    return ProfileArtifactStore(
        data_dir=tmp_path,
        compatibility=CompatibilityPolicy(
            tool_schema_fingerprints={"read": "read", "write": "write"}
        ),
    )


def test_compile_is_content_addressed_and_state_is_derived(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    first = compile_sync(store)
    second = compile_sync(store)
    assert first["artifact_id"] == second["artifact_id"]
    assert first["state"] == "validated"
    assert second["cache_hit"] is True
    directory = tmp_path / "profile-artifacts" / first["artifact_id"]
    assert not (directory / "lifecycle.json").exists()
    assert not (directory / "transaction.json").exists()
    inspected = store.inspect(first["artifact_id"])
    assert inspected["state"] == "validated"
    assert inspected["validation_report"]["valid"] is True


@pytest.mark.asyncio
async def test_llm_compile_repairs_once_and_cache_skips_provider(tmp_path: Path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_with_retry(self, **kwargs: Any) -> Any:
            self.calls += 1
            content = "bad" if self.calls == 1 else _compiled()
            return type("Response", (), {"content": content, "usage": {}})()

    provider = Provider()
    store = store_for(tmp_path)
    result = await store.compile(PROFILE, compiler="llm", provider=provider, model="test")
    assert result["state"] == "validated"
    assert provider.calls == 2
    await store.compile(PROFILE, compiler="llm", provider=provider, model="test")
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_failed_llm_compile_is_quarantined(tmp_path: Path) -> None:
    class Provider:
        async def chat_with_retry(self, **kwargs: Any) -> Any:
            return type("Response", (), {"content": "bad", "usage": {}})()

    store = store_for(tmp_path)
    with pytest.raises(ProfileCompilationError) as exc:
        await store.compile(PROFILE, compiler="llm", provider=Provider(), model="test")
    assert exc.value.artifact_id is not None
    assert store.inspect(exc.value.artifact_id)["state"] == "quarantined"


def test_approval_activation_and_rollback_use_pointer_and_audit(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    first = compile_sync(store)
    store.approve(first["artifact_id"], approved_by="operator")
    assert store.inspect(first["artifact_id"])["state"] == "approved"
    activated = store.activate(first["artifact_id"])
    assert activated["state"] == "active"
    assert store.status("coding-team")["artifact_id"] == first["artifact_id"]
    assert store.load_active("coding-team").profile_id == "coding-team"
    assert list((tmp_path / "profile-artifacts" / "activation-audit").glob("*.json"))
    assert store.rollback(first["artifact_id"])["rollback"] is True


def test_reactivating_current_artifact_is_idempotent(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    artifact = compile_sync(store)
    store.approve(artifact["artifact_id"])
    first = store.activate(artifact["artifact_id"])
    second = store.activate(artifact["artifact_id"])

    assert second == first
    assert second["generation"] == 1
    assert len(list(store.audit_dir.glob("*.json"))) == 1


def test_activation_requires_approval_and_compatible_tools(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    artifact = compile_sync(store)
    with pytest.raises(ArtifactLifecycleError):
        store.activate(artifact["artifact_id"])
    store.approve(artifact["artifact_id"])
    incompatible = ProfileArtifactStore(
        data_dir=tmp_path,
        compatibility=CompatibilityPolicy(tool_schema_fingerprints={"read": "changed"}),
    )
    with pytest.raises(ArtifactCompatibilityError):
        incompatible.activate(artifact["artifact_id"])


def test_pointer_or_audit_tampering_is_rejected(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    artifact = compile_sync(store)
    store.approve(artifact["artifact_id"])
    store.activate(artifact["artifact_id"])
    pointer = tmp_path / "profile-artifacts" / "active" / "coding-team.json"
    data = json.loads(pointer.read_text())
    data["artifact_digest"] = "0" * 64
    pointer.write_text(json.dumps(data))
    with pytest.raises(ArtifactIntegrityError):
        store.load_active("coding-team")


def _compiled() -> str:
    return """class CompiledProfile:
    profile_schema_version = 1
    compiled_api_version = 1
    profile_id = "coding-team"
    revision = 1
    description = "Coding team"
    default_agent_id = "implementer"
    max_handoffs = 3
    master_prompt = "Route to the best role."
    shared_prompt = "Shared constraints."
    agents = (
        {
            "id": "planner",
            "description": "Plan the work",
            "prompt": "Produce a plan.",
            "tools": ("read",),
        },
        {
            "id": "implementer",
            "description": "Implement the work",
            "prompt": "Implement and verify.",
            "tools": ("read", "write"),
        },
    )
"""
