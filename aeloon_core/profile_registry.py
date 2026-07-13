"""Prompt-free discovery of active Worker profile artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from aeloon_core.profile_artifacts import ProfileArtifactStore
from aeloon_core.profiles import RuntimeProfileSpec
from aeloon_core.worker_sessions import ProfileHandle


@dataclass(frozen=True)
class ProfileDescriptor:
    profile: ProfileHandle
    description: str
    capability_tags: tuple[str, ...]
    requested_tools: tuple[str, ...]
    side_effect_level: str
    supports_parallel: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize only capability metadata; prompts are intentionally absent."""

        return {
            "profile": self.profile.model_dump(mode="json"),
            "description": self.description,
            "capability_tags": list(self.capability_tags),
            "requested_tools": list(self.requested_tools),
            "side_effect_level": self.side_effect_level,
            "supports_parallel": self.supports_parallel,
        }


class ProfileRegistry:
    """Read active compatible profiles without exposing their instructions."""

    def __init__(self, store: ProfileArtifactStore) -> None:
        self.store = store

    def discover(self) -> list[ProfileDescriptor]:
        descriptors: list[ProfileDescriptor] = []
        for status in self.store.list_active():
            try:
                runtime = self.store.load_active(str(status["profile_id"]))
            except Exception:
                continue
            descriptors.append(self._descriptor(runtime, status))
        return descriptors

    @staticmethod
    def _descriptor(runtime: RuntimeProfileSpec, status: dict[str, Any]) -> ProfileDescriptor:
        tools = tuple(sorted({tool for agent in runtime.agents for tool in agent.tools}))
        contract = {
            "profile_id": runtime.profile_id,
            "description": runtime.description,
            "tools": tools,
            "control_protocol_version": runtime.control_protocol_version,
        }
        contract_hash = hashlib.sha256(
            json.dumps(contract, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        return ProfileDescriptor(
            profile=ProfileHandle(
                profile_id=runtime.profile_id,
                artifact_id=runtime.artifact_id or "",
                generation=runtime.generation,
                activation_audit_id=str(status.get("audit_id") or runtime.artifact_id or "legacy"),
                contract_hash=contract_hash,
            ),
            description=runtime.description,
            capability_tags=tuple(
                sorted({runtime.profile_id, *(tool.split("_")[0] for tool in tools)})
            ),
            requested_tools=tools,
            side_effect_level=(
                "read_only"
                if all(
                    tool in {"read", "glob", "grep", "web_fetch", "web_search"}
                    for tool in tools
                )
                else "mutating"
            ),
            supports_parallel=runtime.control_protocol_version >= 2,
        )
