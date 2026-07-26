"""Agent definitions, presets, and runtime construction."""

from aeloon_core.harness.agent.base import (
    ReviewFinding,
    ReviewReport,
    Role,
    RoleDefinitionError,
    RoleRegistry,
    RoleSnapshot,
    WorkerEvidence,
    WorkerReport,
)
from aeloon_core.harness.agent.factory import (
    RoleAgentFactory,
    history_capability,
    master_harness_capabilities,
)
from aeloon_core.harness.agent.presets import BUILTIN_ROLES
from aeloon_core.harness.agent.prompt import (
    MASTER_SYSTEM_MARKER,
    MASTER_USER_REQUEST_MARKER,
    SYSTEM_PROMPT,
    master_system_prompt,
)

__all__ = [
    "BUILTIN_ROLES",
    "MASTER_SYSTEM_MARKER",
    "MASTER_USER_REQUEST_MARKER",
    "ReviewFinding",
    "ReviewReport",
    "Role",
    "RoleAgentFactory",
    "RoleDefinitionError",
    "RoleRegistry",
    "RoleSnapshot",
    "SYSTEM_PROMPT",
    "WorkerEvidence",
    "WorkerReport",
    "history_capability",
    "master_harness_capabilities",
    "master_system_prompt",
]
