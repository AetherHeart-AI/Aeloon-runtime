"""Master prompt surface.

Reusable specialists now live under :mod:`aeloon_core.harness.expert` as
ExpertSkills rather than public Role/Workflow abstractions.
"""

from aeloon_core.harness.agent.prompt import (
    MASTER_SYSTEM_MARKER,
    MASTER_USER_REQUEST_MARKER,
    SYSTEM_PROMPT,
    master_system_prompt,
)

__all__ = [
    "MASTER_SYSTEM_MARKER",
    "MASTER_USER_REQUEST_MARKER",
    "SYSTEM_PROMPT",
    "master_system_prompt",
]
