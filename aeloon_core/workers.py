"""Compatibility exports for the Python Role system.

New code should import from :mod:`aeloon_core.customization`.
"""

from aeloon_core.customization.roles import (
    ALLOWED_ROLE_CAPABILITIES,
    MAX_ROLE_DESCRIPTION_CHARS,
    MAX_ROLE_PROMPT_CHARS,
    ROLE_ID_PATTERN,
    ReviewFinding,
    ReviewReport,
    Role,
    RoleDefinitionError,
    RoleRegistry,
    RoleSnapshot,
    WorkerDefinitionError,
    WorkerEvidence,
    WorkerRegistry,
    WorkerReport,
    WorkerSnapshot,
    role_digest,
    snapshot_role,
)

# Legacy constant aliases retained for import compatibility. Markdown parsing itself
# has intentionally been removed.
WORKER_ID_PATTERN = ROLE_ID_PATTERN
MAX_WORKER_DESCRIPTION_CHARS = MAX_ROLE_DESCRIPTION_CHARS
MAX_WORKER_PROMPT_CHARS = MAX_ROLE_PROMPT_CHARS
MAX_WORKER_SOURCE_CHARS = MAX_ROLE_PROMPT_CHARS
MAX_WORKER_SOURCE_METADATA_CHARS = 4_096


def parse_worker(*args: object, **kwargs: object) -> None:
    """Reject the removed Markdown worker format with an actionable error."""

    del args, kwargs
    raise WorkerDefinitionError(
        "Markdown worker definitions are no longer supported; define a Role subclass "
        "and export it from .aeloon-core/catalog.py"
    )


def worker_digest(
    *,
    worker_id: str,
    description: str,
    prompt: str,
) -> str:
    """Compatibility digest for callers that have not migrated to Role definitions."""

    from pydantic import BaseModel

    return role_digest(
        role_id=worker_id,
        description=description,
        system_prompt=prompt,
        output_model=BaseModel,
        model_tier="strong",
        capabilities=(),
        concurrency_mode="exclusive",
    )


__all__ = [
    "ALLOWED_ROLE_CAPABILITIES",
    "MAX_ROLE_DESCRIPTION_CHARS",
    "MAX_ROLE_PROMPT_CHARS",
    "MAX_WORKER_DESCRIPTION_CHARS",
    "MAX_WORKER_PROMPT_CHARS",
    "MAX_WORKER_SOURCE_CHARS",
    "MAX_WORKER_SOURCE_METADATA_CHARS",
    "ROLE_ID_PATTERN",
    "ReviewFinding",
    "ReviewReport",
    "Role",
    "RoleDefinitionError",
    "RoleRegistry",
    "RoleSnapshot",
    "WORKER_ID_PATTERN",
    "WorkerDefinitionError",
    "WorkerEvidence",
    "WorkerRegistry",
    "WorkerReport",
    "WorkerSnapshot",
    "parse_worker",
    "role_digest",
    "snapshot_role",
    "worker_digest",
]
