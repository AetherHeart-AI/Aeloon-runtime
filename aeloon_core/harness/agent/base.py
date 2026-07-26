"""Agent Role contracts, structured reports, and immutable registries."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ROLE_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
MAX_ROLE_DESCRIPTION_CHARS = 1_000
MAX_ROLE_PROMPT_CHARS = 128_000
ALLOWED_ROLE_CAPABILITIES = frozenset(
    {"filesystem", "shell", "repo_context", "planning"}
)


class RoleDefinitionError(ValueError):
    """Raised when a Python role definition or catalog is invalid."""


class WorkerEvidence(BaseModel):
    """One bounded evidence item returned by an isolated role."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    kind: Literal["file", "test", "typecheck", "lint", "runtime", "source"]
    locator: str = Field(min_length=1, max_length=1_000)
    claim: str = Field(min_length=1, max_length=1_000)
    status: Literal["passed", "failed", "observed", "not_applicable"]
    method: str | None = Field(default=None, min_length=1, max_length=2_000)
    finding_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )


class WorkerReport(BaseModel):
    """Default structured result returned by an isolated role."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    status: Literal["completed", "partial", "blocked"] = "completed"
    summary: str = Field(min_length=1, max_length=8_000)
    artifacts: tuple[str, ...] = Field(default=(), max_length=32)
    evidence: tuple[WorkerEvidence, ...] = Field(default=(), max_length=32)
    unresolved: tuple[str, ...] = Field(default=(), max_length=32)
    next_steps: tuple[str, ...] = Field(default=(), max_length=32)


class ReviewFinding(BaseModel):
    """One actionable code-review finding used by fixed review workflows."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    severity: Literal["critical", "high", "medium", "low"]
    location: str = Field(min_length=1, max_length=1_000)
    impact: str = Field(min_length=1, max_length=2_000)
    reproduction: str = Field(min_length=1, max_length=2_000)


class ReviewReport(WorkerReport):
    """Review output whose findings can drive bounded revision templates."""

    findings: tuple[ReviewFinding, ...] = Field(default=(), max_length=64)


RoleOutputT = TypeVar("RoleOutputT", bound=BaseModel)


class Role(Generic[RoleOutputT]):
    """Trusted Python definition for one reusable agent responsibility.

    Subclasses configure an agent. Runtime construction, budgets, lifecycle events,
    and execution remain host-owned.
    """

    id: ClassVar[str]
    description: ClassVar[str]
    system_prompt: ClassVar[str]
    output_model: ClassVar[type[BaseModel]] = WorkerReport
    model_tier: ClassVar[Literal["fast", "strong"]] = "strong"
    capabilities: ClassVar[tuple[str, ...]] = (
        "filesystem",
        "shell",
        "repo_context",
        "planning",
    )
    concurrency_mode: ClassVar[Literal["parallel_safe", "exclusive"]] = "exclusive"


class RoleSnapshot(BaseModel):
    """Immutable, validated startup snapshot of one Python Role class."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str
    description: str
    system_prompt: str
    output_model: type[BaseModel]
    model_tier: Literal["fast", "strong"]
    capabilities: tuple[str, ...]
    concurrency_mode: Literal["parallel_safe", "exclusive"]
    source: str
    digest: str

    def descriptor(self) -> dict[str, Any]:
        """Return bounded metadata safe to expose to the Master."""

        return {
            "id": self.id,
            "description": self.description,
            "source": self.source,
            "digest": self.digest,
            "model_tier": self.model_tier,
            "capabilities": list(self.capabilities),
            "concurrency_mode": self.concurrency_mode,
            "output_schema": self.output_model.model_json_schema(),
        }


def snapshot_role(role_type: type[Role[Any]], *, source: str) -> RoleSnapshot:
    """Validate one Role subclass and freeze its canonical definition."""

    if not isinstance(role_type, type) or not issubclass(role_type, Role):
        raise RoleDefinitionError("role entries must be Role subclasses")
    role_id = _required_text(role_type, "id", max_length=64)
    if re.fullmatch(ROLE_ID_PATTERN, role_id) is None:
        raise RoleDefinitionError(f"invalid role id {role_id!r}")
    description = _required_text(
        role_type,
        "description",
        max_length=MAX_ROLE_DESCRIPTION_CHARS,
    )
    system_prompt = _required_text(
        role_type,
        "system_prompt",
        max_length=MAX_ROLE_PROMPT_CHARS,
    )
    output_model = getattr(role_type, "output_model", None)
    if not isinstance(output_model, type) or not issubclass(output_model, BaseModel):
        raise RoleDefinitionError(f"role {role_id!r} output_model must be a BaseModel type")
    model_tier = getattr(role_type, "model_tier", None)
    if model_tier not in {"fast", "strong"}:
        raise RoleDefinitionError(f"role {role_id!r} has invalid model_tier")
    concurrency_mode = getattr(role_type, "concurrency_mode", None)
    if concurrency_mode not in {"parallel_safe", "exclusive"}:
        raise RoleDefinitionError(f"role {role_id!r} has invalid concurrency_mode")
    raw_capabilities = getattr(role_type, "capabilities", None)
    if not isinstance(raw_capabilities, tuple) or not all(
        isinstance(name, str) and name for name in raw_capabilities
    ):
        raise RoleDefinitionError(f"role {role_id!r} capabilities must be a tuple of names")
    if len(set(raw_capabilities)) != len(raw_capabilities):
        raise RoleDefinitionError(f"role {role_id!r} has duplicate capabilities")
    unknown = sorted(set(raw_capabilities) - ALLOWED_ROLE_CAPABILITIES)
    if unknown:
        raise RoleDefinitionError(
            f"role {role_id!r} has unknown capabilities: {', '.join(unknown)}"
        )
    canonical_source = source.strip()
    if not canonical_source:
        raise RoleDefinitionError("role source must be nonempty")
    digest = role_digest(
        role_id=role_id,
        description=description,
        system_prompt=system_prompt,
        output_model=output_model,
        model_tier=model_tier,
        capabilities=raw_capabilities,
        concurrency_mode=concurrency_mode,
    )
    return RoleSnapshot(
        id=role_id,
        description=description,
        system_prompt=system_prompt,
        output_model=output_model,
        model_tier=model_tier,
        capabilities=raw_capabilities,
        concurrency_mode=concurrency_mode,
        source=canonical_source,
        digest=digest,
    )


def role_digest(
    *,
    role_id: str,
    description: str,
    system_prompt: str,
    output_model: type[BaseModel],
    model_tier: str,
    capabilities: tuple[str, ...],
    concurrency_mode: str,
) -> str:
    """Return a stable digest of every runtime-relevant Role field."""

    payload = json.dumps(
        {
            "id": role_id,
            "description": description,
            "system_prompt": system_prompt,
            "output_schema": output_model.model_json_schema(),
            "model_tier": model_tier,
            "capabilities": capabilities,
            "concurrency_mode": concurrency_mode,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RoleRegistry:
    """Immutable Role catalog assembled at process startup."""

    __slots__ = ("_roles",)

    def __init__(self, roles: Mapping[str, RoleSnapshot]) -> None:
        self._roles = MappingProxyType(dict(roles))

    @classmethod
    def from_types(
        cls,
        builtin_types: Iterable[type[Role[Any]]],
        project_types: Iterable[type[Role[Any]]] = (),
        *,
        project_source: str = "<project-catalog>",
    ) -> RoleRegistry:
        discovered: dict[str, RoleSnapshot] = {}
        for role_type in builtin_types:
            snapshot = snapshot_role(
                role_type,
                source=f"builtin:{role_type.__module__}.{role_type.__qualname__}",
            )
            if snapshot.id in discovered:
                raise RoleDefinitionError(f"duplicate built-in role id {snapshot.id!r}")
            discovered[snapshot.id] = snapshot
        project_seen: set[str] = set()
        for role_type in project_types:
            snapshot = snapshot_role(
                role_type,
                source=f"{project_source}#{role_type.__qualname__}",
            )
            if snapshot.id in project_seen:
                raise RoleDefinitionError(f"duplicate project role id {snapshot.id!r}")
            project_seen.add(snapshot.id)
            discovered[snapshot.id] = snapshot
        return cls(discovered)

    @property
    def roles(self) -> Mapping[str, RoleSnapshot]:
        return self._roles

    def list(self) -> tuple[RoleSnapshot, ...]:
        return tuple(self._roles[role_id] for role_id in sorted(self._roles))

    def get(self, role_id: str) -> RoleSnapshot:
        try:
            return self._roles[role_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._roles))
            raise KeyError(f"unknown role {role_id!r}; available: {available}") from exc


def _required_text(role_type: type[Role[Any]], field: str, *, max_length: int) -> str:
    value = getattr(role_type, field, None)
    if not isinstance(value, str) or not value.strip():
        raise RoleDefinitionError(
            f"role {role_type.__qualname__!r} requires nonempty {field}"
        )
    normalized = value.strip()
    if len(normalized) > max_length:
        raise RoleDefinitionError(
            f"role {getattr(role_type, 'id', role_type.__qualname__)!r} "
            f"{field} exceeds {max_length} characters"
        )
    return normalized


__all__ = [
    "ALLOWED_ROLE_CAPABILITIES",
    "MAX_ROLE_DESCRIPTION_CHARS",
    "MAX_ROLE_PROMPT_CHARS",
    "ROLE_ID_PATTERN",
    "ReviewFinding",
    "ReviewReport",
    "Role",
    "RoleDefinitionError",
    "RoleRegistry",
    "RoleSnapshot",
    "WorkerEvidence",
    "WorkerReport",
    "role_digest",
    "snapshot_role",
]
