"""Pure Worker lifecycle contracts shared by scheduling and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from aeloon_core.workers import WorkerSnapshot

ReportText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]
ReportItem = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
]


class WorkerRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    WAITING_FOR_CONTEXT = "waiting_for_context"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Return whether no continuation semantics are attached to this Run."""

        return self in {
            self.COMPLETED,
            self.PARTIAL,
            self.FAILED,
            self.CANCELLED,
        }

    @property
    def settled(self) -> bool:
        """Return whether awaiters should stop waiting for this Run."""

        return self is self.WAITING_FOR_CONTEXT or self.terminal


class WorkerSessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    ARCHIVED = "archived"


class WorkerOperation(StrEnum):
    SPAWN = "spawn"
    REUSE = "reuse"
    RESUME = "resume"


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different scheduling request."""


class WorkerRunFencedError(RuntimeError):
    """A Run lost durable execution authority before a tool boundary."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


EvidenceKind = Literal[
    "legacy",
    "file",
    "test",
    "typecheck",
    "lint",
    "runtime",
    "source",
]
TerminalEvidenceKind = Literal[
    "file",
    "test",
    "typecheck",
    "lint",
    "runtime",
    "source",
]
EvidenceStatus = Literal["passed", "failed", "observed", "not_applicable"]


class EvidenceItem(_FrozenModel):
    """One bounded, machine-checkable claim returned by a Worker."""

    kind: EvidenceKind
    locator: str = Field(min_length=1, max_length=1_000)
    claim: str = Field(min_length=1, max_length=1_000)
    status: EvidenceStatus
    method: str | None = Field(default=None, min_length=1, max_length=2_000)
    finding_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )

    @classmethod
    def from_legacy(cls, value: str) -> EvidenceItem:
        text = value.strip()
        return cls(
            kind="legacy",
            locator=text,
            claim=text,
            status="observed",
        )


class TerminalEvidenceItem(EvidenceItem):
    """Evidence shape accepted from new Worker terminal output."""

    kind: TerminalEvidenceKind


class PermissionSnapshot(_FrozenModel):
    tool_names: tuple[str, ...] = ()
    skills_enabled: bool = True


RelatedContextSection = Literal[
    "objective",
    "summary",
    "artifacts",
    "evidence",
    "unresolved",
]


class RelatedWorkerContext(_FrozenModel):
    """Bounded reference material from an explicitly associated WorkerRun."""

    source_kind: Literal["worker_run", "flow_node"]
    source_id: str = Field(min_length=1, max_length=128)
    relation: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    worker_id: str = Field(min_length=1, max_length=128)
    worker_type_id: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=64)
    included_sections: tuple[RelatedContextSection, ...] = ()
    objective: str | None = Field(default=None, max_length=2_000)
    summary: str | None = Field(default=None, max_length=4_000)
    artifacts: tuple[ReportItem, ...] = Field(default=(), max_length=8)
    evidence: tuple[EvidenceItem, ...] = Field(default=(), max_length=8)
    unresolved: tuple[ReportItem, ...] = Field(default=(), max_length=4)

    @field_validator("evidence", mode="before")
    @classmethod
    def _legacy_evidence_is_readable(cls, value: Any) -> Any:
        return _normalize_evidence(value)


class BudgetGrant(_FrozenModel):
    # Cumulative Run limits are optional. They are deliberately separate from
    # the model's per-request context window.
    max_requests: int = Field(default=25, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_seconds: int = Field(default=3_600, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)


class BudgetIncrease(_FrozenModel):
    """Master-authored target limits for an exact checkpoint continuation."""

    max_requests: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_seconds: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _contains_target(self) -> BudgetIncrease:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("a budget increase requires at least one target limit")
        return self

    def apply(self, current: BudgetGrant) -> BudgetGrant:
        """Return a strictly larger grant without silently reducing another limit."""

        updates: dict[str, int] = {}
        for field_name, target in self.model_dump().items():
            if target is None:
                continue
            existing = getattr(current, field_name)
            if existing is None:
                if field_name == "max_output_tokens":
                    updates[field_name] = target
                    continue
                raise ValueError(f"{field_name} is already unlimited and cannot be increased")
            if target <= existing:
                raise ValueError(
                    f"{field_name} must increase from {existing}; requested {target}"
                )
            updates[field_name] = target
        return current.model_copy(update=updates)


class ContextEnvelope(_FrozenModel):
    objective: str = Field(min_length=1, max_length=64_000)
    permissions: PermissionSnapshot
    budget: BudgetGrant
    budget_increase_count: int = Field(default=0, ge=0)
    related_contexts: tuple[RelatedWorkerContext, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def _related_context_is_bounded(self) -> ContextEnvelope:
        encoded = json.dumps(
            [item.model_dump(mode="json") for item in self.related_contexts],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded) > 64_000:
            raise ValueError("related Worker context exceeds 64000 characters")
        return self


class WorkerReport(_FrozenModel):
    """Bounded model-authored data returned to the Master."""

    summary: ReportText
    artifacts: tuple[ReportItem, ...] = Field(default=(), max_length=32)
    evidence: tuple[EvidenceItem, ...] = Field(default=(), max_length=32)
    unresolved: tuple[ReportItem, ...] = Field(default=(), max_length=32)

    @field_validator("evidence", mode="before")
    @classmethod
    def _legacy_evidence_is_readable(cls, value: Any) -> Any:
        return _normalize_evidence(value)


class WaitingRequest(_FrozenModel):
    summary: ReportText
    question: str = Field(min_length=1, max_length=1_000)


class ResultEnvelope(_FrozenModel):
    worker_id: str
    run_id: str
    status: WorkerRunStatus
    report: WorkerReport | None = None
    tool_outcome: Literal["known", "unknown", "none"] = "none"
    usage: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = Field(default=0, ge=0)
    role: str | None = Field(default=None, max_length=128)
    resolved_model: str | None = Field(default=None, max_length=256)
    request_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    budget_request_limit: int | None = Field(default=None, ge=1)
    budget_request_utilization: float | None = Field(default=None, ge=0)
    partial_reason: str | None = Field(default=None, max_length=2_000)


@dataclass(frozen=True)
class WorkerSessionRecord:
    worker_id: str
    base_session_id: str
    snapshot: WorkerSnapshot
    status: WorkerSessionStatus
    created_at: str


@dataclass(frozen=True)
class WorkerRunRecord:
    run_id: str
    worker_id: str
    base_turn_id: str | None
    status: WorkerRunStatus
    context: ContextEnvelope
    idempotency_key: str
    created_at: str
    result: ResultEnvelope | None = None
    run_sequence: int = 1
    source_run_id: str | None = None
    waiting_request: WaitingRequest | None = None
    cancel_requested_at: str | None = None
    activated_at: str | None = None
    active_tool_count: int = 0
    execution_owner_token: str | None = None


def _normalize_evidence(value: Any) -> Any:
    if not isinstance(value, list | tuple):
        return value
    return tuple(
        EvidenceItem.from_legacy(item) if isinstance(item, str) else item
        for item in value
    )


__all__ = [
    "BudgetGrant",
    "BudgetIncrease",
    "ContextEnvelope",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceStatus",
    "IdempotencyConflictError",
    "PermissionSnapshot",
    "ReportItem",
    "ReportText",
    "RelatedContextSection",
    "RelatedWorkerContext",
    "ResultEnvelope",
    "TerminalEvidenceItem",
    "TerminalEvidenceKind",
    "WaitingRequest",
    "WorkerOperation",
    "WorkerReport",
    "WorkerRunFencedError",
    "WorkerRunRecord",
    "WorkerRunStatus",
    "WorkerSessionRecord",
    "WorkerSessionStatus",
]
