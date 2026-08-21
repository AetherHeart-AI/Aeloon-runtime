"""Shared operation lifecycle states and transition predicates."""

from __future__ import annotations

from typing import Literal

OperationStatus = Literal[
    "queued",
    "active",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "interrupted"})
IN_FLIGHT_STATUSES: frozenset[str] = frozenset({"queued", "active", "cancelling"})


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def is_in_flight(status: str) -> bool:
    return status in IN_FLIGHT_STATUSES


def is_cancellable(status: str) -> bool:
    return status in {"queued", "active", "cancelling"}


__all__ = [
    "IN_FLIGHT_STATUSES",
    "OperationStatus",
    "TERMINAL_STATUSES",
    "is_cancellable",
    "is_in_flight",
    "is_terminal",
]
