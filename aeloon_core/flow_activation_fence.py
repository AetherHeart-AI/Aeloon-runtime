"""Cross-process linearization for Flow activation, claims, and cancellation."""

from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def flow_activation_fence(data_dir: Path, flow_id: str) -> Iterator[None]:
    """Lock one Flow's short activation/cancellation critical section."""

    lock_dir = Path(data_dir) / ".flow-activation-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(flow_id.encode("utf-8")).hexdigest()
    with (lock_dir / f"{digest}.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def flow_id_from_turn_id(base_turn_id: str | None) -> str | None:
    """Return the owning Flow id encoded in a WorkerRun turn id."""

    if base_turn_id is None or not base_turn_id.startswith("flow:"):
        return None
    flow_id = base_turn_id.removeprefix("flow:")
    return flow_id or None


__all__ = ["flow_activation_fence", "flow_id_from_turn_id"]
