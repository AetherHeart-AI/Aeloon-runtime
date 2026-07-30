"""Harness registry for the unified CLI."""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness.aeloon import AeloonHarness
from benchmarks.harness.base import Harness
from benchmarks.harness.claude import ClaudeHarness
from benchmarks.harness.codex import CodexHarness
from benchmarks.harness.pi import PiHarness
from benchmarks.progress import info

HARNESS_TYPES: dict[str, type[Harness]] = {
    "aeloon": AeloonHarness,
    "pi": PiHarness,
    "codex": CodexHarness,
    "claude": ClaudeHarness,
}
HARNESS_NAMES = tuple(HARNESS_TYPES)


def get_harness(name: str, *, project_root: Path, model: str) -> Harness:
    try:
        harness_type = HARNESS_TYPES[name]
    except KeyError:
        available = ", ".join(HARNESS_NAMES)
        raise RuntimeError(f"Unknown harness {name!r}; available harnesses: {available}") from None
    return harness_type(project_root=project_root, model=model)


def get_harnesses(
    names: list[str],
    *,
    project_root: Path,
    model: str,
) -> list[Harness]:
    selected = list(HARNESS_NAMES) if "all" in names else list(dict.fromkeys(names))
    harnesses: list[Harness] = []
    for name in selected:
        info("Checking harness %s...", name)
        harness = get_harness(name, project_root=project_root, model=model)
        info("Harness %s ready; version=%s", name, harness.version or "unknown")
        harnesses.append(harness)
    return harnesses
