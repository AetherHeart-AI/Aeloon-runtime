"""Trusted Expert runner registration and project extension loading."""

from __future__ import annotations

import ast
import inspect
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

from aeloon_core.harness.expert.base import ExpertRunner
from aeloon_core.harness.expert.langgraph import LangGraphExpertRunner
from aeloon_core.harness.expert.runners.coding import CodingExpertRunner
from aeloon_core.harness.expert.runners.prompt import PromptExpertRunner
from aeloon_core.harness.expert.runners.research import ResearchExpertRunner
from aeloon_core.harness.skill.base import RUNNER_ID_PATTERN


class ExpertCatalogError(ValueError):
    """Raised when trusted project runner registration is invalid."""


class ExpertRunnerRegistry:
    """Immutable mapping from manifest runner ids to trusted implementations."""

    __slots__ = ("_runners", "project_source")

    def __init__(
        self,
        runners: Mapping[str, ExpertRunner],
        *,
        project_source: str | None = None,
    ) -> None:
        self._runners = MappingProxyType(dict(runners))
        self.project_source = project_source

    @classmethod
    def discover(cls, workspace: Path) -> ExpertRunnerRegistry:
        runners: dict[str, ExpertRunner] = {
            "builtin.research": ResearchExpertRunner(),
            "builtin.coding": CodingExpertRunner(),
            "builtin.prompt": PromptExpertRunner(),
        }
        catalog_path = workspace / ".aeloon-core" / "catalog.py"
        project_source: str | None = None
        if catalog_path.is_file():
            legacy_exports = _legacy_catalog_exports(catalog_path)
            if legacy_exports:
                raise ExpertCatalogError(
                    f"project catalog {catalog_path} uses removed entries "
                    f"{', '.join(legacy_exports)}; migrate to EXPERT_RUNNERS"
                )
            module = _load_catalog(catalog_path)
            legacy = [name for name in ("ROLES", "WORKFLOWS") if hasattr(module, name)]
            if legacy:
                raise ExpertCatalogError(
                    f"project catalog {catalog_path} uses removed entries "
                    f"{', '.join(legacy)}; migrate to EXPERT_RUNNERS"
                )
            raw = getattr(module, "EXPERT_RUNNERS", {})
            if not isinstance(raw, Mapping):
                raise ExpertCatalogError(
                    "project catalog EXPERT_RUNNERS must be a mapping"
                )
            for runner_id, entry in raw.items():
                if (
                    not isinstance(runner_id, str)
                    or re.fullmatch(RUNNER_ID_PATTERN, runner_id) is None
                ):
                    raise ExpertCatalogError(
                        "EXPERT_RUNNERS keys must be canonical runner ids"
                    )
                if runner_id in runners:
                    raise ExpertCatalogError(
                        f"project runner id collides with built-in {runner_id!r}"
                    )
                runners[runner_id] = _coerce_runner(entry, runner_id=runner_id)
            project_source = str(catalog_path)
        return cls(runners, project_source=project_source)

    def require(self, runner_id: str) -> ExpertRunner:
        runner = self._runners.get(runner_id)
        if runner is None:
            available = ", ".join(sorted(self._runners))
            raise ExpertCatalogError(
                f"unknown Expert runner {runner_id!r}; registered: {available}"
            )
        return runner

    def ids(self) -> tuple[str, ...]:
        return tuple(self._runners)


def _legacy_catalog_exports(path: Path) -> tuple[str, ...]:
    """Detect removed exports before imports referencing removed APIs execute."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return ()
    found: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"ROLES", "WORKFLOWS"}:
                found.add(target.id)
    return tuple(sorted(found))


def _coerce_runner(entry: Any, *, runner_id: str) -> ExpertRunner:
    if isinstance(entry, type):
        try:
            entry = entry()
        except Exception as exc:
            raise ExpertCatalogError(
                f"could not construct Expert runner {runner_id!r}: {exc}"
            ) from exc
    if callable(getattr(entry, "ainvoke", None)):
        return LangGraphExpertRunner(entry)
    if isinstance(entry, ExpertRunner):
        if not inspect.iscoroutinefunction(entry.run):
            raise ExpertCatalogError(
                f"EXPERT_RUNNERS[{runner_id!r}].run must be async"
            )
        return entry
    raise ExpertCatalogError(
        f"EXPERT_RUNNERS[{runner_id!r}] must implement async run() or be a "
        "compiled graph with ainvoke()"
    )


def _load_catalog(path: Path) -> ModuleType:
    module_name = f"_aeloon_expert_catalog_{abs(hash(path))}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except Exception as exc:
        raise ExpertCatalogError(
            f"project catalog {path} raised {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    return module


__all__ = ["ExpertCatalogError", "ExpertRunnerRegistry"]
