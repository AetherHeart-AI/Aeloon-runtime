"""Discover project and preset Agent/Workflow definitions at startup."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from aeloon_core.harness.agent.base import (
    Role,
    RoleDefinitionError,
    RoleRegistry,
)
from aeloon_core.harness.agent.presets import BUILTIN_ROLES
from aeloon_core.harness.workflow.base import (
    WorkflowDefinitionError,
    WorkflowRegistry,
    WorkflowTemplate,
)
from aeloon_core.harness.workflow.presets import BUILTIN_WORKFLOWS


class CatalogDefinitionError(ValueError):
    """Raised when a project Python catalog cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class Catalog:
    """Process-scoped immutable Role and Workflow Template registries."""

    roles: RoleRegistry
    workflows: WorkflowRegistry
    project_root: Path
    project_source: str | None = None

    @classmethod
    def discover(cls, project_root: Path | str) -> Catalog:
        root = Path(project_root).expanduser().resolve()
        catalog_path = root / ".aeloon-core" / "catalog.py"
        project_roles: tuple[type[Role[Any]], ...] = ()
        project_workflows: tuple[type[WorkflowTemplate[Any, Any]], ...] = ()
        project_source: str | None = None
        if catalog_path.is_file():
            module = _load_project_catalog(catalog_path)
            project_roles = _catalog_entries(module, "ROLES", Role)
            project_workflows = _catalog_entries(
                module,
                "WORKFLOWS",
                WorkflowTemplate,
            )
            project_source = str(catalog_path)
        try:
            roles = RoleRegistry.from_types(
                BUILTIN_ROLES,
                project_roles,
                project_source=project_source or "<project-catalog>",
            )
            workflows = WorkflowRegistry.from_types(
                BUILTIN_WORKFLOWS,
                project_workflows,
                project_source=project_source or "<project-catalog>",
            )
        except (RoleDefinitionError, WorkflowDefinitionError) as exc:
            raise CatalogDefinitionError(str(exc)) from exc
        return cls(
            roles=roles,
            workflows=workflows,
            project_root=root,
            project_source=project_source,
        )


def _load_project_catalog(path: Path) -> ModuleType:
    module_name = f"_aeloon_project_catalog_{abs(hash(path))}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        source = path.read_bytes()
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception as exc:
        raise CatalogDefinitionError(
            f"project catalog {path} raised {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    return module


def _catalog_entries(
    module: ModuleType,
    name: str,
    base: type[Any],
) -> tuple[Any, ...]:
    raw = getattr(module, name, ())
    if not isinstance(raw, tuple):
        raise CatalogDefinitionError(f"project catalog {name} must be a tuple")
    for entry in raw:
        if not isinstance(entry, type) or not issubclass(entry, base):
            raise CatalogDefinitionError(
                f"project catalog {name} entries must be {base.__name__} subclasses"
            )
    return raw


__all__ = ["Catalog", "CatalogDefinitionError"]
