from __future__ import annotations

import ast
from pathlib import Path

from aeloon_core.customization import Role, WorkflowTemplate
from aeloon_core.harness import HarnessAgentRuntime, ModelRouter
from aeloon_core.model_router import ModelRouter as CompatibilityModelRouter
from aeloon_core.pydantic_runtime import (
    HarnessAgentRuntime as CompatibilityHarnessAgentRuntime,
)
from aeloon_core.roles import Role as CompatibilityRole
from aeloon_core.workflow_templates import (
    WorkflowTemplate as CompatibilityWorkflowTemplate,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "aeloon_core"
LEGACY_IMPLEMENTATION_MODULES = {
    "aeloon_core.catalog",
    "aeloon_core.harness_runtime",
    "aeloon_core.model_router",
    "aeloon_core.pydantic_model",
    "aeloon_core.pydantic_runtime",
    "aeloon_core.roles",
    "aeloon_core.tools.workflows",
    "aeloon_core.workflow_runtime",
    "aeloon_core.workflow_templates",
}


def test_customization_layer_does_not_depend_on_harness_runtime() -> None:
    imports = _imports_under(PACKAGE_ROOT / "customization")

    assert not {
        module for module in imports if module.startswith("aeloon_core.harness")
    }


def test_harness_layer_uses_canonical_modules_instead_of_compatibility_facades() -> None:
    imports = _imports_under(PACKAGE_ROOT / "harness")

    assert not (imports & LEGACY_IMPLEMENTATION_MODULES)


def test_legacy_public_imports_remain_identity_compatible() -> None:
    assert CompatibilityRole is Role
    assert CompatibilityWorkflowTemplate is WorkflowTemplate
    assert CompatibilityModelRouter is ModelRouter
    assert CompatibilityHarnessAgentRuntime is HarnessAgentRuntime


def _imports_under(root: Path) -> set[str]:
    imported: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
    return imported
