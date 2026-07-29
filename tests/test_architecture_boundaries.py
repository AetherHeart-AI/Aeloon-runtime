from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "aeloon_core"
EXPECTED_ROOT_MODULES = {
    "__init__.py",
    "__main__.py",
    "config.py",
    "orchestrator.py",
}
EXPECTED_HARNESS_MODULES = {"__init__.py", "catalog.py"}
EXPECTED_HARNESS_FEATURES = {
    "agent": {"__init__.py", "base.py", "factory.py", "presets.py", "prompt.py"},
    "execution": {
        "__init__.py",
        "engine.py",
        "events.py",
        "stuck.py",
        "transitions.py",
    },
    "model": {"__init__.py", "router.py"},
    "provider": {"__init__.py", "base.py", "deepseek.py"},
    "tool": {"__init__.py", "base.py", "filesystem.py", "registry.py", "search.py"},
    "workflow": {"__init__.py", "base.py", "presets.py", "runner.py", "tools.py"},
}
EXPECTED_SUPPORT_FEATURES = {
    "conversation": {"__init__.py", "history.py", "session.py"},
    "web": {"__init__.py", "bridge.py", "events.py", "launcher.py", "output.py"},
}


def test_harness_is_grouped_by_feature() -> None:
    harness_root = PACKAGE_ROOT / "harness"

    assert {path.name for path in harness_root.glob("*.py")} == EXPECTED_HARNESS_MODULES
    assert {
        path.name
        for path in harness_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } == set(EXPECTED_HARNESS_FEATURES)
    for feature, expected_files in EXPECTED_HARNESS_FEATURES.items():
        assert {
            path.name for path in (harness_root / feature).glob("*.py")
        } == expected_files


def test_support_code_is_grouped_by_feature() -> None:
    for feature, expected_files in EXPECTED_SUPPORT_FEATURES.items():
        assert {
            path.name for path in (PACKAGE_ROOT / feature).glob("*.py")
        } == expected_files


def test_feature_packages_do_not_import_removed_horizontal_layers() -> None:
    imports = _imports_under(PACKAGE_ROOT / "harness")

    assert not {
        module
        for module in imports
        if module.startswith(("aeloon_core.customization", "aeloon_core.tools"))
    }


def test_package_root_contains_only_entrypoints_and_composition_modules() -> None:
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} == EXPECTED_ROOT_MODULES


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
