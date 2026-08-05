from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "aeloon_core"


def test_runtime_is_python_only_and_removed_subsystems_are_absent() -> None:
    forbidden_paths = (
        PACKAGE / "pi_runtime",
        PACKAGE / "web",
        PACKAGE / "orchestrator.py",
        PACKAGE / "harness" / "expert",
        PACKAGE / "harness" / "mcp",
        PACKAGE / "harness" / "model",
        PACKAGE / "harness" / "execution",
        PACKAGE / "harness",
        PACKAGE / "service.py",
        PACKAGE / "providers.py",
        PACKAGE / "rename_session.py",
    )
    assert all(not path.exists() for path in forbidden_paths)
    assert not list(PACKAGE.rglob("*.js"))
    assert not list(PACKAGE.rglob("*.ts"))
    assert not list(PACKAGE.rglob("package.json"))
    assert not list(PACKAGE.rglob("bun.lock"))


def test_package_imports_have_no_removed_runtime_edges() -> None:
    imports: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
    forbidden = ("mcp", "langgraph", "aeloon_core.web", "aeloon_core.orchestrator")
    assert not {name for name in imports if name.startswith(forbidden)}


def test_four_module_dependencies_point_inward() -> None:
    forbidden_by_module = {
        "core": ("aeloon_core.runtime", "aeloon_core.bridge", "aeloon_core.cloud"),
        "runtime": ("aeloon_core.bridge", "aeloon_core.cloud"),
        "cloud": ("aeloon_core.runtime", "aeloon_core.bridge", "aeloon_core.config"),
    }
    for module, forbidden in forbidden_by_module.items():
        imports: set[str] = set()
        for path in (PACKAGE / module).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
                elif isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
        assert not {
            name
            for name in imports
            if any(name.startswith(prefix) for prefix in forbidden)
        }, module


def test_runtime_has_no_wire_dispatch_or_bridge_errors() -> None:
    service = (PACKAGE / "runtime" / "service.py").read_text(encoding="utf-8")
    tree = ast.parse(service)
    assert "BridgeError" not in service
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "dispatch"
        for node in ast.walk(tree)
    )


def test_artifact_delivery_is_runtime_owned() -> None:
    assert (PACKAGE / "runtime" / "artifacts.py").is_file()
    core_source = "\n".join(
        path.read_text(encoding="utf-8") for path in (PACKAGE / "core").rglob("*.py")
    )
    assert "present_files" not in core_source
    assert "artifact_delivery" not in core_source


def test_shared_config_does_not_load_cloud_implementation() -> None:
    config = (PACKAGE / "config.py").read_text(encoding="utf-8")
    assert "aeloon_core.cloud" not in config


def test_dependency_manifest_has_no_removed_runtime_dependencies() -> None:
    manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "@earendil-works" not in manifest
    assert '"mcp' not in manifest
    assert "langgraph" not in manifest
    assert "pillow" in manifest
