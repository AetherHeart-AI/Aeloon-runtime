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


def test_dependency_manifest_has_no_pi_bun_mcp_or_langgraph() -> None:
    manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "@earendil-works" not in manifest
    assert "pi-agent-core" not in manifest
    assert '"mcp' not in manifest
    assert "langgraph" not in manifest
    assert "pillow" in manifest
