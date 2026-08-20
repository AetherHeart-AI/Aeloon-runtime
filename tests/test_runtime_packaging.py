from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from aeloon_runtime.version import runtime_version


def test_runtime_distribution_metadata_has_release_command_and_version() -> None:
    metadata = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "aeloon-runtime"' in metadata
    assert 'version = "0.1.0"' in metadata
    assert 'aeloon-runtime = "aeloon_runtime.__main__:main"' in metadata


def test_runtime_entrypoint_reports_runtime_identity() -> None:
    environment = {**os.environ, "AELOON_RUNTIME_MODE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "aeloon_runtime", "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "aeloon-runtime 0.1.0"


def test_runtime_diagnostics_use_the_independent_runtime_version(monkeypatch) -> None:
    monkeypatch.setenv("AELOON_RUNTIME_MODE", "1")
    assert runtime_version() == "0.1.0"
    monkeypatch.delenv("AELOON_RUNTIME_MODE")
    assert runtime_version() == "0.0.16"


def test_protocol_package_is_generated_from_the_v3_manifest() -> None:
    root = Path(__file__).parents[1]
    package = (root / "packages" / "protocol" / "package.json").read_text(encoding="utf-8")
    tsconfig = (root / "packages" / "protocol" / "tsconfig.json").read_text(encoding="utf-8")
    source = (root / "packages" / "protocol" / "src" / "index.ts").read_text(encoding="utf-8")
    assert '"name": "@aeloon/protocol"' in package
    assert '"version": "3.0.0"' in package
    assert '"main": "dist/index.js"' in package
    assert '"emitDeclarationOnly"' not in tsconfig
    assert "export interface RuntimeRpcMethodMap" in source
    assert '"plugins.configure"' in source
    assert '"plugin.cloud.account_login"' in source
    assert "terminal_id?: string | null" in source


def test_runtime_release_bundle_installs_wheel_dependencies_into_runtime_site() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "runtime-release.yml"
    ).read_text(encoding="utf-8")
    assert 'UV_PYTHON="$runtime_python" UV_PYTHON_DOWNLOADS=never' in workflow
    assert '"$root/components/uv/uv" pip install' in workflow
    assert '--target "$root/runtime-site"' in workflow
    assert "uv export --frozen --no-dev --no-emit-project" in workflow
    assert "astral-sh/python-build-standalone/releases/download" in workflow
    assert "indygreg/python-build-standalone" not in workflow
    assert '--requirement "$RUNNER_TEMP/runtime-requirements.txt"' in workflow
    assert '"$(find "$RUNNER_TEMP/wheel" -name \'*.whl\' -print -quit)"' in workflow
    assert 'export AELOON_RUNTIME_COMMIT="__RUNTIME_COMMIT__"' in workflow
    assert 'sed -i.bak "s/__RUNTIME_COMMIT__/${GITHUB_SHA}/"' in workflow
    assert 'export AELOON_RUNTIME_COMMIT="${GITHUB_SHA}"' not in workflow
    # The checked-out desktop repository is core-ui; dispatching to the old
    # aeloon-ui name would silently leave the published Runtime unpinned.
    assert "repos/AetherHeart-AI/core-ui/dispatches" in workflow


def test_runtime_release_publishes_the_runtime_wheel_alongside_bundles() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "runtime-release.yml"
    ).read_text(encoding="utf-8")
    assert "name: Build aeloon-runtime wheel" in workflow
    assert "name: aeloon-runtime-wheel" in workflow
    assert "needs: [bundle, protocol-package, wheel]" in workflow
    assert 'gh release upload "$tag" release/*.whl --clobber' in workflow


def test_package_workflow_checks_protocol_compatibility_against_base_revision() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "python-package.yml"
    ).read_text(encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}" in workflow
    assert "git show \"$BASE_SHA:aeloon_runtime/rpc/aeloon-rpc-v3.manifest.json\"" in workflow
    assert "tools/check_rpc_compat.py" in workflow
