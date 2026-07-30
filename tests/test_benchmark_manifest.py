from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from benchmarks.adapters.base import BenchmarkAdapter
from benchmarks.adapters.registry import ADAPTER_PATHS, get_adapter
from benchmarks.harness.registry import HARNESS_TYPES


def test_benchmark_and_harness_registries_expose_supported_integrations() -> None:
    assert set(ADAPTER_PATHS) == {"refactorbench", "livecodebench", "repoqa"}
    assert set(HARNESS_TYPES) == {"aeloon", "pi", "codex", "claude"}
    assert all(
        isinstance(get_adapter(name, project_root=Path.cwd()), BenchmarkAdapter)
        for name in ADAPTER_PATHS
    )


def test_benchmark_manifest_records_source_and_harness_versions(
    tmp_path: Path,
) -> None:
    adapter = get_adapter("refactorbench", project_root=tmp_path)
    harnesses = [
        SimpleNamespace(
            name="aeloon",
            version="aeloon-core@abc",
            model="deepseek-v4-flash",
        ),
        SimpleNamespace(
            name="codex",
            version="codex-cli 1",
            model="deepseek-v4-flash",
        ),
    ]

    manifest = adapter.manifest(harnesses, status="running")

    assert manifest["schema_version"] == 1
    assert manifest["benchmark"] == "refactorbench"
    assert manifest["status"] == "running"
    assert manifest["source"] == {
        "repository": "https://github.com/microsoft/RefactorBench.git",
        "checkout": str(adapter.run.source_dir),
        "revision": None,
    }
    assert manifest["harnesses"] == [
        {
            "id": "aeloon",
            "version": "aeloon-core@abc",
            "model": "deepseek-v4-flash",
        },
        {
            "id": "codex",
            "version": "codex-cli 1",
            "model": "deepseek-v4-flash",
        },
    ]
