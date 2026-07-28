"""Benchmark registry for the unified CLI."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from benchmarks.adapters.base import BenchmarkAdapter

ADAPTER_PATHS = {
    "refactorbench": "benchmarks.adapters.refactorbench:RefactorBenchAdapter",
    "livecodebench": "benchmarks.adapters.livecodebench:LiveCodeBenchAdapter",
}
BENCHMARK_NAMES = tuple(ADAPTER_PATHS)


def get_adapter(name: str, *, project_root: Path) -> BenchmarkAdapter:
    try:
        module_name, type_name = ADAPTER_PATHS[name].split(":", maxsplit=1)
    except KeyError:
        available = ", ".join(BENCHMARK_NAMES)
        raise RuntimeError(
            f"Unknown benchmark {name!r}; available benchmarks: {available}"
        ) from None
    module = import_module(module_name)
    adapter_type = getattr(module, type_name)
    return adapter_type(project_root=project_root)
