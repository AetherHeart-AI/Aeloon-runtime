"""Compatibility import for the legacy LiveCodeBench runner."""

from __future__ import annotations

import sys

from benchmarks.livecodebench import runner as _runner

if __name__ == "__main__":
    _runner.main()
else:
    sys.modules[__name__] = _runner
