"""Compatibility import for the legacy RefactorBench runner."""

from __future__ import annotations

import sys

from benchmarks.refactorbench import runner as _runner

if __name__ == "__main__":
    _runner.main()
else:
    sys.modules[__name__] = _runner
