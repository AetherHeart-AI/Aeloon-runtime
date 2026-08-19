"""Package build identity shared without importing the application service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

__version__ = "0.0.16"
RUNTIME_VERSION = "0.1.0"


def runtime_version() -> str:
    """Return the standalone Runtime release identity when serving v3.

    The legacy command line and v2 RPC surface retain the Core 0.0.16
    identity during the migration window.  A bundled/standalone Runtime must
    report its independently versioned 0.1.0 line in v3 diagnostics.
    """

    if os.environ.get("AELOON_RUNTIME_MODE") == "1" or Path(sys.argv[0]).name == "aeloon-runtime":
        return RUNTIME_VERSION
    return __version__


@lru_cache(maxsize=1)
def core_commit() -> str:
    """Return the embedded release commit, with source-tree fallbacks for development."""

    build_info = Path(__file__).with_name("_build_info.json")
    try:
        value = json.loads(build_info.read_text(encoding="utf-8"))
        commit = str(value.get("commit") or "")
        if _valid_commit(commit):
            return commit
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    environment_commit = os.environ.get("AELOON_CORE_COMMIT", "")
    if _valid_commit(environment_commit):
        return environment_commit
    repository = Path(__file__).parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        commit = result.stdout.strip()
        if result.returncode == 0 and _valid_commit(commit):
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _valid_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


__all__ = ["__version__", "RUNTIME_VERSION", "runtime_version", "core_commit"]
