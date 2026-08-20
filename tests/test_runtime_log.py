from __future__ import annotations

import json
from pathlib import Path

from aeloon_runtime.runtime_log import RuntimeLog


def test_runtime_log_rotates_private_lifecycle_file(tmp_path: Path) -> None:
    path = tmp_path / "runtime.log"
    path.write_text("x" * 20, encoding="utf-8")
    path.chmod(0o600)
    log = RuntimeLog(tmp_path, max_bytes=10)
    log.write("started", pid=123, socket="/tmp/runtime.sock")
    log.close()
    assert (tmp_path / "runtime.log.1").read_text(encoding="utf-8") == "x" * 20
    assert (tmp_path / "runtime.log").stat().st_mode & 0o777 == 0o600
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "started"
    assert record["pid"] == 123


def test_runtime_log_is_best_effort_after_close(tmp_path: Path) -> None:
    log = RuntimeLog(tmp_path)
    log.close()
    log.write("ignored")
    assert (tmp_path / "runtime.log").read_text(encoding="utf-8") == ""
