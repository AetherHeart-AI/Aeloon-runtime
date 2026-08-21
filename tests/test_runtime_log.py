from __future__ import annotations

import json
from pathlib import Path

from aeloon_runtime.runtime_log import RuntimeLog, read_runtime_logs


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


def test_runtime_log_reader_skips_secrets_and_corrupt_lines(tmp_path: Path) -> None:
    previous = tmp_path / "runtime.log.1"
    previous.write_text(
        json.dumps({"at": "2026-01-01T00:00:00.000Z", "event": "started", "pid": 1}) + "\n",
        encoding="utf-8",
    )
    log = RuntimeLog(tmp_path)
    log.write(
        "connected",
        transport="wss",
        device_id="dev-1",
        source="127.0.0.1",
        token="must-not-appear",
        pairing_url="aeloon://pair?code=secret",
    )
    log.close()
    current = (tmp_path / "runtime.log").read_bytes()
    (tmp_path / "runtime.log").write_bytes(current + b"{truncated")
    result = read_runtime_logs(tmp_path, limit=10)
    assert result["truncated"] is False
    assert [item["event"] for item in result["entries"]] == ["connected", "started"]
    assert result["entries"][0]["fields"]["transport"] == "wss"
    assert "token" not in result["entries"][0]["fields"]
    assert "pairing_url" not in result["entries"][0]["fields"]
    dumped = json.dumps(result)
    assert "must-not-appear" not in dumped
    assert "secret" not in dumped
