from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = connection.recv(size - len(value))
        if not chunk:
            raise RuntimeError("Runtime closed the lifecycle smoke socket")
        value.extend(chunk)
    return bytes(value)


def _request(
    connection: socket.socket,
    identifier: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.dumps(
        {"id": str(identifier), "method": method, "params": params or {}},
        separators=(",", ":"),
    ).encode()
    connection.sendall(struct.pack("!I", len(payload)) + payload)
    while True:
        size = struct.unpack("!I", _recv_exact(connection, 4))[0]
        value = json.loads(_recv_exact(connection, size))
        if value.get("method") == "event":
            continue
        return value


def _connect(socket_path: Path) -> socket.socket:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(socket_path))
            return connection
        except OSError:
            connection.close()
            time.sleep(0.05)
    raise AssertionError(f"Runtime socket did not become ready: {socket_path}")


def _handshake(connection: socket.socket, identifier: int = 1) -> None:
    response = _request(
        connection,
        identifier,
        "system.handshake",
        {
            "protocol": {"min": "3.0.0-draft.3", "max": "3.0.0"},
            "client": {"name": "lifecycle-smoke", "version": "1", "platform": sys.platform},
        },
    )
    assert response.get("result", {}).get("protocol") == "3.0.0-draft.3"


def test_standalone_runtime_survives_client_disconnect_and_restores_projection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "runtime-data"
    # macOS limits AF_UNIX paths to a little over 100 bytes; pytest's deeply
    # nested tmp_path can exceed that before Runtime even gets to bind.
    socket_path = Path("/tmp") / f"aeloon-lifecycle-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    environment = {**os.environ, "AELOON_RUNTIME_MODE": "1", "PYTHONPATH": str(Path.cwd())}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aeloon_core",
            "serve",
            "--unix",
            str(socket_path),
            "--data-dir",
            str(data_dir),
            "--workspace-root",
            str(workspace),
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    first: socket.socket | None = None
    second: socket.socket | None = None
    try:
        first = _connect(socket_path)
        _handshake(first)
        project = _request(first, 2, "project.add", {"path": str(workspace)})["result"]["project"]
        thread = _request(
            first,
            3,
            "thread.create",
            {"project_id": project["id"], "kind": "standard", "title": "persistent"},
        )["result"]["thread"]
        first_snapshot = _request(first, 4, "system.snapshot")["result"]
        assert any(item["id"] == thread["id"] for item in first_snapshot["threads"])

        metadata_path = data_dir / "runtime.pid.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["pid"] == process.pid
        assert socket_path.stat().st_mode & 0o777 == 0o600
        first.close()
        first = None
        time.sleep(0.1)
        assert process.poll() is None, "closing the client must not stop Runtime"

        second = _connect(socket_path)
        _handshake(second, identifier=5)
        second_snapshot = _request(second, 6, "system.snapshot")["result"]
        assert any(item["id"] == thread["id"] for item in second_snapshot["threads"])
        assert json.loads(metadata_path.read_text(encoding="utf-8"))["pid"] == process.pid
        assert _request(second, 7, "system.shutdown")["result"]["accepted"] is True
        second.close()
        second = None
        assert process.wait(timeout=5) == 0
        assert not socket_path.exists()
        assert not metadata_path.exists()
    finally:
        first and first.close()
        second and second.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
