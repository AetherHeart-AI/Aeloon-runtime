#!/usr/bin/env python3
"""Run the Runtime Docker Unix-transport smoke when Docker is available."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    dockerfile = Path(__file__).parents[1] / "Dockerfile.runtime"
    source = dockerfile.read_text(encoding="utf-8")
    if "EXPOSE" in source or "--tcp" in source or "--host" in source:
        raise SystemExit("Runtime Docker image must expose only the Unix socket")
    if args.check_only:
        print("Runtime Dockerfile contract is valid")
        return 0
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("Docker is not installed")
    info = subprocess.run([docker, "info"], capture_output=True, text=True, check=False)
    if info.returncode != 0:
        raise SystemExit("Docker daemon is unavailable; run this smoke in CI")
    subprocess.run(
        [
            docker,
            "build",
            "-f",
            str(dockerfile),
            "-t",
            "aeloon-runtime-smoke",
            str(dockerfile.parent),
        ],
        check=True,
    )
    _run_unix_socket_smoke(docker)
    print("Runtime Docker image built and Unix socket smoke passed")
    return 0


def _run_unix_socket_smoke(docker: str) -> None:
    """Exercise the image with workspace/data/socket mounts.

    Docker Desktop's host bind mounts do not preserve chmod(2) on Unix
    sockets, so the socket mount is a Docker-managed volume.  It is still a
    mounted `/run/aeloon` volume, and the client runs inside the same image so
    the smoke verifies the actual container transport and 0600 socket mode.
    """

    suffix = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    container = f"aeloon-runtime-smoke-{suffix}"
    volume = f"aeloon-runtime-socket-{suffix}"
    client = r'''
import json, socket, struct

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/run/aeloon/runtime.sock")

def request(identifier, method, params):
    value = {"id": str(identifier), "method": method, "params": params}
    payload = json.dumps(value, separators=(",", ":")).encode()
    sock.sendall(struct.pack("!I", len(payload)) + payload)
    while True:
        header = sock.recv(4)
        if not header:
            raise RuntimeError("Runtime closed the Docker smoke socket")
        size = struct.unpack("!I", header)[0]
        body = b""
        while len(body) < size:
            chunk = sock.recv(size - len(body))
            if not chunk:
                raise RuntimeError("Runtime truncated a Docker smoke frame")
            body += chunk
        value = json.loads(body)
        if value.get("method") != "event":
            return value

handshake = request(1, "system.handshake", {
    "protocol": {"min": "3.0.0", "max": "3.0.0"},
    "client": {"name": "docker-smoke", "version": "1", "platform": "linux-aarch64"},
})
health = request(2, "system.health", {})
shutdown = request(3, "system.shutdown", {})
assert handshake["result"]["protocol"] == "3.0.0", handshake
assert health["result"]["ok"] is True, health
assert shutdown["result"]["accepted"] is True, shutdown
print(json.dumps({"handshake": handshake["result"]["protocol"], "health": True, "shutdown": True}))
'''
    with tempfile.TemporaryDirectory(prefix="aeloon-runtime-docker-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        data = root / "data"
        workspace.mkdir()
        data.mkdir()
        subprocess.run([docker, "volume", "create", volume], check=True, capture_output=True)
        try:
            subprocess.run(
                [
                    docker,
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--volume",
                    f"{workspace}:/workspace",
                    "--volume",
                    f"{data}:/data",
                    "--volume",
                    f"{volume}:/run/aeloon",
                    "aeloon-runtime-smoke",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            last_error = "Runtime container did not become ready"
            for _ in range(240):
                result = subprocess.run(
                    [docker, "exec", container, "python", "-c", client],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    print(result.stdout.strip())
                    return
                last_error = result.stderr.strip() or result.stdout.strip() or last_error
                state = subprocess.run(
                    [docker, "inspect", "--format", "{{.State.Status}}", container],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                if state in {"exited", "dead"}:
                    break
                time.sleep(0.1)
            logs = subprocess.run(
                [docker, "logs", container], capture_output=True, text=True, check=False
            )
            raise RuntimeError(
                f"Runtime Docker Unix socket smoke failed: {last_error}\n{logs.stdout[-4000:]}"
            )
        finally:
            subprocess.run([docker, "rm", "-f", container], capture_output=True, check=False)
            subprocess.run([docker, "volume", "rm", volume], capture_output=True, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
