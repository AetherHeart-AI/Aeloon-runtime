from __future__ import annotations

from pathlib import Path


def test_runtime_dockerfile_is_unix_transport_only() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile.runtime").read_text(encoding="utf-8")
    assert "EXPOSE" not in dockerfile
    assert "--unix" in dockerfile
    assert "--workspace-root" in dockerfile
    assert "aeloon-runtime" in dockerfile


def test_docker_smoke_exercises_mounted_runtime_transport() -> None:
    smoke = (Path(__file__).parents[1] / "tools" / "docker_smoke.py").read_text(
        encoding="utf-8"
    )
    assert '"volume", "create"' in smoke
    assert '"/run/aeloon/runtime.sock"' in smoke
    assert '"system.handshake"' in smoke
    assert '"system.shutdown"' in smoke
