"""Suite-wide environment isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

# Keep file-based config tests deterministic regardless of the host shell.
_CONFIG_ENV_OVERRIDES = (
    "AELOON_CORE_CONFIG",
    "DEEPSEEK_API_KEY",
)


@pytest.fixture(autouse=True)
def clear_config_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Isolate load_config() from developer credentials and persistent config."""

    for name in _CONFIG_ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AELOON_CORE_CONFIG", str(tmp_path / "isolated-config.json"))
