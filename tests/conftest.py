"""Suite-wide model safety boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

# Keep file-based config tests deterministic regardless of the host shell.
_CONFIG_ENV_OVERRIDES = (
    "AELOON_CORE_PROVIDER",
    "AELOON_CORE_CONFIG",
    "AELOON_CORE_WORKSPACE",
    "AELOON_CORE_DATA_DIR",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ARK_API_KEY",
    "ARK_BASE_URL",
    "ARK_MODEL",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "EXA_API_KEY",
)

@pytest.fixture(autouse=True)
def clear_config_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Isolate load_config() from developer shell credentials and model defaults."""

    for name in _CONFIG_ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AELOON_CORE_CONFIG", str(tmp_path / "isolated-config.json"))
