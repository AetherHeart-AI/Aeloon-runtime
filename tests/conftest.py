"""Suite-wide model safety boundaries."""

from __future__ import annotations

import pydantic_ai.models
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
)


@pytest.fixture(autouse=True)
def forbid_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every model-driving test must use TestModel or FunctionModel explicitly."""

    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture(autouse=True)
def clear_config_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate load_config() from developer shell credentials and model defaults."""

    for name in _CONFIG_ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
