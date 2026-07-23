"""Suite-wide model safety boundaries."""

from __future__ import annotations

import pydantic_ai.models
import pytest


@pytest.fixture(autouse=True)
def forbid_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every model-driving test must use TestModel or FunctionModel explicitly."""

    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
