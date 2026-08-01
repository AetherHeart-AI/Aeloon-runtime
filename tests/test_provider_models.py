"""Tests for pi-ai provider/model construction."""

from __future__ import annotations

import json
from pathlib import Path

from aeloon_core.config import DeepSeekProviderConfig
from aeloon_core.harness.provider import PiModel, build_deepseek_model


def test_builds_deepseek_model_for_pi_ai_with_transport_settings() -> None:
    bundle = build_deepseek_model(
        provider=DeepSeekProviderConfig(
            api_key="sk-test",
            extra_headers={"x-routing-key": "tenant-a"},
            proxy="http://127.0.0.1:7890",
        ),
        model_name="deepseek-v4-flash",
        temperature=0.2,
        reasoning_effort="high",
        timeout=17,
    )

    assert isinstance(bundle.model, PiModel)
    assert bundle.model.provider == "deepseek"
    assert bundle.model.model_id == "deepseek-v4-flash"
    assert bundle.model.proxy == "http://127.0.0.1:7890"
    assert bundle.model.to_runtime()["api_key"] == "sk-test"
    assert bundle.settings == {
        "temperature": 0.2,
        "timeout_ms": 17_000,
        "reasoning": "high",
        "headers": {"x-routing-key": "tenant-a"},
    }


def test_deepseek_model_without_reasoning_effort() -> None:
    bundle = build_deepseek_model(
        provider=DeepSeekProviderConfig(api_key="sk-test"),
        model_name="deepseek-v4-flash",
        temperature=0.7,
        reasoning_effort=None,
        timeout=10,
    )

    assert "reasoning" not in bundle.settings
    assert bundle.settings["temperature"] == 0.7


def test_pi_packages_are_exactly_pinned_to_the_same_release() -> None:
    package_path = (
        Path(__file__).resolve().parents[1]
        / "aeloon_core"
        / "pi_runtime"
        / "package.json"
    )
    dependencies = json.loads(package_path.read_text(encoding="utf-8"))["dependencies"]

    assert dependencies == {
        "@earendil-works/pi-agent-core": "0.83.0",
        "@earendil-works/pi-ai": "0.83.0",
    }
