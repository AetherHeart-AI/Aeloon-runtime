"""Provider-neutral Pi model bundles and request settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

PiModelSettings = dict[str, Any]


@runtime_checkable
class PiModelLike(Protocol):
    """Small Python-side model contract consumed by the Pi runtime bridge."""

    provider: str
    model_id: str

    def to_runtime(self) -> dict[str, Any]:
        """Return the JSON payload used to resolve a pi-ai model."""


@dataclass(frozen=True, slots=True)
class PiModel:
    """A provider/model selection resolved by pi-ai in the Bun process."""

    provider: str
    model_id: str
    api_key: str = field(repr=False)
    proxy: str | None = None

    @property
    def model_name(self) -> str:
        """Compatibility spelling used by existing routing and display code."""

        return self.model_id

    def to_runtime(self) -> dict[str, Any]:
        return {
            "kind": "pi-ai",
            "provider": self.provider,
            "model_id": self.model_id,
            "api_key": self.api_key,
            "proxy": self.proxy,
        }


@dataclass(frozen=True, slots=True)
class ScriptedPiModel:
    """Deterministic pi-core test model backed by serialized responses."""

    responses: tuple[dict[str, Any], ...]
    provider: str = "aeloon-test"
    model_id: str = "scripted"

    @property
    def model_name(self) -> str:
        return self.model_id

    def to_runtime(self) -> dict[str, Any]:
        return {
            "kind": "scripted",
            "provider": self.provider,
            "model_id": self.model_id,
            "responses": list(self.responses),
        }


@dataclass(slots=True)
class PiModelBundle:
    """A pi-ai model reference and settings owned by Aeloon."""

    model: PiModel
    settings: PiModelSettings

    async def close(self) -> None:
        """Pi transports are process-local and close with each bridge run."""


def _base_settings(
    *,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
    extra_headers: dict[str, str],
) -> PiModelSettings:
    settings: PiModelSettings = {
        "temperature": temperature,
        "timeout_ms": int(timeout * 1_000),
    }
    if reasoning_effort:
        settings["reasoning"] = reasoning_effort
    if extra_headers:
        settings["headers"] = dict(extra_headers)
    return settings


__all__ = [
    "PiModel",
    "PiModelBundle",
    "PiModelLike",
    "PiModelSettings",
    "ScriptedPiModel",
]
