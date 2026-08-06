"""Runtime-safe projection and redaction primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from aeloon_core.config import Config, provider_secret_values


class RuntimeProjection:
    def __init__(self, config: Callable[[], Config], *, output_limit: int) -> None:
        self._config = config
        self.output_limit = output_limit

    def safe_mapping(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): self.json_safe(item) for key, item in value.items()}

    def json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self.json_safe(item)
                for key, item in value.items()
                if str(key).lower() not in {"api_key", "authorization", "systemprompt", "payload"}
            }
        if isinstance(value, list | tuple):
            return [self.json_safe(item) for item in value]
        return str(value)[: self.output_limit]

    def sanitize(self, message: str) -> str:
        value = message[:4_000]
        for secret in provider_secret_values(self._config()):
            value = value.replace(secret, "***")
        return value


__all__ = ["RuntimeProjection"]
