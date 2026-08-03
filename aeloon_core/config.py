"""Configuration for the standalone Python harness."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DeepSeekConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = "no-key"
    base_url: str = "https://api.deepseek.com"
    proxy: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)


class LocalModelConfig(BaseModel):
    """One model exposed by an OpenAI-compatible local API provider."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str | None = None
    reasoning: bool = False
    supports_image: bool = False
    context_window: int = Field(default=128_000, ge=1)
    max_tokens: int = Field(default=32_768, ge=1)


class LocalProviderConfig(BaseModel):
    """A user-added OpenAI-compatible API endpoint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    api_key: str = "no-key"
    proxy: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    models: list[LocalModelConfig] = Field(min_length=1)


class CloudConfig(BaseModel):
    """Aeloon's optional account-backed model service."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    base_url: str = "https://api.aetherheart.com"
    proxy: str | None = None
    device_name: str = "Aeloon Core"
    allow_insecure_http: bool = False


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_retries: int = Field(default=3, ge=0, le=20)
    base_delay_ms: int = Field(default=2_000, ge=0)
    max_retry_delay_ms: int = Field(default=60_000, ge=0)


class CompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    reserve_tokens: int = Field(default=16_384, ge=1)
    keep_recent_tokens: int = Field(default=20_000, ge=1)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Empty means "use the first connected model". A non-empty value pins a default.
    model: str = ""
    thinking_level: Literal["off", "minimal", "low", "medium", "high", "max"] = "off"
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    timeout_ms: int | None = Field(default=None, ge=0)
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    retry: RetryConfig = Field(default_factory=RetryConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)


class ResourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roots: list[Path] = Field(default_factory=list)
    no_skills: bool = False
    no_prompt_templates: bool = False
    no_context_files: bool = False


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shell_path: str | None = None
    auto_resize_images: bool = True


class Config(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    workspace: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Field(default_factory=lambda: Path("~/.aeloon-core").expanduser())
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    local_providers: dict[str, LocalProviderConfig] = Field(default_factory=dict)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)

    def normalized(self) -> Config:
        workspace = self.workspace.expanduser().resolve(strict=False)
        model_id = self.agent.model.strip()
        roots = [
            (root.expanduser() if root.expanduser().is_absolute() else workspace / root).resolve(
                strict=False
            )
            for root in self.resources.roots
        ]
        return self.model_copy(
            update={
                "workspace": workspace,
                "data_dir": self.data_dir.expanduser().resolve(strict=False),
                "agent": self.agent.model_copy(update={"model": model_id}),
                "resources": self.resources.model_copy(update={"roots": roots}),
            }
        )


def default_config_path() -> Path:
    raw = os.environ.get("AELOON_CORE_CONFIG")
    return Path(raw).expanduser() if raw else Path("~/.aeloon-core/config.json").expanduser()


def resolve_config_path(path: Path | str | None = None) -> Path:
    return Path(path).expanduser() if path is not None else default_config_path()


def load_config(
    path: Path | str | None = None,
) -> Config:
    resolved = resolve_config_path(path)
    if resolved.is_file():
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        config = Config.model_validate(raw)
    else:
        config = Config()
    return config.normalized()


def save_config(config: Config, path: Path | str | None = None, *, force: bool = True) -> Path:
    resolved = resolve_config_path(path)
    if resolved.exists() and not force:
        raise FileExistsError(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    normalized = config.normalized()
    payload = json.dumps(normalized.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
    return resolved


def public_config(config: Config, *, show_secrets: bool = False) -> dict[str, Any]:
    value = config.model_dump(mode="json")
    if not show_secrets:
        if value["deepseek"]["api_key"] != "no-key":
            value["deepseek"]["api_key"] = "***"
        _redact_secret_headers(value["deepseek"]["extra_headers"])
        for provider in value["local_providers"].values():
            if provider["api_key"] != "no-key":
                provider["api_key"] = "***"
            _redact_secret_headers(provider["extra_headers"])
    return value


def _redact_secret_headers(headers: dict[str, str]) -> None:
    for name in tuple(headers):
        if name.lower() in {"authorization", "api-key", "x-api-key"}:
            headers[name] = "***"


__all__ = [
    "AgentConfig",
    "CompactionConfig",
    "Config",
    "CloudConfig",
    "DeepSeekConfig",
    "LocalModelConfig",
    "LocalProviderConfig",
    "ResourceConfig",
    "RetryConfig",
    "ToolConfig",
    "default_config_path",
    "load_config",
    "public_config",
    "resolve_config_path",
    "save_config",
]
