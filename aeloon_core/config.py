"""Configuration for the Aeloon application runtime."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str | None = None
    reasoning: bool = False
    supports_image: bool = False
    context_window: int = Field(default=128_000, ge=1)
    max_output_tokens: int = Field(
        default_factory=lambda data: max(
            1, min(8_192, int(data.get("context_window", 128_000) * 0.25))
        ),
        ge=1,
    )
    cost: dict[str, float] = Field(default_factory=dict)

class DeepSeekProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: Literal["deepseek"] = "deepseek"
    name: str = "DeepSeek"
    enabled: bool = True
    endpoint: str = "https://api.deepseek.com"
    api_key: str | None = None
    proxy: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    models: list[ProviderModelConfig] = Field(default_factory=list)

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: Any) -> Any:
        return _normalize_api_key(value)


class CustomProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: Literal["custom"] = "custom"
    backend: Literal["openai", "llamacpp", "ollama", "vllm"]
    name: str
    enabled: bool = True
    endpoint: str
    api_key: str | None = None
    proxy: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    models: list[ProviderModelConfig] = Field(default_factory=list)

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: Any) -> Any:
        return _normalize_api_key(value)


class CloudProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: Literal["cloud"] = "cloud"
    name: str = "Aeloon Cloud"
    enabled: bool = True
    endpoint: str = "https://api.aetherheart.com"
    proxy: str | None = None
    device_name: str = "Aeloon Core"
    allow_insecure_http: bool = False


ProviderConfig = Annotated[
    DeepSeekProviderConfig
    | CustomProviderConfig
    | CloudProviderConfig,
    Field(discriminator="driver"),
]

_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SENSITIVE_HEADERS = {
    "api-key",
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
}


def _normalize_api_key(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    return normalized or None


def is_sensitive_header(name: str) -> bool:
    return name.strip().lower() in _SENSITIVE_HEADERS


def redact_sensitive_headers(headers: dict[str, str]) -> None:
    for name in tuple(headers):
        if is_sensitive_header(name):
            headers[name] = "***"


def provider_secret_values(config: Config) -> tuple[str, ...]:
    values: list[str] = []
    for provider in config.providers.values():
        api_key = getattr(provider, "api_key", None)
        if api_key:
            values.append(api_key)
        headers = getattr(provider, "headers", {})
        values.extend(
            value for name, value in headers.items() if value and is_sensitive_header(name)
        )
    return tuple(dict.fromkeys(values))


def _default_providers() -> dict[str, ProviderConfig]:
    return {
        "deepseek": DeepSeekProviderConfig(),
        "aeloon-cloud": CloudProviderConfig(),
    }


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
    # None selects every discovered skill; an explicit list is subtractive.
    enabled_skills: list[str] | None = None
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
    providers: dict[str, ProviderConfig] = Field(default_factory=_default_providers)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)

    @model_validator(mode="after")
    def validate_reserved_providers(self) -> Config:
        for provider_id, provider in self.providers.items():
            if not _PROVIDER_ID.fullmatch(provider_id):
                raise ValueError(f"Invalid Provider id: {provider_id}")
            parsed = urlsplit(provider.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"Provider endpoint must be HTTP(S): {provider_id}")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    f"Provider endpoint must not contain credentials, query, or fragment: "
                    f"{provider_id}"
                )
            if (
                isinstance(provider, CloudProviderConfig)
                and parsed.scheme != "https"
                and not provider.allow_insecure_http
            ):
                raise ValueError(
                    "Aeloon Cloud endpoint must use HTTPS unless allow_insecure_http is true"
                )
        required = {"deepseek": "deepseek", "aeloon-cloud": "cloud"}
        for provider_id, driver in required.items():
            provider = self.providers.get(provider_id)
            if provider is None:
                raise ValueError(f"Reserved Provider is required: {provider_id}")
            if provider.driver != driver:
                raise ValueError(f"Reserved Provider {provider_id} must use driver {driver}")
        exclusive_drivers = {"deepseek": "deepseek", "cloud": "aeloon-cloud"}
        for provider_id, provider in self.providers.items():
            owner = exclusive_drivers.get(provider.driver)
            if owner is not None and provider_id != owner:
                raise ValueError(f"Provider driver {provider.driver} is reserved for id {owner}")
        return self

    def normalized(self) -> Config:
        workspace = self.workspace.expanduser().resolve(strict=False)
        model_id = self.agent.model.strip()
        roots = [
            (root.expanduser() if root.expanduser().is_absolute() else workspace / root).resolve(
                strict=False
            )
            for root in self.resources.roots
        ]
        enabled_skills = (
            None
            if self.resources.enabled_skills is None
            else list(
                dict.fromkeys(
                    name for item in self.resources.enabled_skills if (name := item.strip())
                )
            )
        )
        return self.model_copy(
            update={
                "workspace": workspace,
                "data_dir": self.data_dir.expanduser().resolve(strict=False),
                "agent": self.agent.model_copy(update={"model": model_id}),
                "resources": self.resources.model_copy(
                    update={"roots": roots, "enabled_skills": enabled_skills}
                ),
                "providers": {
                    provider_id: provider.model_copy(
                        update={
                            "endpoint": provider.endpoint.rstrip("/"),
                            **(
                                {"api_key": getattr(provider, "api_key", None) or None}
                                if hasattr(provider, "api_key")
                                else {}
                            ),
                        }
                    )
                    for provider_id, provider in self.providers.items()
                },
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
        for provider in value["providers"].values():
            if provider.get("api_key"):
                provider["api_key"] = "***"
            redact_sensitive_headers(provider.get("headers") or {})
    return value


__all__ = [
    "AgentConfig",
    "CompactionConfig",
    "Config",
    "CloudProviderConfig",
    "DeepSeekProviderConfig",
    "CustomProviderConfig",
    "ProviderConfig",
    "ProviderModelConfig",
    "ResourceConfig",
    "RetryConfig",
    "ToolConfig",
    "default_config_path",
    "load_config",
    "is_sensitive_header",
    "provider_secret_values",
    "public_config",
    "redact_sensitive_headers",
    "resolve_config_path",
    "save_config",
]
