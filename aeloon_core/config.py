"""Runtime configuration for the standalone Aeloon Core project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CustomProviderConfig(BaseModel):
    """OpenAI-compatible provider settings."""

    api_key: str = "no-key"
    api_base: str = "http://localhost:8000/v1"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    proxy: str | None = None


class ProvidersConfig(BaseModel):
    """Provider namespace."""

    custom: CustomProviderConfig = Field(default_factory=CustomProviderConfig)


class AgentDefaultsConfig(BaseModel):
    """Default generation settings."""

    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    chat_timeout: int = 3600
    context_window_tokens: int = 128_000
    max_iterations: int = 25


class AgentsConfig(BaseModel):
    """Agent namespace."""

    defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)


class ExecToolConfig(BaseModel):
    """Shell execution settings."""

    timeout: int = 60


class WebToolConfig(BaseModel):
    """Web fetch and web search settings."""

    fetch_timeout: int = 20
    search_api_url: str | None = None
    search_api_key: str | None = None
    max_results: int = 5


class ToolsConfig(BaseModel):
    """Tool namespace."""

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    web: WebToolConfig = Field(default_factory=WebToolConfig)


class Config(BaseModel):
    """Top-level runtime config."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    workspace: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Field(default_factory=lambda: Path("~/.aeloon-core").expanduser())

    def normalized(self) -> Config:
        """Return a copy with filesystem paths expanded and resolved."""

        return self.model_copy(
            update={
                "workspace": self.workspace.expanduser().resolve(strict=False),
                "data_dir": self.data_dir.expanduser().resolve(strict=False),
            }
        )


def default_config_path() -> Path:
    """Return the default config path."""

    raw = os.environ.get("AELOON_CORE_CONFIG")
    if raw:
        return Path(raw).expanduser()
    return Path("~/.aeloon-core/config.json").expanduser()


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Return the explicit or default config path."""

    return Path(path).expanduser() if path is not None else default_config_path()


def load_config(path: Path | str | None = None) -> Config:
    """Load config from JSON and environment overrides."""

    config_path = resolve_config_path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    config = Config.model_validate(data)
    updates: dict[str, Any] = {}

    if api_key := os.environ.get("AELOON_CORE_API_KEY"):
        updates.setdefault("providers", {}).setdefault("custom", {})["api_key"] = api_key
    if api_base := os.environ.get("AELOON_CORE_API_BASE"):
        updates.setdefault("providers", {}).setdefault("custom", {})["api_base"] = api_base
    if model := os.environ.get("AELOON_CORE_MODEL"):
        updates.setdefault("agents", {}).setdefault("defaults", {})["model"] = model
    if workspace := os.environ.get("AELOON_CORE_WORKSPACE"):
        updates["workspace"] = workspace
    if data_dir := os.environ.get("AELOON_CORE_DATA_DIR"):
        updates["data_dir"] = data_dir

    if updates:
        merged = config.model_dump(mode="json")
        _deep_update(merged, updates)
        config = Config.model_validate(merged)

    return config.normalized()


def save_config(config: Config, path: Path | str | None = None) -> Path:
    """Persist config JSON and return the written path."""

    config_path = resolve_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config.normalized().model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
