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


class ContextCompactionConfig(BaseModel):
    """Automatic model-context compaction settings."""

    enabled: bool = True
    trigger_ratio: float = Field(default=0.9, ge=0.1, le=1.0)
    buffer_tokens: int = Field(default=20_000, ge=0)
    preserve_recent_turns: int = Field(default=2, ge=1)
    preserve_recent_tokens: int | None = Field(default=None, ge=1)
    summary_max_tokens: int = Field(default=4096, ge=256)


class AgentDefaultsConfig(BaseModel):
    """Default generation settings."""

    model: str = "default"
    temperature: float = 0.7
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    chat_timeout: int = 3600
    context_window_tokens: int = 128_000
    max_iterations: int = 25
    max_auto_continue_iterations: int = 25
    max_finalization_iterations: int = 2
    context_compaction: ContextCompactionConfig = Field(
        default_factory=ContextCompactionConfig
    )


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


class SkillsConfig(BaseModel):
    """Skill discovery settings."""

    enabled: bool = True
    external: bool = True
    claude_code: bool = True
    paths: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """Top-level runtime config."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
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
    if max_tokens := os.environ.get("AELOON_CORE_MAX_TOKENS"):
        updates.setdefault("agents", {}).setdefault("defaults", {})["max_tokens"] = (
            _parse_optional_int(max_tokens)
        )
    if workspace := os.environ.get("AELOON_CORE_WORKSPACE"):
        updates["workspace"] = workspace
    if data_dir := os.environ.get("AELOON_CORE_DATA_DIR"):
        updates["data_dir"] = data_dir
    if skills_enabled := os.environ.get("AELOON_CORE_SKILLS_ENABLED"):
        updates.setdefault("skills", {})["enabled"] = _parse_bool(skills_enabled)
    if disable_external := os.environ.get("AELOON_CORE_DISABLE_EXTERNAL_SKILLS"):
        updates.setdefault("skills", {})["external"] = not _parse_bool(disable_external)
    if disable_claude := os.environ.get("AELOON_CORE_DISABLE_CLAUDE_CODE_SKILLS"):
        updates.setdefault("skills", {})["claude_code"] = not _parse_bool(disable_claude)
    if skill_paths := os.environ.get("AELOON_CORE_SKILL_PATHS"):
        updates.setdefault("skills", {})["paths"] = _split_env_list(skill_paths)

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


def _parse_optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return None
    return int(value)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _split_env_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(os.pathsep) if item.strip()]


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
