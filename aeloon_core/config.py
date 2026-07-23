"""Runtime configuration for the standalone Aeloon Core project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProviderName = Literal["anthropic", "volcengine"]
KNOWN_PROVIDERS: frozenset[str] = frozenset({"anthropic", "volcengine"})

_REMOVED_V1_AGENT_DEFAULTS = frozenset(
    {"base_profile_id", "profile_id", "max_handoffs"}
)


class AnthropicProviderConfig(BaseModel):
    """Anthropic Messages API provider settings."""

    api_key: str = "no-key"
    base_url: str = "https://api.anthropic.com"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    proxy: str | None = None
    prompt_caching: bool = True


class VolcengineProviderConfig(BaseModel):
    """Volcano Engine Ark Agent Plan settings for the OpenAI Responses API."""

    api_key: str = "no-key"
    base_url: str = "https://ark.cn-beijing.volces.com/api/plan/v3"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    proxy: str | None = None


class ProvidersConfig(BaseModel):
    """Provider credential namespace (no global active switch)."""

    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)
    volcengine: VolcengineProviderConfig = Field(default_factory=VolcengineProviderConfig)


class ContextCompactionConfig(BaseModel):
    """Automatic model-context compaction settings."""

    enabled: bool = True
    trigger_ratio: float = Field(default=0.9, ge=0.1, le=1.0)
    preserve_recent_turns: int = Field(default=2, ge=1)
    preserve_recent_tokens: int | None = Field(default=None, ge=1)
    summary_max_tokens: int = Field(default=4096, ge=256)


class AgentRuntimePolicy(BaseModel):
    """Host policy layered around the PydanticAI execution loop."""

    transition_trace_enabled: bool = True
    stuck_detection_enabled: bool = True
    stuck_detection_threshold: int = Field(default=4, ge=3, le=20)


class AgentDefaultsConfig(BaseModel):
    """Default generation settings, including the default provider/model pair."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    reasoning_effort: str | None = None
    chat_timeout: int = 3600
    context_window_tokens: int = 128_000
    # Per-request completion ceiling. Anthropic-compatible SDKs default to 4096
    # when unset, which thinking models routinely exhaust before tool output.
    max_output_tokens: int = Field(default=32_768, ge=256)
    max_iterations: int = 25
    context_compaction: ContextCompactionConfig = Field(
        default_factory=ContextCompactionConfig
    )
    runtime: AgentRuntimePolicy = Field(default_factory=AgentRuntimePolicy)

    def model_ref(self) -> str:
        """Return the default selection as `provider/model`."""

        return format_model_ref(self.provider, self.model)


class AgentRoutingConfig(BaseModel):
    """Optional provider/model overrides for Master and Worker roles.

    Values may be bare model names (inherit `agents.defaults.provider`) or
    explicit `provider/model` refs. The model router falls back to the config
    default pair when an override cannot be used.
    """

    model_config = ConfigDict(extra="forbid")

    master: str | None = Field(default=None, min_length=1)
    workers: dict[str, str] = Field(default_factory=dict)

    @field_validator("master")
    @classmethod
    def _master_route_is_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("master model route requires a nonempty model ref")
        return normalized

    @field_validator("workers")
    @classmethod
    def _worker_routes_are_nonempty(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for worker_type_id, model_name in value.items():
            worker_id = worker_type_id.strip()
            model = model_name.strip()
            if not worker_id or not model:
                raise ValueError("worker model routes require nonempty ids and model names")
            normalized[worker_id] = model
        return normalized


class AgentBudgetConfig(BaseModel):
    """Optional request-budget overrides by orchestration responsibility."""

    model_config = ConfigDict(extra="forbid")

    master: int | None = Field(default=None, ge=1)
    workers: dict[str, int] = Field(default_factory=dict)

    @field_validator("workers")
    @classmethod
    def _worker_budgets_are_positive(cls, value: dict[str, int]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for worker_type_id, max_iterations in value.items():
            worker_id = worker_type_id.strip()
            if not worker_id:
                raise ValueError("worker budget overrides require nonempty ids")
            if max_iterations < 1:
                raise ValueError("worker budget overrides must be positive")
            normalized[worker_id] = max_iterations
        return normalized


class AgentsConfig(BaseModel):
    """Agent namespace."""

    defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)
    routing: AgentRoutingConfig = Field(default_factory=AgentRoutingConfig)
    budgets: AgentBudgetConfig = Field(default_factory=AgentBudgetConfig)


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


def parse_model_ref(
    value: str,
    *,
    default_provider: ProviderName,
) -> tuple[ProviderName, str]:
    """Parse a bare model name or explicit `provider/model` ref.

    Only the first path segment is treated as a provider when it is a known
    provider id. Other slashes remain part of the model name.
    """

    text = value.strip()
    if not text:
        raise ValueError("model ref must be nonempty")
    provider_candidate, separator, remainder = text.partition("/")
    if separator and provider_candidate in KNOWN_PROVIDERS:
        model = remainder.strip()
        if not model:
            raise ValueError(f"model ref {value!r} is missing a model name")
        return provider_candidate, model  # type: ignore[return-value]
    if default_provider not in KNOWN_PROVIDERS:
        raise ValueError(f"unknown default provider: {default_provider!r}")
    return default_provider, text  # type: ignore[return-value]


def format_model_ref(provider: ProviderName | str, model: str) -> str:
    """Format a provider/model pair for config and display."""

    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model name must be nonempty")
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(f"unknown provider: {provider!r}")
    return f"{provider}/{normalized_model}"


def load_config(path: Path | str | None = None) -> Config:
    """Load config from JSON and environment overrides."""

    config_path = resolve_config_path(path)
    data: Any = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        _drop_removed_v1_settings(data)
        _migrate_agent_runtime_settings(data)
        _migrate_provider_settings(data)
        _migrate_active_provider(data)

    config = Config.model_validate(data)
    updates: dict[str, Any] = {}

    default_provider = os.environ.get(
        "AELOON_CORE_PROVIDER",
        config.agents.defaults.provider,
    )
    if default_provider not in KNOWN_PROVIDERS:
        raise ValueError(
            "AELOON_CORE_PROVIDER must be 'anthropic' or 'volcengine', "
            f"got {default_provider!r}"
        )
    updates.setdefault("agents", {}).setdefault("defaults", {})[
        "provider"
    ] = default_provider

    # Credential env vars apply to their own provider independently of the default.
    if api_key := os.environ.get("ARK_API_KEY"):
        updates.setdefault("providers", {}).setdefault("volcengine", {})[
            "api_key"
        ] = api_key
    if base_url := os.environ.get("ARK_BASE_URL"):
        updates.setdefault("providers", {}).setdefault("volcengine", {})[
            "base_url"
        ] = base_url
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        updates.setdefault("providers", {}).setdefault("anthropic", {})[
            "api_key"
        ] = api_key
    if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
        updates.setdefault("providers", {}).setdefault("anthropic", {})[
            "base_url"
        ] = base_url

    if default_provider == "volcengine":
        if model := os.environ.get("ARK_MODEL"):
            updates.setdefault("agents", {}).setdefault("defaults", {})["model"] = model
    else:
        if model := os.environ.get("ANTHROPIC_MODEL"):
            updates.setdefault("agents", {}).setdefault("defaults", {})[
                "model"
            ] = model
    max_context_tokens = os.environ.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS")
    auto_compact_window = os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    parsed_max_context = (
        _parse_positive_int(
            max_context_tokens,
            name="CLAUDE_CODE_MAX_CONTEXT_TOKENS",
        )
        if max_context_tokens
        else None
    )
    parsed_auto_compact = (
        _parse_positive_int(
            auto_compact_window,
            name="CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        )
        if auto_compact_window
        else None
    )
    context_tokens = parsed_max_context or parsed_auto_compact
    if context_tokens is not None:
        updates.setdefault("agents", {}).setdefault("defaults", {})[
            "context_window_tokens"
        ] = context_tokens
    if parsed_auto_compact is not None and context_tokens is not None:
        updates.setdefault("agents", {}).setdefault("defaults", {}).setdefault(
            "context_compaction", {}
        )["trigger_ratio"] = max(
            0.1,
            min(1.0, parsed_auto_compact / context_tokens),
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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _split_env_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(os.pathsep) if item.strip()]


def _parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _drop_removed_v1_settings(data: Any) -> None:
    """Ignore only the Profile settings removed by the v2 runtime."""

    if not isinstance(data, dict):
        return
    agents = data.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    if not isinstance(defaults, dict):
        return
    for key in _REMOVED_V1_AGENT_DEFAULTS:
        defaults.pop(key, None)


def _migrate_provider_settings(data: Any) -> None:
    """Upgrade the former custom provider config into Anthropic naming."""

    if not isinstance(data, dict):
        return
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    legacy = providers.pop("custom", None)
    if not isinstance(legacy, dict) or "anthropic" in providers:
        return
    migrated = dict(legacy)
    if "api_base" in migrated and "base_url" not in migrated:
        migrated["base_url"] = migrated.pop("api_base")
    providers["anthropic"] = migrated


def _migrate_active_provider(data: Any) -> None:
    """Move the removed `providers.active` switch onto `agents.defaults.provider`."""

    if not isinstance(data, dict):
        return
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    active = providers.pop("active", None)
    if active not in KNOWN_PROVIDERS:
        return
    agents = data.setdefault("agents", {})
    if not isinstance(agents, dict):
        return
    defaults = agents.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        return
    if "provider" not in defaults:
        defaults["provider"] = active


def _migrate_agent_runtime_settings(data: Any) -> None:
    """Move safe legacy policy fields and discard removed reviewer behavior."""

    if not isinstance(data, dict):
        return
    agents = data.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    if not isinstance(defaults, dict):
        return
    legacy = defaults.pop("uasm", None)
    if not isinstance(legacy, dict) or "runtime" in defaults:
        return
    defaults["runtime"] = {
        key: legacy[key]
        for key in (
            "transition_trace_enabled",
            "stuck_detection_enabled",
            "stuck_detection_threshold",
        )
        if key in legacy
    }


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
