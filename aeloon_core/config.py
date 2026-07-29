"""Runtime configuration for the standalone Aeloon Core project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

ProviderName = Literal["deepseek"]
KNOWN_PROVIDERS: frozenset[str] = frozenset({"deepseek"})
SkillRootId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$"),
]

_REMOVED_V1_AGENT_DEFAULTS = frozenset(
    {"base_profile_id", "profile_id", "max_handoffs"}
)


class DeepSeekProviderConfig(BaseModel):
    """Credentials and transport settings for Pydantic AI's DeepSeek provider."""

    api_key: str = "no-key"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    proxy: str | None = None


class ProvidersConfig(BaseModel):
    """Provider credential namespace (no global active switch)."""

    deepseek: DeepSeekProviderConfig = Field(default_factory=DeepSeekProviderConfig)


class ContextCompactionConfig(BaseModel):
    """Harness sliding-window settings."""

    enabled: bool = True
    trigger_ratio: float = Field(default=0.9, ge=0.1, le=1.0)
    preserve_recent_tokens: int | None = Field(default=None, ge=1)


class AgentRuntimePolicy(BaseModel):
    """Host policy layered around the PydanticAI execution loop."""

    transition_trace_enabled: bool = True
    stuck_detection_enabled: bool = True
    stuck_detection_threshold: int = Field(default=4, ge=3, le=20)


class AgentDefaultsConfig(BaseModel):
    """Default generation settings, including the default provider/model pair."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = "deepseek"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.7
    reasoning_effort: str | None = None
    chat_timeout: int = 3600
    context_window_tokens: int = 128_000
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
    """Optional provider/model overrides for Master and Expert stages.

    Values may be bare model names (inherit `agents.defaults.provider`) or
    explicit `provider/model` refs. The model router falls back to the config
    default pair when an override cannot be used.
    """

    model_config = ConfigDict(extra="forbid")

    master: str | None = Field(default=None, min_length=1)
    experts: dict[str, str] = Field(default_factory=dict)

    @field_validator("master")
    @classmethod
    def _master_route_is_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("master model route requires a nonempty model ref")
        return normalized

    @field_validator("experts")
    @classmethod
    def _expert_routes_are_nonempty(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for route_key, model_name in value.items():
            expert_route = route_key.strip()
            model = model_name.strip()
            if not expert_route or not model:
                raise ValueError("expert model routes require nonempty ids and model names")
            normalized[expert_route] = model
        return normalized


class AgentsConfig(BaseModel):
    """Model defaults and routing for the Master and Expert stages."""

    defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)
    routing: AgentRoutingConfig = Field(default_factory=AgentRoutingConfig)


class SkillRootConfig(BaseModel):
    """One explicitly trusted source of discoverable Skill manifests."""

    model_config = ConfigDict(extra="forbid")

    id: SkillRootId
    path: Path


class SkillsConfig(BaseModel):
    """Skill discovery roots and the plain-Skill scope granted to Master."""

    model_config = ConfigDict(extra="forbid")

    roots: list[SkillRootConfig] = Field(default_factory=list)
    master_allowlist: list[str] = Field(default_factory=list)

    @field_validator("roots")
    @classmethod
    def _root_ids_are_unique(
        cls,
        roots: list[SkillRootConfig],
    ) -> list[SkillRootConfig]:
        ids = [root.id for root in roots]
        if len(ids) != len(set(ids)):
            raise ValueError("skill root ids must be unique")
        if "builtin" in ids or "workspace" in ids:
            raise ValueError("'builtin' and 'workspace' are reserved skill root ids")
        return roots

    @field_validator("master_allowlist")
    @classmethod
    def _master_skill_ids_are_nonempty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("master skill allowlist entries must be nonempty")
        return list(dict.fromkeys(normalized))


class ExpertsConfig(BaseModel):
    """Turn-scoped ExpertSkill execution policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=lambda: ["builtin:research", "builtin:coding"])
    max_calls_per_turn: int = Field(default=8, ge=1, le=128)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    stage_request_limit: int = Field(default=25, ge=1, le=100)
    timeout_seconds: float = Field(default=1800.0, ge=1.0, le=7200.0)
    max_upstream_chars: int = Field(default=32_000, ge=1_000, le=256_000)
    web_backend: Literal["exa"] = "exa"

    @field_validator("enabled")
    @classmethod
    def _enabled_expert_ids_are_nonempty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("enabled expert ids must be nonempty")
        return list(dict.fromkeys(normalized))


class ExecToolConfig(BaseModel):
    """Shell execution settings."""

    timeout: int = 60


class ToolsConfig(BaseModel):
    """Tool namespace."""

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)


class Config(BaseModel):
    """Top-level runtime config."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    experts: ExpertsConfig = Field(default_factory=ExpertsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    workspace: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Field(default_factory=lambda: Path("~/.aeloon-core").expanduser())

    def normalized(self) -> Config:
        """Return a copy with filesystem paths expanded and resolved."""

        workspace = self.workspace.expanduser().resolve(strict=False)
        roots = [
            root.model_copy(
                update={
                    "path": (
                        root.path.expanduser()
                        if root.path.expanduser().is_absolute()
                        else workspace / root.path.expanduser()
                    ).resolve(strict=False)
                }
            )
            for root in self.skills.roots
        ]
        return self.model_copy(
            update={
                "workspace": workspace,
                "data_dir": self.data_dir.expanduser().resolve(strict=False),
                "skills": self.skills.model_copy(update={"roots": roots}),
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
        raise ValueError("model ref should be nonempty")
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
        _drop_removed_orchestration_settings(data)
        _migrate_agent_runtime_settings(data)
        _migrate_active_provider(data)

    config = Config.model_validate(data)
    updates: dict[str, Any] = {}

    default_provider = os.environ.get(
        "AELOON_CORE_PROVIDER",
        config.agents.defaults.provider,
    )
    if default_provider not in KNOWN_PROVIDERS:
        raise ValueError(
            "AELOON_CORE_PROVIDER must be 'deepseek', "
            f"got {default_provider!r}"
        )
    updates.setdefault("agents", {}).setdefault("defaults", {})[
        "provider"
    ] = default_provider

    if api_key := os.environ.get("DEEPSEEK_API_KEY"):
        updates.setdefault("providers", {}).setdefault("deepseek", {})[
            "api_key"
        ] = api_key

    if model := os.environ.get("DEEPSEEK_MODEL"):
        updates.setdefault("agents", {}).setdefault("defaults", {})["model"] = model

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


def _drop_removed_orchestration_settings(data: Any) -> None:
    """Migrate safe orchestration fields and discard removed DAG settings."""

    if not isinstance(data, dict):
        return
    skills = data.get("skills")
    if isinstance(skills, dict):
        skills.pop("enabled", None)
    agents = data.get("agents")
    if not isinstance(agents, dict):
        return
    agents.pop("budgets", None)
    harness = agents.pop("harness", None)
    if isinstance(harness, dict):
        experts = data.setdefault("experts", {})
        if isinstance(experts, dict):
            if "sub_agent_request_limit" in harness:
                experts.setdefault(
                    "stage_request_limit",
                    harness["sub_agent_request_limit"],
                )
            if "max_agent_calls" in harness:
                experts.setdefault("max_calls_per_turn", harness["max_agent_calls"])
    agents.pop("templates", None)
    routing = agents.get("routing")
    if isinstance(routing, dict):
        workers = routing.pop("workers", None)
        if isinstance(workers, dict) and "experts" not in routing:
            legacy_routes = {
                "builder": "builtin:coding/build",
                "reviewer": "builtin:coding/review",
                "explorer": "builtin:research",
                "researcher": "builtin:research/docs",
            }
            routing["experts"] = {
                legacy_routes[worker_id]: model
                for worker_id, model in workers.items()
                if worker_id in legacy_routes
            }


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
