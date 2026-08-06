"""Public session, capability, and application runtime API."""

from __future__ import annotations

from typing import Any

from aeloon_core.runtime.artifacts import (
    PRESENT_FILES_TOOL_NAME,
    Artifact,
    PresentFilesTool,
)
from aeloon_core.runtime.coordinator import OperationCoordinator
from aeloon_core.runtime.input import PreparedTurn, TurnInputResolver
from aeloon_core.runtime.projection import RuntimeProjection
from aeloon_core.runtime.prompt import build_system_prompt, format_skills_for_system_prompt
from aeloon_core.runtime.providers import (
    BaseProvider,
    CloudProvider,
    CustomProvider,
    DeepSeekProvider,
    OpenAICompatibleProvider,
    ProviderManager,
    ProviderManagerFactory,
    normalize_model_id,
    provider_manager_factory,
    qualify_model_id,
    resolve_model_id,
    split_model_id,
    validate_provider_id,
)
from aeloon_core.runtime.resources import (
    LoadedSkill,
    PromptTemplate,
    ResourceLoader,
    RuntimeResources,
    Skill,
)
from aeloon_core.runtime.session import (
    JsonlSessionRepository,
    Session,
    SessionContext,
    SessionError,
    SessionMetadata,
)
from aeloon_core.runtime.tooling import RuntimeToolSet
from aeloon_core.runtime.types import (
    OperationSnapshot,
    RuntimeEvent,
    RuntimeFailure,
    SessionInfo,
    SessionSnapshot,
    TurnInput,
)


def __getattr__(name: str) -> Any:
    if name == "RuntimeService":
        from aeloon_core.runtime.service import RuntimeService

        return RuntimeService
    raise AttributeError(name)


__all__ = [
    "Artifact",
    "BaseProvider",
    "CloudProvider",
    "CustomProvider",
    "DeepSeekProvider",
    "JsonlSessionRepository",
    "LoadedSkill",
    "OperationSnapshot",
    "OperationCoordinator",
    "OpenAICompatibleProvider",
    "PRESENT_FILES_TOOL_NAME",
    "PresentFilesTool",
    "PreparedTurn",
    "PromptTemplate",
    "ProviderManager",
    "ProviderManagerFactory",
    "ResourceLoader",
    "RuntimeEvent",
    "RuntimeFailure",
    "RuntimeProjection",
    "RuntimeResources",
    "RuntimeService",
    "RuntimeToolSet",
    "Session",
    "SessionContext",
    "SessionError",
    "SessionInfo",
    "SessionMetadata",
    "SessionSnapshot",
    "Skill",
    "TurnInput",
    "TurnInputResolver",
    "build_system_prompt",
    "format_skills_for_system_prompt",
    "normalize_model_id",
    "provider_manager_factory",
    "qualify_model_id",
    "resolve_model_id",
    "split_model_id",
    "validate_provider_id",
]
