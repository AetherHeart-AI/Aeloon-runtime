"""Static wire shapes used to publish the aeloon-rpc-v2 contract.

These definitions are deliberately runtime-light.  Pydantic is only imported by
the manifest exporter and contract tests; the RPC server uses these objects as
typing metadata and never validates responses or streaming events at runtime.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, Required, TypedDict

JsonObject = dict[str, Any]


class EmptyParams(TypedDict):
    pass


class ClientDescriptor(TypedDict):
    name: str
    version: str


class HandshakeParams(TypedDict):
    protocol: Literal["aeloon-rpc-v2"]
    client: NotRequired[ClientDescriptor]
    attachment_roots: NotRequired[list[str]]


class HandshakeLimits(TypedDict):
    prompt_chars: int
    attachments: int
    image_bytes: int
    file_bytes: int
    request_bytes: int
    retained_events: int


class HandshakeResult(TypedDict):
    protocol: Literal["aeloon-rpc-v2"]
    core_version: str
    core_commit: str
    server_instance_id: str
    methods: list[str]
    events: list[str]
    attachment_roots: list[str]
    config_path: str
    data_dir: str
    limits: HandshakeLimits


class HealthResult(TypedDict):
    status: Literal["running", "stopping"]
    pid: int
    started_at: str
    active_operations: int
    current_seq: int


class ShutdownResult(TypedDict):
    status: Literal["stopping"]


class EventsSubscribeParams(TypedDict):
    session_ids: NotRequired[list[str]]
    after_seq: NotRequired[int]


class EventCursor(TypedDict):
    server_instance_id: str
    seq: int


class ArtifactReference(TypedDict):
    path: str
    name: NotRequired[str]
    mime_type: NotRequired[str]
    size_bytes: NotRequired[int]
    kind: NotRequired[Literal["presentation", "document", "spreadsheet", "pdf", "image"]]


class ContentBlock(TypedDict):
    id: str
    type: Literal["text", "thinking", "tool_call"]
    role: NotRequired[str]
    content: NotRequired[str]
    name: NotRequired[str]
    arguments: NotRequired[JsonObject]
    status: NotRequired[str]
    result: NotRequired[str]
    duration_ms: NotRequired[int]
    details: NotRequired[JsonObject]
    artifacts: NotRequired[list[ArtifactReference]]


class CoreTurnAttachment(TypedDict):
    id: str
    type: Literal["image", "file", "assistant_selection"]
    display_name: str
    mime_type: NotRequired[str]
    size_bytes: NotRequired[int]
    text: NotRequired[str]


class AttachmentDescriptor(TypedDict):
    id: str
    type: Literal["image", "file"]
    display_name: str
    mime_type: str
    size_bytes: int
    source_path: str


class PromptTurnInput(TypedDict):
    kind: Literal["prompt"]
    text: str
    attachments: NotRequired[list[CoreTurnAttachment]]


class SkillTurnInput(TypedDict):
    kind: Literal["skill"]
    name: str
    additional_instructions: NotRequired[str]


class PromptTemplateTurnInput(TypedDict):
    kind: Literal["prompt_template"]
    name: str
    arguments: NotRequired[list[str]]


TurnInput = PromptTurnInput | SkillTurnInput | PromptTemplateTurnInput


class SessionMetadata(TypedDict):
    session_id: str
    workspace: str
    created_at: str
    title: str | None
    schema_version: int


class SessionState(TypedDict):
    leaf_id: str | None
    model_id: str
    thinking_level: str
    active_tools: list[str]


class ContextMessageStats(TypedDict):
    messageCount: int
    estimatedTokens: int
    percentage: float


class ContextWindowStats(TypedDict):
    usedTokens: int
    windowTokens: int | None
    remainingTokens: int | None
    usagePercent: float | None


class CacheStats(TypedDict):
    inputTokens: int
    readTokens: int
    writeTokens: int
    cacheableTokens: int
    hitTokenPercent: float
    requestCount: int
    hitRequestCount: int
    hitRequestPercent: float


class SessionStats(TypedDict):
    messageCount: int
    totalTokens: int
    costTotal: float
    contextWindow: ContextWindowStats
    messageTypes: dict[str, ContextMessageStats]
    cache: CacheStats


class TimelineTurn(TypedDict):
    type: Literal["turn"]
    turn_id: str
    status: str
    input: TurnInput
    blocks: list[ContentBlock]
    usage: dict[str, float]
    model_id: str
    thinking_level: str
    created_at: str
    completed_at: str | None
    duration_ms: int | None
    final_content: str | None
    error: str | None


class TimelineEntry(TypedDict):
    type: str


class ActiveOperation(TypedDict):
    operation_id: str
    kind: str
    status: str
    input: JsonObject
    blocks: list[ContentBlock]
    usage: dict[str, float]
    created_at: str


class SessionSnapshot(TypedDict):
    metadata: SessionMetadata
    state: SessionState
    stats: SessionStats
    timeline: list[TimelineTurn | TimelineEntry]
    active_operations: list[ActiveOperation]


class SessionCreateParams(TypedDict):
    session_id: str
    workspace: str
    title: NotRequired[str | None]


class SessionListParams(TypedDict):
    workspace: NotRequired[str]


class SessionListResult(TypedDict):
    sessions: list[SessionMetadata]


class SessionIdParams(TypedDict):
    session_id: str


class SessionDeleteResult(TypedDict):
    session_id: str
    deleted: bool


class SessionRenameParams(SessionIdParams):
    title: str


class SessionRenameResult(TypedDict):
    session_id: str
    title: str | None


class SessionConfigureParams(TypedDict, total=False):
    session_id: Required[str]
    model_id: str
    thinking_level: str
    active_tools: list[str]


class SessionConfigureResult(TypedDict):
    session_id: str
    model_id: str
    thinking_level: str
    active_tools: list[str]


class SessionTreeResult(TypedDict):
    session_id: str
    leaf_id: str | None
    entries: list[JsonObject]


class SessionNavigateParams(SessionIdParams):
    entry_id: str


class SessionNavigateResult(TypedDict):
    session_id: str
    leaf_id: str | None


class SessionCompactParams(TypedDict, total=False):
    session_id: Required[str]
    force: bool


class SessionCompactResult(TypedDict):
    session_id: str
    compacted: bool
    summary: NotRequired[str]


class SessionNextTurnParams(SessionIdParams):
    entry_id: str


class SessionNextTurnResult(TypedDict):
    session_id: str
    entry_id: str
    accepted: bool


class TurnStartParams(TypedDict):
    session_id: str
    input: TurnInput


class TurnStartResult(TypedDict):
    operation_id: str
    queue_position: int
    attachment_ids: list[str]
    skill_id: NotRequired[str]


class OperationIdParams(TypedDict):
    operation_id: str


class TurnCancelResult(TypedDict):
    operation_id: str
    cancelled: bool


class TurnTextParams(OperationIdParams):
    text: str


class TurnAcceptedResult(TypedDict):
    operation_id: str
    accepted: bool


class CatalogParams(TypedDict):
    workspace: NotRequired[str]
    session_id: NotRequired[str]


class CatalogItem(TypedDict):
    id: str
    name: str
    description: NotRequired[str]


class SkillCatalogItem(CatalogItem):
    command: str
    source: str
    location: str
    selected: bool
    enabled: bool
    explicit_invocation_enabled: bool
    model_invocation_enabled: bool
    content_loading: Literal["on_demand"]


class ModelCatalogItem(CatalogItem):
    provider_id: str
    thinking_levels: list[str]
    supports_image: bool
    context_window: int
    max_output_tokens: int


class CloudUser(TypedDict):
    id: str
    username: str
    display_name: str
    avatar_url: NotRequired[str | None]
    tier: NotRequired[str | None]


class ProviderSummary(TypedDict):
    id: str
    name: str
    driver: Literal["deepseek", "cloud", "custom"]
    backend: NotRequired[Literal["openai", "llamacpp", "ollama", "vllm"]]
    kind: Literal["local", "cloud"]
    endpoint: str
    enabled: bool
    authenticated: bool | None
    credential_configured: bool
    model_ids: list[str]
    user: NotRequired[CloudUser | None]


class CatalogResult(TypedDict):
    default_model_id: NotRequired[str]
    default_thinking_level: NotRequired[str]
    providers: list[ProviderSummary]
    models: list[ModelCatalogItem]
    tools: list[CatalogItem]
    skills: list[SkillCatalogItem]
    prompt_templates: list[CatalogItem]


class ProviderListResult(TypedDict):
    providers: list[ProviderSummary]


class ProviderIdParams(TypedDict):
    provider_id: str
    revision: NotRequired[int]


class ProviderMutationResult(TypedDict):
    provider: ProviderSummary
    revision: int


class ProviderAddParams(TypedDict):
    provider_id: str
    endpoint: str
    driver: NotRequired[Literal["custom"]]
    backend: NotRequired[Literal["openai", "llamacpp", "ollama", "vllm"]]
    name: NotRequired[str]
    api_key: NotRequired[str]
    proxy: NotRequired[str]
    headers: NotRequired[dict[str, str]]
    models: NotRequired[list[str]]
    max_output_tokens: NotRequired[int]
    revision: NotRequired[int]


class ProviderRemoveResult(TypedDict):
    provider_id: str
    removed: bool
    revision: int


class ProviderModelSettings(TypedDict):
    id: str
    name: NotRequired[str | None]
    reasoning: NotRequired[bool]
    supports_image: NotRequired[bool]
    context_window: NotRequired[int]
    max_output_tokens: NotRequired[int]
    cost: NotRequired[dict[str, float]]


class ProviderSettings(TypedDict):
    driver: Literal["deepseek", "cloud", "custom"]
    name: str
    enabled: bool
    endpoint: str
    proxy: str | None
    headers: NotRequired[dict[str, str]]
    credential_configured: bool
    device_name: NotRequired[str]
    allow_insecure_http: NotRequired[bool]
    models: NotRequired[list[ProviderModelSettings]]


class RetrySettings(TypedDict):
    enabled: bool
    max_retries: int
    base_delay_ms: int
    max_retry_delay_ms: int


class CompactionSettings(TypedDict):
    enabled: bool
    reserve_tokens: int
    keep_recent_tokens: int


class ResourceSettings(TypedDict):
    roots: list[str]
    load_skills: bool
    enabled_skill_ids: list[str]
    load_prompt_templates: bool
    load_context_files: bool


class ToolSettings(TypedDict):
    shell_path: str | None
    auto_resize_images: bool
    web: dict[str, Any]


class ToolsSearchTestParams(TypedDict):
    workspace: NotRequired[str]


class ToolsSearchTestResult(TypedDict):
    ok: bool
    provider: str
    result_count: int
    latency_ms: int
    message: NotRequired[str]


class SettingsResult(TypedDict):
    revision: int
    config_path: str
    default_model_id: str
    default_thinking_level: str
    retry: RetrySettings
    compaction: CompactionSettings
    resources: ResourceSettings
    tools: ToolSettings
    providers: dict[str, ProviderSettings]


class SettingsGetParams(TypedDict):
    workspace: NotRequired[str]


class SecretAction(TypedDict):
    path: str
    action: Literal["set", "clear"]
    value: NotRequired[str]


class SettingsUpdateParams(TypedDict):
    revision: int
    patch: JsonObject
    secret_actions: NotRequired[list[SecretAction]]
    workspace: NotRequired[str]


class CloudStatusResult(TypedDict):
    enabled: bool
    authenticated: bool
    user: CloudUser | None
    base_url: str
    vault_kind: str
    ok: NotRequired[bool]


class CloudLoginParams(TypedDict):
    username: str
    password: str


class OperationPayload(TypedDict):
    kind: str
    duration_ms: NotRequired[int]
    error: NotRequired[str]
    code: NotRequired[str]


class ContentStartedPayload(TypedDict):
    block: ContentBlock


class ContentDeltaPayload(TypedDict):
    block_id: str
    delta: str


class BlockPatchPayload(TypedDict):
    block_id: str
    patch: JsonObject


class UsagePayload(TypedDict):
    usage: NotRequired[dict[str, float]]
    stats: NotRequired[SessionStats]


class QueuePayload(TypedDict):
    queued_operation_ids: NotRequired[list[str]]
    active_operation_id: NotRequired[str | None]


class SessionRenamedPayload(TypedDict):
    title: str | None
    source: NotRequired[str]


class RevisionPayload(TypedDict):
    revision: int


class ProviderUpdatedPayload(TypedDict):
    provider_id: str
    action: str


class ShutdownPayload(TypedDict):
    intentional: bool
    reason: str


class LogPayload(TypedDict, total=False):
    level: str
    message: str
    data: JsonObject


class CoreEventBase(TypedDict):
    seq: int
    time: str
    workspace: str | None
    session_id: str | None
    operation_id: str | None


class EventsSubscribeResult(TypedDict):
    server_instance_id: str
    current_seq: int
    replay_complete: bool
    events: list[JsonObject]
    cursor: EventCursor


__all__ = [name for name in globals() if not name.startswith("_")]
