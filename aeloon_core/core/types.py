"""Vendor-neutral contracts for one stateless inference run."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "max"]
QueueMode = Literal["all", "one-at-a-time"]
StopReason = Literal["stop", "toolUse", "length", "error", "aborted"]
RunEventType = Literal[
    "abort",
    "agent_end",
    "agent_start",
    "auto_retry_end",
    "auto_retry_start",
    "before_agent_start",
    "before_inference_context",
    "before_inference_request",
    "compaction_end",
    "compaction_start",
    "context",
    "message_end",
    "message_start",
    "message_update",
    "queue_update",
    "context_compacted",
    "settled",
    "tool_call",
    "tool_execution_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_result",
    "turn_end",
    "turn_start",
]


class RunError(RuntimeError):
    """Stable public failure raised while preparing or running an agent."""

    def __init__(self, code: str, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class InferenceError(RunError):
    """Inference or transport failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        cause: Exception | None = None,
        retryable: bool = False,
        retry_after_ms: int | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(code, message, cause=cause)
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class ImageContent:
    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"


UserContent: TypeAlias = TextContent | ImageContent
AssistantContent: TypeAlias = TextContent | ThinkingContent | ToolCall
ToolResultContent: TypeAlias = TextContent | ImageContent


@dataclass(frozen=True, slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    reasoning: int | None = None
    cost: dict[str, float] = field(
        default_factory=lambda: {
            "input": 0.0,
            "output": 0.0,
            "cacheRead": 0.0,
            "cacheWrite": 0.0,
            "total": 0.0,
        }
    )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> Usage:
        raw = value or {}
        prompt = int(raw.get("input", raw.get("prompt_tokens", 0)) or 0)
        completion = int(raw.get("output", raw.get("completion_tokens", 0)) or 0)
        prompt_details = raw.get("prompt_tokens_details")
        completion_details = raw.get("completion_tokens_details")
        cache_read = int(
            raw.get(
                "cacheRead",
                raw.get(
                    "cache_read",
                    raw.get(
                        "prompt_cache_hit_tokens",
                        prompt_details.get("cached_tokens", 0)
                        if isinstance(prompt_details, Mapping)
                        else 0,
                    ),
                ),
            )
            or 0
        )
        cache_write = int(raw.get("cacheWrite", raw.get("cache_write", 0)) or 0)
        total = int(raw.get("totalTokens", raw.get("total_tokens", 0)) or 0)
        input_tokens = prompt if "input" in raw else max(0, prompt - cache_read)
        reasoning = raw.get("reasoning")
        if reasoning is None and isinstance(completion_details, Mapping):
            reasoning = completion_details.get("reasoning_tokens")
        cost = raw.get("cost")
        default_cost = {
            "input": 0.0,
            "output": 0.0,
            "cacheRead": 0.0,
            "cacheWrite": 0.0,
            "total": 0.0,
        }
        return cls(
            input=input_tokens,
            output=completion,
            cache_read=cache_read,
            cache_write=cache_write,
            total_tokens=total
            or prompt + completion + cache_write + (cache_read if "input" in raw else 0),
            reasoning=(int(reasoning) if reasoning is not None else None),
            cost=dict(cost) if isinstance(cost, Mapping) else default_cost,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "input": self.input,
            "output": self.output,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "totalTokens": self.total_tokens,
            "cost": dict(self.cost),
        }
        if self.reasoning is not None:
            value["reasoning"] = self.reasoning
        return value


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str | tuple[UserContent, ...]
    timestamp: int = field(default_factory=now_ms)
    role: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[AssistantContent, ...]
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["assistant"] = "assistant"

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.content if isinstance(part, TextContent))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(part for part in self.content if isinstance(part, ToolCall))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: tuple[ToolResultContent, ...]
    is_error: bool = False
    usage: Usage | None = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["toolResult"] = "toolResult"


AgentMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


def content_to_dict(content: Any) -> dict[str, Any]:
    if isinstance(content, ImageContent):
        return {"type": "image", "data": content.data, "mimeType": content.mime_type}
    if isinstance(content, ToolCall):
        return {
            "type": "toolCall",
            "id": content.id,
            "name": content.name,
            "arguments": content.arguments,
        }
    return asdict(content)


def message_to_dict(message: AgentMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        content: Any = message.content
        if isinstance(content, tuple):
            content = [content_to_dict(part) for part in content]
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, AssistantMessage):
        value: dict[str, Any] = {
            "role": "assistant",
            "content": [content_to_dict(part) for part in message.content],
            "provider": message.provider,
            "model": message.model,
            "usage": message.usage.to_dict(),
            "stopReason": message.stop_reason,
            "timestamp": message.timestamp,
        }
        if message.error_message:
            value["errorMessage"] = message.error_message
        return value
    value = {
        "role": "toolResult",
        "toolCallId": message.tool_call_id,
        "toolName": message.tool_name,
        "content": [content_to_dict(part) for part in message.content],
        "isError": message.is_error,
        "timestamp": message.timestamp,
    }
    if message.usage is not None:
        value["usage"] = message.usage.to_dict()
    return value


def content_from_dict(value: Mapping[str, Any]) -> Any:
    kind = value.get("type")
    if kind == "text":
        return TextContent(str(value.get("text") or ""))
    if kind == "thinking":
        return ThinkingContent(str(value.get("thinking") or ""))
    if kind == "image":
        return ImageContent(
            data=str(value.get("data") or ""),
            mime_type=str(value.get("mimeType", value.get("mime_type", "image/png"))),
        )
    if kind == "toolCall":
        arguments = value.get("arguments")
        return ToolCall(
            id=str(value.get("id") or "tool-call"),
            name=str(value.get("name") or "tool"),
            arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
        )
    raise ValueError(f"unsupported content type: {kind!r}")


def message_from_dict(value: Mapping[str, Any]) -> AgentMessage:
    role = value.get("role")
    if role == "user":
        raw_content = value.get("content", "")
        content: str | tuple[UserContent, ...]
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, Sequence):
            content = tuple(
                content_from_dict(part) for part in raw_content if isinstance(part, Mapping)
            )
        else:
            content = str(raw_content)
        return UserMessage(content=content, timestamp=int(value.get("timestamp") or 0))
    if role == "assistant":
        raw_content = value.get("content")
        return AssistantMessage(
            content=tuple(
                content_from_dict(part) for part in raw_content or [] if isinstance(part, Mapping)
            ),
            provider=str(value.get("provider") or "unknown"),
            model=str(value.get("model") or "unknown"),
            usage=Usage.from_dict(
                value.get("usage") if isinstance(value.get("usage"), Mapping) else None
            ),
            stop_reason=str(value.get("stopReason") or "stop"),  # type: ignore[arg-type]
            error_message=(str(value["errorMessage"]) if value.get("errorMessage") else None),
            timestamp=int(value.get("timestamp") or 0),
        )
    if role == "toolResult":
        raw_content = value.get("content")
        return ToolResultMessage(
            tool_call_id=str(value.get("toolCallId") or "tool-call"),
            tool_name=str(value.get("toolName") or "tool"),
            content=tuple(
                content_from_dict(part) for part in raw_content or [] if isinstance(part, Mapping)
            ),
            is_error=bool(value.get("isError", False)),
            usage=(
                Usage.from_dict(value["usage"]) if isinstance(value.get("usage"), Mapping) else None
            ),
            timestamp=int(value.get("timestamp") or 0),
        )
    raise ValueError(f"unsupported message role: {role!r}")


@dataclass(frozen=True, slots=True)
class Model:
    id: str
    name: str
    provider: str
    reasoning: bool = False
    input: tuple[str, ...] = ("text",)
    context_window: int = 128_000
    max_output_tokens: int = 32_768
    cost: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamOptions:
    timeout_ms: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    thinking_level: ThinkingLevel = "off"
    max_retries: int | None = None
    base_delay_ms: int = 2_000
    max_retry_delay_ms: int = 60_000
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InferenceContext:
    system_prompt: str
    messages: tuple[AgentMessage, ...]
    tools: tuple[dict[str, Any], ...]
    session_id: str


@dataclass(frozen=True, slots=True)
class AssistantStreamEvent:
    type: Literal[
        "start",
        "text_delta",
        "thinking_delta",
        "toolcall_delta",
        "done",
        "error",
    ]
    delta: str = ""
    content_index: int | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    message: AssistantMessage | None = None


class InferencePort(Protocol):
    def stream(
        self,
        model: Model,
        context: InferenceContext,
        options: StreamOptions,
    ) -> AsyncIterator[AssistantStreamEvent]: ...


ToolUpdateCallback: TypeAlias = Callable[["ToolResult"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: tuple[ToolResultContent, ...]
    details: Any = None
    is_error: bool = False
    terminate: bool = False
    usage: Usage | None = None

    @classmethod
    def text(
        cls,
        value: str,
        *,
        details: Any = None,
        is_error: bool = False,
        terminate: bool = False,
    ) -> ToolResult:
        return cls((TextContent(value),), details, is_error, terminate)


class Tool(Protocol):
    """Executable tool contract consumed by the stateless run engine."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    execution_mode: Literal["parallel", "sequential"]

    def definition(self) -> dict[str, Any]: ...

    def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def execute(
        self,
        call_id: str,
        arguments: dict[str, Any],
        on_update: ToolUpdateCallback | None,
    ) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class RunEvent:
    type: RunEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**self.data, "type": self.type}


RunEventSink: TypeAlias = Callable[[RunEvent], Awaitable[None] | None]
RunHook: TypeAlias = Callable[[RunEvent], Awaitable[dict[str, Any] | None] | dict[str, Any] | None]

__all__ = [
    "AgentMessage",
    "AssistantContent",
    "AssistantMessage",
    "AssistantStreamEvent",
    "RunError",
    "RunEvent",
    "RunEventSink",
    "RunEventType",
    "RunHook",
    "ImageContent",
    "Model",
    "InferenceContext",
    "InferenceError",
    "InferencePort",
    "QueueMode",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "content_from_dict",
    "content_to_dict",
    "message_from_dict",
    "message_to_dict",
    "now_ms",
]
