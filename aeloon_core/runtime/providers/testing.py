"""Deterministic inference Provider for tests and dependency injection."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Sequence

from aeloon_core.core import (
    AssistantMessage,
    AssistantStreamEvent,
    InferenceContext,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from aeloon_core.runtime.providers.base import BaseProvider


class ScriptedProvider(BaseProvider):
    def __init__(
        self,
        responses: Iterable[AssistantMessage | Sequence[AssistantStreamEvent]],
        *,
        models: Iterable[Model] = (),
        provider_id: str | None = None,
        name: str = "Scripted Provider",
    ) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[Model, InferenceContext, StreamOptions]] = []
        inferred = [
            Model(
                item.model
                if item.model.startswith(f"{item.provider}/")
                else f"{item.provider}/{item.model}",
                item.model,
                item.provider,
            )
            for item in self.responses
            if isinstance(item, AssistantMessage)
        ]
        configured = tuple(models) or tuple({item.id: item for item in inferred}.values())
        resolved_id = provider_id or (configured[0].provider if configured else "scripted")
        super().__init__(
            provider_id=resolved_id,
            name=name,
            endpoint="scripted://local",
        )
        self._models = {item.id: item for item in configured}

    async def models(self) -> dict[str, Model]:
        return dict(self._models)

    def stream(
        self,
        model: Model,
        context: InferenceContext,
        options: StreamOptions,
    ) -> AsyncIterator[AssistantStreamEvent]:
        return self._stream(model, context, options)

    async def _stream(
        self,
        model: Model,
        context: InferenceContext,
        options: StreamOptions,
    ) -> AsyncIterator[AssistantStreamEvent]:
        self.requests.append((model, context, options))
        if not self.responses:
            message = AssistantMessage(
                content=(),
                provider=model.provider,
                model=model.id,
                stop_reason="error",
                error_message="ScriptedProvider has no response left",
            )
            yield AssistantStreamEvent("start")
            yield AssistantStreamEvent("error", message=message)
            return
        response = self.responses.pop(0)
        if isinstance(response, AssistantMessage):
            yield AssistantStreamEvent("start")
            for index, part in enumerate(response.content):
                if isinstance(part, TextContent):
                    yield AssistantStreamEvent("text_delta", delta=part.text, content_index=index)
                elif isinstance(part, ThinkingContent):
                    yield AssistantStreamEvent(
                        "thinking_delta", delta=part.thinking, content_index=index
                    )
                elif isinstance(part, ToolCall):
                    yield AssistantStreamEvent(
                        "toolcall_delta",
                        delta=json.dumps(part.arguments),
                        content_index=index,
                        tool_call_index=index,
                        tool_call_id=part.id,
                        tool_name=part.name,
                    )
            event_type = "error" if response.stop_reason in {"error", "aborted"} else "done"
            yield AssistantStreamEvent(event_type, message=response)
            return
        for event in response:
            yield event

    async def close(self) -> None:
        return None


__all__ = ["ScriptedProvider"]
