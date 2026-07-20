"""Single model-visible context pipeline.

Canonical history stays intact between explicit compaction checkpoints.  Every
provider-only projection lives here so model rounds do not quietly stack
independent context policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aeloon_core.config import ContextCompactionConfig
from aeloon_core.context_compaction import (
    estimate_request_tokens,
    maybe_compact_messages,
)
from aeloon_core.runtime_support import shrink_answered_tool_args_for_provider


@dataclass(frozen=True)
class ContextView:
    """Canonical checkpoint plus the exact request view sent to a provider."""

    canonical_messages: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    original_tokens: int
    visible_tokens: int
    usage: dict[str, int] = field(default_factory=dict)
    transformations: tuple[str, ...] = ()
    prefix_reset_reason: str | None = None


class ContextViewPipeline:
    """Render one prefix-stable model request from canonical messages."""

    def __init__(
        self,
        *,
        provider: Any,
        model: str,
        compaction: ContextCompactionConfig | None = None,
        context_window_tokens: int = 128_000,
    ) -> None:
        self.provider = provider
        self.model = model
        self.compaction = compaction or ContextCompactionConfig(enabled=False)
        self.context_window_tokens = max(1, int(context_window_tokens))

    async def render(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        additional_messages: list[dict[str, Any]] | None = None,
    ) -> ContextView:
        """Return the only derived view used by a normal model round.

        Rolling/minimal history selection is deliberately absent.  A new Worker
        already receives the minimal dispatch envelope (objective, permissions,
        and budget), while a running Master or Worker keeps an append-only prefix
        until an explicit context-window compaction checkpoint is necessary.
        """

        additional = list(additional_messages or [])
        projected_before_compaction = shrink_answered_tool_args_for_provider(messages)
        original_tokens = estimate_request_tokens(
            [*messages, *additional],
            tools=tools,
            model=self.model,
        )
        canonical = messages
        usage: dict[str, int] = {}
        transformations: list[str] = []
        prefix_reset_reason: str | None = None

        if self.compaction.enabled:
            compacted = await maybe_compact_messages(
                provider=self.provider,
                model=self.model,
                messages=messages,
                visible_messages=projected_before_compaction,
                tools=tools,
                additional_messages=additional,
                config=self.compaction,
                context_window_tokens=self.context_window_tokens,
            )
            canonical = compacted.messages
            usage = dict(compacted.usage)
            if compacted.compacted:
                transformations.append("compaction")
                prefix_reset_reason = "compaction"

        # Oversized answered tool arguments are projected before their first
        # appearance in a subsequent request, so the provider-visible prefix is
        # stable while canonical history retains the complete action.
        provider_messages = shrink_answered_tool_args_for_provider(canonical)
        if provider_messages is not canonical:
            transformations.append("answered_tool_arguments")
        request_messages = [*provider_messages, *additional]
        visible_tokens = estimate_request_tokens(
            request_messages,
            tools=tools,
            model=self.model,
        )
        return ContextView(
            canonical_messages=canonical,
            messages=request_messages,
            tools=list(tools),
            original_tokens=original_tokens,
            visible_tokens=visible_tokens,
            usage=usage,
            transformations=tuple(transformations),
            prefix_reset_reason=prefix_reset_reason,
        )


__all__ = ["ContextView", "ContextViewPipeline"]
