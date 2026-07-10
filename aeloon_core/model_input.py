"""Typed boundary for canonical model-input preparation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, TypeAlias, runtime_checkable

from aeloon_core.transitions import normalize_usage

Message: TypeAlias = dict[str, Any]


class PreparedModelInput(Protocol):
    """A rich preparation result that preserves canonical messages and usage."""

    messages: list[Message]
    usage: Mapping[str, Any]


@runtime_checkable
class TokenCountedPreparedModelInput(PreparedModelInput, Protocol):
    """A preparation result that already measured its visible request tokens."""

    compacted_tokens: int


PreparedModelInputResult: TypeAlias = list[Message] | PreparedModelInput
PrepareModelInput: TypeAlias = Callable[
    [list[Message], list[Message], list[Message]],
    Awaitable[PreparedModelInputResult],
]


def unpack_prepared_model_input(
    prepared: PreparedModelInputResult,
) -> tuple[list[Message], dict[str, int], int | None]:
    """Validate and unpack either the legacy list or rich result shape."""

    if isinstance(prepared, list):
        return prepared, {}, None
    messages = prepared.messages
    if not isinstance(messages, list):
        raise TypeError("prepared model input messages must be a list")
    measured_tokens = (
        max(0, prepared.compacted_tokens)
        if isinstance(prepared, TokenCountedPreparedModelInput)
        else None
    )
    return messages, normalize_usage(prepared.usage), measured_tokens
