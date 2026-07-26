"""Bounded, deterministic detection of repeated Harness tool exchanges."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

_MAX_EXCHANGES = 20
_DEFAULT_THRESHOLD = 4
_DIGEST_CHARS = 64

StuckPattern = Literal["repeated_action_observation"]


@dataclass(frozen=True)
class StuckDetection:
    """Small, safe evidence for one deterministic stuck-loop match."""

    pattern: StuckPattern
    repetitions: int
    threshold: int
    scanned_exchanges: int
    distinct_steps: int
    tool_name: str
    action_digest: str
    observation_digest: str
    exchange_digest: str
    argument_chars: int
    observation_chars: int

    def to_payload(self) -> dict[str, Any]:
        """Return bounded telemetry without raw arguments or results."""

        return {
            "pattern": self.pattern,
            "repetitions": max(0, int(self.repetitions)),
            "threshold": max(0, int(self.threshold)),
            "scanned_exchanges": min(
                _MAX_EXCHANGES,
                max(0, int(self.scanned_exchanges)),
            ),
            "distinct_steps": max(0, int(self.distinct_steps)),
            "tool_name": self.tool_name[:120],
            "action_digest": self.action_digest[:_DIGEST_CHARS],
            "observation_digest": self.observation_digest[:_DIGEST_CHARS],
            "exchange_digest": self.exchange_digest[:_DIGEST_CHARS],
            "argument_chars": max(0, int(self.argument_chars)),
            "observation_chars": max(0, int(self.observation_chars)),
        }


@dataclass(frozen=True)
class _ToolExchange:
    step: int
    tool_name: str
    action_digest: str
    observation_digest: str
    exchange_digest: str
    argument_chars: int
    observation_chars: int
    successful_read_only: bool


def detect_repeated_tool_exchanges(
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_modes: Mapping[str, str],
    threshold: int = _DEFAULT_THRESHOLD,
) -> StuckDetection | None:
    """Detect an identical successful read-only action/result tail.

    Only the latest 20 complete tool exchanges after the last real user prompt
    are retained. Tool call IDs and assistant thought text are deliberately not
    part of equality. Errors, mutating calls, and incomplete protocol pairs can
    never contribute to a match.
    """

    resolved_threshold = int(threshold)
    if not 2 <= resolved_threshold <= _MAX_EXCHANGES:
        raise ValueError(f"threshold must be between 2 and {_MAX_EXCHANGES}")

    exchanges = _recent_complete_exchanges(
        messages,
        tool_modes=tool_modes,
        limit=_MAX_EXCHANGES,
    )
    if len(exchanges) < resolved_threshold:
        return None

    latest = exchanges[-1]
    if not latest.successful_read_only:
        return None

    repeated: list[_ToolExchange] = []
    for exchange in reversed(exchanges):
        if not exchange.successful_read_only or exchange.exchange_digest != latest.exchange_digest:
            break
        repeated.append(exchange)

    # The detector is intentionally cross-step: a model emitting the same call
    # several times in one batch does not by itself prove a loop.
    distinct_steps = len({exchange.step for exchange in repeated})
    if len(repeated) < resolved_threshold or distinct_steps < resolved_threshold:
        return None

    return StuckDetection(
        pattern="repeated_action_observation",
        repetitions=len(repeated),
        threshold=resolved_threshold,
        scanned_exchanges=len(exchanges),
        distinct_steps=distinct_steps,
        tool_name=latest.tool_name,
        action_digest=latest.action_digest,
        observation_digest=latest.observation_digest,
        exchange_digest=latest.exchange_digest,
        argument_chars=latest.argument_chars,
        observation_chars=latest.observation_chars,
    )


def _recent_complete_exchanges(
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_modes: Mapping[str, str],
    limit: int,
) -> list[_ToolExchange]:
    start = _current_assignment_start(messages)
    pending_results: dict[str, Mapping[str, Any]] = {}
    newest_first: list[_ToolExchange] = []

    # Walk backwards so a long assignment never materializes more than the
    # configured evidence window of complete exchanges.
    for message_index in range(len(messages) - 1, start - 1, -1):
        message = messages[message_index]
        content = message.get("content")
        if not isinstance(content, list):
            continue

        if message.get("role") == "user":
            for block in reversed(content):
                if not isinstance(block, Mapping) or block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                if isinstance(call_id, str) and call_id:
                    pending_results.setdefault(call_id, block)
            continue

        if message.get("role") != "assistant":
            continue
        for block in reversed(content):
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                continue
            result = pending_results.pop(call_id, None)
            if result is None:
                continue
            newest_first.append(
                _build_exchange(
                    step=message_index,
                    action=block,
                    result=result,
                    tool_modes=tool_modes,
                )
            )
            if len(newest_first) >= limit:
                return list(reversed(newest_first))

    return list(reversed(newest_first))


def _current_assignment_start(messages: Sequence[Mapping[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if _is_real_user_prompt(messages[index]):
            return index + 1
    return 0


def _is_real_user_prompt(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return True
    return not any(
        isinstance(block, Mapping) and block.get("type") == "tool_result" for block in content
    )


def _build_exchange(
    *,
    step: int,
    action: Mapping[str, Any],
    result: Mapping[str, Any],
    tool_modes: Mapping[str, str],
) -> _ToolExchange:
    raw_name = action.get("name")
    tool_name = raw_name if isinstance(raw_name, str) else ""
    arguments = action.get("input")
    observation = result.get("content")
    serialized_arguments = _canonical_json(arguments)
    serialized_observation = _canonical_observation(observation)
    action_digest = _digest(
        _canonical_json(
            {
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
    )
    observation_digest = _digest(serialized_observation)
    exchange_digest = _digest(f"{action_digest}:{observation_digest}")
    return _ToolExchange(
        step=step,
        tool_name=tool_name,
        action_digest=action_digest,
        observation_digest=observation_digest,
        exchange_digest=exchange_digest,
        argument_chars=len(serialized_arguments),
        observation_chars=len(serialized_observation),
        successful_read_only=(
            bool(tool_name)
            and tool_modes.get(tool_name, "exclusive") == "read_only"
            and not _tool_result_failed(result)
        ),
    )


def _tool_result_failed(result: Mapping[str, Any]) -> bool:
    if result.get("is_error") is True:
        return True
    content = result.get("content")
    text = content if isinstance(content, str) else _canonical_json(content)
    normalized = text.lstrip().lower()
    return normalized.startswith("error") or normalized.startswith("skipped duplicate call")


def _canonical_observation(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return _canonical_json(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


__all__ = ["StuckDetection", "detect_repeated_tool_exchanges"]
