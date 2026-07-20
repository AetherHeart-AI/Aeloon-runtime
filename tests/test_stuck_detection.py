"""Focused tests for bounded cross-step stuck detection."""

from __future__ import annotations

import json
from typing import Any

from aeloon_core.stuck_detection import detect_repeated_tool_exchanges

_TOOL_MODES = {"read": "read_only", "write": "mutating"}


def _prompt(text: str = "inspect the repository") -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _append_exchange(
    messages: list[dict[str, Any]],
    *,
    call_id: str,
    tool_name: str = "read",
    arguments: dict[str, Any] | None = None,
    result: str = "same contents",
    thought: str = "I should inspect this.",
    is_error: bool = False,
) -> None:
    messages.extend(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": thought},
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": tool_name,
                        "input": arguments or {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": result,
                        "is_error": is_error,
                    }
                ],
            },
        ]
    )


def test_detects_at_four_but_not_three_and_ignores_ids_and_thoughts() -> None:
    messages = _prompt()
    for index in range(3):
        _append_exchange(
            messages,
            call_id=f"call-{index}",
            thought=f"Different wording {index}",
        )

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None

    _append_exchange(
        messages,
        call_id="call-3",
        thought="A fourth, differently worded thought",
    )
    detection = detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES)

    assert detection is not None
    assert detection.pattern == "repeated_action_observation"
    assert detection.repetitions == 4
    assert detection.threshold == 4
    assert detection.distinct_steps == 4
    assert detection.tool_name == "read"
    assert len(detection.action_digest) == 64
    assert len(detection.observation_digest) == 64


def test_new_real_user_prompt_resets_the_detection_window() -> None:
    messages = _prompt("first assignment")
    for index in range(4):
        _append_exchange(messages, call_id=f"old-{index}")
    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES)

    messages.append({"role": "user", "content": "new assignment"})
    _append_exchange(messages, call_id="new-0")

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_changed_arguments_break_the_repeated_tail() -> None:
    messages = _prompt()
    for index in range(3):
        _append_exchange(messages, call_id=f"same-{index}")
    _append_exchange(
        messages,
        call_id="changed",
        arguments={"path": "pyproject.toml"},
    )

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_changed_result_breaks_the_repeated_tail() -> None:
    messages = _prompt()
    for index in range(3):
        _append_exchange(messages, call_id=f"same-{index}")
    _append_exchange(messages, call_id="changed", result="new contents")

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_mutating_exchanges_never_match() -> None:
    messages = _prompt()
    for index in range(4):
        _append_exchange(
            messages,
            call_id=f"write-{index}",
            tool_name="write",
            arguments={"path": "notes.txt", "content": "x"},
            result="written",
        )

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_repeated_calls_in_one_batch_are_not_cross_step_stuck() -> None:
    messages = _prompt()
    tool_uses = [
        {
            "type": "tool_use",
            "id": f"batch-{index}",
            "name": "read",
            "input": {"path": "README.md"},
        }
        for index in range(4)
    ]
    tool_results = [
        {
            "type": "tool_result",
            "tool_use_id": f"batch-{index}",
            "content": "same contents",
            "is_error": False,
        }
        for index in range(4)
    ]
    messages.extend(
        [
            {"role": "assistant", "content": tool_uses},
            {"role": "user", "content": tool_results},
        ]
    )

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_error_exchanges_never_match_or_preserve_an_older_match() -> None:
    messages = _prompt()
    for index in range(4):
        _append_exchange(messages, call_id=f"ok-{index}")
    _append_exchange(
        messages,
        call_id="error",
        result="Error [READ_FAILED]: unavailable",
    )

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_incomplete_tool_uses_do_not_contribute_to_the_threshold() -> None:
    messages = _prompt()
    for index in range(3):
        _append_exchange(messages, call_id=f"complete-{index}")
    for index in range(4):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"incomplete-{index}",
                        "name": "read",
                        "input": {"path": "README.md"},
                    }
                ],
            }
        )

    assert detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES) is None


def test_large_outputs_produce_small_digest_only_evidence() -> None:
    result = "x" * 1_000_000
    messages = _prompt()
    for index in range(4):
        _append_exchange(messages, call_id=f"large-{index}", result=result)

    detection = detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES)
    assert detection is not None
    payload = detection.to_payload()
    serialized = json.dumps(payload, sort_keys=True)

    assert len(serialized) < 1_000
    assert result[:1_000] not in serialized
    assert payload["observation_chars"] == len(result)
    assert len(payload["exchange_digest"]) == 64


def test_only_the_latest_twenty_complete_exchanges_are_reported() -> None:
    messages = _prompt()
    for index in range(25):
        _append_exchange(messages, call_id=f"call-{index}")

    detection = detect_repeated_tool_exchanges(messages, tool_modes=_TOOL_MODES)

    assert detection is not None
    assert detection.scanned_exchanges == 20
    assert detection.repetitions == 20
