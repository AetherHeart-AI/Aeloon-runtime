"""Shared discovery-time filters for Agent-capable models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_EXCLUDED_MODEL_TERMS = (
    "embed",
    "bge-",
    "gte-",
    "m3e",
    "rerank",
    "seedream",
    "dall-e",
    "dalle",
    "stable-diffusion",
    "sdxl",
    "sd3",
    "midjourney",
    "kolors",
    "cogview",
    "wanx-image",
    "flux",
    "imagen",
    "image-01",
    "seedance",
    "cogvideo",
    "wanx-video",
    "hailuo",
    "kling",
    "video-01",
    "text-to-video",
    "tts",
    "text-to-speech",
    "speech-to-text",
    "whisper",
    "asr",
    "cosyvoice",
    "sambert",
    "paraformer",
    "moderation",
    "guard",
)
_TOOL_CALL_TERMS = ("tool", "function-call")
_TOOL_BOOLEAN_KEYS = (
    "supports_tools",
    "supportsTools",
    "supports_tool_calls",
    "supportsToolCalls",
    "tool_calling",
    "toolCalling",
    "tool_calls",
    "toolCalls",
    "function_calling",
    "functionCalling",
    "supports_function_calling",
    "supportsFunctionCalling",
)


def is_excluded_model_name(*names: str) -> bool:
    """Return whether any supplied model name identifies a non-Agent model family."""
    return any(term in _normalize(name) for name in names for term in _EXCLUDED_MODEL_TERMS)


def supports_tool_calls(capabilities: Any) -> bool | None:
    """Read tool-call support from an authoritative capability enumeration."""
    if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes, bytearray)):
        return None
    if not capabilities:
        return None
    return any(_mentions_tool_calls(str(capability)) for capability in capabilities)


def is_agent_capable_model(
    model_id: str,
    name: str,
    value: Mapping[str, Any],
) -> bool:
    """Return whether discovery metadata permits using a model as an Agent model."""
    if is_excluded_model_name(model_id, name):
        return False

    direct = _first_bool(value, *_TOOL_BOOLEAN_KEYS)
    if direct is False:
        return False

    chat_template_caps = value.get("chat_template_caps")
    if isinstance(chat_template_caps, Mapping):
        tools = _first_bool(chat_template_caps, "tools")
        if tools is False:
            return False

    for key in ("capabilities", "supported_parameters"):
        detected = supports_tool_calls(value.get(key))
        if detected is False:
            return False

    return True


def _mentions_tool_calls(value: str) -> bool:
    normalized = _normalize(value)
    return any(term in normalized for term in _TOOL_CALL_TERMS)


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _first_bool(value: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in value:
            parsed = _bool_value(value[key])
            if parsed is not None:
                return parsed
    return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


__all__ = [
    "is_agent_capable_model",
    "is_excluded_model_name",
    "supports_tool_calls",
]
