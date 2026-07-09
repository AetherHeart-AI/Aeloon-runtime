"""Model limit lookup from the public LiteLLM metadata table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

GENERIC_DEFAULT_MAX_TOKENS = 4096
LITELLM_MODEL_TABLE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_FETCH_TIMEOUT_SECONDS = 10

_litellm_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelLimits:
    """Provider/model token limits from a metadata table."""

    output_tokens: int | None
    context_tokens: int | None
    source: str


async def resolve_max_tokens_for_model(
    model: str,
    configured_max_tokens: int | None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> int:
    """Return max_tokens, using the LiteLLM table when set to auto."""

    if configured_max_tokens is not None:
        return max(1, configured_max_tokens)

    limits = await resolve_model_limits(model, http_client=http_client)
    if limits is not None and limits.output_tokens is not None:
        return max(1, limits.output_tokens)
    return GENERIC_DEFAULT_MAX_TOKENS


async def resolve_model_limits(
    model: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> ModelLimits | None:
    """Resolve model limits from the LiteLLM metadata table."""

    try:
        table = await _litellm_table(http_client=http_client)
    except Exception as exc:
        logger.debug("Model metadata lookup failed via litellm: {}", exc)
        return None
    return litellm_limits_from_table(table, model)


def litellm_limits_from_table(table: Any, model: str) -> ModelLimits | None:
    """Extract LiteLLM model table limits for one model."""

    if not isinstance(table, dict):
        return None

    candidates = _model_candidates(model)
    for exact in (True, False):
        for key, value in table.items():
            matcher = _matches_model_id_exact if exact else _matches_model_id_suffix
            if not matcher(key, candidates) or not isinstance(value, dict):
                continue
            output = _positive_int(value.get("max_output_tokens")) or _positive_int(
                value.get("max_tokens")
            )
            context = _positive_int(value.get("max_input_tokens"))
            return ModelLimits(output_tokens=output, context_tokens=context, source="litellm")
    return None


async def _litellm_table(
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    global _litellm_cache
    if _litellm_cache is None:
        _litellm_cache = await _fetch_json(LITELLM_MODEL_TABLE_URL, http_client=http_client)
    return _litellm_cache


async def _fetch_json(
    url: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    if http_client is not None:
        return await _get_json(http_client, url)
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client:
        return await _get_json(client, url)


async def _get_json(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    response = await client.get(url, timeout=_FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _model_candidates(model: str) -> set[str]:
    normalized = _normalize_model_id(model)
    candidates = {normalized}
    if "/" in normalized:
        candidates.add(normalized.split("/", 1)[1])
        candidates.add(normalized.rsplit("/", 1)[1])
    if "deepseek-v4" in normalized and not normalized.startswith("deepseek/"):
        candidates.add(f"deepseek/{normalized}")
    if "deepseek-v4flash" in normalized:
        candidates.add(normalized.replace("deepseek-v4flash", "deepseek-v4-flash"))
    if "deepseek-v4pro" in normalized:
        candidates.add(normalized.replace("deepseek-v4pro", "deepseek-v4-pro"))
    return {candidate for candidate in candidates if candidate}


def _matches_model_id_exact(value: Any, candidates: set[str]) -> bool:
    if not isinstance(value, str):
        return False
    normalized = _normalize_model_id(value)
    return normalized in candidates


def _matches_model_id_suffix(value: Any, candidates: set[str]) -> bool:
    if not isinstance(value, str):
        return False
    normalized = _normalize_model_id(value)
    return any(normalized.endswith(f"/{candidate}") for candidate in candidates)


def _normalize_model_id(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
