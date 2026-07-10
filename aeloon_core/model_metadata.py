"""Context-window lookup from the public LiteLLM metadata table."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

LITELLM_MODEL_TABLE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_FETCH_TIMEOUT_SECONDS = 10

_litellm_cache: dict[str, Any] | None = None


async def resolve_context_window(
    model: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> int | None:
    """Resolve a model context window from the LiteLLM metadata table."""

    try:
        table = await _litellm_table(http_client=http_client)
    except Exception as exc:
        logger.debug("Model metadata lookup failed via litellm: {}", exc)
        return None
    return litellm_context_window_from_table(table, model)


def litellm_context_window_from_table(table: Any, model: str) -> int | None:
    """Extract one context window from the LiteLLM model table."""

    if not isinstance(table, dict):
        return None

    candidates = _model_candidates(model)
    for exact in (True, False):
        for key, value in table.items():
            matcher = _matches_model_id_exact if exact else _matches_model_id_suffix
            if not matcher(key, candidates) or not isinstance(value, dict):
                continue
            context = _positive_int(value.get("max_input_tokens"))
            if context is not None:
                return context
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
