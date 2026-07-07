"""Model limit lookup from public metadata tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

GENERIC_DEFAULT_MAX_TOKENS = 4096
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
LITELLM_MODEL_TABLE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_FETCH_TIMEOUT_SECONDS = 10

_openrouter_cache: dict[str, Any] | None = None
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
    api_base: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> int:
    """Return max_tokens, using OpenRouter/LiteLLM tables when set to auto."""

    if configured_max_tokens is not None:
        return max(1, configured_max_tokens)

    limits = await resolve_model_limits(model, api_base=api_base, http_client=http_client)
    if limits is not None and limits.output_tokens is not None:
        return max(1, limits.output_tokens)
    return GENERIC_DEFAULT_MAX_TOKENS


async def resolve_model_limits(
    model: str,
    *,
    api_base: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ModelLimits | None:
    """Resolve model limits from OpenRouter and LiteLLM metadata tables."""

    sources = (
        ("openrouter", _lookup_openrouter_limits),
        ("litellm", _lookup_litellm_limits),
    )
    if not _prefers_openrouter(api_base):
        sources = tuple(reversed(sources))

    for source, lookup in sources:
        try:
            limits = await lookup(model, http_client=http_client)
        except Exception as exc:
            logger.debug("Model metadata lookup failed via {}: {}", source, exc)
            continue
        if limits is not None and limits.output_tokens is not None:
            return limits
    return None


def openrouter_limits_from_payload(payload: Any, model: str) -> ModelLimits | None:
    """Extract OpenRouter top-provider limits for one model."""

    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return None

    candidates = _model_candidates(model)
    for exact in (True, False):
        item = _matching_openrouter_item(data, candidates, exact=exact)
        if item is None:
            continue
        top_provider = item.get("top_provider")
        if not isinstance(top_provider, dict):
            top_provider = {}
        output = _positive_int(top_provider.get("max_completion_tokens"))
        context = _positive_int(top_provider.get("context_length")) or _positive_int(
            item.get("context_length")
        )
        return ModelLimits(output_tokens=output, context_tokens=context, source="openrouter")
    return None


def _matching_openrouter_item(
    data: list[Any],
    candidates: set[str],
    *,
    exact: bool,
) -> dict[str, Any] | None:
    for item in data:
        if not isinstance(item, dict):
            continue
        ids = [
            item.get("id"),
            item.get("canonical_slug"),
            item.get("hugging_face_id"),
        ]
        matcher = _matches_model_id_exact if exact else _matches_model_id_suffix
        if any(matcher(value, candidates) for value in ids):
            return item
    return None


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


async def _lookup_openrouter_limits(
    model: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> ModelLimits | None:
    payload = await _openrouter_payload(http_client=http_client)
    return openrouter_limits_from_payload(payload, model)


async def _lookup_litellm_limits(
    model: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> ModelLimits | None:
    table = await _litellm_table(http_client=http_client)
    return litellm_limits_from_table(table, model)


async def _openrouter_payload(
    *,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    global _openrouter_cache
    if _openrouter_cache is None:
        _openrouter_cache = await _fetch_json(OPENROUTER_MODELS_URL, http_client=http_client)
    return _openrouter_cache


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
        response = await http_client.get(url, timeout=_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}


def _prefers_openrouter(api_base: str | None) -> bool:
    if not api_base:
        return False
    host = urlparse(api_base).netloc.lower()
    return "openrouter.ai" in host


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


def _matches_model_id(value: Any, candidates: set[str]) -> bool:
    return _matches_model_id_exact(value, candidates) or _matches_model_id_suffix(
        value, candidates
    )


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
