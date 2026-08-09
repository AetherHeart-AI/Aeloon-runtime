"""Custom OpenAI-compatible Provider with model and image-capability discovery."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from aeloon_core.core import InferenceError, Model
from aeloon_core.runtime.providers.openai import OpenAICompatibleProvider

MODEL_DISCOVERY_TIMEOUT = 15
CAPABILITY_PROBE_TIMEOUT = 15
CAPABILITY_PROBE_CONCURRENCY = 4
_PROBE_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAATUlEQVR42u3PQQ0AAAgEILV/"
    "5zOFDzdoQCepz6aeExAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "ELi3cqoDfaKuZM4AAAAASUVORK5CYII="
)
_IMAGE_TERMS = ("image", "vision", "multimodal", "multi-modal")


@dataclass(frozen=True, slots=True)
class _DiscoveredModel:
    model: Model
    supports_image: bool | None


class CustomProvider(OpenAICompatibleProvider):
    """A URL-configured OpenAI-compatible Provider."""

    driver = "custom"

    def __init__(self, *, backend: str = "openai", endpoint: str, **kwargs: Any) -> None:
        if backend not in {"openai", "llamacpp", "ollama", "vllm"}:
            raise ValueError(f"Unsupported custom Provider backend: {backend}")
        self.backend = backend
        self.discovery_endpoint = endpoint.rstrip("/")
        super().__init__(endpoint=_inference_endpoint(backend, endpoint), **kwargs)

    async def _discover_models(self) -> list[Model]:
        headers = dict(self.headers)
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        client = await self._get_client()
        if self.backend == "ollama":
            return await self._discover_ollama(client, headers)
        last_error: Exception | None = None
        for endpoint in _endpoint_candidates(self.discovery_endpoint):
            try:
                response = await client.get(
                    f"{endpoint}/models",
                    headers=headers,
                    timeout=MODEL_DISCOVERY_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
                if self.backend == "llamacpp":
                    payload = await self._llamacpp_payload(client, endpoint, payload, headers)
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                continue
            discovered = _models_from_payload(payload, self.id)
            if not discovered:
                last_error = ValueError("the model list contained no usable models")
                continue
            self.endpoint = _inference_endpoint(self.backend, endpoint)
            return await self._resolve_image_capabilities(discovered, headers)
        detail = f": {self._sanitize(str(last_error))}" if last_error is not None else ""
        sanitized_cause = (
            ValueError(self._sanitize(str(last_error))) if last_error is not None else None
        )
        raise InferenceError(
            "model_discovery",
            f"Could not load models from {self.name}{detail}",
            cause=sanitized_cause,
        )

    async def _discover_ollama(
        self,
        client: httpx.AsyncClient,
        headers: Mapping[str, str],
    ) -> list[Model]:
        root = _without_v1(self.discovery_endpoint)
        try:
            response = await client.get(
                f"{root}/api/tags", headers=headers, timeout=MODEL_DISCOVERY_TIMEOUT
            )
            response.raise_for_status()
            payload = response.json()
            raw_models = payload.get("models") if isinstance(payload, Mapping) else None
            if not isinstance(raw_models, list):
                raise ValueError("the Ollama model list was invalid")
            values: list[dict[str, Any]] = []
            for raw in raw_models:
                if not isinstance(raw, Mapping):
                    continue
                model_id = str(raw.get("model") or raw.get("name") or "").strip()
                if not model_id:
                    continue
                detail_response = await client.post(
                    f"{root}/api/show",
                    headers=headers,
                    json={"model": model_id, "verbose": False},
                    timeout=MODEL_DISCOVERY_TIMEOUT,
                )
                detail_response.raise_for_status()
                detail = detail_response.json()
                detail_value = detail if isinstance(detail, Mapping) else {}
                capabilities = detail_value.get("capabilities")
                model_info = detail_value.get("model_info")
                values.append(
                    {
                        **dict(raw),
                        "id": model_id,
                        "capabilities": capabilities,
                        "context_window": _ollama_context_window(model_info),
                        "reasoning": _sequence_mentions(capabilities, ("thinking", "reasoning")),
                    }
                )
        except (httpx.HTTPError, ValueError) as exc:
            sanitized = self._sanitize(str(exc))
            raise InferenceError(
                "model_discovery",
                f"Could not load models from {self.name}: {sanitized}",
                cause=ValueError(sanitized),
            ) from None
        discovered = _models_from_payload({"models": values}, self.id)
        if not discovered:
            raise InferenceError(
                "model_discovery", f"Could not load models from {self.name}: no usable models"
            )
        self.endpoint = f"{root}/v1"
        return await self._resolve_image_capabilities(discovered, headers)

    async def _llamacpp_payload(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: Any,
        headers: Mapping[str, str],
    ) -> Any:
        if not isinstance(payload, Mapping):
            return payload
        raw_values = payload.get("data")
        if not isinstance(raw_values, list):
            raw_values = payload.get("models")
        if not isinstance(raw_values, list):
            return payload
        root = _without_v1(endpoint)
        values: list[Any] = []
        for raw in raw_values:
            if not isinstance(raw, Mapping):
                values.append(raw)
                continue
            model_id = str(raw.get("id") or raw.get("model") or raw.get("name") or "")
            try:
                response = await client.get(
                    f"{root}/props",
                    headers=headers,
                    params={"model": model_id} if model_id else None,
                    timeout=MODEL_DISCOVERY_TIMEOUT,
                )
                response.raise_for_status()
                props = response.json()
            except (httpx.HTTPError, ValueError):
                props = {}
            settings = (
                props.get("default_generation_settings") if isinstance(props, Mapping) else {}
            )
            values.append(
                {
                    **dict(raw),
                    "context_window": (
                        settings.get("n_ctx") if isinstance(settings, Mapping) else None
                    ),
                    "modalities": props.get("modalities") if isinstance(props, Mapping) else None,
                    "chat_template_caps": (
                        props.get("chat_template_caps") if isinstance(props, Mapping) else None
                    ),
                }
            )
        return {"data": values}

    async def _resolve_image_capabilities(
        self,
        discovered: list[_DiscoveredModel],
        headers: Mapping[str, str],
    ) -> list[Model]:
        semaphore = asyncio.Semaphore(CAPABILITY_PROBE_CONCURRENCY)

        async def resolve(item: _DiscoveredModel) -> Model:
            supports_image = item.supports_image
            if supports_image is None:
                async with semaphore:
                    supports_image = await self._probe_image_support(item.model, headers)
            return replace(
                item.model,
                input=("text", "image") if supports_image else ("text",),
            )

        return list(await asyncio.gather(*(resolve(item) for item in discovered)))

    async def _probe_image_support(
        self,
        model: Model,
        headers: Mapping[str, str],
    ) -> bool:
        payload = {
            "model": self.request_model_id(model),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with OK."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PROBE_IMAGE}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "max_tokens": 1,
        }
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.endpoint}{self.chat_path}",
                json=payload,
                headers=headers,
                timeout=CAPABILITY_PROBE_TIMEOUT,
            )
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError):
            return False
        return isinstance(value, Mapping) and value.get("error") is None


def _endpoint_candidates(endpoint: str) -> tuple[str, ...]:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        return (base,)
    return (base, f"{base}/v1")


def _without_v1(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _inference_endpoint(backend: str, endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        return base
    if backend in {"openai", "llamacpp", "ollama", "vllm"}:
        return f"{base}/v1"
    return base


def _models_from_payload(payload: Any, provider_id: str) -> list[_DiscoveredModel]:
    if not isinstance(payload, Mapping):
        return []
    values = payload.get("data")
    if not isinstance(values, list):
        values = payload.get("models")
    if not isinstance(values, list):
        return []

    models: list[_DiscoveredModel] = []
    seen: set[str] = set()
    prefix = f"{provider_id}/"
    for raw in values:
        value: Mapping[str, Any]
        if isinstance(raw, str):
            value = {"id": raw}
        elif isinstance(raw, Mapping):
            value = raw
        else:
            continue
        raw_id = str(
            value.get("id")
            or value.get("model")
            or value.get("model_key")
            or value.get("name")
            or ""
        ).strip().lstrip("/")
        if not raw_id:
            continue
        local_id = raw_id.removeprefix(prefix)
        model_id = f"{prefix}{local_id}"
        if model_id in seen:
            continue
        seen.add(model_id)
        context_window = _positive_int(
            _first(value, "context_window", "contextWindow", "max_model_len"),
            128_000,
        )
        max_tokens = min(
            _positive_int(_first(value, "max_tokens", "maxTokens"), 32_768),
            context_window,
        )
        reasoning = _first_bool(
            value,
            "reasoning",
            "supports_reasoning",
            "supportsReasoning",
        )
        if reasoning is None:
            reasoning = _sequence_mentions(
                value.get("capabilities"), ("thinking", "reasoning")
            )
        cost = value.get("cost")
        model = Model(
            id=model_id,
            name=str(
                value.get("display_name")
                or value.get("displayName")
                or value.get("name")
                or raw_id
            ),
            provider=provider_id,
            reasoning=bool(reasoning),
            context_window=context_window,
            max_tokens=max_tokens,
            cost=dict(cost) if isinstance(cost, Mapping) else {},
        )
        models.append(_DiscoveredModel(model, _image_capability(value)))
    return models


def _image_capability(value: Mapping[str, Any]) -> bool | None:
    sources = [value]
    for key in ("architecture", "meta"):
        source = value.get(key)
        if isinstance(source, Mapping):
            sources.append(source)
    for source in sources:
        direct = _first_bool(
            source,
            "supports_image",
            "supportsImage",
            "supports_vision",
            "supportsVision",
            "allow_image",
            "allowImage",
            "vision",
            "multimodal",
        )
        if direct is not None:
            return direct
        for key in (
            "input_modalities",
            "inputModalities",
            "input",
            "modalities",
            "modality",
        ):
            if key in source:
                detected = _contains_image(source[key])
                if detected is not None:
                    return detected
        if "capabilities" in source:
            detected = _capabilities_include_image(source["capabilities"])
            if detected is not None:
                return detected
    return None


def _capabilities_include_image(value: Any) -> bool | None:
    if isinstance(value, Mapping):
        recognized: list[bool] = []
        for key, enabled in value.items():
            if _mentions_image(str(key)):
                parsed = _bool_value(enabled)
                recognized.append(True if parsed is None else parsed)
        return any(recognized) if recognized else False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_mentions_image(str(item)) for item in value)
    if isinstance(value, str):
        return _mentions_image(value)
    return None


def _contains_image(value: Any) -> bool | None:
    if isinstance(value, Mapping):
        for key in ("input", "inputs", "input_modalities", "inputModalities"):
            if key in value:
                return _contains_image(value[key])
        recognized = [
            True if (parsed := _bool_value(enabled)) is None else parsed
            for key, enabled in value.items()
            if _mentions_image(str(key))
        ]
        return any(recognized) if recognized else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_mentions_image(str(item)) for item in value)
    if isinstance(value, str):
        return _mentions_image(value)
    return None


def _mentions_image(value: str) -> bool:
    normalized = value.strip().lower().replace("_", "-")
    return any(term in normalized for term in _IMAGE_TERMS)


def _first(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _ollama_context_window(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    for key, item in value.items():
        if str(key).endswith(".context_length"):
            parsed = _positive_int(item, 0)
            if parsed:
                return parsed
    return None


def _sequence_mentions(value: Any, terms: Sequence[str]) -> bool | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    normalized = [str(item).strip().lower() for item in value]
    return any(any(term in item for term in terms) for item in normalized)


__all__ = ["CustomProvider"]
