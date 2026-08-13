"""Aeloon Cloud runtime Provider backed by an injected account gateway."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from aeloon_core.core import (
    AssistantStreamEvent,
    InferenceContext,
    InferenceError,
    Model,
    StreamOptions,
)
from aeloon_core.runtime.ports import AccountGateway
from aeloon_core.runtime.providers.base import BaseProvider
from aeloon_core.runtime.providers.openai import OpenAICompatibleProvider

CLOUD_PROVIDER_ID = "aeloon-cloud"


class CloudProvider(BaseProvider):
    driver = "cloud"
    kind = "cloud"

    def __init__(
        self,
        account: AccountGateway,
        *,
        name: str = "Aeloon Cloud",
        endpoint: str,
        enabled: bool = True,
        proxy: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            provider_id=CLOUD_PROVIDER_ID,
            name=name,
            endpoint=endpoint,
            enabled=enabled,
        )
        self.account = account
        self.proxy = proxy
        self._client = client or httpx.AsyncClient(proxy=proxy, timeout=None)
        self._owns_client = client is None

    async def models(self) -> dict[str, Model]:
        return self._models_from_values(await self.account.models())

    async def discover_models(self) -> list[Model]:
        return list(self._models_from_values(await self.account.models(force=True)).values())

    @staticmethod
    def _models_from_values(values: list[dict[str, Any]]) -> dict[str, Model]:
        result: dict[str, Model] = {}
        for value in values:
            if not isinstance(value, Mapping):
                continue
            raw_id = (
                str(
                    value.get("model_key")
                    or value.get("modelKey")
                    or value.get("id")
                    or value.get("model")
                    or ""
                )
                .strip()
                .lstrip("/")
            )
            if not raw_id:
                continue
            model_id = f"{CLOUD_PROVIDER_ID}/{raw_id.removeprefix(f'{CLOUD_PROVIDER_ID}/')}"
            context_window = _positive_int(
                value.get("context_window") or value.get("contextWindow"),
                128_000,
            )
            result[model_id] = Model(
                id=model_id,
                name=str(value.get("display_name") or value.get("name") or raw_id),
                provider=CLOUD_PROVIDER_ID,
                reasoning=bool(value.get("reasoning") or value.get("supports_reasoning")),
                input=("text", "image")
                if value.get("supports_image") or value.get("supportsImage")
                else ("text",),
                context_window=context_window,
                max_output_tokens=min(
                    _positive_int(
                        value.get("max_tokens") or value.get("maxTokens"),
                        32_768,
                    ),
                    context_window,
                ),
                cost=dict(value.get("cost") or {}),
            )
        return result

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
        for force in (False, True):
            try:
                token = await self.account.access_token(force=force)
            except Exception as exc:
                raise InferenceError("auth", str(exc), cause=exc) from exc
            inference = OpenAICompatibleProvider(
                provider_id=self.id,
                name=self.name,
                endpoint=self.endpoint,
                api_key=token,
                client=self._client,
                chat_path="/proxy/v1/chat",
                request_model_id=lambda item: item.id.removeprefix(f"{CLOUD_PROVIDER_ID}/"),
                prepare_payload=self._prepare_payload,
                thinking_level_map={"high": "high", "max": "max"},
            )
            try:
                async for event in inference.stream(model, context, options):
                    yield event
                return
            except InferenceError as exc:
                if force or "HTTP 401" not in str(exc) and "HTTP 403" not in str(exc):
                    raise
        raise InferenceError("auth", "Aeloon Cloud authentication failed")

    def status(self) -> dict[str, Any]:
        account = self.account.status()
        return {
            **super().status(),
            "authenticated": bool(account.get("authenticated")),
            "credential_configured": bool(account.get("authenticated")),
            "user": account.get("user"),
        }

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _prepare_payload(payload: dict[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        value.pop("stream_options", None)
        value["task_id"] = f"cloud_{uuid.uuid4().hex}"
        return value


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


__all__ = ["CLOUD_PROVIDER_ID", "CloudProvider"]
