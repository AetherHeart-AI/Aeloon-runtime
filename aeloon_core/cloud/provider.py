"""Provider adapter that keeps cloud credentials inside Core."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx

from aeloon_core.cloud.account import CLOUD_PROVIDER_ID, CloudAccountService
from aeloon_core.cloud.client import CloudError
from aeloon_core.core import (
    AssistantStreamEvent,
    DeepSeekProvider,
    Model,
    ProviderContext,
    ProviderError,
    StreamOptions,
)


class CloudProvider:
    """OpenAI-compatible cloud proxy backed by the Core account session."""

    def __init__(
        self,
        account: CloudAccountService,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.account = account
        self._client = client or httpx.AsyncClient(proxy=account.config.proxy, timeout=None)
        self._owns_client = client is None

    def stream(
        self, model: Model, context: ProviderContext, options: StreamOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        return self._stream(model, context, options)

    async def _stream(
        self, model: Model, context: ProviderContext, options: StreamOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        for force in (False, True):
            try:
                token = await self.account.access_token(force=force)
            except CloudError as exc:
                raise ProviderError("auth", str(exc), cause=exc) from exc
            provider = DeepSeekProvider(
                api_key=token,
                base_url=self.account.client.base_url,
                client=self._client,
                chat_path="/proxy/v1/chat",
                display_name="Aeloon Cloud",
                request_model_id=lambda item: item.id.removeprefix(f"{CLOUD_PROVIDER_ID}/"),
                prepare_payload=self._prepare_payload,
            )
            try:
                async for event in provider.stream(model, context, options):
                    yield event
                return
            except ProviderError as exc:
                if force or "HTTP 401" not in str(exc) and "HTTP 403" not in str(exc):
                    raise
        raise ProviderError("auth", "Aeloon Cloud authentication failed")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _prepare_payload(payload: dict[str, object]) -> dict[str, object]:
        payload = dict(payload)
        payload.pop("stream_options", None)
        payload["task_id"] = f"cloud_{uuid.uuid4().hex}"
        return payload
