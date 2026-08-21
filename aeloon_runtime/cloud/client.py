"""Small HTTP client for the Aeloon Cloud account and model APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import httpx


class CloudError(RuntimeError):
    """A sanitized Aeloon Cloud request failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_auth_failure(self) -> bool:
        return self.status_code in {401, 403}


@dataclass(frozen=True, slots=True)
class CloudTokenBundle:
    access_token: str
    refresh_token: str | None
    expires_at: datetime
    user: dict[str, Any]


class CloudClient:
    """Typed client for the minimum account-backed provider surface."""

    def __init__(
        self,
        base_url: str,
        *,
        proxy: str | None = None,
        allow_insecure_http: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = self._validated_base_url(base_url, allow_insecure_http)
        self.proxy = proxy
        self._client = client
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def login(
        self,
        *,
        username: str,
        password: str,
        device_id: str,
        device_name: str,
    ) -> CloudTokenBundle:
        payload = await self._post(
            "/auth/v1/login",
            {
                "username": username,
                "password": password,
                "device_id": device_id,
                "device_name": device_name,
            },
        )
        return self._token_bundle(payload, fallback_username=username)

    async def refresh(
        self,
        *,
        refresh_token: str,
        device_id: str,
        device_name: str,
    ) -> CloudTokenBundle:
        payload = await self._post(
            "/auth/v1/refresh",
            {
                "refresh_token": refresh_token,
                "device_id": device_id,
                "device_name": device_name,
            },
        )
        return self._token_bundle(payload)

    async def models(self, access_token: str) -> dict[str, Any]:
        return self._payload_data(await self._get("/proxy/v1/models", access_token))

    async def search(self, access_token: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._payload_data(
            await self._post_authed("/proxy/v1/search", payload, access_token)
        )

    async def _post_authed(
        self, path: str, body: Mapping[str, Any], access_token: str
    ) -> dict[str, Any]:
        client = await self._http_client()
        try:
            response = await client.post(
                f"{self.base_url}{path}",
                json=dict(body),
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {access_token}",
                },
                timeout=60.0,
            )
        except httpx.RequestError as exc:
            raise CloudError(f"Unable to reach Aeloon Cloud: {exc}") from exc
        return self._response(response)

    async def _get(self, path: str, access_token: str) -> dict[str, Any]:
        client = await self._http_client()
        try:
            response = await client.get(
                f"{self.base_url}{path}",
                headers={"authorization": f"Bearer {access_token}"},
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise CloudError(f"Unable to reach Aeloon Cloud: {exc}") from exc
        return self._response(response)

    async def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        client = await self._http_client()
        try:
            response = await client.post(
                f"{self.base_url}{path}",
                json=dict(body),
                headers={"content-type": "application/json"},
                timeout=60.0,
            )
        except httpx.RequestError as exc:
            raise CloudError(f"Unable to reach Aeloon Cloud: {exc}") from exc
        return self._response(response)

    async def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(proxy=self.proxy, follow_redirects=True)
        return self._client

    @staticmethod
    def _validated_base_url(value: str, allow_insecure_http: bool) -> str:
        base_url = value.strip().rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("cloud.base_url must be an HTTP(S) URL")
        if parsed.scheme == "http" and not allow_insecure_http:
            raise ValueError("cloud.base_url must use HTTPS unless insecure HTTP is enabled")
        return base_url

    @classmethod
    def _response(cls, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code >= 400:
            raise CloudError(
                cls._error_message(payload, response.text or f"HTTP {response.status_code}"),
                status_code=response.status_code,
            )
        if not isinstance(payload, dict):
            raise CloudError("Aeloon Cloud returned an invalid response")
        code = payload.get("code")
        if code not in (None, 0, "0"):
            raise CloudError(cls._error_message(payload, f"Aeloon Cloud error {code}"))
        return payload

    @classmethod
    def _token_bundle(
        cls, payload: dict[str, Any], *, fallback_username: str = ""
    ) -> CloudTokenBundle:
        data = cls._payload_data(payload)
        access = data.get("access_token") or data.get("accessToken")
        if not isinstance(access, str) or not access:
            raise CloudError("Aeloon Cloud login response omitted the access token")
        refresh = data.get("refresh_token") or data.get("refreshToken")
        return CloudTokenBundle(
            access_token=access,
            refresh_token=refresh if isinstance(refresh, str) and refresh else None,
            expires_at=cls._expiry(data),
            user=cls._public_user(data, fallback_username=fallback_username),
        )

    @staticmethod
    def _payload_data(payload: dict[str, Any]) -> dict[str, Any]:
        return payload["data"] if isinstance(payload.get("data"), dict) else payload

    @staticmethod
    def _expiry(payload: dict[str, Any]) -> datetime:
        raw = payload.get("expires_at") or payload.get("expiresAt")
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            except ValueError:
                pass
        if isinstance(raw, int | float):
            timestamp = float(raw) / 1000 if raw > 10_000_000_000 else float(raw)
            try:
                return datetime.fromtimestamp(timestamp, tz=UTC)
            except (OSError, OverflowError, ValueError):
                pass
        ttl = payload.get("expires_in") or payload.get("expiresIn") or 300
        try:
            seconds = max(1, int(ttl))
        except (TypeError, ValueError):
            seconds = 300
        return datetime.now(UTC) + timedelta(seconds=seconds)

    @staticmethod
    def _public_user(payload: dict[str, Any], *, fallback_username: str) -> dict[str, Any]:
        raw = payload.get("user") or payload.get("account") or payload.get("profile")
        user = raw if isinstance(raw, dict) else payload
        username = str(user.get("username") or user.get("email") or fallback_username or "user")
        return {
            "id": str(user.get("id") or user.get("account_id") or user.get("uid") or username),
            "username": username,
            "display_name": str(user.get("display_name") or user.get("displayName") or username),
            "avatar_url": user.get("avatar_url") or user.get("avatarUrl"),
            "tier": user.get("tier"),
        }

    @staticmethod
    def _error_message(payload: Any, fallback: str) -> str:
        if isinstance(payload, dict):
            for container in (payload, payload.get("error"), payload.get("data")):
                if isinstance(container, dict):
                    value = (
                        container.get("message")
                        or container.get("detail")
                        or container.get("error")
                    )
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            value = payload.get("message") or payload.get("error")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return fallback.strip()[:1_000] or "Aeloon Cloud request failed"
