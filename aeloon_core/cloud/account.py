"""Aeloon Cloud account state and dynamic model catalog."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeloon_core.cloud.client import CloudClient, CloudError, CloudTokenBundle
from aeloon_core.cloud.config import CloudConfig
from aeloon_core.cloud.vault import TokenVault, default_token_vault

CATALOG_TTL_SECONDS = 3_600


@dataclass(frozen=True, slots=True)
class AccountState:
    user: dict[str, Any]
    base_url: str
    device_id: str


class CloudAccountService:
    """The single owner of cloud identity, refresh credentials, and catalog state."""

    def __init__(
        self,
        config: CloudConfig,
        *,
        data_dir: Path,
        client: CloudClient | None = None,
        vault: TokenVault | None = None,
    ) -> None:
        self.config = config
        self.data_dir = data_dir.expanduser().resolve(strict=False)
        self.state_path = self.data_dir / "cloud-account.json"
        self.client = client or CloudClient(
            config.base_url,
            proxy=config.proxy,
            allow_insecure_http=config.allow_insecure_http,
        )
        account = hashlib.sha256(f"{self.data_dir}|{self.client.base_url}".encode()).hexdigest()
        self.vault = vault or default_token_vault(self.data_dir, account)
        self._access_token: str | None = None
        self._access_expires_at: datetime | None = None
        self._lock = asyncio.Lock()
        self._catalog: dict[str, Any] | None = None
        self._catalog_at = 0.0

    @property
    def device_id(self) -> str:
        return hashlib.sha256(str(self.data_dir).encode()).hexdigest()[:32]

    async def login(self, *, username: str, password: str) -> dict[str, Any]:
        if not self.config.enabled:
            raise CloudError("Aeloon Cloud is disabled in Core settings")
        if not username.strip() or not password:
            raise CloudError("Username and password are required")
        bundle = await self.client.login(
            username=username.strip(),
            password=password,
            device_id=self.device_id,
            device_name=self.config.device_name,
        )
        await self._accept(bundle)
        return self.status()

    async def access_token(self, force: bool = False) -> str:
        now = datetime.now(UTC)
        if not force and self._access_token and self._access_expires_at:
            if (self._access_expires_at - now).total_seconds() > 60:
                return self._access_token
        async with self._lock:
            now = datetime.now(UTC)
            if not force and self._access_token and self._access_expires_at:
                if (self._access_expires_at - now).total_seconds() > 60:
                    return self._access_token
            state = self._load_state()
            refresh_token = await asyncio.to_thread(self.vault.load)
            if state is None or not refresh_token:
                raise CloudError(
                    "Sign in to Aeloon Cloud before using cloud models",
                    status_code=401,
                )
            bundle = await self.client.refresh(
                refresh_token=refresh_token,
                device_id=state.device_id,
                device_name=self.config.device_name,
            )
            await self._accept(bundle, fallback_user=state.user)
            if not self._access_token:
                raise CloudError("Aeloon Cloud token refresh failed", status_code=401)
            return self._access_token

    async def models(self, *, force: bool = False) -> list[dict[str, Any]]:
        if not force and self._catalog is not None:
            if time.monotonic() - self._catalog_at < CATALOG_TTL_SECONDS:
                return self._models_from_payload(self._catalog)
        token = await self.access_token()
        try:
            payload = await self.client.models(token)
        except CloudError as exc:
            if not exc.is_auth_failure:
                raise
            token = await self.access_token(force=True)
            payload = await self.client.models(token)
        self._catalog = payload
        self._catalog_at = time.monotonic()
        return self._models_from_payload(payload)

    def status(self) -> dict[str, Any]:
        state = self._load_state()
        try:
            credential_available = self.vault.load() is not None
        except Exception:
            credential_available = False
        authenticated = state is not None and credential_available
        return {
            "enabled": self.config.enabled,
            "authenticated": authenticated,
            "user": state.user if authenticated and state is not None else None,
            "base_url": self.client.base_url,
            "vault_kind": self.vault.kind,
        }

    def logout(self) -> dict[str, Any]:
        self.vault.delete()
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass
        self._access_token = None
        self._access_expires_at = None
        self._catalog = None
        self._catalog_at = 0.0
        return {**self.status(), "ok": True}

    async def close(self) -> None:
        await self.client.close()

    async def _accept(
        self, bundle: CloudTokenBundle, *, fallback_user: dict[str, Any] | None = None
    ) -> None:
        user = bundle.user
        if user.get("username") == "user" and fallback_user:
            user = fallback_user
        if bundle.refresh_token:
            await asyncio.to_thread(self.vault.save, bundle.refresh_token)
        state = AccountState(user=user, base_url=self.client.base_url, device_id=self.device_id)
        await asyncio.to_thread(self._save_state, state)
        self._access_token = bundle.access_token
        self._access_expires_at = bundle.expires_at

    def _load_state(self) -> AccountState | None:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get("base_url") != self.client.base_url:
            return None
        user = raw.get("user")
        device_id = raw.get("device_id")
        if not isinstance(user, dict) or not isinstance(device_id, str):
            return None
        return AccountState(user=user, base_url=self.client.base_url, device_id=device_id)

    def _save_state(self, state: AccountState) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path.write_text(
            json.dumps(
                {"user": state.user, "base_url": state.base_url, "device_id": state.device_id},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if os.name != "nt":
            os.chmod(self.state_path, 0o600)

    @staticmethod
    def _models_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            data = payload.get("data")
            raw_models = data.get("models") if isinstance(data, dict) else []
        return [dict(item) for item in raw_models if isinstance(item, dict)]
