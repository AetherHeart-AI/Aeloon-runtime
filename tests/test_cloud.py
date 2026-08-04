from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from aeloon_core.cloud import (
    CloudAccountService,
    CloudProvider,
    CloudTokenBundle,
    InMemoryTokenVault,
)
from aeloon_core.config import CloudConfig, Config, save_config
from aeloon_core.harness import Model, ProviderContext, StreamOptions, UserMessage
from aeloon_core.harness.providers import collect_assistant
from aeloon_core.service import CoreService


class FakeCloudClient:
    base_url = "https://cloud.example"

    def __init__(self) -> None:
        self.login_calls: list[dict[str, str]] = []
        self.refresh_calls: list[str] = []

    async def login(self, **values: str) -> CloudTokenBundle:
        self.login_calls.append(values)
        return bundle("access-1", "refresh-1", "alice")

    async def refresh(self, **values: str) -> CloudTokenBundle:
        self.refresh_calls.append(values["refresh_token"])
        return bundle("access-2", None, "alice")

    async def models(self, token: str) -> dict[str, Any]:
        assert token in {"access-1", "access-2"}
        return {
            "models": [
                {
                    "model_key": "reasoner",
                    "name": "Cloud Reasoner",
                    "context_window": 200_000,
                    "max_tokens": 40_000,
                    "supports_reasoning": True,
                }
            ]
        }

    async def close(self) -> None:
        pass


def bundle(access: str, refresh: str | None, username: str) -> CloudTokenBundle:
    return CloudTokenBundle(
        access_token=access,
        refresh_token=refresh,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        user={"id": username, "username": username, "display_name": username.title()},
    )


def account(tmp_path: Path) -> tuple[CloudAccountService, FakeCloudClient, InMemoryTokenVault]:
    client = FakeCloudClient()
    vault = InMemoryTokenVault()
    service = CloudAccountService(
        CloudConfig(base_url=client.base_url),
        data_dir=tmp_path / "data",
        client=client,  # type: ignore[arg-type]
        vault=vault,
    )
    return service, client, vault


@pytest.mark.asyncio
async def test_account_owns_refresh_token_and_qualified_catalog(tmp_path: Path) -> None:
    service, client, vault = account(tmp_path)
    assert service.status()["authenticated"] is False

    status = await service.login(username="alice", password="secret")

    assert status["authenticated"] is True
    assert status["user"]["display_name"] == "Alice"
    assert vault.load() == "refresh-1"
    assert "secret" not in service.state_path.read_text(encoding="utf-8")
    assert "refresh-1" not in service.state_path.read_text(encoding="utf-8")
    assert service.state_path.stat().st_mode & 0o777 == 0o600
    models = await service.models()
    assert list(models) == ["aeloon-cloud/reasoner"]
    assert models["aeloon-cloud/reasoner"].context_window == 200_000
    assert client.login_calls[0]["device_name"] == "Aeloon Core"

    service._access_expires_at = datetime.now(UTC)
    assert await service.access_token() == "access-2"
    assert client.refresh_calls == ["refresh-1"]
    assert vault.load() == "refresh-1"


@pytest.mark.asyncio
async def test_bridge_exposes_account_without_credentials_and_adds_cloud_models(
    tmp_path: Path,
) -> None:
    cloud, _, _ = account(tmp_path)
    config_path = tmp_path / "config.json"
    save_config(Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)
    service = CoreService(config_path=config_path, cloud_account_service=cloud)

    logged_in = await service.dispatch(
        "cloud.account.login", {"username": "alice", "password": "secret"}
    )
    assert logged_in["authenticated"] is True
    assert "secret" not in json.dumps(logged_in)
    assert "access-1" not in json.dumps(logged_in)
    catalog = await service.catalog_get({})
    cloud_model = next(item for item in catalog["models"] if item["provider_id"] == "aeloon-cloud")
    assert cloud_model["id"] == "aeloon-cloud/reasoner"

    session = await service.session_create({"workspace": str(tmp_path)})
    configured = await service.session_configure(
        {"session_id": session["session_id"], "model_id": cloud_model["id"]}
    )
    assert configured["model_id"] == "aeloon-cloud/reasoner"
    replay = await service.events_subscribe({"session_ids": [], "after_seq": 0})
    assert any(event["name"] == "cloud.account.updated" for event in replay["events"])

    logged_out = await service.dispatch("cloud.account.logout")
    assert logged_out["authenticated"] is False
    logged_out_catalog = await service.catalog_get({})
    assert all(item["provider_id"] != "aeloon-cloud" for item in logged_out_catalog["models"])


class ProviderAccount:
    def __init__(self) -> None:
        self.config = CloudConfig(base_url="https://cloud.example")
        self.client = type("Client", (), {"base_url": "https://cloud.example"})()
        self.forces: list[bool] = []

    async def access_token(self, force: bool = False) -> str:
        self.forces.append(force)
        return "fresh" if force else "expired"


@pytest.mark.asyncio
async def test_cloud_provider_strips_provider_prefix_and_refreshes_after_401() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["authorization"] == "Bearer expired":
            return httpx.Response(401, json={"message": "expired"})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"},'
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    account = ProviderAccount()
    provider = CloudProvider(account, client=client)  # type: ignore[arg-type]
    model = Model(
        id="aeloon-cloud/reasoner",
        name="Cloud Reasoner",
        provider="aeloon-cloud",
        base_url="https://cloud.example",
        reasoning=True,
        thinking_level_map={"high": "high", "max": "max"},
    )
    message = await collect_assistant(
        provider,
        model,
        ProviderContext("", (UserMessage("hello"),), (), "session"),
        StreamOptions(max_retries=0, thinking_level="high"),
    )
    await client.aclose()

    assert message.text == "ok"
    assert account.forces == [False, True]
    assert [request.url.path for request in requests] == ["/proxy/v1/chat", "/proxy/v1/chat"]
    payload = json.loads(requests[-1].content)
    assert payload["model"] == "reasoner"
    assert payload["reasoning_effort"] == "high"
    assert payload["task_id"].startswith("cloud_")
    assert "stream_options" not in payload
