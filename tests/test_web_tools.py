from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from aeloon_runtime.config import Config, public_config, save_config
from aeloon_runtime.rpc import AeloonRpcAdapter
from aeloon_runtime.runtime import RuntimeService
from aeloon_runtime.tool.web.backends import SearchOutcome, normalize, search_provider
from aeloon_runtime.tool.web.cache import WebCache
from aeloon_runtime.tool.web.fetch import WebFetchTool
from aeloon_runtime.tool.web.safety import validate_url_target
from aeloon_runtime.tool.web.search import WebSearchTool


@pytest.mark.parametrize(
    ("provider", "payload", "title"),
    [
        (
            "tavily",
            {"results": [{"title": "Tavily", "url": "https://t.example", "content": "t"}]},
            "Tavily",
        ),
        (
            "brave",
            {
                "web": {
                    "results": [{"title": "Brave", "url": "https://b.example", "description": "b"}]
                }
            },
            "Brave",
        ),
        (
            "bocha",
            {
                "data": {
                    "webPages": {
                        "value": [{"name": "Bocha", "url": "https://bo.example", "snippet": "b"}]
                    }
                }
            },
            "Bocha",
        ),
        (
            "aeloon-cloud",
            {"data": {"results": [{"title": "Cloud", "link": "https://c.example"}]}},
            "Cloud",
        ),
    ],
)
def test_provider_response_normalization(
    provider: str, payload: dict[str, Any], title: str
) -> None:
    outcome = normalize(payload, provider, 5)
    assert outcome.error is None
    assert outcome.results[0].title == title
    assert outcome.results[0].engine == provider


@pytest.mark.asyncio
async def test_provider_adapters_send_expected_authentication() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"results": [{"title": "ok", "url": "https://example.com"}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        for provider in ("tavily", "brave", "bocha"):
            outcome = await search_provider(
                client,
                provider,
                "query",
                1,
                api_key="secret",
                base_url="https://search.example",
                cloud_search=None,
            )
            assert outcome.results
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[1].headers["x-subscription-token"] == "secret"
    assert requests[2].headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_cache_ttl_and_turn_budget_reset() -> None:
    cache: WebCache[str] = WebCache(0.01, 1)
    await cache.put("key", "value")
    assert await cache.get("key") == "value"
    await asyncio.sleep(0.02)
    assert await cache.get("key") is None

    calls = 0

    async def cloud(_payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"results": [{"title": "result", "url": f"https://example.com/{calls}"}]}

    tool = WebSearchTool(
        provider="aeloon-cloud",
        api_key=None,
        base_url=None,
        max_results=5,
        timeout_s=1,
        max_searches_per_turn=1,
        cache_ttl_s=0,
        cache_size=0,
        cloud_search=cloud,
    )
    tool.begin_turn("one")
    assert not (await tool.search("first")).error
    assert "budget exhausted" in str((await tool.search("second")).error)
    tool.begin_turn("two")
    assert not (await tool.search("second")).error


@pytest.mark.asyncio
async def test_ssrf_rejects_local_targets() -> None:
    for url in ("http://127.0.0.1", "http://[::1]", "http://169.254.169.254/latest/meta-data"):
        with pytest.raises(ValueError, match="Private, local"):
            await validate_url_target(url)


@pytest.mark.asyncio
async def test_fetch_extracts_readable_html_and_reports_redirect_location() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=(
                    "<html><body><article><h1>Useful title</h1>"
                    "<p>Useful body</p></article></body></html>"
                ),
            ),
            httpx.Response(301, headers={"location": "https://example.com/new"}),
        ]
    )
    tool = WebFetchTool(
        default_mode="skim",
        default_max_chars=6000,
        full_max_chars=50000,
        timeout_s=1,
        cache_ttl_s=0,
        cache_size=0,
    )
    await tool.client.aclose()
    tool.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses)), follow_redirects=False
    )
    result = await tool.execute("fetch", {"url": "https://93.184.216.34/page"}, None)
    assert "Useful body" in result.content[0].text
    with pytest.raises(ValueError, match="https://example.com/new"):
        await tool.execute("redirect", {"url": "https://93.184.216.34/redirect"}, None)


@pytest.mark.asyncio
async def test_web_settings_merge_secret_channel_and_redaction(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = Config(
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        tools={"web": {"search": {"api_key": "old-secret"}}},
    )
    save_config(config, config_path)
    runtime = RuntimeService(config_path=config_path)
    runtime._models = lambda: _empty_models()  # type: ignore[method-assign]
    raw = config.model_dump(mode="json")
    runtime._apply_settings_patch(
        raw, {"tools": {"web": {"search": {"max_results": 9}}}}, valid_model_ids=[]
    )
    assert raw["tools"]["web"]["search"]["api_key"] == "old-secret"
    with pytest.raises(Exception, match="secret_actions"):
        runtime._apply_settings_patch(
            raw, {"tools": {"web": {"search": {"api_key": "leak"}}}}, valid_model_ids=[]
        )

    settings = await runtime.settings_get({})
    assert "api_key" not in settings["tools"]["web"]["search"]
    assert settings["tools"]["web"]["search"]["credential_configured"] is True
    assert public_config(config)["tools"]["web"]["search"]["api_key"] == "***"
    assert "old-secret" not in AeloonRpcAdapter(runtime)._sanitize("failed old-secret")

    updated = await runtime.settings_update(
        {
            "revision": settings["revision"],
            "patch": {},
            "secret_actions": [
                {"path": "tools.web.search.api_key", "action": "set", "value": "new-secret"}
            ],
        }
    )
    assert updated["tools"]["web"]["search"]["credential_configured"] is True
    assert "new-secret" not in str(updated)
    await runtime.close()


@pytest.mark.asyncio
async def test_tools_search_test_returns_structured_failure(tmp_path: Path) -> None:
    runtime = RuntimeService(config_path=tmp_path / "config.json", data_dir=tmp_path / "data")
    outcome = SearchOutcome([], "upstream failed", "tavily")
    runtime._tool_set = lambda **_kwargs: SimpleNamespace(
        by_name={"web_search": SimpleNamespace(search=lambda *_args: _outcome(outcome))}
    )  # type: ignore[method-assign]
    result = await runtime.tools_search_test({})
    assert result == {
        "ok": False,
        "provider": "tavily",
        "result_count": 0,
        "latency_ms": result["latency_ms"],
        "message": "upstream failed",
    }
    await runtime.close()


async def _empty_models() -> dict[str, Any]:
    return {}


async def _outcome(value: SearchOutcome) -> SearchOutcome:
    return value
