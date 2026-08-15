from __future__ import annotations

import html
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

CloudSearchCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    content: str
    engine: str


@dataclass(slots=True)
class SearchOutcome:
    results: list[SearchResult]
    error: str | None
    engine: str


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("results", "webPages", "web", "data"):
            value = payload.get(key)
            if key == "webPages" and isinstance(value, dict):
                value = value.get("value")
            if key == "web" and isinstance(value, dict):
                value = value.get("results")
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _items(value)
                if nested:
                    return nested
    return []


def normalize(payload: Any, engine: str, count: int) -> SearchOutcome:
    results: list[SearchResult] = []
    for item in _items(payload)[:count]:
        url = item.get("url") or item.get("link")
        if not isinstance(url, str) or not url:
            continue
        results.append(
            SearchResult(
                str(item.get("title") or item.get("name") or url),
                url,
                str(item.get("content") or item.get("description") or item.get("snippet") or ""),
                engine,
            )
        )
    return SearchOutcome(results, None if results else "Search returned no results", engine)


async def search_provider(
    client: httpx.AsyncClient,
    provider: str,
    query: str,
    count: int,
    *,
    api_key: str | None,
    base_url: str | None,
    cloud_search: CloudSearchCallback | None = None,
) -> SearchOutcome:
    try:
        if provider == "aeloon-cloud":
            if cloud_search is None:
                return SearchOutcome([], "Aeloon Cloud is not authenticated", provider)
            return normalize(await cloud_search({"query": query, "count": count}), provider, count)
        if provider != "duckduckgo" and not api_key:
            return SearchOutcome([], f"An API key is required for {provider}", provider)
        if provider == "tavily":
            response = await client.post(
                base_url or "https://api.tavily.com/search",
                json={"query": query, "max_results": count},
                headers={"Authorization": f"Bearer {api_key}"},
            )
        elif provider == "brave":
            response = await client.get(
                base_url or "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"X-Subscription-Token": api_key or ""},
            )
        elif provider == "bocha":
            response = await client.post(
                base_url or "https://api.bochaai.com/v1/web-search",
                json={"query": query, "count": count},
                headers={"Authorization": f"Bearer {api_key}"},
            )
        elif provider == "duckduckgo":
            response = await client.post(
                base_url or "https://lite.duckduckgo.com/lite/", data={"q": query}
            )
            response.raise_for_status()
            results = []
            links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.I | re.S)
            for url, title in links:
                if url.startswith("//duckduckgo.com/l/?"):
                    continue
                results.append(
                    SearchResult(
                        re.sub("<.*?>", "", html.unescape(title)).strip(),
                        urljoin(str(response.url), html.unescape(url)),
                        "",
                        provider,
                    )
                )
                if len(results) >= count:
                    break
            return SearchOutcome(
                results, None if results else "Search returned no results", provider
            )
        else:
            return SearchOutcome([], f"Unsupported search provider: {provider}", provider)
        response.raise_for_status()
        return normalize(response.json(), provider, count)
    except (httpx.HTTPError, ValueError) as exc:
        return SearchOutcome([], f"Search request failed: {exc}", provider)
