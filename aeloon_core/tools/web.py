"""Minimal web fetch and search tools."""

from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import quote_plus, urlparse

import html2text
import httpx

from aeloon_core.config import WebToolConfig
from aeloon_core.tools.base import Tool


def _validate_http_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "URL must be absolute and use http or https"
    return None


class WebFetchTool(Tool):
    """Fetch a web page and return readable text."""

    def __init__(self, *, config: WebToolConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "webfetch"

    @property
    def concurrency_mode(self) -> str:
        return "read_only"

    @property
    def description(self) -> str:
        return "Fetch an HTTP(S) URL and return readable text content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP(S) URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "minimum": 500,
                    "maximum": 100000,
                    "description": "Maximum characters to return.",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, max_chars: int = 12000, **kwargs: Any) -> str:
        del kwargs
        if error := _validate_http_url(url):
            return f"Error: {error}"
        try:
            async with httpx.AsyncClient(
                timeout=self.config.fetch_timeout,
                follow_redirects=True,
                headers={"User-Agent": "AeloonCore/0.1"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            text = response.text
            if "html" in content_type.lower() or "<html" in text[:500].lower():
                converter = html2text.HTML2Text()
                converter.ignore_images = True
                converter.ignore_links = False
                converter.body_width = 0
                text = converter.handle(text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n... truncated {len(text) - max_chars} chars ..."
            return f"URL: {response.url}\nStatus: {response.status_code}\n\n{text}"
        except Exception as exc:
            return f"Error fetching URL: {exc}"


class WebSearchTool(Tool):
    """Search the web via a configured search API or DuckDuckGo HTML fallback."""

    def __init__(self, *, config: WebToolConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "websearch"

    @property
    def concurrency_mode(self) -> str:
        return "read_only"

    @property
    def description(self) -> str:
        return (
            "Search the web. If tools.web.search_api_url is set, POSTs JSON to that API; "
            "otherwise uses a lightweight DuckDuckGo HTML fallback."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum result count.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int | None = None, **kwargs: Any) -> str:
        del kwargs
        limit = min(max_results or self.config.max_results, 10)
        try:
            if self.config.search_api_url:
                results = await self._search_configured_api(query, limit)
            else:
                results = await self._search_duckduckgo(query, limit)
            if not results:
                return "(no results)"
            return "\n\n".join(
                f"{index}. {item.get('title', 'Untitled')}\n"
                f"{item.get('url', '')}\n"
                f"{item.get('snippet', '')}".strip()
                for index, item in enumerate(results[:limit], start=1)
            )
        except Exception as exc:
            return f"Error searching web: {exc}"

    async def _search_configured_api(self, query: str, limit: int) -> list[dict[str, str]]:
        assert self.config.search_api_url is not None
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.search_api_key:
            headers["Authorization"] = f"Bearer {self.config.search_api_key}"
        payload = {"query": query, "q": query, "max_results": limit, "limit": limit}
        async with httpx.AsyncClient(timeout=self.config.fetch_timeout) as client:
            response = await client.post(self.config.search_api_url, headers=headers, json=payload)
            response.raise_for_status()
        return _normalize_search_payload(response.json())

    async def _search_duckduckgo(self, query: str, limit: int) -> list[dict[str, str]]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        async with httpx.AsyncClient(
            timeout=self.config.fetch_timeout,
            follow_redirects=True,
            headers={"User-Agent": "AeloonCore/0.1"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
        html = response.text
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?>(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        results: list[dict[str, str]] = []
        for match in pattern.finditer(html):
            results.append(
                {
                    "title": _clean_html(match.group("title")),
                    "url": unescape(match.group("url")),
                    "snippet": _clean_html(match.group("snippet")),
                }
            )
            if len(results) >= limit:
                break
        return results


def _normalize_search_payload(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, list):
        raw_results = payload
    elif isinstance(payload, dict):
        raw_results = (
            payload.get("results")
            or payload.get("organic_results")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    else:
        raw_results = []

    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or item.get("headline") or "Untitled"
        url = item.get("url") or item.get("link") or item.get("href") or ""
        snippet = item.get("snippet") or item.get("content") or item.get("description") or ""
        results.append(
            {
                "title": str(title),
                "url": str(url),
                "snippet": str(snippet),
            }
        )
    return results


def _clean_html(value: str) -> str:
    text = re.sub(r"<.*?>", "", value)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()
