from __future__ import annotations

from typing import Any

import httpx

from aeloon_core.core.types import ToolResult, ToolUpdateCallback
from aeloon_core.tool.base import BaseTool, object_schema
from aeloon_core.tool.web.backends import CloudSearchCallback, SearchOutcome, search_provider
from aeloon_core.tool.web.cache import WebCache


class WebSearchTool(BaseTool):
    name = "web_search"
    label = "Web search"
    description = (
        "Search the public web for current information and return titles, URLs, and snippets."
    )
    prompt_snippet = "Search the public web for current information"
    parameters = object_schema(
        {"query": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 20}},
        ("query",),
    )

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        max_results: int,
        timeout_s: float,
        max_searches_per_turn: int,
        cache_ttl_s: int,
        cache_size: int,
        cloud_search: CloudSearchCallback | None,
    ) -> None:
        self.provider, self.api_key, self.base_url = provider, api_key, base_url
        self.max_results, self.max_searches_per_turn = max_results, max_searches_per_turn
        self.cloud_search = cloud_search
        self.client = httpx.AsyncClient(timeout=timeout_s, follow_redirects=True)
        self.cache: WebCache[SearchOutcome] = WebCache(cache_ttl_s, cache_size)
        self._budget: dict[str, int] = {}
        self._run_id = ""

    def begin_turn(self, run_id: str) -> None:
        self._run_id, self._budget = run_id, {run_id: 0}

    async def search(self, query: str, count: int | None = None) -> SearchOutcome:
        provider = self.provider
        if provider == "auto":
            provider = "aeloon-cloud" if self.cloud_search is not None else "duckduckgo"
        limit = min(max(1, count or self.max_results), self.max_results)
        key = f"{provider}:{limit}:{query.strip()}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        used = self._budget.get(self._run_id, 0)
        if self.max_searches_per_turn and used >= self.max_searches_per_turn:
            return SearchOutcome([], "Web search budget exhausted for this turn", provider)
        self._budget[self._run_id] = used + 1
        outcome = await search_provider(
            self.client,
            provider,
            query,
            limit,
            api_key=self.api_key,
            base_url=self.base_url,
            cloud_search=self.cloud_search,
        )
        if outcome.results:
            await self.cache.put(key, outcome)
        return outcome

    async def execute(
        self, _call_id: str, arguments: dict[str, Any], _on_update: ToolUpdateCallback | None
    ) -> ToolResult:
        outcome = await self.search(
            str(arguments["query"]), int(arguments["count"]) if arguments.get("count") else None
        )
        if outcome.error:
            return ToolResult.text(
                outcome.error, is_error=True, details={"provider": outcome.engine}
            )
        text = "\n\n".join(
            f"{i}. {r.title}\n{r.url}\n{r.content}" for i, r in enumerate(outcome.results, 1)
        )
        return ToolResult.text(
            text, details={"provider": outcome.engine, "resultCount": len(outcome.results)}
        )
