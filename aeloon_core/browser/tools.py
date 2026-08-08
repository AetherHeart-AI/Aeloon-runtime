"""Core-owned Browser Use tools backed by the Electron Browser Runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from aeloon_core.browser.client import BrowserRuntimeError, execute_browser_tool
from aeloon_core.browser.protocol import BrowserContext
from aeloon_core.core.types import ImageContent, TextContent, ToolResult, ToolUpdateCallback
from aeloon_core.tool.base import BaseTool


def _load_catalogue() -> tuple[dict[str, Any], ...]:
    path = files("aeloon_core.browser").joinpath("browser-tool-catalogue-v1.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("Browser tool catalogue is invalid")
    return tuple(dict(item) for item in raw)


BROWSER_TOOL_CATALOGUE = _load_catalogue()
BROWSER_TOOL_NAMES = tuple(str(item["name"]) for item in BROWSER_TOOL_CATALOGUE)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _browser_error(error: BrowserRuntimeError) -> dict[str, Any]:
    if error.envelope:
        return dict(error.envelope)
    code = {
        "unavailable": "BrowserRuntimeUnavailable",
        "timeout": "BrowserTimeout",
        "malformed": "BrowserMalformedResponse",
        "transport": "BrowserUnavailable",
    }.get(error.kind, "BrowserRuntimeFailure")
    return {
        "type": "aeloon_browser_error",
        "code": code,
        "message": str(error),
        "retryable": error.kind in {"unavailable", "timeout", "transport"},
        "phase": "transport",
        "effectMayHaveCommitted": error.kind == "timeout",
    }


@dataclass(frozen=True, slots=True)
class _BrowserDefinition:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    host_output_schema: dict[str, Any]
    default_timeout_ms: int
    maximum_timeout_ms: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _BrowserDefinition:
        return cls(
            name=str(value["name"]),
            title=str(value.get("title") or value["name"]),
            description=str(value.get("description") or ""),
            input_schema=dict(value.get("inputSchema") or {}),
            host_output_schema=dict(value.get("hostOutputSchema") or {}),
            default_timeout_ms=int(value.get("defaultTimeoutMs") or 10_000),
            maximum_timeout_ms=int(value.get("maximumTimeoutMs") or 30_000),
        )


class BrowserProxyTool(BaseTool):
    execution_mode = "sequential"

    def __init__(self, context: BrowserContext, definition: _BrowserDefinition) -> None:
        self.context = context
        self.browser_definition = definition
        self.name = definition.name
        self.label = definition.title
        self.description = definition.description
        self.parameters = definition.input_schema
        self.prompt_snippet = definition.description

    async def execute(
        self,
        call_id: str,
        arguments: dict[str, Any],
        on_update: ToolUpdateCallback | None,
    ) -> ToolResult:
        normalized = dict(arguments)
        requested_timeout = normalized.get("timeoutMs")
        timeout_ms = (
            min(int(requested_timeout), self.browser_definition.maximum_timeout_ms)
            if isinstance(requested_timeout, int) and not isinstance(requested_timeout, bool)
            else self.browser_definition.default_timeout_ms
        )
        if not normalized.get("idempotencyKey"):
            digest = hashlib.sha256(
                f"{self.context.session_id}\0{call_id}\0{self.name}\0{_stable_json(normalized)}".encode()
            ).hexdigest()
            normalized["idempotencyKey"] = digest
        try:
            host_value = await execute_browser_tool(
                self.context,
                call_id=call_id,
                name=self.name,
                arguments=normalized,
                timeout_ms=timeout_ms,
            )
            Draft202012Validator(self.browser_definition.host_output_schema).validate(host_value)
        except BrowserRuntimeError as exc:
            value = _browser_error(exc)
            return ToolResult(
                content=(TextContent(_stable_json(value)),),
                details=value,
                is_error=True,
            )
        except Exception as exc:
            value = {
                "type": "aeloon_browser_error",
                "code": "BrowserMalformedResponse",
                "message": f"Browser Runtime returned an invalid response: {type(exc).__name__}",
                "retryable": False,
                "phase": "runtime",
                "effectMayHaveCommitted": True,
            }
            return ToolResult(
                content=(TextContent(_stable_json(value)),),
                details=value,
                is_error=True,
            )

        envelope = host_value if isinstance(host_value, dict) else {"structuredContent": host_value}
        structured = envelope.get("structuredContent", host_value)
        content: list[TextContent | ImageContent] = [TextContent(_stable_json(structured))]
        image = envelope.get("image")
        if (
            isinstance(image, dict)
            and image.get("mimeType") == "image/png"
            and isinstance(image.get("data"), str)
            and image["data"]
        ):
            content[0] = TextContent(
                f"{content[0].text}\n\n"
                "Visual pixels are attached for image-capable models. If they are unavailable, "
                "use browser_snapshot to inspect the page as structured text."
            )
            content.append(ImageContent(image["data"], "image/png"))
        return ToolResult(content=tuple(content), details=structured)


class BrowserToolSet:
    names = BROWSER_TOOL_NAMES

    def __init__(self, context: BrowserContext) -> None:
        definitions = tuple(_BrowserDefinition.from_dict(item) for item in BROWSER_TOOL_CATALOGUE)
        self.tools = tuple(BrowserProxyTool(context, item) for item in definitions)
        self.by_name = {tool.name: tool for tool in self.tools}


__all__ = ["BROWSER_TOOL_CATALOGUE", "BROWSER_TOOL_NAMES", "BrowserProxyTool", "BrowserToolSet"]
