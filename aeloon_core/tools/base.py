"""Base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel


class Tool(ABC):
    """Abstract base class for agent tools.

    Each tool declares its parameters as a Pydantic model (``args_model``), which
    provides validation, lax coercion (e.g. a stringified ``"5"`` becomes ``5``),
    and the JSON schema advertised to the model — all in one place.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    concurrency_mode: ClassVar[Literal["read_only", "mutating", "exclusive"]] = "exclusive"

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with validated parameters."""

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema format."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.args_model.model_json_schema()),
            },
        }


def _strip_titles(node: Any) -> Any:
    """Drop Pydantic's auto-generated ``title`` keys for a leaner tool schema."""

    if isinstance(node, dict):
        return {key: _strip_titles(value) for key, value in node.items() if key != "title"}
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


class WorkspaceTool(Tool):
    """A tool bound to a workspace directory, with shared path resolution."""

    def __init__(
        self,
        *,
        workspace: Path,
        denied_paths: Iterable[Path] = (),
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.denied_paths = tuple(
            Path(path).expanduser().resolve(strict=False) for path in denied_paths
        )

    def _resolve(self, path: str | None) -> Path:
        """Resolve a path against the workspace; empty/None resolves to the root."""

        if not path:
            return self.workspace
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=False)
        if self.denied_paths and not resolved.is_relative_to(self.workspace):
            raise ValueError(f"path escapes workspace: {path}")
        for denied in self.denied_paths:
            if resolved == denied or resolved.is_relative_to(denied):
                raise PermissionError(f"path is protected from agent tools: {path}")
        return resolved
