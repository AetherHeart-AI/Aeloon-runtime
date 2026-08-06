"""Turn input validation and slash-Skill resolution."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeloon_core.config import Config
from aeloon_core.runtime.resources import ResourceLoader
from aeloon_core.runtime.session import Session
from aeloon_core.runtime.types import RuntimeFailure

_SKILL_COMMAND = re.compile(r"^/([A-Za-z0-9][A-Za-z0-9._:-]*)(?:\s+([\s\S]*))?$")


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    input: dict[str, Any]

    @property
    def skill_id(self) -> str | None:
        value = self.input.get("skill_id")
        return str(value) if value else None


class TurnInputResolver:
    def __init__(self, *, prompt_limit: int, attachment_limit: int) -> None:
        self.prompt_limit = prompt_limit
        self.attachment_limit = attachment_limit

    def parse(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise RuntimeFailure("invalid_argument", "turn.start.input must be an object")
        kind = raw.get("kind")
        if kind == "prompt":
            return self.prompt(raw)
        if kind == "skill":
            return {
                "kind": "skill",
                "name": self._required_string(raw, "name"),
                "additional_instructions": str(raw.get("additional_instructions") or "") or None,
            }
        if kind == "prompt_template":
            arguments = raw.get("arguments") or []
            if not isinstance(arguments, list) or any(
                not isinstance(item, str) for item in arguments
            ):
                raise RuntimeFailure("invalid_argument", "template arguments must be strings")
            return {
                "kind": "prompt_template",
                "name": self._required_string(raw, "name"),
                "arguments": arguments,
            }
        raise RuntimeFailure(
            "invalid_argument",
            "input.kind must be prompt, skill, or prompt_template",
        )

    def prompt(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise RuntimeFailure("invalid_argument", "input must be an object")
        text = str(raw.get("text") or "")
        if not text.strip() or len(text) > self.prompt_limit:
            raise RuntimeFailure(
                "invalid_argument",
                f"Prompt must contain 1 to {self.prompt_limit:,} characters",
            )
        attachments = raw.get("attachments") or []
        if not isinstance(attachments, list) or len(attachments) > self.attachment_limit:
            raise RuntimeFailure(
                "invalid_attachment",
                f"At most {self.attachment_limit} attachments are allowed",
            )
        return {"kind": "prompt", "text": text, "attachments": attachments}

    async def resolve_slash_skill(
        self,
        *,
        session: Session,
        value: dict[str, Any],
        config: Config,
        resource_loader: Callable[[Config], ResourceLoader],
    ) -> PreparedTurn:
        if value.get("kind") != "prompt":
            return PreparedTurn(value)
        match = _SKILL_COMMAND.fullmatch(str(value.get("text") or ""))
        if match is None:
            return PreparedTurn(value)
        name = match.group(1)
        effective = config.model_copy(update={"workspace": Path(session.metadata.cwd)}).normalized()
        loader = resource_loader(effective)
        resources = await asyncio.to_thread(loader.reload)
        skill = next((item for item in loader.available_skills if item.name == name), None)
        if skill is None:
            return PreparedTurn(value)
        if not any(item.name == name for item in resources.skills):
            raise RuntimeFailure(
                "invalid_argument",
                f"Skill '{name}' is available but disabled in Runtime settings",
            )
        return PreparedTurn(
            {
                **value,
                "skill_id": name,
                "_skill_instructions": (match.group(2) or "").strip() or None,
            }
        )

    @staticmethod
    def _required_string(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeFailure("invalid_argument", f"{key} is required")
        return value.strip()


__all__ = ["PreparedTurn", "TurnInputResolver"]
