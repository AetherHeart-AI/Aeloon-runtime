"""Strict, startup-scoped Worker definition discovery.

Worker definitions are prompt configuration, not executable agent graphs. A registry
loads the built-in definitions and project overrides once, then exposes immutable
snapshots suitable for persisting with a WorkerSession.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, model_validator
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

WORKER_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
MAX_WORKER_SOURCE_CHARS = 256_000
MAX_WORKER_DESCRIPTION_CHARS = 1_000
MAX_WORKER_PROMPT_CHARS = 128_000
MAX_WORKER_SOURCE_METADATA_CHARS = 4_096

WorkerIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=WORKER_ID_PATTERN),
]
WorkerDescription = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_WORKER_DESCRIPTION_CHARS,
    ),
]
WorkerPrompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_WORKER_PROMPT_CHARS,
    ),
]
WorkerSource = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_WORKER_SOURCE_METADATA_CHARS,
    ),
]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\n(?P<yaml>.*?)\n---[ \t]*(?:\n|\Z)(?P<body>.*)\Z",
    re.DOTALL,
)


class WorkerDefinitionError(ValueError):
    """Raised when a Worker definition or catalog violates its contract."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class _WorkerHeader(_FrozenStrictModel):
    id: WorkerIdentifier
    description: WorkerDescription


class WorkerSnapshot(_FrozenStrictModel):
    """An immutable, persistence-ready copy of one Worker definition."""

    id: WorkerIdentifier
    description: WorkerDescription
    prompt: WorkerPrompt
    source: WorkerSource
    digest: Digest

    @model_validator(mode="after")
    def _verify_digest(self) -> Self:
        expected = worker_digest(
            worker_id=self.id,
            description=self.description,
            prompt=self.prompt,
        )
        if self.digest != expected:
            raise ValueError("worker digest does not match its canonical definition")
        return self

    def descriptor(self) -> dict[str, str]:
        """Return the bounded metadata safe to expose to the Master."""

        return {
            "id": self.id,
            "description": self.description,
            "source": self.source,
            "digest": self.digest,
        }


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases and duplicate mapping keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            event = self.get_event()
            raise ConstructorError(
                "while composing Worker YAML",
                event.start_mark,
                "YAML aliases are not allowed",
                event.start_mark,
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, got {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def worker_digest(*, worker_id: str, description: str, prompt: str) -> str:
    """Return the stable SHA-256 digest of a canonical complete definition."""

    payload = json.dumps(
        {
            "id": worker_id,
            "description": description,
            "prompt": prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_worker(text: str, *, source: str = "<memory>") -> WorkerSnapshot:
    """Parse one strict Markdown Worker definition into an immutable snapshot."""

    if not isinstance(text, str):
        raise WorkerDefinitionError("worker definition must be text")
    if len(text) > MAX_WORKER_SOURCE_CHARS:
        raise WorkerDefinitionError(
            f"worker definition exceeds {MAX_WORKER_SOURCE_CHARS} characters"
        )

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    match = _FRONTMATTER_RE.fullmatch(normalized)
    if match is None:
        raise WorkerDefinitionError(
            "worker definition must begin with YAML frontmatter delimited by ---"
        )

    try:
        frontmatter_data = yaml.load(match.group("yaml"), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise WorkerDefinitionError(f"invalid worker YAML: {exc}") from exc
    if not isinstance(frontmatter_data, dict) or not all(
        isinstance(key, str) for key in frontmatter_data
    ):
        raise WorkerDefinitionError("worker frontmatter must be a string-keyed mapping")

    try:
        header = _WorkerHeader.model_validate(frontmatter_data)
    except ValidationError as exc:
        raise WorkerDefinitionError(f"invalid worker frontmatter: {exc}") from exc

    prompt = match.group("body").strip()
    if not prompt:
        raise WorkerDefinitionError(f"worker {header.id!r} must have a nonempty prompt")

    if not isinstance(source, str):
        raise WorkerDefinitionError("worker source must be text")
    canonical_source = source.strip()
    if not canonical_source:
        raise WorkerDefinitionError("worker source must be nonempty")
    digest = worker_digest(
        worker_id=header.id,
        description=header.description,
        prompt=prompt,
    )
    try:
        return WorkerSnapshot(
            id=header.id,
            description=header.description,
            prompt=prompt,
            source=canonical_source,
            digest=digest,
        )
    except ValidationError as exc:
        raise WorkerDefinitionError(f"invalid worker definition: {exc}") from exc


class WorkerRegistry:
    """An immutable catalog discovered once at process startup."""

    __slots__ = ("_workers", "project_root")

    def __init__(self, project_root: Path | str) -> None:
        root = Path(project_root).expanduser().resolve()
        self.project_root = root
        self._workers = MappingProxyType(self._discover(root))

    @classmethod
    def discover(cls, project_root: Path | str) -> WorkerRegistry:
        """Discover built-ins plus project overrides and freeze the result."""

        return cls(project_root)

    @property
    def workers(self) -> Mapping[str, WorkerSnapshot]:
        """Return an immutable id-to-snapshot view."""

        return self._workers

    def list(self) -> tuple[WorkerSnapshot, ...]:
        """Return all definitions in deterministic id order."""

        return tuple(self._workers[worker_id] for worker_id in sorted(self._workers))

    def get(self, worker_id: str) -> WorkerSnapshot:
        """Return one definition by id."""

        try:
            return self._workers[worker_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._workers))
            raise KeyError(f"unknown worker {worker_id!r}; available: {available}") from exc

    @staticmethod
    def _discover(project_root: Path) -> dict[str, WorkerSnapshot]:
        discovered: dict[str, WorkerSnapshot] = {}

        package_root = files("aeloon_core.builtin_workers")
        for entry in sorted(package_root.iterdir(), key=lambda candidate: candidate.name):
            if not entry.name.endswith(".md"):
                continue
            snapshot = parse_worker(
                entry.read_text(encoding="utf-8"),
                source=f"builtin:{entry.name}",
            )
            if snapshot.id in discovered:
                raise WorkerDefinitionError(
                    f"duplicate built-in worker id {snapshot.id!r} in {entry.name!r}"
                )
            discovered[snapshot.id] = snapshot

        project_workers: dict[str, WorkerSnapshot] = {}
        custom_root = project_root / ".aeloon-core" / "workers"
        if custom_root.is_dir():
            for path in sorted(custom_root.glob("*.md")):
                snapshot = parse_worker(
                    path.read_text(encoding="utf-8"),
                    source=str(path.resolve()),
                )
                previous = project_workers.get(snapshot.id)
                if previous is not None:
                    raise WorkerDefinitionError(
                        f"duplicate project worker id {snapshot.id!r}: "
                        f"{previous.source!r} and {snapshot.source!r}"
                    )
                project_workers[snapshot.id] = snapshot

        discovered.update(project_workers)
        return discovered


__all__ = [
    "MAX_WORKER_SOURCE_CHARS",
    "MAX_WORKER_DESCRIPTION_CHARS",
    "MAX_WORKER_PROMPT_CHARS",
    "MAX_WORKER_SOURCE_METADATA_CHARS",
    "WORKER_ID_PATTERN",
    "WorkerDefinitionError",
    "WorkerRegistry",
    "WorkerSnapshot",
    "parse_worker",
    "worker_digest",
]
