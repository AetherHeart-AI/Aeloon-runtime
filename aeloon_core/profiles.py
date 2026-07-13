"""Strict profile source and literal-only compiled artifact contracts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

PROFILE_SCHEMA_VERSION = 1
COMPILED_API_VERSION = 1
PROFILE_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
MAX_PROFILE_SOURCE_CHARS = 256_000
MAX_COMPILED_SOURCE_CHARS = 1_000_000
RESERVED_ROLE_IDS = frozenset(
    {
        "complete_task",
        "context_processing",
        "control",
        "done",
        "handoff_agent",
        "harness",
        "master",
        "profile_master",
        "guard",
        "tool",
        "worker",
    }
)

ProfileIdentifier: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=PROFILE_ID_PATTERN),
]
NonEmptyText: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ToolName: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\n(?P<yaml>.*?)\n---[ \t]*(?:\n|\Z)(?P<body>.*)\Z",
    re.DOTALL,
)
_AGENT_SECTION_RE = re.compile(r"Agent: (?P<agent_id>[a-z][a-z0-9_-]{0,63})\Z")
_COMPILED_FIELDS = (
    "profile_schema_version",
    "compiled_api_version",
    "profile_id",
    "revision",
    "description",
    "default_agent_id",
    "max_handoffs",
    "master_prompt",
    "shared_prompt",
    "agents",
)
_COMPILED_FIELD_SET = frozenset(_COMPILED_FIELDS)


class ProfileValidationError(ValueError):
    """Raised when a source profile or compiled artifact violates its contract."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def _validate_tools(tools: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(tools)) != len(tools):
        raise ValueError("tool names must be unique within an agent")
    return tools


def _validate_agents(
    agents: Sequence[ProfileAgentSource | RuntimeAgentSpec | _ProfileAgentDeclaration],
    *,
    default_agent_id: str,
) -> None:
    agent_ids = [agent.id for agent in agents]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("agent ids must be unique")
    if default_agent_id not in agent_ids:
        raise ValueError("default agent must name a declared agent")


class _ProfileAgentDeclaration(_FrozenStrictModel):
    id: ProfileIdentifier
    description: NonEmptyText
    tools: tuple[ToolName, ...]

    @field_validator("id")
    @classmethod
    def _reject_reserved_id(cls, value: str) -> str:
        if value in RESERVED_ROLE_IDS:
            raise ValueError(f"reserved runtime/control name cannot be an agent id: {value}")
        return value

    @field_validator("tools")
    @classmethod
    def _require_unique_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_tools(value)


class _ProfileFrontmatter(_FrozenStrictModel):
    schema_version: Literal[1]
    id: ProfileIdentifier
    revision: int = Field(ge=1)
    description: NonEmptyText
    default_agent: ProfileIdentifier
    max_handoffs: int = Field(ge=0)
    agents: tuple[_ProfileAgentDeclaration, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _validate_agent_references(self) -> Self:
        _validate_agents(self.agents, default_agent_id=self.default_agent)
        return self


class ProfileAgentSource(_FrozenStrictModel):
    """One validated role declaration plus its Markdown instructions."""

    id: ProfileIdentifier
    description: NonEmptyText
    tools: tuple[ToolName, ...]
    prompt: NonEmptyText

    @field_validator("id")
    @classmethod
    def _reject_reserved_id(cls, value: str) -> str:
        if value in RESERVED_ROLE_IDS:
            raise ValueError(f"reserved runtime/control name cannot be an agent id: {value}")
        return value

    @field_validator("tools")
    @classmethod
    def _require_unique_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_tools(value)


class ProfileSource(_FrozenStrictModel):
    """Canonical, validated content parsed from one ``PROFILE.md`` file."""

    schema_version: Literal[1]
    id: ProfileIdentifier
    revision: int = Field(ge=1)
    description: NonEmptyText
    default_agent: ProfileIdentifier
    max_handoffs: int = Field(ge=0)
    shared_prompt: NonEmptyText
    master_prompt: NonEmptyText
    agents: tuple[ProfileAgentSource, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _validate_agent_references(self) -> Self:
        _validate_agents(self.agents, default_agent_id=self.default_agent)
        return self

    @property
    def agent_map(self) -> Mapping[str, ProfileAgentSource]:
        """Return an immutable role lookup."""

        return MappingProxyType({agent.id: agent for agent in self.agents})

    def agent(self, agent_id: str) -> ProfileAgentSource:
        """Return a declared role or raise ``KeyError``."""

        return self.agent_map[agent_id]


class RuntimeAgentSpec(_FrozenStrictModel):
    """Literal-only role data consumed by the runtime plane."""

    id: ProfileIdentifier
    description: NonEmptyText
    tools: tuple[ToolName, ...]
    prompt: NonEmptyText

    @field_validator("id")
    @classmethod
    def _reject_reserved_id(cls, value: str) -> str:
        if value in RESERVED_ROLE_IDS:
            raise ValueError(f"reserved runtime/control name cannot be an agent id: {value}")
        return value

    @field_validator("tools")
    @classmethod
    def _require_unique_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_tools(value)


class RuntimeProfileSpec(_FrozenStrictModel):
    """Immutable, validated profile artifact passed into the runtime plane."""

    profile_schema_version: Literal[1]
    compiled_api_version: Literal[1]
    profile_id: ProfileIdentifier
    revision: int = Field(ge=1)
    description: NonEmptyText
    default_agent_id: ProfileIdentifier
    max_handoffs: int = Field(ge=0)
    master_prompt: NonEmptyText
    shared_prompt: NonEmptyText
    agents: tuple[RuntimeAgentSpec, ...] = Field(min_length=1, max_length=16)
    artifact_id: NonEmptyText | None = None
    generation: int = Field(default=0, ge=0)
    control_protocol_version: Literal[1, 2] = 1

    @model_validator(mode="after")
    def _validate_agent_references(self) -> Self:
        _validate_agents(self.agents, default_agent_id=self.default_agent_id)
        if self.control_protocol_version == 2 and any(
            "delegate_tasks" in agent.tools for agent in self.agents
        ):
            raise ValueError(
                "control protocol v2 reserves delegate_tasks as an internal tool"
            )
        return self

    @property
    def agent_map(self) -> Mapping[str, RuntimeAgentSpec]:
        """Return an immutable role lookup."""

        return MappingProxyType({agent.id: agent for agent in self.agents})

    def agent(self, agent_id: str) -> RuntimeAgentSpec:
        """Return a declared role or raise ``KeyError``."""

        return self.agent_map[agent_id]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            event = self.get_event()
            raise ConstructorError(
                "while composing profile YAML",
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


def parse_profile(text: str) -> ProfileSource:
    """Parse one strict Markdown/YAML profile into canonical source data."""

    if not isinstance(text, str):
        raise ProfileValidationError("profile source must be text")
    if len(text) > MAX_PROFILE_SOURCE_CHARS:
        raise ProfileValidationError(
            f"profile source exceeds {MAX_PROFILE_SOURCE_CHARS} characters"
        )
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    match = _FRONTMATTER_RE.fullmatch(normalized)
    if match is None:
        raise ProfileValidationError(
            "PROFILE.md must begin with one YAML frontmatter document delimited by ---"
        )

    try:
        frontmatter_data = yaml.load(match.group("yaml"), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ProfileValidationError(f"invalid profile YAML: {exc}") from exc
    if not isinstance(frontmatter_data, dict) or not all(
        isinstance(key, str) for key in frontmatter_data
    ):
        raise ProfileValidationError("profile frontmatter must be a string-keyed mapping")

    try:
        frontmatter = _ProfileFrontmatter.model_validate(frontmatter_data)
    except ValidationError as exc:
        raise ProfileValidationError(f"invalid profile frontmatter: {exc}") from exc

    shared_prompt, master_prompt, agent_prompts = _parse_markdown_sections(
        match.group("body")
    )
    declared_ids = {agent.id for agent in frontmatter.agents}
    section_ids = set(agent_prompts)
    missing = sorted(declared_ids - section_ids)
    undeclared = sorted(section_ids - declared_ids)
    if missing or undeclared:
        details = []
        if missing:
            details.append(f"missing agent sections: {', '.join(missing)}")
        if undeclared:
            details.append(f"undeclared agent sections: {', '.join(undeclared)}")
        raise ProfileValidationError("; ".join(details))

    try:
        return ProfileSource(
            schema_version=frontmatter.schema_version,
            id=frontmatter.id,
            revision=frontmatter.revision,
            description=frontmatter.description,
            default_agent=frontmatter.default_agent,
            max_handoffs=frontmatter.max_handoffs,
            shared_prompt=shared_prompt,
            master_prompt=master_prompt,
            agents=tuple(
                ProfileAgentSource(
                    id=agent.id,
                    description=agent.description,
                    tools=agent.tools,
                    prompt=agent_prompts[agent.id],
                )
                for agent in frontmatter.agents
            ),
        )
    except ValidationError as exc:
        raise ProfileValidationError(f"invalid profile content: {exc}") from exc


def _parse_markdown_sections(body: str) -> tuple[str, str, dict[str, str]]:
    lines = body.splitlines(keepends=True)
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        candidate = line.rstrip("\n")
        if not (candidate.startswith("## ") or candidate == "##"):
            continue
        title = candidate[3:].rstrip() if candidate.startswith("## ") else ""
        if title in {"Shared", "Master"} or _AGENT_SECTION_RE.fullmatch(title):
            headings.append((index, title))
            continue
        raise ProfileValidationError(f"unknown or malformed level-two section: {candidate!r}")

    if not headings:
        raise ProfileValidationError("PROFILE.md must contain Shared, Master, and Agent sections")
    if any(line.strip() for line in lines[: headings[0][0]]):
        raise ProfileValidationError("content before the first profile section is not allowed")

    sections: dict[str, str] = {}
    for position, (line_index, title) in enumerate(headings):
        if title in sections:
            raise ProfileValidationError(f"duplicate Markdown section: {title}")
        next_index = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        content = "".join(lines[line_index + 1 : next_index]).strip()
        if not content:
            raise ProfileValidationError(f"Markdown section must not be empty: {title}")
        sections[title] = content

    missing_required = [title for title in ("Shared", "Master") if title not in sections]
    if missing_required:
        raise ProfileValidationError(
            f"missing required Markdown sections: {', '.join(missing_required)}"
        )
    agent_prompts = {
        match.group("agent_id"): content
        for title, content in sections.items()
        if (match := _AGENT_SECTION_RE.fullmatch(title)) is not None
    }
    return sections["Shared"], sections["Master"], agent_prompts


def canonical_profile_bytes(profile: ProfileSource) -> bytes:
    """Return deterministic UTF-8 JSON bytes for profile identity and caching."""

    payload = profile.model_dump(mode="json")
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_profile_hash(profile: ProfileSource) -> str:
    """Return the hexadecimal SHA-256 digest of canonical profile bytes."""

    return hashlib.sha256(canonical_profile_bytes(profile)).hexdigest()


def emit_compiled_profile(profile: ProfileSource) -> str:
    """Emit the deterministic constant-only Python artifact for ``profile``."""

    lines = [
        '"""Generated Aeloon profile artifact. Do not edit."""',
        "",
        "class CompiledProfile:",
        f"    profile_schema_version = {PROFILE_SCHEMA_VERSION}",
        f"    compiled_api_version = {COMPILED_API_VERSION}",
        f"    profile_id = {_python_string(profile.id)}",
        f"    revision = {profile.revision}",
        f"    description = {_python_string(profile.description)}",
        f"    default_agent_id = {_python_string(profile.default_agent)}",
        f"    max_handoffs = {profile.max_handoffs}",
        f"    master_prompt = {_python_string(profile.master_prompt)}",
        f"    shared_prompt = {_python_string(profile.shared_prompt)}",
        "    agents = (",
    ]
    for agent in profile.agents:
        lines.extend(
            [
                "        {",
                f'            "id": {_python_string(agent.id)},',
                f'            "description": {_python_string(agent.description)},',
                f'            "tools": {_python_tuple(agent.tools)},',
                f'            "prompt": {_python_string(agent.prompt)},',
                "        },",
            ]
        )
    lines.extend(["    )", ""])
    return "\n".join(lines)


def _python_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _python_tuple(values: Sequence[str]) -> str:
    rendered = [_python_string(value) for value in values]
    if not rendered:
        return "()"
    if len(rendered) == 1:
        return f"({rendered[0]},)"
    return f"({', '.join(rendered)})"


def parse_compiled_profile(
    source: str,
    *,
    artifact_id: str | None = None,
    generation: int = 0,
) -> RuntimeProfileSpec:
    """Decode a literal-only compiled class without importing or executing it."""

    if not isinstance(source, str):
        raise ProfileValidationError("compiled profile source must be text")
    if len(source) > MAX_COMPILED_SOURCE_CHARS:
        raise ProfileValidationError(
            f"compiled profile exceeds {MAX_COMPILED_SOURCE_CHARS} characters"
        )
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ProfileValidationError(f"invalid compiled profile syntax: {exc}") from exc
    if module.type_ignores:
        raise ProfileValidationError("type-ignore directives are not allowed")

    body = list(module.body)
    if body and _is_docstring(body[0]):
        body.pop(0)
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        raise ProfileValidationError(
            "compiled source must contain only an optional docstring and CompiledProfile"
        )

    class_node = body[0]
    if class_node.name != "CompiledProfile":
        raise ProfileValidationError("compiled class must be named CompiledProfile")
    if (
        class_node.bases
        or class_node.keywords
        or class_node.decorator_list
        or getattr(class_node, "type_params", ())
    ):
        raise ProfileValidationError(
            "CompiledProfile must not have bases, keywords, decorators, or type parameters"
        )

    assignments: dict[str, Any] = {}
    for statement in class_node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ProfileValidationError(
                "CompiledProfile body may contain only plain literal assignments"
            )
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            raise ProfileValidationError("compiled assignment targets must be plain names")
        name = target.id
        if name not in _COMPILED_FIELD_SET:
            raise ProfileValidationError(f"compiled field is not allowed: {name}")
        if name in assignments:
            raise ProfileValidationError(f"duplicate compiled field: {name}")
        _validate_literal_ast(statement.value)
        try:
            assignments[name] = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
            raise ProfileValidationError(f"invalid literal value for {name}: {exc}") from exc

    missing = [name for name in _COMPILED_FIELDS if name not in assignments]
    if missing:
        raise ProfileValidationError(f"missing compiled fields: {', '.join(missing)}")

    try:
        return RuntimeProfileSpec.model_validate(
            {
                **assignments,
                "artifact_id": artifact_id,
                "generation": generation,
            }
        )
    except ValidationError as exc:
        raise ProfileValidationError(f"invalid compiled profile values: {exc}") from exc


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _validate_literal_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Constant):
        if type(node.value) not in {str, int}:
            raise ProfileValidationError(
                f"compiled literals may contain only strings and integers, got "
                f"{type(node.value).__name__}"
            )
        return
    if isinstance(node, ast.Tuple | ast.List):
        for element in node.elts:
            _validate_literal_ast(element)
        return
    if isinstance(node, ast.Dict):
        seen_keys: set[str] = set()
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or type(key.value) is not str:
                raise ProfileValidationError("compiled mapping keys must be string literals")
            if key.value in seen_keys:
                raise ProfileValidationError(f"duplicate compiled mapping key: {key.value}")
            seen_keys.add(key.value)
            _validate_literal_ast(value)
        return
    raise ProfileValidationError(
        f"compiled values must be literal-only; {type(node).__name__} is not allowed"
    )


__all__ = [
    "COMPILED_API_VERSION",
    "MAX_COMPILED_SOURCE_CHARS",
    "MAX_PROFILE_SOURCE_CHARS",
    "PROFILE_ID_PATTERN",
    "PROFILE_SCHEMA_VERSION",
    "RESERVED_ROLE_IDS",
    "ProfileAgentSource",
    "ProfileSource",
    "ProfileValidationError",
    "RuntimeAgentSpec",
    "RuntimeProfileSpec",
    "canonical_profile_bytes",
    "canonical_profile_hash",
    "emit_compiled_profile",
    "parse_compiled_profile",
    "parse_profile",
]
