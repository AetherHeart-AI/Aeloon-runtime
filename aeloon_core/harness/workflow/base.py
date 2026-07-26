"""Workflow Template contracts, finite plans, validation, and discovery."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aeloon_core.harness.agent.base import ROLE_ID_PATTERN, RoleRegistry

MAX_WORKFLOW_NODES = 16
WORKFLOW_ID_PATTERN = ROLE_ID_PATTERN


class WorkflowDefinitionError(ValueError):
    """Raised when a template or compiled plan violates its contract."""


class EmptyTuning(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutputCondition(BaseModel):
    """A bounded predicate over one upstream node's structured output."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    node_id: str = Field(pattern=ROLE_ID_PATTERN)
    field_path: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$",
    )
    operator: Literal["empty", "non_empty", "equals", "not_equals"]
    value: Any | None = None

    @model_validator(mode="after")
    def _value_matches_operator(self) -> OutputCondition:
        if self.operator in {"empty", "non_empty"} and self.value is not None:
            raise ValueError(f"{self.operator} conditions cannot declare value")
        return self


class WorkflowNode(BaseModel):
    """One role invocation in a compiled fixed workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(pattern=ROLE_ID_PATTERN)
    role_id: str = Field(pattern=ROLE_ID_PATTERN)
    objective: str = Field(min_length=1, max_length=32_000)
    depends_on: tuple[str, ...] = ()
    include_reports: tuple[str, ...] = ()
    condition: OutputCondition | None = None

    @model_validator(mode="after")
    def _report_dependencies_are_explicit(self) -> WorkflowNode:
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on entries must be unique")
        if len(set(self.include_reports)) != len(self.include_reports):
            raise ValueError("include_reports entries must be unique")
        missing = set(self.include_reports) - set(self.depends_on)
        if missing:
            raise ValueError("include_reports must also appear in depends_on")
        return self


class WorkflowPlan(BaseModel):
    """A finite, inspectable execution plan built by trusted Python."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    nodes: tuple[WorkflowNode, ...] = Field(min_length=1, max_length=MAX_WORKFLOW_NODES)
    success_when_any: tuple[OutputCondition, ...] = ()

    @model_validator(mode="after")
    def _validate_graph(self) -> WorkflowPlan:
        indexed = {node.id: node for node in self.nodes}
        if len(indexed) != len(self.nodes):
            raise ValueError("workflow node ids must be unique")
        ids = set(indexed)
        for node in self.nodes:
            unknown = set(node.depends_on) - ids
            if unknown:
                raise ValueError(
                    f"node {node.id!r} has unknown dependencies: {sorted(unknown)}"
                )
            if node.id in node.depends_on:
                raise ValueError(f"node {node.id!r} cannot depend on itself")
            if node.condition is not None:
                if node.condition.node_id not in node.depends_on:
                    raise ValueError(
                        f"node {node.id!r} condition must reference a dependency"
                    )
        topological_layers(self)
        for condition in self.success_when_any:
            if condition.node_id not in ids:
                raise ValueError("success conditions must reference a plan node")
        return self

    def validate_roles(self, roles: RoleRegistry) -> None:
        indexed = {node.id: node for node in self.nodes}
        for node in self.nodes:
            try:
                roles.get(node.role_id)
            except KeyError as exc:
                raise WorkflowDefinitionError(
                    f"node {node.id!r} references unknown role {node.role_id!r}"
                ) from exc
            if node.condition is not None:
                _validate_condition_output(node.condition, indexed=indexed, roles=roles)
        for condition in self.success_when_any:
            _validate_condition_output(condition, indexed=indexed, roles=roles)


def topological_layers(plan: WorkflowPlan) -> tuple[tuple[str, ...], ...]:
    """Return stable topological layers or raise on a cycle."""

    indexed = {node.id: node for node in plan.nodes}
    remaining = set(indexed)
    completed: set[str] = set()
    layers: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(
                node_id
                for node_id in remaining
                if set(indexed[node_id].depends_on) <= completed
            )
        )
        if not ready:
            raise ValueError("workflow plan contains a dependency cycle")
        layers.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)
    return tuple(layers)


InputT = TypeVar("InputT", bound=BaseModel)
TuningT = TypeVar("TuningT", bound=BaseModel)


class WorkflowTemplate(ABC, Generic[InputT, TuningT]):
    """Trusted Python definition that compiles inputs into a validated plan."""

    id: ClassVar[str]
    description: ClassVar[str]
    tags: ClassVar[tuple[str, ...]] = ()
    when_to_use: ClassVar[str]
    avoid_when: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    tuning_model: ClassVar[type[BaseModel]] = EmptyTuning

    @abstractmethod
    def build(self, inputs: InputT, tuning: TuningT) -> WorkflowPlan:
        """Compile one request into a finite execution plan."""


class WorkflowTemplateSnapshot(BaseModel):
    """Validated metadata plus the trusted template instance."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str
    description: str
    tags: tuple[str, ...]
    when_to_use: str
    avoid_when: str
    input_model: type[BaseModel]
    tuning_model: type[BaseModel]
    template: WorkflowTemplate[Any, Any]
    source: str
    digest: str

    def descriptor(self, *, include_schema: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "tags": list(self.tags),
            "when_to_use": self.when_to_use,
            "avoid_when": self.avoid_when,
            "source": self.source,
            "digest": self.digest,
        }
        if include_schema:
            result["input_schema"] = self.input_model.model_json_schema()
            result["tuning_schema"] = self.tuning_model.model_json_schema()
        return result

    def compile(
        self,
        *,
        inputs: Mapping[str, Any],
        tuning: Mapping[str, Any] | None = None,
        roles: RoleRegistry,
        max_nodes: int = MAX_WORKFLOW_NODES,
    ) -> WorkflowPlan:
        try:
            validated_inputs = self.input_model.model_validate(dict(inputs))
            validated_tuning = self.tuning_model.model_validate(dict(tuning or {}))
        except ValidationError as exc:
            raise WorkflowDefinitionError(str(exc)) from exc
        try:
            plan = self.template.build(validated_inputs, validated_tuning)
        except WorkflowDefinitionError:
            raise
        except Exception as exc:
            raise WorkflowDefinitionError(
                f"template {self.id!r} could not compile the supplied values: {exc}"
            ) from exc
        if not isinstance(plan, WorkflowPlan):
            raise WorkflowDefinitionError(
                f"template {self.id!r} build() must return WorkflowPlan"
            )
        if len(plan.nodes) > max_nodes:
            raise WorkflowDefinitionError(
                f"template {self.id!r} produced {len(plan.nodes)} nodes; "
                f"configured maximum is {max_nodes}"
            )
        plan.validate_roles(roles)
        return plan


def snapshot_template(
    template_type: type[WorkflowTemplate[Any, Any]],
    *,
    source: str,
) -> WorkflowTemplateSnapshot:
    """Validate and instantiate one WorkflowTemplate subclass."""

    if not isinstance(template_type, type) or not issubclass(
        template_type,
        WorkflowTemplate,
    ):
        raise WorkflowDefinitionError(
            "workflow entries must be WorkflowTemplate subclasses"
        )
    try:
        template = template_type()
    except Exception as exc:
        raise WorkflowDefinitionError(
            f"cannot construct template {template_type.__qualname__}: {exc}"
        ) from exc
    template_id = _required_template_text(template_type, "id", 64)
    if re.fullmatch(WORKFLOW_ID_PATTERN, template_id) is None:
        raise WorkflowDefinitionError(f"invalid workflow template id {template_id!r}")
    description = _required_template_text(template_type, "description", 1_000)
    when_to_use = _required_template_text(template_type, "when_to_use", 2_000)
    avoid_when = _required_template_text(template_type, "avoid_when", 2_000)
    tags = getattr(template_type, "tags", None)
    if not isinstance(tags, tuple) or not all(
        isinstance(tag, str) and tag.strip() for tag in tags
    ):
        raise WorkflowDefinitionError(
            f"template {template_id!r} tags must be a tuple of strings"
        )
    normalized_tags = tuple(tag.strip().casefold() for tag in tags)
    if len(set(normalized_tags)) != len(normalized_tags):
        raise WorkflowDefinitionError(f"template {template_id!r} has duplicate tags")
    input_model = getattr(template_type, "input_model", None)
    tuning_model = getattr(template_type, "tuning_model", None)
    for name, model in (("input_model", input_model), ("tuning_model", tuning_model)):
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise WorkflowDefinitionError(
                f"template {template_id!r} {name} must be a BaseModel type"
            )
    canonical = {
        "id": template_id,
        "description": description,
        "tags": normalized_tags,
        "when_to_use": when_to_use,
        "avoid_when": avoid_when,
        "input_schema": input_model.model_json_schema(),
        "tuning_schema": tuning_model.model_json_schema(),
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return WorkflowTemplateSnapshot(
        id=template_id,
        description=description,
        tags=normalized_tags,
        when_to_use=when_to_use,
        avoid_when=avoid_when,
        input_model=input_model,
        tuning_model=tuning_model,
        template=template,
        source=source,
        digest=digest,
    )


class WorkflowRegistry:
    """Immutable template catalog with zero-model deterministic search."""

    __slots__ = ("_templates",)

    def __init__(self, templates: Mapping[str, WorkflowTemplateSnapshot]) -> None:
        self._templates = MappingProxyType(dict(templates))

    @classmethod
    def from_types(
        cls,
        builtin_types: Iterable[type[WorkflowTemplate[Any, Any]]],
        project_types: Iterable[type[WorkflowTemplate[Any, Any]]] = (),
        *,
        project_source: str = "<project-catalog>",
    ) -> WorkflowRegistry:
        discovered: dict[str, WorkflowTemplateSnapshot] = {}
        for template_type in builtin_types:
            snapshot = snapshot_template(
                template_type,
                source=(
                    f"builtin:{template_type.__module__}.{template_type.__qualname__}"
                ),
            )
            if snapshot.id in discovered:
                raise WorkflowDefinitionError(
                    f"duplicate built-in workflow id {snapshot.id!r}"
                )
            discovered[snapshot.id] = snapshot
        project_seen: set[str] = set()
        for template_type in project_types:
            snapshot = snapshot_template(
                template_type,
                source=f"{project_source}#{template_type.__qualname__}",
            )
            if snapshot.id in project_seen:
                raise WorkflowDefinitionError(
                    f"duplicate project workflow id {snapshot.id!r}"
                )
            project_seen.add(snapshot.id)
            discovered[snapshot.id] = snapshot
        return cls(discovered)

    def list(self) -> tuple[WorkflowTemplateSnapshot, ...]:
        return tuple(
            self._templates[template_id] for template_id in sorted(self._templates)
        )

    def get(self, template_id: str) -> WorkflowTemplateSnapshot:
        try:
            return self._templates[template_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._templates))
            raise KeyError(
                f"unknown workflow template {template_id!r}; available: {available}"
            ) from exc

    def search(
        self,
        query: str,
        *,
        tags: Iterable[str] = (),
        limit: int = 5,
    ) -> tuple[dict[str, Any], ...]:
        if limit < 1 or limit > 10:
            raise ValueError("workflow search limit must be between 1 and 10")
        normalized_query = query.strip().casefold()
        query_terms = _search_terms(normalized_query)
        requested_tags = {tag.strip().casefold() for tag in tags if tag.strip()}
        ranked: list[tuple[int, str, WorkflowTemplateSnapshot]] = []
        for snapshot in self.list():
            haystack = " ".join(
                (
                    snapshot.id,
                    snapshot.description,
                    snapshot.when_to_use,
                    snapshot.avoid_when,
                    *snapshot.tags,
                )
            ).casefold()
            haystack_terms = _search_terms(haystack)
            score = 0
            if normalized_query and normalized_query in haystack:
                score += 20
            score += 4 * len(query_terms & haystack_terms)
            score += 8 * len(requested_tags & set(snapshot.tags))
            if requested_tags and not requested_tags <= set(snapshot.tags):
                score -= 2
            if score > 0:
                ranked.append((score, snapshot.id, snapshot))
        if not ranked and "delegate" in self._templates:
            ranked.append((0, "delegate", self._templates["delegate"]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            {
                "score": score,
                **snapshot.descriptor(include_schema=True),
            }
            for score, _, snapshot in ranked[:limit]
        )


def _search_terms(text: str) -> set[str]:
    ascii_words = set(re.findall(r"[a-z0-9_-]+", text))
    cjk_terms: set[str] = set()
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        if len(run) == 1:
            cjk_terms.add(run)
        else:
            cjk_terms.update(
                run[index : index + 2] for index in range(len(run) - 1)
            )
    return ascii_words | cjk_terms


def _validate_condition_output(
    condition: OutputCondition,
    *,
    indexed: Mapping[str, WorkflowNode],
    roles: RoleRegistry,
) -> None:
    referenced_node = indexed[condition.node_id]
    output_model = roles.get(referenced_node.role_id).output_model
    root_field = condition.field_path.partition(".")[0]
    if root_field not in output_model.model_fields:
        raise WorkflowDefinitionError(
            f"condition on node {condition.node_id!r} references field "
            f"{condition.field_path!r}, but role {referenced_node.role_id!r} "
            f"returns {output_model.__name__}"
        )


def _required_template_text(
    template_type: type[WorkflowTemplate[Any, Any]],
    field: str,
    max_length: int,
) -> str:
    value = getattr(template_type, field, None)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowDefinitionError(
            f"template {template_type.__qualname__!r} requires nonempty {field}"
        )
    normalized = value.strip()
    if len(normalized) > max_length:
        raise WorkflowDefinitionError(
            f"template {getattr(template_type, 'id', template_type.__qualname__)!r} "
            f"{field} exceeds {max_length} characters"
        )
    return normalized


__all__ = [
    "EmptyTuning",
    "MAX_WORKFLOW_NODES",
    "OutputCondition",
    "WorkflowDefinitionError",
    "WorkflowNode",
    "WorkflowPlan",
    "WorkflowRegistry",
    "WorkflowTemplate",
    "WorkflowTemplateSnapshot",
    "snapshot_template",
    "topological_layers",
]
