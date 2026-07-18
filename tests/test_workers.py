from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aeloon_core.workers import (
    MAX_WORKER_DESCRIPTION_CHARS,
    MAX_WORKER_PROMPT_CHARS,
    MAX_WORKER_SOURCE_METADATA_CHARS,
    WorkerDefinitionError,
    WorkerRegistry,
    WorkerSnapshot,
    parse_worker,
)


def _definition(
    *,
    worker_id: str = "custom",
    description: str = "A custom responsibility",
    prompt: str = "Deliver the requested outcome.",
) -> str:
    return (
        "---\n"
        f"id: {worker_id}\n"
        f"description: {description}\n"
        "---\n"
        f"{prompt}\n"
    )


def _write_project_worker(root: Path, filename: str, content: str) -> Path:
    worker_root = root / ".aeloon-core" / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    path = worker_root / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_builtin_registry_has_four_soft_responsibilities(tmp_path: Path) -> None:
    registry = WorkerRegistry.discover(tmp_path)

    assert [worker.id for worker in registry.list()] == [
        "builder",
        "explorer",
        "researcher",
        "reviewer",
    ]
    assert all(worker.source.startswith("builtin:") for worker in registry.list())
    assert all(len(worker.digest) == 64 for worker in registry.list())
    assert registry.get("builder").description


def test_parse_worker_normalizes_definition_and_is_immutable() -> None:
    worker = parse_worker(
        _definition(description="  Focused work  ", prompt="  First line.\r\n\r\nSecond line.  "),
        source=" project.md ",
    )

    assert worker.id == "custom"
    assert worker.description == "Focused work"
    assert worker.prompt == "First line.\n\nSecond line."
    assert worker.source == "project.md"
    with pytest.raises(ValidationError):
        worker.prompt = "changed"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "---\nid: custom\ndescription: valid\ntools: [exec]\n---\nprompt\n",
            "extra",
        ),
        (
            "---\nid: custom\nid: duplicate\ndescription: valid\n---\nprompt\n",
            "duplicate key",
        ),
        (
            _definition(worker_id="Invalid ID"),
            "invalid worker frontmatter",
        ),
        (
            "---\nid: custom\ndescription: 42\n---\nprompt\n",
            "invalid worker frontmatter",
        ),
        (
            _definition(prompt="   "),
            "nonempty prompt",
        ),
        (
            "id: custom\ndescription: missing delimiters\n",
            "frontmatter delimited",
        ),
    ],
)
def test_worker_definition_is_strict(source: str, message: str) -> None:
    with pytest.raises(WorkerDefinitionError, match=message):
        parse_worker(source)


def test_yaml_aliases_are_rejected() -> None:
    source = "---\nid: &worker custom\ndescription: *worker\n---\nprompt\n"

    with pytest.raises(WorkerDefinitionError, match="aliases are not allowed"):
        parse_worker(source)


def test_source_metadata_must_be_nonempty_text() -> None:
    with pytest.raises(WorkerDefinitionError, match="source must be nonempty"):
        parse_worker(_definition(), source="  ")
    with pytest.raises(WorkerDefinitionError, match="source must be text"):
        parse_worker(_definition(), source=42)  # type: ignore[arg-type]


def test_worker_descriptor_and_prompt_fields_are_bounded() -> None:
    with pytest.raises(WorkerDefinitionError, match="invalid worker frontmatter"):
        parse_worker(
            _definition(description="d" * (MAX_WORKER_DESCRIPTION_CHARS + 1))
        )
    with pytest.raises(WorkerDefinitionError, match="invalid worker definition"):
        parse_worker(_definition(prompt="p" * (MAX_WORKER_PROMPT_CHARS + 1)))
    with pytest.raises(WorkerDefinitionError, match="invalid worker definition"):
        parse_worker(
            _definition(),
            source="s" * (MAX_WORKER_SOURCE_METADATA_CHARS + 1),
        )


def test_project_definition_overrides_builtin_and_registry_does_not_hot_reload(
    tmp_path: Path,
) -> None:
    path = _write_project_worker(
        tmp_path,
        "local-builder.md",
        _definition(
            worker_id="builder",
            description="Project builder",
            prompt="Follow this project's conventions.",
        ),
    )
    registry = WorkerRegistry.discover(tmp_path)
    snapshot = registry.get("builder")

    assert snapshot.description == "Project builder"
    assert snapshot.prompt == "Follow this project's conventions."
    assert snapshot.source == str(path.resolve())

    path.write_text(
        _definition(
            worker_id="builder",
            description="Changed after startup",
            prompt="This must require a new registry.",
        ),
        encoding="utf-8",
    )
    assert registry.get("builder") is snapshot
    assert registry.get("builder").description == "Project builder"
    assert WorkerRegistry.discover(tmp_path).get("builder").description == "Changed after startup"


def test_duplicate_project_ids_are_rejected(tmp_path: Path) -> None:
    _write_project_worker(tmp_path, "one.md", _definition(worker_id="duplicate"))
    _write_project_worker(tmp_path, "two.md", _definition(worker_id="duplicate"))

    with pytest.raises(WorkerDefinitionError, match="duplicate project worker id"):
        WorkerRegistry.discover(tmp_path)


def test_digest_covers_every_canonical_definition_field_but_not_source() -> None:
    original = parse_worker(_definition(), source="first.md")
    same_definition = parse_worker(_definition(), source="second.md")
    changed_id = parse_worker(_definition(worker_id="another"))
    changed_description = parse_worker(_definition(description="Another responsibility"))
    changed_prompt = parse_worker(_definition(prompt="A different outcome."))

    assert original.digest == same_definition.digest
    assert len(
        {
            original.digest,
            changed_id.digest,
            changed_description.digest,
            changed_prompt.digest,
        }
    ) == 4


def test_snapshot_rejects_a_digest_that_does_not_match_definition() -> None:
    worker = parse_worker(_definition())

    with pytest.raises(ValidationError, match="digest does not match"):
        WorkerSnapshot(
            id=worker.id,
            description=worker.description,
            prompt=worker.prompt,
            source=worker.source,
            digest="0" * 64,
        )


def test_registry_workers_mapping_and_unknown_lookup_are_explicit(tmp_path: Path) -> None:
    registry = WorkerRegistry(tmp_path)

    with pytest.raises(TypeError):
        registry.workers["other"] = registry.get("builder")
    with pytest.raises(
        KeyError,
        match="available: builder, explorer, researcher, reviewer",
    ):
        registry.get("missing")
