from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from aeloon_core.customization.catalog import Catalog, CatalogDefinitionError
from aeloon_core.customization.roles import (
    ReviewReport,
    Role,
    RoleDefinitionError,
    RoleRegistry,
    snapshot_role,
)


class CustomOutput(BaseModel):
    result: str


class CustomRole(Role[CustomOutput]):
    id = "custom"
    description = "A custom responsibility"
    system_prompt = "Deliver the requested outcome."
    output_model = CustomOutput
    model_tier = "fast"
    capabilities = ("filesystem", "planning")
    concurrency_mode = "parallel_safe"


def test_builtin_catalog_has_python_roles_and_structured_reviewer(
    tmp_path: Path,
) -> None:
    catalog = Catalog.discover(tmp_path)

    assert [role.id for role in catalog.roles.list()] == [
        "builder",
        "explorer",
        "researcher",
        "reviewer",
    ]
    assert all(role.source.startswith("builtin:") for role in catalog.roles.list())
    assert catalog.roles.get("reviewer").output_model is ReviewReport
    assert catalog.roles.get("builder").concurrency_mode == "exclusive"
    assert catalog.roles.get("explorer").concurrency_mode == "parallel_safe"


def test_role_snapshot_is_immutable_and_digest_covers_runtime_fields() -> None:
    original = snapshot_role(CustomRole, source="one.py")
    same = snapshot_role(CustomRole, source="two.py")

    class ChangedPrompt(CustomRole):
        system_prompt = "Changed."

    class ChangedCapabilities(CustomRole):
        capabilities = ("planning",)

    assert original.digest == same.digest
    assert original.digest != snapshot_role(ChangedPrompt, source="one.py").digest
    assert original.digest != snapshot_role(ChangedCapabilities, source="one.py").digest
    with pytest.raises(ValidationError):
        original.system_prompt = "mutated"


@pytest.mark.parametrize(
    ("role_type", "message"),
    [
        (
            type(
                "InvalidId",
                (Role,),
                {
                    "id": "Invalid ID",
                    "description": "valid",
                    "system_prompt": "valid",
                },
            ),
            "invalid role id",
        ),
        (
            type(
                "NoPrompt",
                (Role,),
                {"id": "no-prompt", "description": "valid", "system_prompt": " "},
            ),
            "requires nonempty system_prompt",
        ),
        (
            type(
                "BadCapability",
                (Role,),
                {
                    "id": "bad-capability",
                    "description": "valid",
                    "system_prompt": "valid",
                    "capabilities": ("unknown",),
                },
            ),
            "unknown capabilities",
        ),
    ],
)
def test_invalid_python_roles_are_rejected(
    role_type: type[Role],
    message: str,
) -> None:
    with pytest.raises(RoleDefinitionError, match=message):
        snapshot_role(role_type, source="test")


def test_project_catalog_overrides_builtin_and_requires_restart(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        """
from aeloon_core.customization.roles import Role

class ProjectBuilder(Role):
    id = "builder"
    description = "Project builder"
    system_prompt = "Follow project conventions."

ROLES = (ProjectBuilder,)
WORKFLOWS = ()
""",
    )
    catalog = Catalog.discover(tmp_path)
    snapshot = catalog.roles.get("builder")

    assert snapshot.description == "Project builder"
    assert snapshot.source.startswith(str(catalog_path))
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8").replace(
            "Project builder",
            "Changed builder",
        ),
        encoding="utf-8",
    )
    assert catalog.roles.get("builder") is snapshot
    assert Catalog.discover(tmp_path).roles.get("builder").description == "Changed builder"


def test_project_catalog_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        """
from aeloon_core.customization.roles import Role

class One(Role):
    id = "duplicate"
    description = "one"
    system_prompt = "one"

class Two(Role):
    id = "duplicate"
    description = "two"
    system_prompt = "two"

ROLES = (One, Two)
WORKFLOWS = ()
""",
    )

    with pytest.raises(CatalogDefinitionError, match="duplicate project role id"):
        Catalog.discover(tmp_path)


def test_legacy_markdown_definitions_fail_with_migration_message(
    tmp_path: Path,
) -> None:
    worker_root = tmp_path / ".aeloon-core" / "workers"
    worker_root.mkdir(parents=True)
    (worker_root / "builder.md").write_text("---\nid: builder\n---\nold\n")

    with pytest.raises(CatalogDefinitionError, match="catalog.py"):
        Catalog.discover(tmp_path)


def test_project_catalog_import_failure_is_actionable(tmp_path: Path) -> None:
    _write_catalog(tmp_path, "raise RuntimeError('broken catalog')\n")

    with pytest.raises(
        CatalogDefinitionError,
        match="RuntimeError: broken catalog",
    ):
        Catalog.discover(tmp_path)


def test_registry_mapping_and_unknown_lookup_are_explicit() -> None:
    registry = RoleRegistry.from_types((CustomRole,))

    with pytest.raises(TypeError):
        registry.roles["other"] = registry.get("custom")
    with pytest.raises(KeyError, match="available: custom"):
        registry.get("missing")


def _write_catalog(root: Path, source: str) -> Path:
    path = root / ".aeloon-core" / "catalog.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.strip() + "\n", encoding="utf-8")
    return path
