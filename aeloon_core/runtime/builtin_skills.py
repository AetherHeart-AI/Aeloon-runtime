"""Provision the skills bundled with the Aeloon Core distribution."""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

BUILTIN_SKILL_IDS = (
    "document-reader",
    "word-docx",
    "powerpoint-pptx",
)


def provision_builtin_skills(data_dir: Path | str) -> tuple[str, ...]:
    """Copy missing bundled skills into ``<data_dir>/skills``.

    Existing paths are never replaced. This lets users customize or replace a
    preset without a later Core startup overwriting their work.
    """

    skill_root = Path(data_dir).expanduser().resolve(strict=False) / "skills"
    skill_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_root = files("aeloon_core.resources").joinpath("skills")
    if not source_root.is_dir():
        raise RuntimeError("Bundled skill resources are missing from the Core package")

    copied: list[str] = []
    for skill_id in BUILTIN_SKILL_IDS:
        destination = skill_root / skill_id
        if os.path.lexists(destination):
            continue
        source = source_root.joinpath(skill_id)
        if not source.is_dir():
            raise RuntimeError(f"Bundled skill resource is missing: {skill_id}")

        staging_root = Path(tempfile.mkdtemp(prefix=f".{skill_id}-", dir=skill_root))
        staged = staging_root / skill_id
        try:
            _copy_resource_tree(source, staged)
            if os.path.lexists(destination):
                continue
            staged.rename(destination)
            copied.append(skill_id)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
    return tuple(copied)


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_resource_tree(child, target)
            continue
        target.write_bytes(child.read_bytes())
        target.chmod(0o600)


__all__ = ["BUILTIN_SKILL_IDS", "provision_builtin_skills"]
