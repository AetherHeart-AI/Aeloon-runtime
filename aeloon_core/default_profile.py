"""Trusted default profile bundled with Aeloon Core."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from aeloon_core.profiles import canonical_profile_hash, parse_profile

if TYPE_CHECKING:
    from aeloon_core.profile_artifacts import ProfileArtifactStore
    from aeloon_core.profiles import RuntimeProfileSpec

DEFAULT_PROFILE_ID = "coding"
DEFAULT_PROFILE_APPROVER = "aeloon-core:builtin:coding"


def coding_profile_source() -> str:
    """Read the immutable coding profile shipped with this package."""

    resource = files("aeloon_core.builtin_profiles").joinpath("coding", "PROFILE.md")
    return resource.read_text(encoding="utf-8")


def materialize_coding_profile(workspace: Path) -> Path:
    """Copy the bundled source into the shared workspace without overwriting edits."""

    target = workspace / ".aeloon-core" / "profiles" / DEFAULT_PROFILE_ID / "PROFILE.md"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(coding_profile_source())
    except FileExistsError:
        pass
    return target


async def load_default_profile(
    store: ProfileArtifactStore,
    *,
    workspace: Path,
) -> RuntimeProfileSpec:
    """Install the trusted default once, then load the active immutable artifact."""

    source = coding_profile_source()
    status = store.status(DEFAULT_PROFILE_ID)
    if status["active"]:
        active = store.inspect(status["artifact_id"])
        approval = active.get("approval") or {}
        package_owned = approval.get("approved_by") == DEFAULT_PROFILE_APPROVER
        source_matches = (
            active["manifest"]["identity"]["canonical_profile_hash"]
            == canonical_profile_hash(parse_profile(source))
        )
        if not package_owned or (active["compatible"] and source_matches):
            return store.load_active(DEFAULT_PROFILE_ID)

    try:
        materialize_coding_profile(workspace)
    except OSError as exc:
        logger.warning("Could not materialize the bundled coding profile: {}", exc)
    artifact = await store.compile(source)
    if artifact["state"] == "validated":
        artifact = store.approve(
            artifact["artifact_id"],
            approved_by=DEFAULT_PROFILE_APPROVER,
        )
    if artifact["state"] == "approved":
        store.activate(artifact["artifact_id"])
    return store.load_active(DEFAULT_PROFILE_ID)
