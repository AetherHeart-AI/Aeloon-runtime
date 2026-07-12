"""Trusted profiles bundled with Aeloon Core."""

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
BUILTIN_PROFILE_IDS = frozenset({"coding", "research"})
DEFAULT_PROFILE_APPROVER = "aeloon-core:builtin:coding"


def builtin_profile_source(profile_id: str) -> str:
    """Read one immutable profile shipped with this package."""

    if profile_id not in BUILTIN_PROFILE_IDS:
        raise ValueError(f"unknown built-in profile: {profile_id!r}")
    resource = files("aeloon_core.builtin_profiles").joinpath(profile_id, "PROFILE.md")
    return resource.read_text(encoding="utf-8")


def coding_profile_source() -> str:
    """Read the immutable coding profile shipped with this package."""

    return builtin_profile_source(DEFAULT_PROFILE_ID)


def research_profile_source() -> str:
    """Read the immutable research profile shipped with this package."""

    return builtin_profile_source("research")


def materialize_builtin_profile(workspace: Path, profile_id: str) -> Path:
    """Copy bundled source into a workspace without overwriting user edits."""

    source = builtin_profile_source(profile_id)
    target = workspace / ".aeloon-core" / "profiles" / profile_id / "PROFILE.md"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(source)
    except FileExistsError:
        pass
    return target


def materialize_coding_profile(workspace: Path) -> Path:
    """Copy the bundled source into the shared workspace without overwriting edits."""

    return materialize_builtin_profile(workspace, DEFAULT_PROFILE_ID)


async def load_builtin_profile(
    store: ProfileArtifactStore,
    *,
    workspace: Path,
    profile_id: str,
) -> RuntimeProfileSpec:
    """Install a trusted built-in once, then load its active artifact."""

    source = builtin_profile_source(profile_id)
    approver = f"aeloon-core:builtin:{profile_id}"
    status = store.status(profile_id)
    if status["active"]:
        active = store.inspect(status["artifact_id"])
        approval = active.get("approval") or {}
        package_owned = approval.get("approved_by") == approver
        source_matches = active["manifest"]["identity"][
            "canonical_profile_hash"
        ] == canonical_profile_hash(parse_profile(source))
        if not package_owned or (active["compatible"] and source_matches):
            return store.load_active(profile_id)

    try:
        materialize_builtin_profile(workspace, profile_id)
    except OSError as exc:
        logger.warning("Could not materialize bundled {} profile: {}", profile_id, exc)
    artifact = await store.compile(source)
    if artifact["state"] == "validated":
        artifact = store.approve(
            artifact["artifact_id"],
            approved_by=approver,
        )
    if artifact["state"] == "approved":
        store.activate(artifact["artifact_id"])
    return store.load_active(profile_id)


async def load_default_profile(
    store: ProfileArtifactStore,
    *,
    workspace: Path,
) -> RuntimeProfileSpec:
    """Install the trusted default once, then load the active immutable artifact."""

    return await load_builtin_profile(
        store,
        workspace=workspace,
        profile_id=DEFAULT_PROFILE_ID,
    )
