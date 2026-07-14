"""Immutable Profile artifacts and explicit approval/activation lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised on Unix; Windows uses msvcrt
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from aeloon_core.profile_compiler import (
    CompileOutcome,
    ProfileCompilerProvider,
    compile_profile,
    compiler_descriptor,
)
from aeloon_core.profiles import (
    LEGACY_EDIT_MIGRATION,
    PROFILE_ID_PATTERN,
    ProfileSource,
    RuntimeProfileSpec,
    canonical_profile_hash,
    parse_compiled_profile,
    parse_profile,
)

ARTIFACT_SCHEMA_VERSION = 2
DEFAULT_COMPILED_API_VERSION = 1
DEFAULT_UASM_API_VERSION = 1
DEFAULT_CONTROL_PROTOCOL_VERSION = 2
DEFAULT_VALIDATOR_VERSION = "1"
DEFAULT_GRAMMAR_VERSION = "1"
DEFAULT_AST_POLICY_VERSION = "1"
DEFAULT_RUNTIME_PROFILE_SPEC_VERSION = "2"

SOURCE_FILENAME = "source_profile.md"
COMPILED_FILENAME = "compiled_profile.py"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "validation_report.json"
SEMANTIC_DIFF_FILENAME = "semantic_diff.json"

JsonWriter = Callable[[Path, Mapping[str, Any]], None]
Clock = Callable[[], datetime]


class ProfileArtifactError(RuntimeError):
    """Base error for Profile artifact operations."""


class ProfileCompilationError(ProfileArtifactError):
    def __init__(self, message: str, *, errors=(), artifact_id: str | None = None) -> None:
        super().__init__(message)
        self.errors = tuple(errors)
        self.artifact_id = artifact_id


class ArtifactNotFoundError(ProfileArtifactError):
    pass


class ArtifactIntegrityError(ProfileArtifactError):
    pass


class ArtifactLifecycleError(ProfileArtifactError):
    pass


class ArtifactCompatibilityError(ProfileArtifactError):
    pass


class ArtifactAuditError(ProfileArtifactError):
    pass


@dataclass(frozen=True)
class CompatibilityPolicy:
    supported_profile_schema_versions: tuple[int, ...] = (1,)
    compiled_api_version: int = DEFAULT_COMPILED_API_VERSION
    uasm_api_version: int = DEFAULT_UASM_API_VERSION
    control_protocol_version: int = DEFAULT_CONTROL_PROTOCOL_VERSION
    validator_version: str = DEFAULT_VALIDATOR_VERSION
    grammar_version: str = DEFAULT_GRAMMAR_VERSION
    ast_policy_version: str = DEFAULT_AST_POLICY_VERSION
    runtime_profile_spec_version: str = DEFAULT_RUNTIME_PROFILE_SPEC_VERSION
    tool_schema_fingerprints: Mapping[str, str] | None = None


class ProfileArtifactStore:
    """Own immutable artifacts and the small mutable approval/pointer surface."""

    def __init__(
        self,
        *,
        data_dir: Path,
        compatibility: CompatibilityPolicy | None = None,
        audit_writer: JsonWriter | None = None,
        ledger_writer: JsonWriter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "profile-artifacts"
        self.cache_dir = self.root / "cache"
        self.approval_dir = self.root / "approvals"
        self.active_dir = self.root / "active"
        self.audit_dir = self.root / "activation-audit"
        self.audit_path = self.root / "activation-audit.jsonl"
        self.compile_ledger_path = self.root / "compile-ledger.jsonl"
        self.activation_lock_path = self.root / "activation.lock"
        self.compatibility = compatibility or CompatibilityPolicy()
        self._audit_writer = audit_writer or _write_json_atomic_exclusive
        self._ledger_writer = ledger_writer or _append_jsonl
        self._clock = clock or (lambda: datetime.now(UTC))
        for directory in (
            self.cache_dir,
            self.approval_dir,
            self.active_dir,
            self.audit_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    async def compile(
        self,
        source: str | Path,
        *,
        compiler: str = "deterministic",
        provider: ProfileCompilerProvider | None = None,
        model: str | None = None,
        compiler_version: str = "1",
        prompt_version: str = "1",
        max_tokens: int = 8_192,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        source_text = _read_source(source)
        source_spec = parse_profile(source_text)
        compiler_info = compiler_descriptor(
            compiler,
            model=model,
            compiler_version=compiler_version,
            prompt_version=prompt_version,
        )
        cache_key = self._cache_key(source_text, source_spec, compiler_info)
        cached = self._cached_summary(cache_key)
        if cached is not None:
            self._record_compile(
                source_spec, compiler_info, True, cached["artifact_id"], started, 0, {}
            )
            return cached

        outcome = await compile_profile(
            source_text,
            source_spec,
            backend=compiler,
            model=model,
            provider=provider,
            compiler_version=compiler_version,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
        )
        with _file_lock(self.activation_lock_path, exclusive=True):
            artifact_id = self._store_outcome(outcome, cache_key)
            if outcome.success:
                _write_json_atomic(
                    self.cache_dir / f"{cache_key}.json",
                    {"artifact_id": artifact_id},
                )
        self._record_compile(
            source_spec,
            outcome.compiler,
            False,
            artifact_id,
            started,
            outcome.repair_count,
            outcome.usage,
            errors=outcome.errors,
        )
        if not outcome.success:
            raise ProfileCompilationError(
                "profile compilation failed",
                errors=outcome.errors,
                artifact_id=artifact_id,
            )
        return self._summary(artifact_id, cache_hit=False)

    def approve(self, artifact_id: str, *, approved_by: str | None = None) -> dict[str, Any]:
        with _file_lock(self.activation_lock_path, exclusive=True):
            manifest = self._verify_integrity(artifact_id)
            if not manifest["identity"]["valid"]:
                raise ArtifactLifecycleError("quarantined artifacts cannot be approved")
            compatible, reasons = self._compatibility(manifest)
            if not compatible:
                raise ArtifactCompatibilityError("; ".join(reasons))
            approval = {
                "artifact_id": artifact_id,
                "artifact_digest": artifact_id,
                "approved_at": self._timestamp(),
                "approved_by": approved_by,
            }
            path = self._approval_path(artifact_id)
            if path.exists():
                existing = _read_json(path)
                if existing != approval and existing.get("artifact_digest") != artifact_id:
                    raise ArtifactIntegrityError("approval is bound to a different digest")
            else:
                _write_json_atomic(path, approval)
            return self._summary(artifact_id)

    def activate(self, artifact_id: str) -> dict[str, Any]:
        with _file_lock(self.activation_lock_path, exclusive=True):
            return self._activate_locked(artifact_id, action="activate")

    def rollback(self, artifact_id: str) -> dict[str, Any]:
        with _file_lock(self.activation_lock_path, exclusive=True):
            result = self._activate_locked(artifact_id, action="rollback")
            result["rollback"] = True
            return result

    def load_active(self, profile_id: str) -> RuntimeProfileSpec:
        with _file_lock(self.activation_lock_path, exclusive=False):
            pointer, manifest = self._validated_active(profile_id)
            return self._load_pinned_locked(pointer, manifest)

    def load_pinned(
        self,
        *,
        profile_id: str,
        artifact_id: str,
        generation: int,
        audit_id: str,
    ) -> RuntimeProfileSpec:
        """Load a historical Worker pin without consulting the current pointer."""

        _validate_identifier(profile_id)
        with _file_lock(self.activation_lock_path, exclusive=False):
            audit = self._read_audit(audit_id)
            if (
                audit.get("profile_id") != profile_id
                or audit.get("artifact_id") != artifact_id
                or audit.get("generation") != generation
            ):
                raise ArtifactIntegrityError("pinned artifact does not match activation audit")
            manifest = self._verify_integrity(artifact_id)
            if manifest["identity"]["profile"]["id"] != profile_id:
                raise ArtifactIntegrityError("pinned artifact belongs to a different profile")
            self._read_approval(artifact_id)
            return self._load_pinned_locked(
                {"artifact_id": artifact_id, "generation": generation}, manifest
            )

    def list_active(self) -> list[dict[str, Any]]:
        """Return active profiles through a public API rather than store layout."""

        with _file_lock(self.activation_lock_path, exclusive=False):
            result: list[dict[str, Any]] = []
            for pointer_path in sorted(self.active_dir.glob("*.json")):
                pointer = self._read_pointer(pointer_path.stem)
                if pointer is None:
                    continue
                result.append(self.status(str(pointer["profile_id"])))
            return result

    def _load_pinned_locked(
        self,
        pointer: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> RuntimeProfileSpec:
        compatible, reasons = self._compatibility(manifest)
        if not compatible:
            raise ArtifactCompatibilityError("; ".join(reasons))
        source = (self._artifact_dir(str(pointer["artifact_id"])) / COMPILED_FILENAME).read_text()
        runtime = parse_compiled_profile(
            source,
            artifact_id=str(pointer["artifact_id"]),
            generation=int(pointer["generation"]),
        )
        return runtime.model_copy(
            update={
                "control_protocol_version": int(
                    manifest["identity"]["compatibility"]["control_protocol_version"]
                )
            }
        )

    def inspect(self, artifact_id: str) -> dict[str, Any]:
        with _file_lock(self.activation_lock_path, exclusive=False):
            manifest = self._verify_integrity(artifact_id)
            directory = self._artifact_dir(artifact_id)
            compatible, reasons = self._compatibility(manifest)
            return {
                "artifact_id": artifact_id,
                "manifest": manifest,
                "state": self._state(artifact_id, manifest),
                "approval": self._read_optional(self._approval_path(artifact_id)),
                "validation_report": _read_json(directory / REPORT_FILENAME),
                "semantic_diff": _read_json(directory / SEMANTIC_DIFF_FILENAME),
                "source": (directory / SOURCE_FILENAME).read_text(),
                "compiled_source": (directory / COMPILED_FILENAME).read_text(),
                "compatible": compatible,
                "compatibility_errors": reasons,
            }

    def status(self, profile_id: str) -> dict[str, Any]:
        _validate_identifier(profile_id)
        with _file_lock(self.activation_lock_path, exclusive=False):
            pointer = self._read_pointer(profile_id)
            if pointer is None:
                return {"profile_id": profile_id, "active": False, "artifact_id": None}
            manifest = self._verify_integrity(pointer["artifact_id"])
            compatible, reasons = self._compatibility(manifest)
            return {
                **pointer,
                "profile_id": profile_id,
                "active": True,
                "state": "active",
                "compatible": compatible,
                "compatibility_errors": reasons,
            }

    def _activate_locked(self, artifact_id: str, *, action: str) -> dict[str, Any]:
        manifest = self._verify_integrity(artifact_id)
        identity = manifest["identity"]
        if not identity["valid"]:
            raise ArtifactLifecycleError("quarantined artifacts cannot be activated")
        profile_id = str(identity["profile"]["id"])
        approval = self._read_approval(artifact_id)
        if approval["artifact_digest"] != artifact_id:
            raise ArtifactIntegrityError("activation approval digest does not match artifact")
        compatible, reasons = self._compatibility(manifest)
        if not compatible:
            raise ArtifactCompatibilityError("; ".join(reasons))
        current = self._read_pointer(profile_id)
        if current is not None:
            self._validated_active(profile_id)
            if action == "activate" and current["artifact_id"] == artifact_id:
                return {
                    **current,
                    "active": True,
                    "state": "active",
                    "compatible": True,
                }
        generation = int(current["generation"] if current else 0) + 1
        payload = {
            "action": action,
            "profile_id": profile_id,
            "artifact_id": artifact_id,
            "artifact_digest": artifact_id,
            "generation": generation,
            "previous_artifact_id": current.get("artifact_id") if current else None,
            "previous_audit_id": current.get("audit_id") if current else None,
            "created_at": self._timestamp(),
        }
        audit_id = _hash_json(payload)
        audit = {"audit_id": audit_id, **payload}
        audit_file = self.audit_dir / f"{audit_id}.json"
        pointer = {
            "profile_id": profile_id,
            "artifact_id": artifact_id,
            "artifact_digest": artifact_id,
            "generation": generation,
            "audit_id": audit_id,
            "activated_at": payload["created_at"],
        }
        try:
            if not audit_file.exists():
                self._audit_writer(audit_file, audit)
            _write_json_atomic(self._pointer_path(profile_id), pointer)
        except Exception as exc:
            if self._read_pointer(profile_id) == pointer:
                return {**pointer, "active": True, "state": "active", "compatible": True}
            if action == "activate":
                raise ArtifactAuditError(f"activation publication failed: {exc}") from exc
            raise
        result = {**pointer, "active": True, "state": "active", "compatible": True}
        if action == "rollback":
            result["rolled_back_from"] = current.get("artifact_id") if current else None
        return result

    def _validated_active(self, profile_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        pointer = self._read_pointer(profile_id)
        if pointer is None:
            raise ArtifactNotFoundError(f"profile {profile_id!r} has no active artifact")
        if (
            pointer.get("profile_id") != profile_id
            or pointer.get("artifact_digest") != pointer.get("artifact_id")
            or not isinstance(pointer.get("generation"), int)
            or pointer["generation"] < 1
            or not isinstance(pointer.get("audit_id"), str)
        ):
            raise ArtifactIntegrityError("active pointer binding is invalid")
        manifest = self._verify_integrity(pointer["artifact_id"])
        if manifest["identity"]["profile"]["id"] != profile_id:
            raise ArtifactIntegrityError("active artifact belongs to a different profile")
        approval = self._read_approval(pointer["artifact_id"])
        if approval["artifact_digest"] != pointer["artifact_id"]:
            raise ArtifactIntegrityError("active approval binding is invalid")
        audit = self._read_audit(pointer["audit_id"])
        for key in ("profile_id", "artifact_id", "artifact_digest", "generation"):
            if audit.get(key) != pointer.get(key):
                raise ArtifactIntegrityError(f"active audit diverges from pointer: {key}")
        return pointer, manifest

    def _store_outcome(self, outcome: CompileOutcome, cache_key: str) -> str:
        source_bytes = outcome.source_text.encode()
        compiled_bytes = outcome.compiled_source.encode()
        report_bytes = _canonical_json_bytes(outcome.report)
        diff_bytes = _canonical_json_bytes(outcome.semantic_diff)
        runtime_spec = (
            outcome.runtime_spec.model_copy(
                update={
                    "control_protocol_version": self.compatibility.control_protocol_version
                }
            )
            if outcome.runtime_spec is not None
            else None
        )
        spec_payload = _to_plain(runtime_spec)
        identity = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_snapshot_hash": _sha256(source_bytes),
            "canonical_profile_hash": canonical_profile_hash(outcome.source),
            "compiled_source_hash": _sha256(compiled_bytes),
            "runtime_spec_hash": _hash_json(spec_payload) if spec_payload is not None else None,
            "validation_report_hash": _sha256(report_bytes),
            "semantic_diff_hash": _sha256(diff_bytes),
            "compiler": outcome.compiler,
            "cache_key": cache_key,
            "compatibility": self._compatibility_descriptor(outcome.source, outcome.runtime_spec),
            "profile": {"id": str(outcome.source.id), "revision": int(outcome.source.revision)},
            "valid": outcome.success,
        }
        artifact_id = _hash_json(identity)
        manifest = {
            "artifact_id": artifact_id,
            "artifact_digest": artifact_id,
            "created_at": self._timestamp(),
            "identity": identity,
        }
        directory = self._artifact_dir(artifact_id)
        if directory.exists():
            self._verify_integrity(artifact_id)
            return artifact_id
        temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=self.root))
        try:
            _write_bytes(temporary / SOURCE_FILENAME, source_bytes)
            _write_bytes(temporary / COMPILED_FILENAME, compiled_bytes)
            _write_bytes(temporary / REPORT_FILENAME, report_bytes + b"\n")
            _write_bytes(temporary / SEMANTIC_DIFF_FILENAME, diff_bytes + b"\n")
            _write_json_atomic(temporary / MANIFEST_FILENAME, manifest)
            os.replace(temporary, directory)
            _fsync_directory(self.root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return artifact_id

    def _compatibility_descriptor(
        self,
        source: ProfileSource,
        runtime: RuntimeProfileSpec | None,
    ) -> dict[str, Any]:
        fingerprints = self.compatibility.tool_schema_fingerprints
        requested = sorted({str(tool) for agent in source.agents for tool in agent.tools})
        return {
            "profile_schema_version": int(source.schema_version),
            "compiled_api_version": int(getattr(runtime, "compiled_api_version", 1)),
            "uasm_api_version": self.compatibility.uasm_api_version,
            "control_protocol_version": self.compatibility.control_protocol_version,
            "validator_version": self.compatibility.validator_version,
            "grammar_version": self.compatibility.grammar_version,
            "ast_policy_version": self.compatibility.ast_policy_version,
            "runtime_profile_spec_version": self.compatibility.runtime_profile_spec_version,
            "requested_tool_schema_fingerprints": {
                name: fingerprints.get(name) if fingerprints is not None else None
                for name in requested
            },
        }

    def _compatibility(self, manifest: Mapping[str, Any]) -> tuple[bool, list[str]]:
        declared = manifest["identity"]["compatibility"]
        errors: list[str] = []
        if (
            declared["profile_schema_version"]
            not in self.compatibility.supported_profile_schema_versions
        ):
            errors.append("unsupported profile schema version")
        expected = {
            "compiled_api_version": self.compatibility.compiled_api_version,
            "uasm_api_version": self.compatibility.uasm_api_version,
            "control_protocol_version": self.compatibility.control_protocol_version,
            "validator_version": self.compatibility.validator_version,
            "grammar_version": self.compatibility.grammar_version,
            "ast_policy_version": self.compatibility.ast_policy_version,
            "runtime_profile_spec_version": self.compatibility.runtime_profile_spec_version,
        }
        errors.extend(
            f"{key} mismatch: artifact={declared.get(key)!r}, host={value!r}"
            for key, value in expected.items()
            if declared.get(key) != value
        )
        host = self.compatibility.tool_schema_fingerprints
        requested_fingerprints = declared.get("requested_tool_schema_fingerprints", {})
        if (
            declared.get("control_protocol_version") == 2
            and "delegate_tasks" in requested_fingerprints
        ):
            errors.append(
                "requested tool conflicts with profile control operation: delegate_tasks"
            )
        for name, fingerprint in requested_fingerprints.items():
            current = host.get(name) if host is not None else None
            if name == "edit":
                errors.append(LEGACY_EDIT_MIGRATION)
                continue
            if fingerprint is None or current is None:
                errors.append(f"requested tool is missing: {name}")
            elif current != fingerprint:
                if name == "write":
                    errors.append(
                        "write tool schema is incompatible; increment the profile revision, "
                        "then compile, approve, and activate a new artifact"
                    )
                else:
                    errors.append(f"tool schema fingerprint changed: {name}")
        return not errors, errors

    def _cache_key(self, text: str, source: ProfileSource, compiler: Mapping[str, Any]) -> str:
        return _hash_json(
            {
                "source_snapshot_hash": _sha256(text.encode()),
                "canonical_profile_hash": canonical_profile_hash(source),
                "compiler": dict(compiler),
                "compatibility": self._compatibility_descriptor(source, None),
            }
        )

    def _cached_summary(self, cache_key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            artifact_id = _read_json(path)["artifact_id"]
            manifest = self._verify_integrity(artifact_id)
            if manifest["identity"]["cache_key"] != cache_key:
                return None
            return self._summary(artifact_id, cache_hit=True)
        except (KeyError, OSError, ValueError, ProfileArtifactError):
            return None

    def _summary(self, artifact_id: str, *, cache_hit: bool = False) -> dict[str, Any]:
        manifest = self._verify_integrity(artifact_id)
        identity = manifest["identity"]
        compatible, reasons = self._compatibility(manifest)
        return {
            "artifact_id": artifact_id,
            "artifact_digest": artifact_id,
            "profile_id": identity["profile"]["id"],
            "revision": identity["profile"]["revision"],
            "state": self._state(artifact_id, manifest),
            "compiler_backend": identity["compiler"]["backend"],
            "cache_hit": cache_hit,
            "compatible": compatible,
            "compatibility_errors": reasons,
        }

    def _state(self, artifact_id: str, manifest: Mapping[str, Any]) -> str:
        if not manifest["identity"]["valid"]:
            return "quarantined"
        profile_id = manifest["identity"]["profile"]["id"]
        pointer = self._read_pointer(profile_id)
        if pointer is not None and pointer.get("artifact_id") == artifact_id:
            return "active"
        return "approved" if self._approval_path(artifact_id).exists() else "validated"

    def _read_approval(self, artifact_id: str) -> dict[str, Any]:
        path = self._approval_path(artifact_id)
        if not path.exists():
            raise ArtifactLifecycleError("artifact must be approved before activation")
        approval = _read_json(path)
        if approval.get("artifact_id") != artifact_id:
            raise ArtifactIntegrityError("approval artifact id mismatch")
        return approval

    def _read_audit(self, audit_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", audit_id):
            raise ArtifactIntegrityError("invalid audit id")
        audit = _read_json(self.audit_dir / f"{audit_id}.json")
        payload = {key: value for key, value in audit.items() if key != "audit_id"}
        if audit.get("audit_id") != _hash_json(payload):
            raise ArtifactIntegrityError("audit digest mismatch")
        return audit

    def _verify_integrity(self, artifact_id: str) -> dict[str, Any]:
        directory = self._artifact_dir(artifact_id)
        if not directory.is_dir():
            raise ArtifactNotFoundError(f"artifact not found: {artifact_id}")
        try:
            manifest = _read_json(directory / MANIFEST_FILENAME)
            identity = manifest["identity"]
            checks = {
                "source_snapshot_hash": _sha256((directory / SOURCE_FILENAME).read_bytes()),
                "compiled_source_hash": _sha256((directory / COMPILED_FILENAME).read_bytes()),
                "validation_report_hash": _sha256(
                    _canonical_json_bytes(_read_json(directory / REPORT_FILENAME))
                ),
                "semantic_diff_hash": _sha256(
                    _canonical_json_bytes(_read_json(directory / SEMANTIC_DIFF_FILENAME))
                ),
            }
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"invalid artifact {artifact_id}: {exc}") from exc
        for key, actual in checks.items():
            if identity.get(key) != actual:
                raise ArtifactIntegrityError(f"artifact {artifact_id} failed {key} check")
        if (
            manifest.get("artifact_id") != artifact_id
            or manifest.get("artifact_digest") != artifact_id
        ):
            raise ArtifactIntegrityError("artifact manifest id mismatch")
        if _hash_json(identity) != artifact_id:
            raise ArtifactIntegrityError("artifact identity digest mismatch")
        return manifest

    def _artifact_dir(self, artifact_id: str) -> Path:
        if len(artifact_id) != 64 or any(char not in "0123456789abcdef" for char in artifact_id):
            raise ValueError(f"invalid artifact id: {artifact_id!r}")
        return self.root / artifact_id

    def _approval_path(self, artifact_id: str) -> Path:
        return self.approval_dir / f"{artifact_id}.json"

    def _pointer_path(self, profile_id: str) -> Path:
        _validate_identifier(profile_id)
        return self.active_dir / f"{profile_id}.json"

    def _read_pointer(self, profile_id: str) -> dict[str, Any] | None:
        path = self._pointer_path(profile_id)
        return _read_json(path) if path.exists() else None

    def _record_compile(
        self,
        source: ProfileSource,
        compiler: Mapping[str, Any],
        cache_hit: bool,
        artifact_id: str,
        started: float,
        repair_count: int,
        usage: Mapping[str, int],
        errors=(),
    ) -> None:
        self._ledger_writer(
            self.compile_ledger_path,
            {
                "type": "profile_compile",
                "profile_id": str(source.id),
                "revision": int(source.revision),
                "compiler": dict(compiler),
                "cache_hit": cache_hit,
                "success": not errors,
                "artifact_id": artifact_id,
                "duration_ms": round((time.perf_counter() - started) * 1_000, 3),
                "repair_count": repair_count,
                "usage": dict(usage),
                "errors": list(errors),
                "created_at": self._timestamp(),
            },
        )

    def _read_optional(self, path: Path) -> dict[str, Any] | None:
        return _read_json(path) if path.exists() else None

    def _timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()


@contextmanager
def _file_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        import msvcrt  # pragma: no cover

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _read_source(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if isinstance(source, str):
        return source
    raise TypeError("profile source must be text or a Path")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_bytes(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(path, _canonical_json_bytes(payload) + b"\n")


def _write_json_atomic_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ArtifactIntegrityError(f"immutable record already exists: {path.name}")
    _write_json_atomic(path, payload)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_json_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _to_plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _hash_json(value: Any) -> str:
    return _sha256(_canonical_json_bytes(value))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_plain(model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    raise TypeError(f"value is not JSON serializable: {type(value).__qualname__}")


def _validate_identifier(value: str) -> None:
    if re.fullmatch(PROFILE_ID_PATTERN, value) is None:
        raise ValueError(f"invalid profile id: {value!r}")


__all__ = [
    "ArtifactAuditError",
    "ArtifactCompatibilityError",
    "ArtifactIntegrityError",
    "ArtifactLifecycleError",
    "ArtifactNotFoundError",
    "CompatibilityPolicy",
    "ProfileArtifactError",
    "ProfileArtifactStore",
    "ProfileCompilationError",
]
