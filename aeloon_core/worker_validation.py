"""Deterministic validation for Worker terminal evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic_ai import ModelRetry

from aeloon_core.worker_state import EvidenceItem
from aeloon_core.worker_terminal_tools import (
    CompleteWorkArgs,
    RequestMasterArgs,
    TerminalEvidenceItem,
)

_EXECUTION_KINDS = frozenset({"test", "typecheck", "lint", "runtime"})
_MUTATION_TOOLS = frozenset({"write", "str_replace", "exec"})
_FILE_LOCATOR = re.compile(r"^(?P<path>.+):(?P<start>\d+)(?:-(?P<end>\d+))?$")
_FINDING_ID = re.compile(r"\[(F-\d+)\]", re.IGNORECASE)
_EXIT_CODE = re.compile(r"(?:^|\n)Exit code:\s*(-?\d+)\s*$", re.MULTILINE)


def validate_worker_terminal_output(
    ctx: Any,
    output: CompleteWorkArgs | RequestMasterArgs,
    *,
    worker_type_id: str,
    workspace: Path,
    inherited_evidence: tuple[EvidenceItem, ...] = (),
) -> CompleteWorkArgs | RequestMasterArgs:
    """Reject unverifiable terminal claims while leaving judgment to the Worker."""

    current_evidence = tuple(
        EvidenceItem.model_validate(item.model_dump(mode="json"))
        for item in output.evidence
    )
    observations = tuple(getattr(ctx.deps, "tool_observations", ()))
    mutated = bool(_MUTATION_TOOLS.intersection(getattr(ctx.deps, "tools_used", ())))
    trusted_inherited = (
        tuple(item for item in inherited_evidence if item.kind != "legacy")
        if not mutated
        else ()
    )
    evidence = _merge_evidence(current_evidence, trusted_inherited)
    inherited_keys = {_evidence_key(item) for item in trusted_inherited}
    for item in evidence:
        if item.kind == "file" and item.status != "not_applicable":
            _validate_file_locator(item, workspace=workspace)
        if item.kind in _EXECUTION_KINDS and item.status in {"passed", "failed"}:
            observation = _matching_exec_observation(item, observations)
            if observation is None and _evidence_key(item) in inherited_keys:
                continue
            if observation is None:
                raise ModelRetry(
                    f"Evidence {item.locator!r} is not backed by an exec call from "
                    "this WorkerRun. Run the check and report its exact command in method."
                )
            exit_code = _exec_exit_code(str(observation.result))
            if item.status == "passed" and exit_code != 0:
                raise ModelRetry(
                    f"Evidence {item.locator!r} claims passed, but its command did "
                    f"not exit successfully (exit_code={exit_code!r})."
                )
            if item.status == "failed" and exit_code == 0:
                raise ModelRetry(
                    f"Evidence {item.locator!r} claims failed, but its command exited 0."
                )

    if evidence != current_evidence:
        output = output.model_copy(
            update={
                "evidence": [
                    TerminalEvidenceItem.model_validate(item.model_dump(mode="json"))
                    for item in evidence
                ]
            }
        )
    if not isinstance(output, CompleteWorkArgs):
        return output
    if worker_type_id == "builder":
        _validate_builder_completion(ctx, evidence)
    elif worker_type_id == "reviewer":
        _validate_reviewer_completion(output, evidence, observations)
    return output


def _merge_evidence(
    current: tuple[EvidenceItem, ...],
    inherited: tuple[EvidenceItem, ...],
) -> tuple[EvidenceItem, ...]:
    merged = list(current)
    seen = {_evidence_key(item) for item in current}
    for item in inherited:
        key = _evidence_key(item)
        if key in seen or len(merged) >= 32:
            continue
        merged.append(item)
        seen.add(key)
    return tuple(merged)


def _evidence_key(item: EvidenceItem) -> tuple[str | None, ...]:
    return (
        item.kind,
        item.locator,
        item.claim,
        item.status,
        item.method,
        item.finding_id,
    )


def _validate_builder_completion(ctx: Any, evidence: tuple[EvidenceItem, ...]) -> None:
    mutated = bool(_MUTATION_TOOLS.intersection(getattr(ctx.deps, "tools_used", ())))
    if not mutated:
        return
    by_kind = {
        kind: [item for item in evidence if item.kind == kind]
        for kind in ("test", "typecheck", "lint")
    }
    missing = [
        kind
        for kind, items in by_kind.items()
        if not any(item.status in {"passed", "not_applicable"} for item in items)
    ]
    if missing:
        raise ModelRetry(
            "Builder modified the workspace but did not account for required "
            f"verification categories: {missing}. Run them or report not_applicable "
            "with a concrete reason."
        )
    if not any(
        item.kind in _EXECUTION_KINDS and item.status == "passed"
        for item in evidence
    ):
        raise ModelRetry(
            "Builder modified the workspace but supplied no successful executable "
            "verification evidence."
        )


def _validate_reviewer_completion(
    output: CompleteWorkArgs,
    evidence: tuple[EvidenceItem, ...],
    observations: tuple[Any, ...],
) -> None:
    executed = [
        item
        for item in evidence
        if item.kind in {"test", "runtime"}
        and item.status in {"passed", "failed"}
        and _matching_exec_observation(item, observations) is not None
    ]
    if not executed:
        raise ModelRetry(
            "Reviewer completion requires an actual test or runtime check with the "
            "exact executed command in evidence.method."
        )
    finding_ids = {match.upper() for match in _FINDING_ID.findall(output.summary)}
    evidenced_ids = {
        item.finding_id.upper()
        for item in evidence
        if item.finding_id is not None
    }
    missing = sorted(finding_ids - evidenced_ids)
    if missing:
        raise ModelRetry(
            "Every reviewer finding ID must have linked evidence; missing "
            f"finding_id values: {missing}."
        )


def _matching_exec_observation(item: EvidenceItem, observations: tuple[Any, ...]) -> Any | None:
    method = item.method.strip() if item.method is not None else None
    if not method:
        return None
    for observation in observations:
        if getattr(observation, "name", None) != "exec":
            continue
        arguments = getattr(observation, "arguments", {})
        if str(arguments.get("command") or "").strip() == method:
            return observation
    return None


def _exec_exit_code(result: str) -> int | None:
    match = _EXIT_CODE.search(result)
    return int(match.group(1)) if match is not None else None


def _validate_file_locator(item: EvidenceItem, *, workspace: Path) -> None:
    match = _FILE_LOCATOR.fullmatch(item.locator)
    if match is None:
        raise ModelRetry(
            f"File evidence locator {item.locator!r} must use path:line or "
            "path:start-end."
        )
    raw_path = Path(match.group("path")).expanduser()
    candidate = raw_path if raw_path.is_absolute() else workspace / raw_path
    resolved = candidate.resolve(strict=False)
    root = workspace.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ModelRetry(
            f"File evidence {item.locator!r} points outside the shared workspace."
        ) from exc
    if not resolved.is_file():
        raise ModelRetry(f"File evidence target does not exist: {item.locator!r}.")
    try:
        line_count = len(resolved.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError) as exc:
        raise ModelRetry(
            f"File evidence target is not readable UTF-8 text: {item.locator!r}."
        ) from exc
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if start < 1 or end < start or end > line_count:
        raise ModelRetry(
            f"File evidence line range is invalid for {item.locator!r}; "
            f"file has {line_count} lines."
        )


__all__ = ["validate_worker_terminal_output"]
