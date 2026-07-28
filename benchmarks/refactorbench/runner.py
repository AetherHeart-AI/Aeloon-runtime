"""Legacy-compatible RefactorBench runner used by the unified adapter.

The runner deliberately keeps RefactorBench tests outside the agent workspace.
It reads the benchmark's official mapping files, prepares a reusable repository
workspace, invokes each selected harness, and then executes the mapped AST test.
Every run is archived with a manifest, summary, JSONL ledgers, patches, and raw
process logs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INSTRUCTION_SETS = {
    "base": "base_mapping.py",
    "descriptive": "descriptive_mapping.py",
    "lazy": "lazy_mapping.py",
}
HARNESS_NAMES = ("aeloon", "pi", "codex", "claude")
EXTERNAL_HARNESS_EXECUTABLES = {
    "pi": "pi",
    "codex": "codex",
    "claude": "claude",
}
CACHE_MARKER = ".aeloon-refactorbench-cache.json"
MAX_CAPTURE_CHARS = 20_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "benchmarks" / "results" / "refactorbench"


@dataclass(frozen=True)
class RefactorCase:
    """One prompt/test pair from an official RefactorBench mapping."""

    instruction_set: str
    instance_id: str
    repository: str
    prompt_path: Path
    test_path: Path


@dataclass(frozen=True)
class ProcessOutcome:
    """Bounded result of one child process."""

    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


@dataclass(frozen=True)
class HarnessInvocation:
    """One non-interactive harness invocation."""

    command: list[str]
    cwd: Path
    input_text: str | None = None
    prompt_argument: bool = False


@dataclass(frozen=True)
class HarnessArtifacts:
    """Archive paths owned by one harness in one benchmark run."""

    results_path: Path
    patch_root: Path
    session_root: Path
    log_root: Path


class WorkspaceCache:
    """Own reusable git-backed workspaces created from benchmark repositories."""

    def __init__(self, root: Path, *, refactorbench_root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.refactorbench_root = refactorbench_root.expanduser().resolve()
        self.repositories_root = self.root / "repositories"
        self.marker_path = self.root / CACHE_MARKER
        self._metadata = self._open_owned_cache()

    def prepare(self, case: RefactorCase) -> tuple[Path, str]:
        """Return a clean workspace and its immutable baseline commit."""

        repository = case.repository
        source = self.refactorbench_root / "repositories" / repository
        if not source.is_dir():
            raise RuntimeError(f"Missing RefactorBench repository: {source}")

        entry = self._metadata["repositories"].get(repository)
        workspace = self.repositories_root / repository
        if entry is None:
            if workspace.exists():
                raise RuntimeError(
                    f"Unregistered cache workspace exists: {workspace}. "
                    "Use a new --cache-dir or remove it after inspection."
                )
            baseline = self._initialize_repository(source, workspace)
            self._metadata["repositories"][repository] = {"baseline": baseline}
            self._write_metadata()
        else:
            baseline = str(entry.get("baseline") or "")
            if not workspace.is_dir() or not baseline:
                raise RuntimeError(
                    f"Incomplete cache entry for {repository}. "
                    "Use a new --cache-dir or remove the cache after inspection."
                )

        _run_checked(
            ["git", "cat-file", "-e", f"{baseline}^{{commit}}"],
            cwd=workspace,
        )
        _run_checked(["git", "reset", "--hard", "--quiet", baseline], cwd=workspace)
        _run_checked(["git", "clean", "-ffdx", "--quiet"], cwd=workspace)
        return workspace, baseline

    def _open_owned_cache(self) -> dict[str, Any]:
        if self.marker_path.exists():
            try:
                metadata = json.loads(self.marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid workspace cache marker: {exc}") from None
            expected_root = str(self.refactorbench_root)
            if metadata.get("refactorbench_root") != expected_root:
                raise RuntimeError(
                    "The workspace cache belongs to a different RefactorBench checkout: "
                    f"{metadata.get('refactorbench_root')!r}. Use another --cache-dir."
                )
            if not isinstance(metadata.get("repositories"), dict):
                raise RuntimeError("Workspace cache marker has invalid repositories data.")
            return metadata

        if self.root.exists() and any(self.root.iterdir()):
            raise RuntimeError(
                f"Refusing to claim non-empty cache directory without {CACHE_MARKER}: "
                f"{self.root}"
            )
        self.repositories_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "refactorbench_root": str(self.refactorbench_root),
            "repositories": {},
        }
        self._metadata = metadata
        self._write_metadata()
        return metadata

    def _initialize_repository(self, source: Path, destination: Path) -> str:
        self.repositories_root.mkdir(parents=True, exist_ok=True)
        staging = self.repositories_root / (
            f".{destination.name}.initializing-{uuid.uuid4().hex[:8]}"
        )
        try:
            shutil.copytree(source, staging, symlinks=True)
            _run_checked(["git", "init", "--quiet"], cwd=staging)
            _run_checked(["git", "add", "--force", "--all"], cwd=staging)
            _run_checked(
                [
                    "git",
                    "-c",
                    "user.name=Aeloon RefactorBench",
                    "-c",
                    "user.email=refactorbench@aeloon.local",
                    "commit",
                    "--quiet",
                    "--message",
                    "RefactorBench baseline",
                ],
                cwd=staging,
            )
            baseline = _run_checked(["git", "rev-parse", "HEAD"], cwd=staging).strip()
            os.replace(staging, destination)
            return baseline
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _write_metadata(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.marker_path.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
        temporary.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.marker_path)


def load_cases(refactorbench_root: Path, instruction_set: str) -> list[RefactorCase]:
    """Load cases without importing or executing benchmark mapping code."""

    root = refactorbench_root.expanduser().resolve()
    mapping_name = INSTRUCTION_SETS.get(instruction_set)
    if mapping_name is None:
        raise ValueError(f"Unknown instruction set: {instruction_set}")
    mapping_path = root / "scripts" / mapping_name
    if not mapping_path.is_file():
        raise RuntimeError(f"Missing official RefactorBench mapping: {mapping_path}")

    mapping = _literal_file_mapping(mapping_path)
    cases: list[RefactorCase] = []
    seen_ids: set[str] = set()
    for raw_test, raw_prompt in mapping.items():
        test_path = _resolve_mapped_path(root, mapping_path.parent, raw_test)
        prompt_path = _resolve_mapped_path(root, mapping_path.parent, raw_prompt)
        if not test_path.is_file():
            raise RuntimeError(f"Mapped RefactorBench test does not exist: {test_path}")
        if not prompt_path.is_file():
            raise RuntimeError(f"Mapped RefactorBench prompt does not exist: {prompt_path}")

        repository = prompt_path.parent.name
        if test_path.parent.name != repository:
            raise RuntimeError(
                f"Prompt/test repository mismatch: {prompt_path} -> {test_path}"
            )
        task_name = prompt_path.stem
        if task_name.endswith("-task"):
            task_name = task_name[: -len("-task")]
        instance_id = f"{repository}/{task_name}"
        if instance_id in seen_ids:
            raise RuntimeError(f"Duplicate RefactorBench instance id: {instance_id}")
        seen_ids.add(instance_id)
        cases.append(
            RefactorCase(
                instruction_set=instruction_set,
                instance_id=instance_id,
                repository=repository,
                prompt_path=prompt_path,
                test_path=test_path,
            )
        )
    return sorted(cases, key=lambda item: item.instance_id)


def select_cases(
    cases: list[RefactorCase],
    *,
    repositories: list[str],
    instance_ids: list[str],
    limit: int | None,
) -> list[RefactorCase]:
    """Apply deterministic command-line filters."""

    selected = [
        case
        for case in cases
        if (not repositories or case.repository in repositories)
        and (not instance_ids or case.instance_id in instance_ids)
    ]
    if instance_ids:
        missing = sorted(set(instance_ids) - {case.instance_id for case in selected})
        if missing:
            raise RuntimeError(f"Unknown or filtered RefactorBench cases: {', '.join(missing)}")
    return selected[:limit] if limit is not None else selected


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run selected cases and archive one durable JSON record per harness/case."""

    benchmark_root = args.refactorbench_root.expanduser().resolve()
    harnesses = _selected_harnesses(args.harness)
    cases = select_cases(
        load_cases(benchmark_root, args.instruction_set),
        repositories=args.repository,
        instance_ids=args.case,
        limit=args.limit,
    )
    if not cases:
        raise RuntimeError("No RefactorBench cases matched the filters.")

    if args.list:
        return {
            "schema_version": 2,
            "benchmark": "refactorbench",
            "instruction_set": args.instruction_set,
            "harnesses": harnesses,
            "cases": [
                {
                    "instance_id": case.instance_id,
                    "repository": case.repository,
                    "prompt_path": str(case.prompt_path),
                    "test_path": str(case.test_path),
                }
                for case in cases
            ],
        }

    if args.results is not None and len(harnesses) != 1:
        raise RuntimeError("--results is only supported when exactly one harness is selected.")

    archive_root = _resolve_archive_root(args)
    existing_manifest = _validate_archive_root(
        archive_root,
        resume=args.resume,
        overwrite=args.overwrite,
        benchmark_root=benchmark_root,
        instruction_set=args.instruction_set,
    )
    if existing_manifest is not None:
        _validate_archive_selection(
            existing_manifest,
            harnesses=harnesses,
            cases=cases,
        )
    artifacts = {
        harness: _harness_artifacts(
            archive_root,
            harness,
            legacy_results=args.results if len(harnesses) == 1 else None,
        )
        for harness in harnesses
    }
    executables = {
        harness: _resolve_harness_executable(harness) for harness in harnesses
    }
    versions = {
        harness: _harness_version(harness, executables[harness])
        for harness in harnesses
    }
    if args.overwrite:
        for harness_artifacts in artifacts.values():
            _clear_artifacts(harness_artifacts)
    completed_by_harness = {
        harness: _prepare_results_file(
            harness_artifacts.results_path,
            resume=args.resume,
            overwrite=args.overwrite,
            instruction_set=args.instruction_set,
            harness=harness,
        )
        for harness, harness_artifacts in artifacts.items()
    }
    previously_completed = {
        harness: set(completed) for harness, completed in completed_by_harness.items()
    }
    harness_metadata = [
        {
            "id": harness,
            "executable": executables[harness],
            "version": versions[harness],
        }
        for harness in harnesses
    ]
    now = datetime.now(UTC).isoformat()
    manifest = {
        **(existing_manifest or {}),
        "schema_version": 2,
        "status": "running",
        "run_id": archive_root.name,
        "created_at": (existing_manifest or {}).get("created_at", now),
        "last_started_at": now,
        "benchmark": "refactorbench",
        "refactorbench_root": str(benchmark_root),
        "instruction_set": args.instruction_set,
        "harnesses": harness_metadata,
        "cases": [case.instance_id for case in cases],
        "aeloon_core_commit": _git_revision(PROJECT_ROOT),
    }
    _write_json(archive_root / "manifest.json", manifest)

    executed_by_harness: dict[str, list[dict[str, Any]]] = {
        harness: [] for harness in harnesses
    }

    try:
        cache = WorkspaceCache(args.cache_dir, refactorbench_root=benchmark_root)
        for harness in harnesses:
            harness_artifacts = artifacts[harness]
            harness_artifacts.patch_root.mkdir(parents=True, exist_ok=True)
            harness_artifacts.session_root.mkdir(parents=True, exist_ok=True)
            harness_artifacts.log_root.mkdir(parents=True, exist_ok=True)
            completed = completed_by_harness[harness]
            pending = [case for case in cases if case.instance_id not in completed]
            for index, case in enumerate(pending, start=1):
                print(
                    f"[{harness} {index}/{len(pending)}] "
                    f"{case.instruction_set}:{case.instance_id}",
                    file=sys.stderr,
                    flush=True,
                )
                workspace, baseline = cache.prepare(case)
                record = _run_case(
                    case,
                    harness=harness,
                    executable=executables[harness],
                    cli_version=versions[harness],
                    workspace=workspace,
                    baseline=baseline,
                    config_path=args.config,
                    archive_root=archive_root,
                    artifacts=harness_artifacts,
                    agent_timeout=args.agent_timeout,
                    test_timeout=args.test_timeout,
                )
                _append_jsonl(harness_artifacts.results_path, record)
                completed[case.instance_id] = record
                executed_by_harness[harness].append(record)
                verdict = "PASS" if record["evaluation"]["passed"] else "FAIL"
                print(
                    f"  {verdict} agent={record['agent']['status']} "
                    f"wall={record['agent']['wall_time_ms']}ms",
                    file=sys.stderr,
                    flush=True,
                )
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(archive_root / "manifest.json", manifest)
        raise

    harness_summaries = {
        harness: _summarize_harness(
            cases=cases,
            completed=completed_by_harness[harness],
            previously_completed=previously_completed[harness],
            executed=executed_by_harness[harness],
            results_path=artifacts[harness].results_path,
        )
        for harness in harnesses
    }
    summary: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "refactorbench",
        "run_id": archive_root.name,
        "instruction_set": args.instruction_set,
        "selected_cases": len(cases),
        "archive": str(archive_root),
        "harnesses": harness_summaries,
    }
    if len(harnesses) == 1:
        summary.update(harness_summaries[harnesses[0]])
    _write_json(archive_root / "summary.json", summary)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(archive_root / "manifest.json", manifest)
    return summary


def _run_case(
    case: RefactorCase,
    *,
    harness: str,
    executable: str,
    cli_version: str | None,
    workspace: Path,
    baseline: str,
    config_path: Path | None,
    archive_root: Path,
    artifacts: HarnessArtifacts,
    agent_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    safe_id = case.instance_id.replace("/", "__")
    prompt = case.prompt_path.read_text(encoding="utf-8")
    invocation = _build_harness_invocation(
        harness,
        executable=executable,
        workspace=workspace,
        prompt=prompt,
        data_dir=artifacts.session_root / safe_id,
        config_path=config_path,
    )
    agent_process = _run_process(
        invocation.command,
        cwd=invocation.cwd,
        timeout=agent_timeout,
        input_text=invocation.input_text,
    )
    stdout_path = artifacts.log_root / f"{case.instruction_set}__{safe_id}.stdout.log"
    stderr_path = artifacts.log_root / f"{case.instruction_set}__{safe_id}.stderr.log"
    stdout_path.write_text(agent_process.stdout, encoding="utf-8")
    stderr_path.write_text(agent_process.stderr, encoding="utf-8")
    agent = _interpret_harness_output(harness, agent_process)
    agent.update(
        {
            "harness": harness,
            "version": cli_version,
            "command": _display_command(invocation),
            "wall_time_ms": agent_process.duration_ms,
            "returncode": agent_process.returncode,
            "timed_out": agent_process.timed_out,
            "stdout": (
                _bounded(agent_process.stdout) if agent.get("payload_error") else None
            ),
            "stderr": _bounded(agent_process.stderr) if agent_process.stderr else None,
            "stdout_path": _archive_path(stdout_path, archive_root),
            "stderr_path": _archive_path(stderr_path, archive_root),
        }
    )

    patch, changed_files, patch_error = _capture_patch(workspace, baseline)
    patch_path = artifacts.patch_root / f"{case.instruction_set}__{safe_id}.patch"
    patch_path.write_text(patch, encoding="utf-8")

    evaluation = _run_official_test(
        test_path=case.test_path,
        workspace=workspace,
        timeout=test_timeout,
    )
    status = agent["status"]
    oracle_passed = evaluation.returncode == 0 and not evaluation.timed_out
    passed = status == "completed" and oracle_passed
    return {
        "schema_version": 2,
        "recorded_at": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "benchmark": "refactorbench",
        "harness": harness,
        "harness_version": cli_version,
        "instruction_set": case.instruction_set,
        "instance_id": case.instance_id,
        "repository": case.repository,
        "prompt": prompt,
        "prompt_path": str(case.prompt_path),
        "test_path": str(case.test_path),
        "workspace": str(workspace),
        "baseline_commit": baseline,
        "aeloon_core_commit": _git_revision(PROJECT_ROOT),
        "config_path": (
            str(config_path.expanduser().resolve())
            if harness == "aeloon" and config_path is not None
            else None
        ),
        "agent": agent,
        "evaluation": {
            "passed": passed,
            "oracle_passed": oracle_passed,
            "returncode": evaluation.returncode,
            "timed_out": evaluation.timed_out,
            "duration_ms": evaluation.duration_ms,
            "stdout": _bounded(evaluation.stdout),
            "stderr": _bounded(evaluation.stderr),
        },
        "changed_files": changed_files,
        "patch_path": _archive_path(patch_path, archive_root),
        "patch_error": patch_error,
        "false_completed": status == "completed" and not oracle_passed,
    }


def _selected_harnesses(raw_harnesses: list[str] | None) -> list[str]:
    requested = raw_harnesses or ["aeloon"]
    if "all" in requested:
        return list(HARNESS_NAMES)
    selected: list[str] = []
    for harness in requested:
        if harness not in HARNESS_NAMES:
            raise RuntimeError(f"Unknown harness: {harness}")
        if harness not in selected:
            selected.append(harness)
    return selected


def _resolve_archive_root(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if args.results is not None:
        results_path = args.results.expanduser().resolve()
        return results_path.parent / f"{results_path.stem}.archive"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (DEFAULT_RESULTS_ROOT / f"{timestamp}-{uuid.uuid4().hex[:8]}").resolve()


def _validate_archive_root(
    root: Path,
    *,
    resume: bool,
    overwrite: bool,
    benchmark_root: Path,
    instruction_set: str,
) -> dict[str, Any] | None:
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"Archive path is not a directory: {root}")
    if not root.exists() or not any(root.iterdir()):
        return None
    manifest_path = root / "manifest.json"
    if not resume and not overwrite:
        raise RuntimeError(
            f"Archive directory already exists and is not empty: {root}. "
            "Use --resume or --overwrite."
        )
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Refusing to claim non-empty archive directory without manifest.json: {root}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid archive manifest: {exc}") from None
    if not isinstance(manifest, dict):
        raise RuntimeError("Archive manifest must be a JSON object.")
    if manifest.get("refactorbench_root") != str(benchmark_root):
        raise RuntimeError("Archive belongs to a different RefactorBench checkout.")
    if manifest.get("instruction_set") != instruction_set:
        raise RuntimeError("Archive belongs to a different instruction set.")
    return manifest


def _validate_archive_selection(
    manifest: dict[str, Any],
    *,
    harnesses: list[str],
    cases: list[RefactorCase],
) -> None:
    archived_harnesses = [
        entry.get("id")
        for entry in (manifest.get("harnesses") or [])
        if isinstance(entry, dict)
    ]
    if archived_harnesses != harnesses:
        raise RuntimeError(
            "Archive harness selection differs from this run; use a new --output-dir."
        )
    archived_cases = manifest.get("cases")
    selected_cases = [case.instance_id for case in cases]
    if archived_cases != selected_cases:
        raise RuntimeError(
            "Archive case selection differs from this run; use a new --output-dir."
        )


def _harness_artifacts(
    archive_root: Path,
    harness: str,
    *,
    legacy_results: Path | None,
) -> HarnessArtifacts:
    harness_root = archive_root / harness
    results_path = (
        legacy_results.expanduser().resolve()
        if legacy_results is not None
        else harness_root / "results.jsonl"
    )
    return HarnessArtifacts(
        results_path=results_path,
        patch_root=harness_root / "patches",
        session_root=harness_root / "sessions",
        log_root=harness_root / "logs",
    )


def _clear_artifacts(artifacts: HarnessArtifacts) -> None:
    for path in (artifacts.patch_root, artifacts.session_root, artifacts.log_root):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise RuntimeError(f"Refusing to replace unexpected artifact path: {path}")
        if path.is_dir():
            shutil.rmtree(path)


def _resolve_harness_executable(harness: str) -> str:
    if harness == "aeloon":
        return sys.executable
    command = EXTERNAL_HARNESS_EXECUTABLES[harness]
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(
            f"Harness {harness!r} requires the {command!r} CLI on PATH."
        )
    return executable


def _harness_version(harness: str, executable: str) -> str | None:
    if harness == "aeloon":
        revision = _git_revision(PROJECT_ROOT)
        return f"aeloon-core@{revision}" if revision else "aeloon-core"
    outcome = _run_process(
        [executable, "--version"],
        cwd=PROJECT_ROOT,
        timeout=10.0,
    )
    if outcome.returncode != 0 or outcome.timed_out:
        return None
    version = outcome.stdout.strip() or outcome.stderr.strip()
    return _bounded(version, limit=500) or None


def _build_harness_invocation(
    harness: str,
    *,
    executable: str,
    workspace: Path,
    prompt: str,
    data_dir: Path,
    config_path: Path | None,
) -> HarnessInvocation:
    if harness == "aeloon":
        command = [
            executable,
            "-m",
            "aeloon_core",
            "run",
            "--workspace",
            str(workspace),
            "--data-dir",
            str(data_dir),
            "--stdin",
            "--output",
            "json",
        ]
        if config_path is not None:
            command.extend(["--config", str(config_path.expanduser().resolve())])
        return HarnessInvocation(
            command=command,
            cwd=PROJECT_ROOT,
            input_text=prompt,
        )
    if harness == "pi":
        return HarnessInvocation(
            command=[
                executable,
                "--print",
                "--mode",
                "json",
                "--no-session",
                "--approve",
                prompt,
            ],
            cwd=workspace,
            prompt_argument=True,
        )
    if harness == "codex":
        return HarnessInvocation(
            command=[
                executable,
                "exec",
                "--sandbox",
                "workspace-write",
                "--ephemeral",
                "--json",
                "-",
            ],
            cwd=workspace,
            input_text=prompt,
        )
    if harness == "claude":
        return HarnessInvocation(
            command=[
                executable,
                "--print",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--dangerously-skip-permissions",
                prompt,
            ],
            cwd=workspace,
            prompt_argument=True,
        )
    raise RuntimeError(f"Unknown harness: {harness}")


def _display_command(invocation: HarnessInvocation) -> list[str]:
    if not invocation.prompt_argument:
        return invocation.command
    return [*invocation.command[:-1], "<prompt>"]


def _interpret_harness_output(
    harness: str,
    outcome: ProcessOutcome,
) -> dict[str, Any]:
    if harness == "aeloon":
        payload, payload_error = _parse_agent_payload(outcome)
        return {
            "status": _agent_status(outcome, payload),
            "session_id": (payload or {}).get("session_id"),
            "turn_id": (payload or {}).get("turn_id"),
            "duration_ms": (payload or {}).get("duration_ms"),
            "final_content": (payload or {}).get("final_content"),
            "tools_used": (payload or {}).get("tools_used", []),
            "usage": (payload or {}).get("usage", {}),
            "transitions": (payload or {}).get("transitions", []),
            "models": (payload or {}).get("models", {}),
            "payload_error": payload_error,
        }

    process_error = _process_failure(outcome)
    if process_error is not None:
        return {
            "status": "timeout" if outcome.timed_out else "process_error",
            "final_content": None,
            "usage": {},
            "payload_error": process_error,
        }
    payloads, payload_error = _json_payloads(outcome.stdout)
    if payload_error is not None:
        return {
            "status": "invalid_output",
            "final_content": None,
            "usage": {},
            "payload_error": payload_error,
        }
    if harness == "pi":
        return _interpret_pi_payloads(payloads)
    if harness == "codex":
        return _interpret_codex_payloads(payloads)
    if harness == "claude":
        return _interpret_claude_payloads(payloads)
    raise RuntimeError(f"Unknown harness: {harness}")


def _process_failure(outcome: ProcessOutcome) -> str | None:
    if outcome.timed_out:
        return "agent process timed out"
    if outcome.returncode != 0:
        return f"agent process exited with code {outcome.returncode}"
    return None


def _json_payloads(stdout: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                return [], f"agent JSONL was invalid at line {line_number}: {exc}"
            if not isinstance(value, dict):
                return [], f"agent JSONL line {line_number} was not an object"
            payloads.append(value)
        if not payloads:
            return [], "agent stdout contained no JSON objects"
        return payloads, None
    if not isinstance(payload, dict):
        return [], "agent JSON payload was not an object"
    return [payload], None


def _interpret_pi_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    assistant: dict[str, Any] | None = None
    session_id: str | None = None
    for payload in payloads:
        session_id = session_id or _first_string(
            payload,
            "id",
            "session_id",
            "sessionId",
        )
        if payload.get("type") == "message_end":
            message = payload.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant = message
        if payload.get("type") == "agent_end" and assistant is None:
            messages = payload.get("messages")
            if isinstance(messages, list):
                assistant = next(
                    (
                        message
                        for message in reversed(messages)
                        if isinstance(message, dict)
                        and message.get("role") == "assistant"
                    ),
                    None,
                )
    if assistant is None:
        return {
            "status": "invalid_output",
            "session_id": session_id,
            "final_content": None,
            "usage": {},
            "payload_error": "Pi JSON stream contained no final assistant message",
        }
    stop_reason = assistant.get("stopReason") or assistant.get("stop_reason")
    usage = (
        dict(assistant["usage"])
        if isinstance(assistant.get("usage"), dict)
        else {}
    )
    if isinstance(usage.get("input"), int | float):
        usage["input_tokens"] = usage["input"]
    if isinstance(usage.get("output"), int | float):
        usage["output_tokens"] = usage["output"]
    return {
        "status": (
            "agent_error" if stop_reason in {"error", "aborted"} else "completed"
        ),
        "session_id": session_id,
        "final_content": _content_text(assistant.get("content")),
        "usage": usage,
        "cost_usd": (
            usage.get("cost", {}).get("total")
            if isinstance(usage.get("cost"), dict)
            else None
        ),
        "models": {
            key: assistant[key]
            for key in ("provider", "model")
            if isinstance(assistant.get(key), str)
        },
        "payload_error": (
            str(assistant.get("errorMessage") or stop_reason)
            if stop_reason in {"error", "aborted"}
            else None
        ),
    }


def _interpret_codex_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    final_content: str | None = None
    usage: dict[str, Any] = {}
    thread_id: str | None = None
    completed = False
    failure: str | None = None
    for payload in payloads:
        event_type = payload.get("type")
        if event_type == "thread.started":
            thread_id = _first_string(payload, "thread_id", "threadId", "id")
        elif event_type == "item.completed":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                final_content = _first_string(item, "text", "content")
        elif event_type == "turn.completed":
            completed = True
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
        elif event_type in {"turn.failed", "error"}:
            failure = _first_string(payload, "message", "error") or str(payload)
    return {
        "status": "agent_error" if failure else ("completed" if completed else "invalid_output"),
        "session_id": thread_id,
        "final_content": final_content,
        "usage": usage,
        "payload_error": failure or (None if completed else "Codex JSON stream did not complete"),
    }


def _interpret_claude_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    payload = payloads[-1]
    success = payload.get("type") == "result" and not bool(payload.get("is_error"))
    subtype = payload.get("subtype")
    if isinstance(subtype, str) and subtype not in {"success", "completed"}:
        success = False
    return {
        "status": "completed" if success else "agent_error",
        "session_id": _first_string(payload, "session_id", "sessionId"),
        "duration_ms": payload.get("duration_ms"),
        "final_content": payload.get("result"),
        "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
        "models": (
            payload.get("modelUsage")
            if isinstance(payload.get("modelUsage"), dict)
            else {}
        ),
        "cost_usd": payload.get("total_cost_usd"),
        "payload_error": None if success else str(subtype or "Claude returned an error result"),
    }


def _first_string(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    return None


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return "\n".join(parts) or None


def _archive_path(path: Path, archive_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(archive_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _summarize_harness(
    *,
    cases: list[RefactorCase],
    completed: dict[str, dict[str, Any]],
    previously_completed: set[str],
    executed: list[dict[str, Any]],
    results_path: Path,
) -> dict[str, Any]:
    selected_ids = {case.instance_id for case in cases}
    selected_records = [
        completed[case.instance_id] for case in cases if case.instance_id in completed
    ]
    passed = sum(
        bool(record.get("evaluation", {}).get("passed"))
        for record in selected_records
    )
    false_completed = sum(
        bool(record.get("false_completed")) for record in selected_records
    )
    return {
        "recorded_cases": len(selected_records),
        "skipped_completed": len(previously_completed & selected_ids),
        "executed_cases": len(executed),
        "passed": passed,
        "pass_rate": passed / len(selected_records) if selected_records else None,
        "false_completed": false_completed,
        "agent_wall_time_ms": sum(
            int(record.get("agent", {}).get("wall_time_ms") or 0)
            for record in selected_records
        ),
        "input_tokens": _sum_usage_metric(selected_records, "input_tokens", "inputTokens"),
        "output_tokens": _sum_usage_metric(
            selected_records,
            "output_tokens",
            "outputTokens",
        ),
        "cost_usd": _sum_agent_metric(selected_records, "cost_usd"),
        "results": str(results_path),
    }


def _sum_usage_metric(
    records: list[dict[str, Any]],
    *keys: str,
) -> int | float | None:
    values: list[int | float] = []
    for record in records:
        usage = record.get("agent", {}).get("usage", {})
        if not isinstance(usage, dict):
            continue
        value = next(
            (
                usage[key]
                for key in keys
                if isinstance(usage.get(key), int | float)
                and not isinstance(usage.get(key), bool)
            ),
            None,
        )
        if value is not None:
            values.append(value)
    return sum(values) if values else None


def _sum_agent_metric(
    records: list[dict[str, Any]],
    key: str,
) -> int | float | None:
    values = [
        value
        for record in records
        if isinstance(
            value := record.get("agent", {}).get(key),
            int | float,
        )
        and not isinstance(value, bool)
    ]
    return sum(values) if values else None


def _run_official_test(
    *,
    test_path: Path,
    workspace: Path,
    timeout: float,
) -> ProcessOutcome:
    # RefactorBench tests use ../ paths. A temporary cwd exactly one level under
    # the workspace preserves that contract without copying tests into the repo.
    with tempfile.TemporaryDirectory(
        prefix=".aeloon-refactorbench-verify-",
        dir=workspace,
    ) as verification_dir:
        return _run_process(
            [sys.executable, str(test_path)],
            cwd=Path(verification_dir),
            timeout=timeout,
        )


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
) -> ProcessOutcome:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
        return ProcessOutcome(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        stdout, stderr = process.communicate()
        return ProcessOutcome(
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
            timed_out=True,
        )


def _parse_agent_payload(
    outcome: ProcessOutcome,
) -> tuple[dict[str, Any] | None, str | None]:
    if outcome.timed_out:
        return None, "agent process timed out"
    if outcome.returncode != 0:
        return None, f"agent process exited with code {outcome.returncode}"
    try:
        payload = json.loads(outcome.stdout)
    except json.JSONDecodeError as exc:
        return None, f"agent stdout was not JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "agent JSON payload was not an object"
    return payload, None


def _agent_status(
    outcome: ProcessOutcome,
    payload: dict[str, Any] | None,
) -> str:
    if outcome.timed_out:
        return "timeout"
    if outcome.returncode != 0:
        return "process_error"
    if payload is None:
        return "invalid_output"
    return str(payload.get("status") or "unknown")


def _capture_patch(workspace: Path, baseline: str) -> tuple[str, list[str], str | None]:
    try:
        # Intent-to-add makes new files visible to `git diff` without staging
        # their content. The next case resets the runner-owned index anyway.
        _run_checked(
            ["git", "add", "--intent-to-add", "--force", "--all"],
            cwd=workspace,
        )
        patch = _run_checked(
            ["git", "diff", "--binary", "--no-ext-diff", baseline, "--"],
            cwd=workspace,
        )
        names = _run_checked(
            ["git", "diff", "--name-only", baseline, "--"],
            cwd=workspace,
        )
        return patch, [line for line in names.splitlines() if line], None
    except RuntimeError as exc:
        return "", [], str(exc)


def _literal_file_mapping(mapping_path: Path) -> dict[str, str]:
    try:
        tree = ast.parse(mapping_path.read_text(encoding="utf-8"), filename=str(mapping_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeError(f"Could not parse RefactorBench mapping: {exc}") from None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "file_mapping"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as exc:
            raise RuntimeError(f"file_mapping must be a literal dictionary: {exc}") from None
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise RuntimeError("file_mapping must contain only string paths.")
        return value
    raise RuntimeError(f"No literal file_mapping assignment found in {mapping_path}")


def _resolve_mapped_path(root: Path, scripts_dir: Path, raw_path: str) -> Path:
    path = (scripts_dir / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise RuntimeError(f"Mapped path escapes RefactorBench root: {raw_path}") from None
    return path


def _prepare_results_file(
    path: Path,
    *,
    resume: bool,
    overwrite: bool,
    instruction_set: str,
    harness: str,
) -> dict[str, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {}
    if overwrite:
        path.write_text("", encoding="utf-8")
        return {}
    if not resume:
        raise RuntimeError(
            f"Results file already exists: {path}. Use --resume or --overwrite."
        )

    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from None
        if (
            isinstance(record, dict)
            and record.get("instruction_set") == instruction_set
            and (
                record.get("harness") == harness
                or (harness == "aeloon" and record.get("harness") is None)
            )
            and isinstance(record.get("instance_id"), str)
        ):
            completed[record["instance_id"]] = record
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_checked(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed in {cwd}: {detail}")
    return completed.stdout


def _git_revision(workspace: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _bounded(value: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"[... {omitted} earlier characters omitted ...]\n{value[-limit:]}"


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def _positive_limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official RefactorBench AST tests through coding harness CLIs."
    )
    parser.add_argument(
        "--refactorbench-root",
        type=Path,
        required=True,
        help="Official microsoft/RefactorBench checkout.",
    )
    parser.add_argument(
        "--instruction-set",
        choices=tuple(INSTRUCTION_SETS),
        default="base",
        help="Official prompt variant (default: base).",
    )
    parser.add_argument(
        "--repository",
        action="append",
        default=[],
        help="Run only this repository id; repeat to select more.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this exact repository/task instance id; repeat to select more.",
    )
    parser.add_argument("--limit", type=_positive_limit, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected cases without creating workspaces or running agents.",
    )
    parser.add_argument(
        "--harness",
        action="append",
        choices=(*HARNESS_NAMES, "all"),
        default=None,
        help=(
            "Harness to run; repeat to compare several, or use 'all'. "
            "Defaults to aeloon."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Aeloon Core config JSON path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".benchmark-workspaces" / "refactorbench",
        help="Runner-owned reusable repository workspaces.",
    )
    archive_location = parser.add_mutually_exclusive_group()
    archive_location.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Self-contained run archive. Defaults to a timestamped directory under "
            "benchmarks/results/refactorbench."
        ),
    )
    archive_location.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Legacy single-harness JSONL path. Prefer --output-dir; artifacts are still "
            "written to a sibling .archive directory."
        ),
    )
    parser.add_argument("--agent-timeout", type=_positive_timeout, default=900.0)
    parser.add_argument("--test-timeout", type=_positive_timeout, default=60.0)
    result_mode = parser.add_mutually_exclusive_group()
    result_mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already present in the result ledger.",
    )
    result_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace managed ledgers in an existing compatible archive.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = run_benchmark(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

