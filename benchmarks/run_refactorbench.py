"""Run official RefactorBench tasks through the public Aeloon Core CLI.

The runner deliberately keeps RefactorBench tests outside the agent workspace.
It reads the benchmark's official mapping files, prepares a reusable repository
workspace, invokes ``aeloon-core run``, and then executes the mapped AST test.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
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
CACHE_MARKER = ".aeloon-refactorbench-cache.json"
MAX_CAPTURE_CHARS = 20_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    """Run selected cases and append one durable JSON record per case."""

    benchmark_root = args.refactorbench_root.expanduser().resolve()
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
            "schema_version": 1,
            "benchmark": "refactorbench",
            "instruction_set": args.instruction_set,
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

    results_path = args.results.expanduser().resolve()
    completed = _prepare_results_file(
        results_path,
        resume=args.resume,
        overwrite=args.overwrite,
        instruction_set=args.instruction_set,
    )
    previously_completed = set(completed)
    pending = [case for case in cases if case.instance_id not in completed]
    cache = WorkspaceCache(args.cache_dir, refactorbench_root=benchmark_root)
    patch_root = results_path.parent / f"{results_path.stem}.patches"
    session_root = results_path.parent / f"{results_path.stem}.sessions"
    patch_root.mkdir(parents=True, exist_ok=True)
    session_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, case in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {case.instruction_set}:{case.instance_id}",
            file=sys.stderr,
            flush=True,
        )
        workspace, baseline = cache.prepare(case)
        record = _run_case(
            case,
            workspace=workspace,
            baseline=baseline,
            config_path=args.config,
            patch_root=patch_root,
            session_root=session_root,
            agent_timeout=args.agent_timeout,
            test_timeout=args.test_timeout,
        )
        _append_jsonl(results_path, record)
        completed[case.instance_id] = record
        records.append(record)
        verdict = "PASS" if record["evaluation"]["passed"] else "FAIL"
        print(
            f"  {verdict} agent={record['agent']['status']} "
            f"wall={record['agent']['wall_time_ms']}ms",
            file=sys.stderr,
            flush=True,
        )

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
        "schema_version": 1,
        "benchmark": "refactorbench",
        "instruction_set": args.instruction_set,
        "selected_cases": len(cases),
        "recorded_cases": len(selected_records),
        "skipped_completed": len(previously_completed & {case.instance_id for case in cases}),
        "executed_cases": len(records),
        "passed": passed,
        "pass_rate": passed / len(selected_records) if selected_records else None,
        "false_completed": false_completed,
        "results": str(results_path),
    }


def _run_case(
    case: RefactorCase,
    *,
    workspace: Path,
    baseline: str,
    config_path: Path | None,
    patch_root: Path,
    session_root: Path,
    agent_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    safe_id = case.instance_id.replace("/", "__")
    data_dir = session_root / safe_id
    command = [
        sys.executable,
        "-m",
        "aeloon_core",
        "run",
        "--workspace",
        str(workspace),
        "--data-dir",
        str(data_dir),
        "--prompt-file",
        str(case.prompt_path),
        "--output",
        "json",
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path.expanduser().resolve())])

    agent_process = _run_process(
        command,
        cwd=PROJECT_ROOT,
        timeout=agent_timeout,
    )
    agent_payload, payload_error = _parse_agent_payload(agent_process)

    patch, changed_files, patch_error = _capture_patch(workspace, baseline)
    patch_path = patch_root / f"{case.instruction_set}__{safe_id}.patch"
    patch_path.write_text(patch, encoding="utf-8")

    evaluation = _run_official_test(
        test_path=case.test_path,
        workspace=workspace,
        timeout=test_timeout,
    )
    status = _agent_status(agent_process, agent_payload)
    oracle_passed = evaluation.returncode == 0 and not evaluation.timed_out
    passed = status == "completed" and oracle_passed
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "benchmark": "refactorbench",
        "instruction_set": case.instruction_set,
        "instance_id": case.instance_id,
        "repository": case.repository,
        "prompt": case.prompt_path.read_text(encoding="utf-8"),
        "prompt_path": str(case.prompt_path),
        "test_path": str(case.test_path),
        "workspace": str(workspace),
        "baseline_commit": baseline,
        "aeloon_core_commit": _git_revision(PROJECT_ROOT),
        "config_path": (
            str(config_path.expanduser().resolve()) if config_path is not None else None
        ),
        "agent": {
            "status": status,
            "returncode": agent_process.returncode,
            "timed_out": agent_process.timed_out,
            "wall_time_ms": agent_process.duration_ms,
            "session_id": (agent_payload or {}).get("session_id"),
            "turn_id": (agent_payload or {}).get("turn_id"),
            "duration_ms": (agent_payload or {}).get("duration_ms"),
            "final_content": (agent_payload or {}).get("final_content"),
            "tools_used": (agent_payload or {}).get("tools_used", []),
            "usage": (agent_payload or {}).get("usage", {}),
            "transitions": (agent_payload or {}).get("transitions", []),
            "models": (agent_payload or {}).get("models", {}),
            "stdout": _bounded(agent_process.stdout) if payload_error else None,
            "stderr": _bounded(agent_process.stderr),
            "payload_error": payload_error,
        },
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
        "patch_path": str(patch_path),
        "patch_error": patch_error,
        "false_completed": status == "completed" and not oracle_passed,
    }


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


def _run_process(command: list[str], *, cwd: Path, timeout: float) -> ProcessOutcome:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return ProcessOutcome(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessOutcome(
            returncode=None,
            stdout=_process_text(exc.stdout),
            stderr=_process_text(exc.stderr),
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


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


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
        description="Run official RefactorBench AST tests through Aeloon Core."
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
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results" / "refactorbench.jsonl",
        help="Append-only result ledger.",
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
        help="Truncate an existing result ledger before running.",
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
