"""Pinned RepoQA data, repository workspaces, prompts, and exact grading."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

DATASET_VERSION = "2024-06-23"
DATASET_FILENAME = f"repoqa-{DATASET_VERSION}.json.gz"
DATASET_URL = (
    "https://github.com/evalplus/repoqa_release/releases/download/"
    f"{DATASET_VERSION}/{DATASET_FILENAME}"
)
DATASET_SHA256 = "c050a2ad90a7df89d9dc1f1c3b3b20683edd20a56293b35fcaae43dec115d681"
RESULT_PREFIX = "REPOQA_RESULT:"


@dataclass(frozen=True)
class RepoQACase:
    """One function-location task backed by an official RepoQA repository."""

    instance_id: str
    language: str
    repository: str
    commit_sha: str
    description: str
    target_path: str
    target_symbol: str
    repository_files: dict[str, str]


@dataclass(frozen=True)
class RepoQAAnswer:
    """Structured location returned by an agent."""

    path: str
    symbol: str


def download_dataset(path: Path) -> None:
    """Download and verify the pinned official RepoQA release."""

    path = path.expanduser().resolve()
    if path.is_symlink():
        raise RuntimeError(f"Refusing to use symlinked RepoQA dataset: {path}")
    if path.is_file():
        _verify_dataset(path)
        return
    if path.exists():
        raise RuntimeError(f"RepoQA dataset path is not a regular file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.downloading-{uuid.uuid4().hex[:8]}")
    try:
        with (
            httpx.stream(
                "GET",
                DATASET_URL,
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, connect=30.0),
            ) as response,
            temporary.open("wb") as stream,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes():
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_dataset(temporary)
        os.replace(temporary, path)
    except (httpx.HTTPError, OSError) as exc:
        raise RuntimeError(f"Could not download RepoQA dataset: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_cases(dataset_path: Path) -> list[RepoQACase]:
    """Load all tasks and interleave languages and repositories."""

    try:
        with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
            dataset = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load RepoQA dataset {dataset_path}: {exc}") from exc
    if not isinstance(dataset, dict):
        raise RuntimeError("RepoQA dataset root must be an object.")

    repositories_by_language: dict[str, list[list[RepoQACase]]] = {}
    seen_ids: set[str] = set()
    for language in sorted(dataset):
        raw_repositories = dataset[language]
        if not isinstance(raw_repositories, list):
            raise RuntimeError(f"RepoQA language {language!r} must contain a list.")
        language_repositories: list[list[RepoQACase]] = []
        for raw_repository in sorted(
            raw_repositories,
            key=lambda item: str(item.get("repo", "")) if isinstance(item, dict) else "",
        ):
            repository_cases = _load_repository_cases(
                language=language,
                raw_repository=raw_repository,
                seen_ids=seen_ids,
            )
            language_repositories.append(repository_cases)
        repositories_by_language[language] = language_repositories

    # A small --limit should cover the full language/repository surface before
    # selecting a second needle from any repository.
    languages = sorted(repositories_by_language)
    max_repositories = max(
        (len(repositories) for repositories in repositories_by_language.values()),
        default=0,
    )
    max_needles = max(
        (
            len(repository)
            for repositories in repositories_by_language.values()
            for repository in repositories
        ),
        default=0,
    )
    cases: list[RepoQACase] = []
    for needle_index in range(max_needles):
        for repository_index in range(max_repositories):
            for language in languages:
                repositories = repositories_by_language[language]
                if repository_index >= len(repositories):
                    continue
                repository_cases = repositories[repository_index]
                if needle_index < len(repository_cases):
                    cases.append(repository_cases[needle_index])
    return cases


def build_prompt(case: RepoQACase) -> str:
    """Build a tool-oriented task without exposing the target location."""

    return (
        "You are solving a RepoQA Agent repository-search task.\n\n"
        "Use only the files and local search/read tools available in the current "
        "repository. Do not use the network and do not modify, create, delete, or "
        "rename any repository files.\n\n"
        f"Repository: {case.repository}\n"
        f"Language: {case.language}\n\n"
        "Find the single function or method described below.\n\n"
        f"{case.description.strip()}\n\n"
        "Finish with exactly one result line using a repository-relative POSIX path:\n"
        f'{RESULT_PREFIX} {{"path":"relative/path.ext","symbol":"function_name"}}'
    )


def parse_answer(final_content: str | None) -> tuple[RepoQAAnswer | None, str | None]:
    """Parse the one-line result contract from a harness response."""

    if not isinstance(final_content, str) or not final_content.strip():
        return None, "agent returned no final content"
    result_lines = [
        line.strip()[len(RESULT_PREFIX) :].strip()
        for line in final_content.splitlines()
        if line.strip().startswith(RESULT_PREFIX)
    ]
    if not result_lines:
        return None, f"final content did not contain {RESULT_PREFIX}"
    if len(result_lines) != 1:
        return None, f"final content contained {len(result_lines)} result lines"
    try:
        payload = json.loads(result_lines[0])
    except json.JSONDecodeError as exc:
        return None, f"result JSON was invalid: {exc}"
    if not isinstance(payload, dict):
        return None, "result JSON must be an object"

    path = payload.get("path")
    symbol = payload.get("symbol")
    if not isinstance(path, str) or not isinstance(symbol, str):
        return None, "result JSON requires string path and symbol fields"
    try:
        normalized_path = _relative_path(path.strip())
    except RuntimeError as exc:
        return None, str(exc)
    normalized_symbol = symbol.strip()
    if not normalized_symbol:
        return None, "result symbol must not be empty"
    return RepoQAAnswer(path=normalized_path, symbol=normalized_symbol), None


def evaluate_answer(
    case: RepoQACase,
    final_content: str | None,
    changed_files: list[str],
) -> dict[str, Any]:
    """Grade exact target location and enforce a read-only outcome."""

    answer, parse_error = parse_answer(final_content)
    path_match = answer is not None and answer.path == case.target_path
    symbol_match = answer is not None and answer.symbol == case.target_symbol
    matched_target = path_match and symbol_match
    clean_worktree = not changed_files
    return {
        "matched_target": matched_target,
        "path_match": path_match,
        "symbol_match": symbol_match,
        "clean_worktree": clean_worktree,
        "reported_path": answer.path if answer is not None else None,
        "reported_symbol": answer.symbol if answer is not None else None,
        "parse_error": parse_error,
    }


class WorkspaceCache:
    """Materialize official repository snapshots and reset harness workspaces."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if self.root == Path(self.root.anchor):
            raise RuntimeError(f"Unsafe RepoQA workspace cache root: {self.root}")

    def materialize(self, case: RepoQACase) -> Path:
        source = self._source_path(case)
        if source.is_symlink():
            raise RuntimeError(f"Refusing to use symlinked RepoQA source: {source}")
        if (source / ".git").is_dir():
            return source
        if source.exists() and (not source.is_dir() or any(source.iterdir())):
            raise RuntimeError(f"Refusing to replace non-empty RepoQA source: {source}")

        source.parent.mkdir(parents=True, exist_ok=True)
        staging = source.parent / f".{source.name}.materializing-{uuid.uuid4().hex[:8]}"
        try:
            staging.mkdir()
            for relative_path, content in case.repository_files.items():
                target = staging.joinpath(*PurePosixPath(relative_path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            _git(["init", "--quiet"], cwd=staging)
            _git(["config", "user.name", "RepoQA Agent"], cwd=staging)
            _git(["config", "user.email", "repoqa-agent@localhost"], cwd=staging)
            _git(["add", "--force", "--all"], cwd=staging)
            _git(
                [
                    "commit",
                    "--quiet",
                    "--no-gpg-sign",
                    "-m",
                    f"RepoQA snapshot {case.repository}@{case.commit_sha}",
                ],
                cwd=staging,
            )
            if source.exists():
                source.rmdir()
            os.replace(staging, source)
        finally:
            if staging.is_dir():
                shutil.rmtree(staging)
        return source

    def prepare(self, case: RepoQACase, harness_name: str) -> tuple[Path, str]:
        source = self.materialize(case)
        workspace = self._workspace_path(case, harness_name)
        if workspace.is_symlink():
            raise RuntimeError(f"Refusing to use symlinked RepoQA workspace: {workspace}")
        if not (workspace / ".git").is_dir():
            if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
                raise RuntimeError(f"Refusing to replace non-empty RepoQA workspace: {workspace}")
            workspace.parent.mkdir(parents=True, exist_ok=True)
            _git(["clone", "--quiet", "--no-hardlinks", str(source), str(workspace)], cwd=self.root)
        self._assert_owned_workspace(workspace)
        _git(["reset", "--hard", "--quiet", "HEAD"], cwd=workspace)
        _git(["clean", "-fdqx"], cwd=workspace)
        baseline = _git(["rev-parse", "HEAD"], cwd=workspace).strip()
        return workspace, baseline

    def _source_path(self, case: RepoQACase) -> Path:
        return self.root / "sources" / _repository_key(case)

    def _workspace_path(self, case: RepoQACase, harness_name: str) -> Path:
        harness_key = hashlib.sha256(harness_name.encode()).hexdigest()[:12]
        return self.root / "workspaces" / harness_key / _repository_key(case)

    def _assert_owned_workspace(self, workspace: Path) -> None:
        resolved = workspace.resolve()
        if not resolved.is_relative_to(self.root) or not (resolved / ".git").is_dir():
            raise RuntimeError(f"Unsafe RepoQA workspace: {workspace}")


def changed_files(workspace: Path) -> list[str]:
    """Return tracked and untracked changes made during a read-only task."""

    tracked = _git(["diff", "--name-only", "-z", "HEAD"], cwd=workspace)
    untracked = _git(["ls-files", "--others", "-z"], cwd=workspace)
    return sorted({path for output in (tracked, untracked) for path in output.split("\0") if path})


def _load_repository_cases(
    *,
    language: str,
    raw_repository: Any,
    seen_ids: set[str],
) -> list[RepoQACase]:
    if not isinstance(raw_repository, dict):
        raise RuntimeError(f"RepoQA language {language!r} contains a non-object repository.")
    repository = raw_repository.get("repo")
    commit_sha = raw_repository.get("commit_sha")
    raw_files = raw_repository.get("content")
    raw_needles = raw_repository.get("needles")
    if not isinstance(repository, str) or not repository:
        raise RuntimeError(f"RepoQA language {language!r} contains an invalid repository name.")
    if not isinstance(commit_sha, str) or not commit_sha:
        raise RuntimeError(f"RepoQA repository {repository!r} has no commit SHA.")
    if not isinstance(raw_files, dict) or not isinstance(raw_needles, list):
        raise RuntimeError(f"RepoQA repository {repository!r} has invalid content or needles.")

    repository_files: dict[str, str] = {}
    for raw_path, content in raw_files.items():
        if not isinstance(raw_path, str) or not isinstance(content, str):
            raise RuntimeError(f"RepoQA repository {repository!r} contains an invalid file.")
        path = _relative_path(raw_path)
        if PurePosixPath(path).parts[0] == ".git":
            raise RuntimeError(f"RepoQA repository {repository!r} contains a .git path.")
        repository_files[path] = content

    cases = []
    for needle in sorted(
        raw_needles,
        key=lambda item: (
            str(item.get("path", "")) if isinstance(item, dict) else "",
            int(item.get("start_line", 0)) if isinstance(item, dict) else 0,
            str(item.get("name", "")) if isinstance(item, dict) else "",
        ),
    ):
        if not isinstance(needle, dict):
            raise RuntimeError(f"RepoQA repository {repository!r} contains an invalid needle.")
        target_path = _relative_path(str(needle.get("path", "")))
        target_symbol = needle.get("name")
        description = needle.get("description")
        if target_path not in repository_files:
            raise RuntimeError(
                f"RepoQA target {repository!r}/{target_path} is absent from repository content."
            )
        if not isinstance(target_symbol, str) or not target_symbol:
            raise RuntimeError(f"RepoQA repository {repository!r} has an invalid target symbol.")
        if not isinstance(description, str) or not description.strip():
            raise RuntimeError(f"RepoQA target {repository!r}/{target_symbol} has no description.")
        instance_id = f"{language}::{repository}::{target_symbol}"
        if instance_id in seen_ids:
            raise RuntimeError(f"RepoQA contains duplicate instance id {instance_id!r}.")
        seen_ids.add(instance_id)
        cases.append(
            RepoQACase(
                instance_id=instance_id,
                language=language,
                repository=repository,
                commit_sha=commit_sha,
                description=description,
                target_path=target_path,
                target_symbol=target_symbol,
                repository_files=repository_files,
            )
        )
    return cases


def _relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise RuntimeError("result path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"path must be repository-relative without traversal: {value!r}")
    return path.as_posix()


def _repository_key(case: RepoQACase) -> str:
    identity = f"{case.language}\0{case.repository}\0{case.commit_sha}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    readable = case.repository.replace("/", "__")
    return f"{readable}-{digest}"


def _verify_dataset(path: Path) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Could not read RepoQA dataset {path}: {exc}") from exc
    if digest.hexdigest() != DATASET_SHA256:
        raise RuntimeError(
            f"RepoQA dataset checksum mismatch for {path}: "
            f"expected {DATASET_SHA256}, got {digest.hexdigest()}"
        )


def _git(arguments: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed in {cwd}: {detail}")
    return completed.stdout
