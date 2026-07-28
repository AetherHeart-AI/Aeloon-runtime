"""Shared benchmark lifecycle: acquire, prepare, evaluate, and archive."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.harness.base import Harness
from benchmarks.progress import info


@dataclass(frozen=True)
class BenchmarkRun:
    """Owned paths for one benchmark invocation."""

    run_id: str
    output_dir: Path
    workspace_root: Path
    source_dir: Path


class BenchmarkAdapter(ABC):
    """Base contract implemented by each official benchmark integration.

    Subclasses own dataset acquisition, dependency installation, case loading,
    official evaluation, and benchmark-specific records. The base owns safe
    checkout creation plus durable JSON/JSONL writes.
    """

    name: str
    repository_url: str

    def __init__(
        self,
        *,
        project_root: Path,
        limit: int | None = None,
        workers: int = 1,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.project_root = project_root.expanduser().resolve()
        self.limit = limit
        self.workers = workers
        self._result_lock = threading.Lock()
        workspace_root = self.project_root / ".benchmark-workspaces"
        source_dir = workspace_root / "sources" / self.name
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        self.run = BenchmarkRun(
            run_id=run_id,
            output_dir=(self.project_root / "benchmarks" / "results" / self.name / run_id),
            workspace_root=workspace_root,
            source_dir=source_dir,
        )

    def prepare(self) -> None:
        """Fetch the official source and install adapter dependencies."""

        info("[%s] Preparing official source and dependencies", self.name)
        self.prepare_dataset()
        self.install_dependencies()
        info("[%s] Environment is ready", self.name)

    def prepare_dataset(self) -> None:
        """Clone the official benchmark into the runner-owned source cache."""

        checkout = self.run.source_dir
        if checkout.is_symlink():
            raise RuntimeError(f"Refusing to use symlinked benchmark source directory: {checkout}")
        if (checkout / ".git").is_dir():
            info("[%s] Reusing source checkout: %s", self.name, checkout)
            run_checked(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=checkout,
            )
            return
        if checkout.exists() and (not checkout.is_dir() or any(checkout.iterdir())):
            raise RuntimeError(
                f"Refusing to replace non-empty benchmark source directory: {checkout}"
            )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        staging = checkout.parent / f".{checkout.name}.cloning-{uuid.uuid4().hex[:8]}"
        try:
            info("[%s] Cloning %s", self.name, self.repository_url)
            run_checked(
                ["git", "clone", "--depth", "1", self.repository_url, str(staging)],
                cwd=self.project_root,
            )
            if checkout.exists():
                checkout.rmdir()
            os.replace(staging, checkout)
            info("[%s] Source checkout created: %s", self.name, checkout)
        finally:
            if staging.is_dir():
                shutil.rmtree(staging)

    @abstractmethod
    def install_dependencies(self) -> None:
        """Install the minimal dependencies required by the official evaluator."""

    @abstractmethod
    def load_cases(self) -> list[Any]:
        """Load the deterministic set of cases for this run."""

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Run the benchmark's official test or evaluator."""

    @abstractmethod
    def execute(self, harnesses: list[Harness]) -> dict[str, Any]:
        """Run all selected harnesses and return a JSON-serializable summary."""

    def write_result(self, path: Path, record: dict[str, Any]) -> None:
        """Append one durable JSONL record."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._result_lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Atomically replace one JSON document."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex[:8]}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def source_revision(self) -> str | None:
        if not self.run.source_dir.is_dir():
            return None
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.run.source_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    def selected(self, cases: list[Any]) -> list[Any]:
        return cases[: self.limit] if self.limit is not None else cases

    def manifest(self, harnesses: Iterable[Harness], *, status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run.run_id,
            "benchmark": self.name,
            "status": status,
            "workers": self.workers,
            "source": {
                "repository": self.repository_url,
                "checkout": str(self.run.source_dir),
                "revision": self.source_revision(),
            },
            "harnesses": [
                {"id": harness.name, "version": harness.version} for harness in harnesses
            ],
        }


def run_checked(command: list[str], *, cwd: Path) -> str:
    """Run setup commands with useful failure details."""

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
