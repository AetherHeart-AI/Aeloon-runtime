from __future__ import annotations

import subprocess
from pathlib import Path

from aeloon_core.git_workspace import changes, status


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def test_changes_preserves_rename_source_and_untracked_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Runtime Test")
    (repo / "old.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")

    (repo / "old.txt").rename(repo / "new.txt")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    _git(repo, "add", "-A")
    # Keep the fixture's untracked entry genuinely unstaged after staging the
    # rename; `git reset` is scoped to this temporary repository.
    _git(repo, "reset", "--", "untracked.txt")

    snapshot = changes(repo)
    files = snapshot["staged"]["files"]
    assert files == [
        {
            "path": "new.txt",
            "status": "R",
            "additions": 0,
            "deletions": 0,
            "binary": False,
            "renamed_from": "old.txt",
        }
    ]
    assert snapshot["changes"]["files"][0]["path"] == "untracked.txt"


def test_status_normalizes_unborn_branch_name(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    value = status(repo)
    assert value["ok"] is True
    assert value["branch"] == "main"
