"""Bounded Git subprocess helpers for Runtime workspace methods."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

_gh_auth_cache: tuple[float, bool] | None = None


def git_command(workspace: Path, args: list[str], *, timeout: float = 30.0) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def status(workspace: Path) -> dict[str, object]:
    code, output, stderr = git_command(workspace, ["status", "--porcelain=v1", "-b"])
    if code != 0:
        return {"ok": False, "branch": None, "entries": [], "stderr": stderr[-4000:]}
    lines = output.splitlines()
    branch = None
    entries: list[dict[str, str]] = []
    for line in lines:
        if line.startswith("## "):
            branch = line[3:].split("...", 1)[0] or None
            if branch and branch.startswith("No commits yet on "):
                branch = branch.removeprefix("No commits yet on ")
        elif len(line) >= 3:
            entries.append({"index": line[0], "worktree": line[1], "path": line[3:]})
    return {"ok": True, "branch": branch, "entries": entries, "stderr": ""}


def diff(workspace: Path, scope: str, path: str | None = None) -> dict[str, object]:
    args = ["diff"]
    if scope == "staged":
        args.append("--cached")
    if path:
        args.extend(["--", path])
    code, output, stderr = git_command(workspace, args)
    truncated = len(output) > 2 * 1024 * 1024
    return {
        "ok": code == 0,
        "scope": scope,
        "path": path or ".",
        "patch": output[: 2 * 1024 * 1024],
        "binary": "Binary files" in output,
        "truncated": truncated,
        "stderr": stderr[-4000:],
    }


def branches(workspace: Path) -> dict[str, object]:
    code, output, stderr = git_command(
        workspace, ["for-each-ref", "--format=%(refname:short)", "refs/heads"]
    )
    head_code, head, _head_error = git_command(workspace, ["branch", "--show-current"])
    return {
        "ok": code == 0,
        "branches": output.splitlines() if code == 0 else [],
        "current": head.strip() if head_code == 0 and head.strip() else None,
        "stderr": stderr[-4000:],
    }


def changes(workspace: Path) -> dict[str, object]:
    """Return the stable staged/working-tree projection consumed by the UI."""

    code, output, stderr = git_command(
        workspace, ["status", "--porcelain=v1", "--untracked-files=all", "--branch", "-z"]
    )
    if code != 0:
        raise RuntimeError(stderr[-4000:] or "Git status is unavailable")
    branch: str | None = None
    entries: list[dict[str, object]] = []
    chunks = output.split("\0")
    index = 0
    while index < len(chunks):
        token = chunks[index]
        index += 1
        if not token:
            continue
        if token.startswith("## "):
            branch_value = token[3:].split("...", 1)[0]
            if branch_value.startswith("No commits yet on "):
                branch_value = branch_value.removeprefix("No commits yet on ")
            branch = None if branch_value in {"(detached)", "(unknown)", ""} else branch_value
            continue
        if token.startswith("?? "):
            entries.append({"path": token[3:], "status": "?", "staged": False, "changed": True})
            continue
        if len(token) < 4 or token[0] in {"!"}:
            continue
        xy, path = token[:2], token[3:]
        renamed_from: str | None = None
        # With `status -z`, rename/copy entries use the next NUL-delimited
        # token for the source path (`R  new\0old\0`), rather than the
        # human-readable `old -> new` spelling. Consume that token here so it
        # cannot be reported as a second changed file.
        if "R" in xy or "C" in xy:
            if index < len(chunks) and chunks[index]:
                renamed_from = chunks[index]
                index += 1
        status = next((value for value in xy if value != "."), "M")
        entries.append(
            {
                "path": path,
                "status": "R" if "R" in xy else status,
                "staged": xy[0] not in {".", " "},
                "changed": xy[1] not in {".", " "},
                **({"renamed_from": renamed_from} if renamed_from else {}),
            }
        )
    staged_stats = _numstat(workspace, staged=True)
    change_stats = _numstat(workspace, staged=False)
    for entry in entries:
        if entry["status"] == "?" and entry["path"] not in change_stats:
            stat = _untracked_numstat(workspace, str(entry["path"]))
            if stat is not None:
                change_stats[str(entry["path"])] = stat
    return {
        "branch": branch,
        "staged": _group([item for item in entries if item["staged"]], staged_stats),
        "changes": _group([item for item in entries if item["changed"]], change_stats),
    }


def stage(workspace: Path, paths: list[str]) -> dict[str, object]:
    _validate_paths(paths)
    code, output, stderr = git_command(workspace, ["add", "--", *paths])
    return {"ok": code == 0, "stdout": output, "stderr": stderr[-4000:]}


def unstage(workspace: Path, paths: list[str]) -> dict[str, object]:
    _validate_paths(paths)
    code, output, stderr = git_command(workspace, ["restore", "--staged", "--", *paths])
    return {"ok": code == 0, "stdout": output, "stderr": stderr[-4000:]}


def branch_create(workspace: Path, name: str) -> dict[str, object]:
    if not name or name.startswith("-") or ".." in name:
        raise ValueError("Invalid branch name")
    code, output, stderr = git_command(workspace, ["switch", "-c", name])
    return {"ok": code == 0, "branch": name, "stdout": output, "stderr": stderr[-4000:]}


def commit(workspace: Path, message: str) -> dict[str, object]:
    if not message.strip():
        raise ValueError("Commit message is required")
    code, output, stderr = git_command(workspace, ["commit", "-m", message])
    return {"ok": code == 0, "commit": output.strip(), "stdout": output, "stderr": stderr[-4000:]}


def push(workspace: Path, remote: str = "origin") -> dict[str, object]:
    if not remote or remote.startswith("-"):
        raise ValueError("Invalid remote")
    code, output, stderr = git_command(workspace, ["push", remote])
    return {
        "ok": code == 0,
        "pushed": code == 0,
        "remote": remote,
        "stdout": output,
        "stderr": stderr[-4000:],
    }


def github_status(workspace: Path) -> dict[str, object]:
    global _gh_auth_cache
    code, output, stderr = git_command(workspace, ["remote", "get-url", "--push", "origin"])
    remote = output.strip()
    parsed = urlparse(remote if "://" in remote else "")
    github = remote.lower().startswith("git@github.com:") or parsed.hostname == "github.com"
    authenticated = False
    if github and shutil.which("gh"):
        now = time.monotonic()
        if _gh_auth_cache is None or now - _gh_auth_cache[0] >= 60:
            checked = subprocess.run(
                ["gh", "auth", "status", "--hostname", "github.com"],
                capture_output=True,
                text=True,
                check=False,
            )
            _gh_auth_cache = (now, checked.returncode == 0)
        authenticated = _gh_auth_cache[1]
    return {
        "github_origin": github,
        "authenticated": authenticated,
        "remote": remote,
        "stderr": stderr[-4000:],
    }


def create_worktree(
    project_path: Path,
    worktree_root: Path,
    project_id: str,
    thread_id: str,
    title: str,
    branch: str | None = None,
) -> tuple[Path, str]:
    """Create a managed worktree and return its canonical path and branch."""

    if not _is_git_repository(project_path):
        raise RuntimeError("Parallel worktree threads require a Git repository")
    branch_name = _safe_branch(branch or f"aeloon/{_slug(title)}-{thread_id[:7]}")
    parent = worktree_root.expanduser().resolve(strict=False) / project_id
    workspace = parent / thread_id
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    if workspace.exists():
        raise RuntimeError(f"Worktree path already exists: {workspace}")
    code, output, stderr = git_command(
        project_path, ["worktree", "add", "-b", branch_name, str(workspace)]
    )
    if code != 0:
        raise RuntimeError(stderr[-4000:] or output[-4000:] or "Could not create worktree")
    return workspace.resolve(strict=True), branch_name


def remove_worktree(project_path: Path, workspace: Path, *, force: bool = False) -> None:
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(workspace))
    code, output, stderr = git_command(project_path, args)
    if code != 0:
        raise RuntimeError(stderr[-4000:] or output[-4000:] or "Could not remove worktree")


def worktree_status(workspace: Path) -> dict[str, object]:
    if not workspace.exists():
        return {"exists": False, "dirty": False, "detail": ""}
    code, output, stderr = git_command(workspace, ["status", "--porcelain"])
    if code != 0:
        raise RuntimeError(stderr[-4000:] or output[-4000:] or "Could not inspect worktree")
    detail = output.strip()
    return {"exists": True, "dirty": bool(detail), "detail": detail}


def _is_git_repository(path: Path) -> bool:
    code, output, _stderr = git_command(path, ["rev-parse", "--is-inside-work-tree"])
    return code == 0 and output.strip() == "true"


def _safe_branch(value: str) -> str:
    if (
        not value
        or value.startswith("-")
        or value.endswith(("/", "."))
        or ".." in value
        or "//" in value
        or "@{" in value
        or any(char.isspace() or char in "~^:?*[\\" for char in value)
    ):
        raise ValueError("Invalid branch name")
    return value


def _slug(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return normalized[:48] or "thread"


def pr_create(workspace: Path, title: str, body: str = "") -> dict[str, object]:
    if not title.strip():
        raise ValueError("Pull request title is required")
    if shutil.which("gh") is None:
        return {"ok": False, "url": "", "stdout": "", "stderr": "gh is not installed"}
    args = ["gh", "pr", "create", "--title", title]
    if body:
        args.extend(["--body", body])
    completed = subprocess.run(
        args,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "url": completed.stdout.strip(),
        "stdout": completed.stdout,
        "stderr": completed.stderr[-4000:],
    }


def _validate_paths(paths: list[str]) -> None:
    for path in paths:
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("Git path escapes workspace")


def _numstat(workspace: Path, *, staged: bool) -> dict[str, dict[str, object]]:
    args = ["diff", "--numstat", "-z", "-M"]
    if staged:
        args.append("--cached")
    args.extend(["--", "."])
    code, output, _stderr = git_command(workspace, args)
    if code != 0:
        return {}
    result: dict[str, dict[str, object]] = {}
    chunks = output.split("\0")
    index = 0
    while index < len(chunks):
        line = chunks[index]
        index += 1
        if not line or "\t" not in line:
            continue
        additions, rest = line.split("\t", 1)
        if "\t" not in rest:
            continue
        deletions, path = rest.split("\t", 1)
        if not path and index + 1 < len(chunks):
            index += 1
            path = chunks[index]
        if path:
            result[path] = {
                "additions": int(additions) if additions.isdigit() else 0,
                "deletions": int(deletions) if deletions.isdigit() else 0,
                "binary": additions == "-" or deletions == "-",
            }
    return result


def _untracked_numstat(workspace: Path, path: str) -> dict[str, object] | None:
    target = workspace / path
    if not target.is_file():
        return {"additions": 0, "deletions": 0, "binary": False}
    completed = subprocess.run(
        ["git", "diff", "--numstat", "--no-index", "--", "/dev/null", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return None
    line = completed.stdout.splitlines()[0] if completed.stdout else ""
    additions, _, deletions = line.partition("\t")
    return {
        "additions": int(additions) if additions.isdigit() else 0,
        "deletions": int(deletions) if deletions.isdigit() else 0,
        "binary": additions == "-" or deletions == "-",
    }


def _group(
    entries: list[dict[str, object]], stats: dict[str, dict[str, object]]
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for entry in entries:
        path = str(entry["path"])
        stat = stats.get(path, {"additions": 0, "deletions": 0, "binary": False})
        files.append(
            {
                "path": path,
                "status": entry["status"],
                "additions": stat["additions"],
                "deletions": stat["deletions"],
                "binary": stat["binary"],
                **({"renamed_from": entry["renamed_from"]} if entry.get("renamed_from") else {}),
            }
        )
    return {
        "files": files,
        "additions": sum(int(file["additions"]) for file in files),
        "deletions": sum(int(file["deletions"]) for file in files),
    }


__all__ = [
    "branch_create",
    "branches",
    "changes",
    "commit",
    "diff",
    "git_command",
    "github_status",
    "create_worktree",
    "push",
    "pr_create",
    "remove_worktree",
    "stage",
    "status",
    "unstage",
    "worktree_status",
]
