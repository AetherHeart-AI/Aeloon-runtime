"""Hard read-only and workspace-boundary regressions for observation tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeloon_core.tools.search_grep import GlobTool, GrepTool


@pytest.mark.asyncio
async def test_grep_treats_option_like_pattern_as_data(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("literal --type-list value\n", encoding="utf-8")
    tool = GrepTool(workspace=workspace, denied_paths=(tmp_path / "runtime",))

    result = await tool.execute("--type-list", path=".")

    assert "notes.txt:1:literal --type-list value" in result
    assert "Error running rg" not in result


@pytest.mark.asyncio
async def test_glob_rejects_parent_traversal_and_filters_escaping_symlinks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (workspace / "outside-link.txt").symlink_to(outside)
    (workspace / "inside.txt").write_text("public", encoding="utf-8")
    tool = GlobTool(workspace=workspace, denied_paths=(tmp_path / "runtime",))

    traversal = await tool.execute("../*.txt")
    visible = await tool.execute("*.txt")

    assert traversal.startswith("Error:")
    assert "outside.txt" not in traversal
    assert visible == "inside.txt"


@pytest.mark.asyncio
async def test_python_grep_fallback_does_not_read_escaping_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret-value", encoding="utf-8")
    (workspace / "outside-link.txt").symlink_to(outside)
    (workspace / "inside.txt").write_text("inside-public-value", encoding="utf-8")
    monkeypatch.setattr("aeloon_core.tools.search_grep.shutil.which", lambda _: None)
    tool = GrepTool(workspace=workspace, denied_paths=(tmp_path / "runtime",))

    secret = await tool.execute("outside-secret-value", path=".")
    public = await tool.execute("inside-public-value", path=".")

    assert secret == "(no matches)"
    assert "inside-public-value" in public


@pytest.mark.asyncio
async def test_ripgrep_excludes_denied_paths_with_glob_metacharacters(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    denied = workspace / "runtime[1]"
    denied.mkdir(parents=True)
    (denied / "private.txt").write_text("SHARED_TOKEN", encoding="utf-8")
    (workspace / "public.txt").write_text("SHARED_TOKEN", encoding="utf-8")
    tool = GrepTool(workspace=workspace, denied_paths=(denied,))

    result = await tool.execute("SHARED_TOKEN", path=".")

    assert "public.txt" in result
    assert "private.txt" not in result
