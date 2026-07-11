from __future__ import annotations

import pytest

from aeloon_core.tools.filesystem import ReadTool, WriteTool
from aeloon_core.tools.shell import ExecTool


@pytest.mark.asyncio
async def test_read_tool_reports_next_offset(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ReadTool(workspace=tmp_path)

    result = await tool.execute("notes.txt", limit=2)

    assert "1| one" in result
    assert "2| two" in result
    assert "Use offset=3" in result


@pytest.mark.asyncio
async def test_read_tool_caps_output_on_line_boundary(tmp_path, monkeypatch) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("\n".join(f"line-{index}" for index in range(1, 20)), encoding="utf-8")
    monkeypatch.setattr(ReadTool, "_MAX_CHARS", 32)
    tool = ReadTool(workspace=tmp_path)

    result = await tool.execute("notes.txt", limit=20)

    assert "Output capped at 32 chars" in result
    assert "Use offset=" in result


@pytest.mark.asyncio
async def test_write_tool_refuses_accidental_overwrite(tmp_path) -> None:
    path = tmp_path / "app.py"
    path.write_text("print('old')\n", encoding="utf-8")
    tool = WriteTool(workspace=tmp_path)

    result = await tool.execute("app.py", "print('new')\n")

    assert result.startswith("Error: File already exists")
    assert path.read_text(encoding="utf-8") == "print('old')\n"


@pytest.mark.asyncio
async def test_write_tool_allows_explicit_overwrite(tmp_path) -> None:
    path = tmp_path / "app.py"
    path.write_text("print('old')\n", encoding="utf-8")
    tool = WriteTool(workspace=tmp_path)

    result = await tool.execute("app.py", "print('new')\n", overwrite=True)

    assert result.startswith("Successfully wrote")
    assert path.read_text(encoding="utf-8") == "print('new')\n"


@pytest.mark.asyncio
async def test_write_tool_rejects_large_content_without_marker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(WriteTool, "_LARGE_CONTENT_CHARS", 10)
    tool = WriteTool(workspace=tmp_path)

    result = await tool.execute("large.txt", "x" * 11)

    assert result.startswith("Error: Refusing large write")
    assert not (tmp_path / "large.txt").exists()


@pytest.mark.asyncio
async def test_write_tool_rejects_missing_marker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(WriteTool, "_LARGE_CONTENT_CHARS", 10)
    tool = WriteTool(workspace=tmp_path)

    result = await tool.execute("large.txt", "x" * 11, end_marker="DONE_MARKER")

    assert result.startswith("Error: content does not end with end_marker")
    assert not (tmp_path / "large.txt").exists()


@pytest.mark.asyncio
async def test_write_tool_strips_marker_before_saving(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(WriteTool, "_LARGE_CONTENT_CHARS", 10)
    tool = WriteTool(workspace=tmp_path)

    result = await tool.execute("large.txt", "hello worldDONE_MARKER", end_marker="DONE_MARKER")

    assert result.startswith("Successfully wrote")
    assert (tmp_path / "large.txt").read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_workspace_tools_reject_outside_and_protected_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    protected = workspace / ".runtime-data"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    protected.mkdir()
    outside.write_text("outside")
    (protected / "secret.txt").write_text("secret")
    (workspace / "linked-runtime-data").symlink_to(protected, target_is_directory=True)
    read = ReadTool(workspace=workspace, denied_paths=(protected,))
    write = WriteTool(workspace=workspace, denied_paths=(protected,))

    outside_result = await read.execute(str(outside))
    protected_result = await read.execute("linked-runtime-data/secret.txt")
    write_result = await write.execute(".runtime-data/owned.txt", "owned")

    assert "path escapes workspace" in outside_result
    assert "protected from agent tools" in protected_result
    assert "protected from agent tools" in write_result
    assert not (protected / "owned.txt").exists()


@pytest.mark.asyncio
async def test_unprotected_workspace_tool_preserves_v1_absolute_path_access(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("legacy access", encoding="utf-8")

    result = await ReadTool(workspace=workspace).execute(str(outside))

    assert "legacy access" in result


@pytest.mark.asyncio
async def test_unprotected_exec_preserves_v1_outside_working_directory(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()

    result = await ExecTool(workspace=workspace).execute(
        command="pwd",
        working_dir=str(outside),
    )

    assert str(outside.resolve()) in result
    assert "Exit code: 0" in result
