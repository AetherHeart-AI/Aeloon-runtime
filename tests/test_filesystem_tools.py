from __future__ import annotations

import pytest

from aeloon_core.tools.filesystem import ReadTool, WriteTool


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
