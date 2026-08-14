from __future__ import annotations

import asyncio
import base64
import io
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from aeloon_core.core import ImageContent
from aeloon_core.tool import BuiltinToolSet
from aeloon_core.tool.shell import prune_bash_logs

DEFAULT_ACTIVE_TOOLS = BuiltinToolSet.default_active_names


def create_all_tools(cwd: Path):
    return BuiltinToolSet(cwd).by_name


@pytest.mark.asyncio
async def test_read_text_truncation_continuation_and_absolute_paths(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"line-{index}" for index in range(2_100)), encoding="utf-8")
    tool = create_all_tools(tmp_path)["read"]

    result = await tool.execute("call", {"path": str(path)}, None)
    text = result.content[0].text

    assert "line-0" in text
    assert "line-1999" in text
    assert "line-2000" not in text
    assert "Continue with offset=2001" in text
    assert result.details["sizeBytes"] == path.stat().st_size
    assert result.details["selectedLines"] == 2_000
    continued = await tool.execute("call", {"path": "large.txt", "offset": 2001}, None)
    assert "line-2000" in continued.content[0].text


@pytest.mark.asyncio
async def test_read_text_streams_distant_window_and_bounds_long_lines(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes((b"prefix\n" * 100_000) + b"target\n")
    tool = create_all_tools(tmp_path)["read"]

    result = await tool.execute(
        "call", {"path": "large.txt", "offset": 100_001, "limit": 1}, None
    )
    assert result.content[0].text == "target"
    assert result.details["lineRange"] == {"start": 100_001, "total": 100_001}

    huge = tmp_path / "huge-line.txt"
    huge.write_bytes(b"x" * (2 * 1024 * 1024))
    result = await tool.execute("call", {"path": "huge-line.txt"}, None)
    assert "Line 1 is 2048.0KB" in result.content[0].text
    assert len(result.content[0].text) < 500


@pytest.mark.asyncio
async def test_structured_file_tools_reject_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    tools = create_all_tools(workspace)

    for value in ("../outside/secret.txt", str(secret)):
        with pytest.raises(PermissionError, match="outside the workspace"):
            await tools["read"].execute("read", {"path": value}, None)
    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["write"].execute(
            "write", {"path": "../outside/new.txt", "content": "blocked"}, None
        )
    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["edit"].execute(
            "edit",
            {"path": str(secret), "edits": [{"oldText": "secret", "newText": "stolen"}]},
            None,
        )
    assert secret.read_text(encoding="utf-8") == "secret"
    assert not (outside / "new.txt").exists()


@pytest.mark.asyncio
async def test_structured_file_tools_reject_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    tools = create_all_tools(workspace)

    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["read"].execute("read", {"path": "escape/secret.txt"}, None)
    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["write"].execute(
            "write", {"path": "escape/new.txt", "content": "blocked"}, None
        )
    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["ls"].execute("ls", {"path": "escape"}, None)
    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["grep"].execute(
            "grep", {"pattern": "secret", "path": "escape"}, None
        )
    with pytest.raises(PermissionError, match="outside the workspace"):
        await tools["find"].execute("find", {"pattern": "*.txt", "path": "escape"}, None)
    assert not (outside / "new.txt").exists()


@pytest.mark.asyncio
async def test_read_returns_image_attachment_and_resizes_large_images(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (2_100, 20), "red").save(path)

    result = await create_all_tools(tmp_path)["read"].execute("call", {"path": "image.png"}, None)

    assert isinstance(result.content[1], ImageContent)
    decoded = base64.b64decode(result.content[1].data)
    with Image.open(io.BytesIO(decoded)) as image:
        assert max(image.size) == 2_000
    assert "resized from 2100x20" in result.content[0].text
    assert result.details["sizeBytes"] == path.stat().st_size
    assert result.details["mimeType"] == "image/png"


@pytest.mark.asyncio
async def test_write_creates_parents_and_edit_preserves_bom_crlf(tmp_path: Path) -> None:
    tools = create_all_tools(tmp_path)
    written = await tools["write"].execute(
        "write", {"path": "nested/new.txt", "content": "alpha\nbeta\n"}, None
    )
    assert (tmp_path / "nested/new.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert written.details["sizeBytes"] == 11

    path = tmp_path / "source.txt"
    path.write_bytes(b"\xef\xbb\xbfalpha\r\nbeta\r\ngamma\r\n")
    result = await tools["edit"].execute(
        "edit",
        {
            "path": "source.txt",
            "edits": [
                {"oldText": "alpha", "newText": "ALPHA"},
                {"oldText": "gamma", "newText": "GAMMA"},
            ],
        },
        None,
    )

    assert path.read_bytes() == b"\xef\xbb\xbfALPHA\r\nbeta\r\nGAMMA\r\n"
    assert result.details["firstChangedLine"] == 1
    assert result.details["replacements"] == 2
    assert result.details["sizeAfterBytes"] == len(path.read_bytes())
    assert "-alpha" in result.details["diff"]
    assert "+GAMMA" in result.details["patch"]


@pytest.mark.asyncio
async def test_edit_is_atomic_for_duplicate_or_overlapping_replacements(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("same same\nabcdef\n", encoding="utf-8")
    tool = create_all_tools(tmp_path)["edit"]

    with pytest.raises(ValueError, match="match exactly once"):
        await tool.execute(
            "edit",
            {"path": "source.txt", "edits": [{"oldText": "same", "newText": "x"}]},
            None,
        )
    with pytest.raises(ValueError, match="overlapping"):
        await tool.execute(
            "edit",
            {
                "path": "source.txt",
                "edits": [
                    {"oldText": "abc", "newText": "x"},
                    {"oldText": "bcde", "newText": "y"},
                ],
            },
            None,
        )
    assert path.read_text(encoding="utf-8") == "same same\nabcdef\n"


@pytest.mark.asyncio
async def test_same_file_mutations_are_serialized(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("alpha beta\n", encoding="utf-8")
    tool = create_all_tools(tmp_path)["edit"]

    await asyncio.gather(
        tool.execute(
            "one",
            {"path": "source.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]},
            None,
        ),
        tool.execute(
            "two",
            {"path": "source.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]},
            None,
        ),
    )

    assert path.read_text(encoding="utf-8") == "ALPHA BETA\n"


@pytest.mark.asyncio
async def test_bash_streams_updates_times_out_and_retains_large_output(tmp_path: Path) -> None:
    tool = create_all_tools(tmp_path)["bash"]
    updates: list[str] = []

    async def update(result) -> None:
        updates.append(result.content[0].text)

    result = await tool.execute("bash", {"command": "printf hello"}, update)
    assert result.content[0].text == "hello"
    assert result.details["outputBytes"] == 5
    assert result.details["exitCode"] == 0
    assert "hello" in "".join(updates)

    with pytest.raises(TimeoutError, match="timed out"):
        await tool.execute("bash", {"command": "sleep 2", "timeout": 0.01}, None)

    large = await tool.execute(
        "bash",
        {"command": "python -c \"print('x' * 60000)\""},
        None,
    )
    assert large.details["truncated"] is True
    full_path = Path(large.details["fullOutputPath"])
    assert full_path.is_file()
    assert len(full_path.read_text(encoding="utf-8").rstrip("\n")) == 60_000
    assert full_path.stat().st_mode & 0o777 == 0o600


def test_prune_bash_logs_ignores_active_files_and_applies_age_and_size(tmp_path: Path) -> None:
    old = tmp_path / "bash-old.log"
    first = tmp_path / "bash-first.log"
    second = tmp_path / "bash-second.log"
    active = tmp_path / "bash-active.tmp"
    for path in (old, first, second, active):
        path.write_bytes(b"x" * 10)
    os.utime(old, (1, 1))
    os.utime(first, (90, 90))
    os.utime(second, (95, 95))

    prune_bash_logs(tmp_path, now=100, retention_seconds=50, cap_bytes=10)

    assert not old.exists()
    assert not first.exists()
    assert second.exists()
    assert active.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is unavailable")
async def test_optional_grep_find_and_ls_tools(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = True\n", encoding="utf-8")
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    tools = create_all_tools(tmp_path)

    grep = await tools["grep"].execute("grep", {"pattern": "needle"}, None)
    found = await tools["find"].execute("find", {"pattern": "*.py"}, None)
    listed = await tools["ls"].execute("ls", {}, None)

    assert "src/app.py:1:needle" in grep.content[0].text
    assert grep.details["resultCount"] == 1
    assert "src/app.py" in found.content[0].text
    assert found.details["resultCount"] == 1
    assert ".hidden" in listed.content[0].text
    assert "src/" in listed.content[0].text
    assert listed.details["resultCount"] == 2


def test_tool_names_schemas_and_default_activation_match_pi() -> None:
    tools = create_all_tools(Path.cwd())
    assert tuple(tools) == ("read", "bash", "edit", "write", "grep", "find", "ls")
    assert DEFAULT_ACTIVE_TOOLS == ("read", "bash", "edit", "write")
    for tool in tools.values():
        assert tool.parameters["type"] == "object"
        assert tool.parameters["additionalProperties"] is False
