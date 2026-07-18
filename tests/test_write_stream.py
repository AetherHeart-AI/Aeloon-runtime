"""Regression coverage for the current atomic chunked write contract."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from aeloon_core.tools.filesystem import WriteTool
from aeloon_core.tools.registry import ToolRegistry


def _next_offset(result: str) -> int:
    match = re.search(r"next_offset=(\d+)", result)
    if match is None:
        raise AssertionError(f"write result omitted next_offset: {result}")
    return int(match.group(1))


class ChunkedWriteBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_then_append_preserves_utf8_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            tool = WriteTool(workspace=workspace)
            first = 'quotes: \\"\nUnicode: 中文\n'
            second = "<think>literal text</think>\n"

            created = await tool.execute(path="src/a.txt", content=first)
            self.assertTrue(created.startswith("Successfully wrote"))
            appended = await tool.execute(
                path="src/a.txt",
                content=second,
                expected_offset=_next_offset(created),
            )

            self.assertTrue(appended.startswith("Successfully wrote"))
            self.assertEqual((workspace / "src/a.txt").read_text(), first + second)
            self.assertEqual(_next_offset(appended), len((first + second).encode("utf-8")))

    async def test_stale_offset_has_no_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("user content")
            tool = WriteTool(workspace=workspace)

            result = await tool.execute(path="a.txt", content="agent", expected_offset=0)

            self.assertTrue(result.startswith("Error [OFFSET_CONFLICT]"))
            self.assertEqual(target.read_text(), "user content")

    async def test_create_never_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("original")
            tool = WriteTool(workspace=workspace)

            result = await tool.execute(path="a.txt", content="replacement")

            self.assertTrue(result.startswith("Error [TARGET_EXISTS]"))
            self.assertEqual(target.read_text(), "original")

    async def test_registry_rejects_unknown_write_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(WriteTool(workspace=Path(temporary)))

            result = await registry.execute(
                "write",
                {"path": "a.txt", "content": "ok", "unexpected": True},
            )

            self.assertIn("TOOL_ARGUMENTS_INVALID", result)
            self.assertFalse((Path(temporary) / "a.txt").exists())


if __name__ == "__main__":
    unittest.main()
