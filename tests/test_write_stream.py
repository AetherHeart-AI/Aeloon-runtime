"""Regression coverage for atomic complete-file writes and edit-based creation."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from aeloon_core.tools.filesystem import StrReplaceTool, WriteTool
from aeloon_core.tools.registry import ToolRegistry


class AtomicWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_writes_complete_utf8_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            tool = WriteTool(workspace=workspace)
            content = 'quotes: \\"\nUnicode: 中文\n<think>literal text</think>\n'

            result = await tool.execute(path="src/a.txt", content=content)

            raw = content.encode("utf-8")
            self.assertTrue(result.startswith("Successfully wrote"))
            self.assertIn(f"bytes={len(raw)}", result)
            self.assertIn(f"total_bytes={len(raw)}", result)
            self.assertIn(f"sha256={hashlib.sha256(raw).hexdigest()}", result)
            self.assertEqual((workspace / "src/a.txt").read_text(), content)

    async def test_write_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("original")
            tool = WriteTool(workspace=workspace)

            result = await tool.execute(path="a.txt", content="replacement")

            self.assertTrue(result.startswith("Successfully wrote"))
            self.assertEqual(target.read_text(), "replacement")

    async def test_registry_rejects_removed_expected_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            tool = WriteTool(workspace=Path(temporary))
            registry.register(tool)

            schema = tool.to_schema()["input_schema"]
            self.assertEqual(set(schema["properties"]), {"path", "content"})
            self.assertEqual(set(schema["required"]), {"path", "content"})

            result = await registry.execute(
                "write",
                {"path": "a.txt", "content": "ok", "expected_offset": 0},
            )

            self.assertIn("TOOL_ARGUMENTS_INVALID", result)
            self.assertFalse((Path(temporary) / "a.txt").exists())


class StrReplaceCreateTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_old_str_creates_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            tool = StrReplaceTool(workspace=workspace)
            content = "first\r\nsecond\r\n"

            result = await tool.execute(
                path="src/a.txt",
                old_str="",
                new_str=content,
            )

            self.assertTrue(result.startswith("Successfully created file"))
            self.assertEqual((workspace / "src/a.txt").read_bytes(), content.encode("utf-8"))

    async def test_empty_old_str_refuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("original")
            tool = StrReplaceTool(workspace=workspace)

            result = await tool.execute(path="a.txt", old_str="", new_str="replacement")

            self.assertTrue(result.startswith("Error [TARGET_EXISTS]"))
            self.assertEqual(target.read_text(), "original")

    def test_schema_allows_empty_old_str(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tool = StrReplaceTool(workspace=Path(temporary))

            old_str_schema = tool.to_schema()["input_schema"]["properties"]["old_str"]

            self.assertNotIn("minLength", old_str_schema)

if __name__ == "__main__":
    unittest.main()
