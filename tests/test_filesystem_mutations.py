"""Safety and atomicity tests for filesystem mutation tools."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeloon_core.tools.filesystem import StrReplaceTool, WriteTool


class MutationPathSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_write_rejects_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.txt"
            outside.write_text("operator data")
            (workspace / "alias.txt").symlink_to(outside)

            result = await WriteTool(workspace=workspace).execute(
                path="alias.txt",
                content="agent data",
            )

            self.assertTrue(result.startswith("Error [PATH_SYMLINK]"))
            self.assertEqual(outside.read_text(), "operator data")

    async def test_write_rejects_symlink_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (workspace / "alias").symlink_to(outside, target_is_directory=True)

            result = await WriteTool(workspace=workspace).execute(
                path="alias/agent.txt",
                content="agent data",
            )

            self.assertTrue(result.startswith("Error [PATH_SYMLINK]"))
            self.assertFalse((outside / "agent.txt").exists())

    async def test_write_rejects_protected_path_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            protected = workspace / ".runtime-data"
            protected.mkdir()
            tool = WriteTool(workspace=workspace, denied_paths=(protected,))

            result = await tool.execute(path=".runtime-data/session.json", content="agent data")

            self.assertTrue(result.startswith("Error [PATH_PROTECTED]"))
            self.assertFalse((protected / "session.json").exists())


class AtomicMutationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_create_does_not_overwrite_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            real_fsync = os.fsync

            def create_target(descriptor: int) -> None:
                target.write_text("user content")
                real_fsync(descriptor)

            with patch("aeloon_core.tools.filesystem.os.fsync", side_effect=create_target):
                result = await WriteTool(workspace=workspace).execute(
                    path="a.txt",
                    content="agent content",
                )

            self.assertTrue(result.startswith("Error [CONCURRENT_MODIFICATION]"))
            self.assertEqual(target.read_text(), "user content")
            self.assertEqual(list(workspace.glob(".a.txt.*.tmp")), [])

    async def test_concurrent_overwrite_does_not_replace_newer_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("original")
            real_fsync = os.fsync

            def modify_target(descriptor: int) -> None:
                target.write_text("user edit")
                real_fsync(descriptor)

            with patch("aeloon_core.tools.filesystem.os.fsync", side_effect=modify_target):
                result = await WriteTool(workspace=workspace).execute(
                    path="a.txt",
                    content="agent content",
                )

            self.assertTrue(result.startswith("Error [CONCURRENT_MODIFICATION]"))
            self.assertEqual(target.read_text(), "user edit")
            self.assertEqual(list(workspace.glob(".a.txt.*.tmp")), [])

    @unittest.skipIf(os.name == "nt", "POSIX mode bits are required")
    async def test_write_overwrite_preserves_existing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("before")
            target.chmod(0o640)

            result = await WriteTool(workspace=workspace).execute(
                path="a.txt",
                content="after",
            )

            self.assertTrue(result.startswith("Successfully wrote"))
            self.assertEqual(target.read_text(), "after")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)

    async def test_replace_failure_keeps_original_and_cleans_staging_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("before")

            with patch(
                "aeloon_core.tools.filesystem.os.replace",
                side_effect=OSError("replace failed"),
            ):
                result = await StrReplaceTool(workspace=workspace).execute(
                    path="a.txt",
                    old_str="before",
                    new_str="after",
                )

            self.assertTrue(result.startswith("Error [IO_ERROR]"))
            self.assertEqual(target.read_text(), "before")
            self.assertEqual(list(workspace.glob(".a.txt.*.tmp")), [])

    async def test_str_replace_rejects_concurrent_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("before")
            real_fsync = os.fsync

            def modify_target(descriptor: int) -> None:
                target.write_text("user edit")
                real_fsync(descriptor)

            with patch("aeloon_core.tools.filesystem.os.fsync", side_effect=modify_target):
                result = await StrReplaceTool(workspace=workspace).execute(
                    path="a.txt",
                    old_str="before",
                    new_str="after",
                )

            self.assertTrue(result.startswith("Error [CONCURRENT_MODIFICATION]"))
            self.assertEqual(target.read_text(), "user edit")
            self.assertEqual(list(workspace.glob(".a.txt.*.tmp")), [])

    @unittest.skipIf(os.name == "nt", "POSIX mode bits are required")
    async def test_str_replace_preserves_existing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "a.txt"
            target.write_text("before")
            target.chmod(0o640)

            result = await StrReplaceTool(workspace=workspace).execute(
                path="a.txt",
                old_str="before",
                new_str="after",
            )

            self.assertTrue(result.startswith("Successfully replaced"))
            self.assertEqual(target.read_text(), "after")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
