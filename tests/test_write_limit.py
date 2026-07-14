"""Write/str_replace per-call content budget policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aeloon_core.model_metadata import litellm_max_output_tokens_from_table
from aeloon_core.tools.filesystem import (
    DEFAULT_MAX_ARGUMENT_CHARS,
    StrReplaceTool,
    WriteTool,
    resolve_max_argument_chars,
)


class ResolveMaxArgumentCharsTests(unittest.TestCase):
    def test_default_is_32k(self) -> None:
        self.assertEqual(DEFAULT_MAX_ARGUMENT_CHARS, 32_000)
        self.assertEqual(resolve_max_argument_chars(None), 32_000)

    def test_min_of_default_and_model_max_output(self) -> None:
        self.assertEqual(resolve_max_argument_chars(384_000), 32_000)
        self.assertEqual(resolve_max_argument_chars(8_192), 8_192)

    def test_invalid_model_limit_falls_back(self) -> None:
        self.assertEqual(resolve_max_argument_chars(0), 32_000)
        self.assertEqual(resolve_max_argument_chars(-1), 32_000)


class LiteLLMMaxOutputLookupTests(unittest.TestCase):
    def test_prefers_max_output_tokens(self) -> None:
        table = {
            "deepseek/deepseek-v4-flash": {
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 384_000,
                "max_tokens": 384_000,
            }
        }
        self.assertEqual(
            litellm_max_output_tokens_from_table(table, "deepseek-v4-flash"),
            384_000,
        )

    def test_falls_back_to_max_tokens(self) -> None:
        table = {"vendor/model-x": {"max_tokens": 4096}}
        self.assertEqual(litellm_max_output_tokens_from_table(table, "model-x"), 4096)


class WriteToolLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_and_runtime_respect_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            tool = WriteTool(workspace=workspace, max_content_chars=100)
            schema = tool.to_schema()
            content_schema = schema["function"]["parameters"]["properties"]["content"]
            self.assertEqual(content_schema.get("maxLength"), 100)

            failed = await tool.execute(path="big.txt", content="x" * 101)
            self.assertTrue(failed.startswith("Error [CONTENT_TOO_LARGE]"))
            self.assertIn("limit=100", failed)

            ok = await tool.execute(path="ok.txt", content="hello")
            self.assertTrue(ok.startswith("Successfully wrote"))
            self.assertEqual((workspace / "ok.txt").read_text(), "hello")

    async def test_str_replace_shares_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "file.txt"
            target.write_text("abcdef")
            tool = StrReplaceTool(workspace=workspace, max_content_chars=3)
            failed = await tool.execute(path="file.txt", old_str="abc", new_str="wxyz")
            self.assertTrue(failed.startswith("Error [CONTENT_TOO_LARGE]"))


if __name__ == "__main__":
    unittest.main()
