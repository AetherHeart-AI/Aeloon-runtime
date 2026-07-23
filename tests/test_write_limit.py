"""Write/str_replace per-call content budget policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aeloon_core.model_metadata import litellm_max_output_tokens_from_table
from aeloon_core.tools.filesystem import (
    DEFAULT_MAX_ARGUMENT_CHARS,
    HOST_MAX_ARGUMENT_CHARS,
    StrReplaceTool,
    WriteTool,
    resolve_max_argument_chars,
)


class ResolveMaxArgumentCharsTests(unittest.TestCase):
    def test_unknown_model_defaults_to_128k(self) -> None:
        self.assertEqual(DEFAULT_MAX_ARGUMENT_CHARS, 128_000)
        self.assertEqual(resolve_max_argument_chars(None), 128_000)

    def test_converts_model_output_tokens_to_characters(self) -> None:
        self.assertEqual(resolve_max_argument_chars(384_000), 1_536_000)
        self.assertEqual(resolve_max_argument_chars(8_192), 32_768)
        self.assertEqual(resolve_max_argument_chars(4_096), 16_384)

    def test_model_budget_never_exceeds_host_ceiling(self) -> None:
        self.assertEqual(resolve_max_argument_chars(10_000_000), HOST_MAX_ARGUMENT_CHARS)

    def test_invalid_model_limit_falls_back(self) -> None:
        self.assertEqual(resolve_max_argument_chars(0), 128_000)
        self.assertEqual(resolve_max_argument_chars(-1), 128_000)


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
    async def test_default_accepts_content_just_above_old_32k_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            tool = WriteTool(workspace=workspace)
            content = "x" * 32_963

            result = await tool.execute(path="index.html", content=content)

            self.assertTrue(result.startswith("Successfully wrote"))
            self.assertEqual((workspace / "index.html").read_text(), content)

    async def test_schema_and_runtime_respect_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            tool = WriteTool(workspace=workspace, max_content_chars=100)
            schema = tool.to_schema()
            content_schema = schema["input_schema"]["properties"]["content"]
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
