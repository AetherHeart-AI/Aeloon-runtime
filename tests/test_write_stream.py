from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from aeloon_core.providers.base import ContentStreamSink, LLMProvider, LLMResponse
from aeloon_core.state_machine import run_agent_loop
from aeloon_core.tools.filesystem import EditTool, WriteTool
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.write_protocol import WriteFrameDecoder, WriteProtocolError
from aeloon_core.write_runtime import WriteCoordinator, WriteRuntimeError


def framed(tx: str, *, file_id: str, path: str, mode: str, body: str) -> str:
    header = json.dumps(
        {"tx": tx, "id": file_id, "path": path, "mode": mode},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<<<AELOON_WRITE_V1 {header}>>>\n{body}<<<END_AELOON_WRITE_V1:{tx}:{file_id}>>>"


class WriteProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.data_dir = root / "data"
        self.workspace.mkdir()
        self.coordinator = WriteCoordinator(
            workspace=self.workspace,
            data_dir=self.data_dir,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    async def test_every_chunk_boundary_preserves_body_and_delays_target(self) -> None:
        tx = "tx-boundaries"
        body = 'quotes: \\"\nUnicode: 中文\n<think>literal file tag</think>'
        payload = framed(tx, file_id="f1", path="src/a.txt", mode="create", body=body)
        payload += f"<<<END_AELOON_WRITE_BATCH_V1:{tx}>>>"

        for split in range(len(payload) + 1):
            attempt = self.coordinator.start_attempt(tx)
            decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
            decoder.feed(payload[:split])
            self.assertFalse((self.workspace / "src/a.txt").exists())
            decoder.feed(payload[split:])
            result = decoder.finalize(finish_reason="stop", has_tool_calls=False)
            self.assertIsNotNone(result.batch)
            assert result.batch is not None
            self.assertEqual(result.batch.files[0].staging_path.read_text(), body)
            self.coordinator.discard_batch(result.batch)

        attempt = self.coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        for char in payload:
            decoder.feed(char)
        result = decoder.finalize(finish_reason="stop", has_tool_calls=False)
        assert result.batch is not None
        await self.coordinator.commit_batch(result.batch)
        self.assertEqual((self.workspace / "src/a.txt").read_text(), body)

    def test_think_state_cannot_trigger_write_but_body_is_opaque(self) -> None:
        tx = "tx-think"
        hidden = framed(tx, file_id="bad", path="bad.txt", mode="create", body="bad")
        attempt = self.coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        decoder.feed(f"<think>{hidden}</think>visible")
        result = decoder.finalize(finish_reason="stop", has_tool_calls=False)
        self.assertEqual(result.visible_content, "visible")
        self.assertIsNone(result.batch)
        self.assertFalse((self.workspace / "bad.txt").exists())

    def test_incomplete_or_mixed_batch_is_discarded(self) -> None:
        tx = "tx-incomplete"
        attempt = self.coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        decoder.feed(framed(tx, file_id="f1", path="a.txt", mode="create", body="body"))
        with self.assertRaises(WriteProtocolError):
            decoder.finalize(finish_reason="length", has_tool_calls=False)
        self.assertFalse((self.workspace / "a.txt").exists())
        self.assertFalse(attempt.staging_dir.exists())

        attempt = self.coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        decoder.feed(
            framed(tx, file_id="f1", path="a.txt", mode="create", body="body")
            + f"<<<END_AELOON_WRITE_BATCH_V1:{tx}>>>"
        )
        with self.assertRaises(WriteProtocolError):
            decoder.finalize(finish_reason="stop", has_tool_calls=True)
        self.assertFalse((self.workspace / "a.txt").exists())

    def test_path_policy_rejects_escape_and_symlink(self) -> None:
        with self.assertRaises(WriteRuntimeError):
            self.coordinator.resolve_target("../outside.txt")
        with self.assertRaises(WriteRuntimeError):
            self.coordinator.resolve_target("/tmp/outside.txt")
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.workspace / "link").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(WriteRuntimeError):
            self.coordinator.resolve_target("link/file.txt")

    async def test_overwrite_detects_concurrent_change(self) -> None:
        target = self.workspace / "a.txt"
        target.write_text("old")
        tx = "tx-conflict"
        attempt = self.coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        decoder.feed(
            framed(tx, file_id="f1", path="a.txt", mode="overwrite", body="new")
            + f"<<<END_AELOON_WRITE_BATCH_V1:{tx}>>>"
        )
        result = decoder.finalize(finish_reason="stop", has_tool_calls=False)
        assert result.batch is not None
        target.write_text("user change")
        with self.assertRaises(WriteRuntimeError):
            await self.coordinator.commit_batch(result.batch)
        self.assertEqual(target.read_text(), "user change")

    async def test_commit_failure_rolls_back_every_target(self) -> None:
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("old-first")
        second.write_text("old-second")
        tx = "tx-rollback"
        attempt = self.coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        decoder.feed(
            framed(tx, file_id="f1", path="first.txt", mode="overwrite", body="new-first")
            + framed(tx, file_id="f2", path="second.txt", mode="overwrite", body="new-second")
            + f"<<<END_AELOON_WRITE_BATCH_V1:{tx}>>>"
        )
        result = decoder.finalize(finish_reason="stop", has_tool_calls=False)
        assert result.batch is not None
        original_replace = __import__("os").replace

        def fail_second_target(source: str | Path, destination: str | Path) -> None:
            source_path = Path(source)
            destination_path = Path(destination)
            if destination_path.resolve(
                strict=False
            ) == second.resolve() and not source_path.name.startswith(".aeloon-backup-"):
                raise OSError("injected second-target failure")
            original_replace(source, destination)

        with mock.patch("aeloon_core.write_runtime.os.replace", side_effect=fail_second_target):
            with self.assertRaises(OSError):
                await self.coordinator.commit_batch(result.batch)

        self.assertEqual(first.read_text(), "old-first")
        self.assertEqual(second.read_text(), "old-second")
        self.assertFalse(list(self.workspace.glob(".aeloon-*")))

    async def test_formatter_failure_leaves_workspace_unchanged(self) -> None:
        def failing_formatter(_path: Path, _logical_path: str) -> None:
            raise RuntimeError("formatter failed")

        coordinator = WriteCoordinator(
            workspace=self.workspace,
            data_dir=self.data_dir / "formatted",
            formatter=failing_formatter,
        )
        tx = "tx-format"
        attempt = coordinator.start_attempt(tx)
        decoder = WriteFrameDecoder(transaction_id=tx, attempt=attempt)
        decoder.feed(
            framed(tx, file_id="f1", path="formatted.txt", mode="create", body="body")
            + f"<<<END_AELOON_WRITE_BATCH_V1:{tx}>>>"
        )
        result = decoder.finalize(finish_reason="stop", has_tool_calls=False)
        assert result.batch is not None
        with self.assertRaises(RuntimeError):
            await coordinator.commit_batch(result.batch)
        self.assertFalse((self.workspace / "formatted.txt").exists())


class StreamingProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(self, *, fail_first_attempt: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self.fail_first_attempt = fail_first_attempt
        self.body = 'unique-body-sentinel\\"\n第二行\n'

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        del response_format
        raise AssertionError("the write-capable headless path must use streaming")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Any = None,
        on_reasoning_delta: Any = None,
        content_sink: ContentStreamSink | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        del on_reasoning_delta
        self.calls += 1
        write_call = self.calls == (2 if self.fail_first_attempt else 1)
        if self.fail_first_attempt and self.calls == 1:
            guidance = next(
                str(item.get("content") or "")
                for item in reversed(messages)
                if "[aeloon-core:write-protocol-v1]" in str(item.get("content") or "")
            )
            match = re.search(r'"tx":"([a-f0-9]+)"', guidance)
            assert match is not None
            partial = framed(
                match.group(1),
                file_id="partial",
                path="partial.txt",
                mode="create",
                body="partial body",
            )[:-12]
            if content_sink is not None:
                await content_sink.feed(partial)
            return LLMResponse(
                content="HTTP 429: rate limit exceeded",
                finish_reason="error",
            )
        if write_call:
            guidance = next(
                str(item.get("content") or "")
                for item in reversed(messages)
                if "[aeloon-core:write-protocol-v1]" in str(item.get("content") or "")
            )
            match = re.search(r'"tx":"([a-f0-9]+)"', guidance)
            assert match is not None
            tx = match.group(1)
            content = (
                "Writing one file.\n"
                + framed(tx, file_id="f1", path="generated.txt", mode="create", body=self.body)
                + f"<<<END_AELOON_WRITE_BATCH_V1:{tx}>>>"
            )
        else:
            content = "Done."
        visible_parts: list[str] = []
        for offset in range(0, len(content), 3):
            delta = content[offset : offset + 3]
            visible = await content_sink.feed(delta) if content_sink is not None else delta
            if visible:
                visible_parts.append(visible)
                if on_delta is not None:
                    await on_delta(visible)
        return LLMResponse(content="".join(visible_parts), finish_reason="stop")


class WriteStreamIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_uasm_commits_streamed_body_without_persisting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            coordinator = WriteCoordinator(workspace=workspace, data_dir=root / "data")
            registry = ToolRegistry()
            registry.register(WriteTool(workspace=workspace))
            registry.register(EditTool(workspace=workspace))
            provider = StreamingProvider()

            state = await run_agent_loop(
                provider=provider,
                model="test",
                tools=registry,
                messages=[{"role": "user", "content": "create the generated file"}],
                max_iterations=4,
                transition_trace_enabled=False,
                write_coordinator=coordinator,
            )

            self.assertEqual((workspace / "generated.txt").read_text(), provider.body)
            self.assertEqual(state.metadata.final_content, "Done.")
            serialized = json.dumps(state.messages, ensure_ascii=False)
            self.assertNotIn("unique-body-sentinel", serialized)
            self.assertIn("sha256", serialized)
            self.assertIn("write", state.tools_used)

    async def test_transient_retry_discards_failed_attempt_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            coordinator = WriteCoordinator(workspace=workspace, data_dir=root / "data")
            registry = ToolRegistry()
            registry.register(WriteTool(workspace=workspace))
            provider = StreamingProvider(fail_first_attempt=True)

            state = await run_agent_loop(
                provider=provider,
                model="test",
                tools=registry,
                messages=[{"role": "user", "content": "create the generated file"}],
                max_iterations=4,
                transition_trace_enabled=False,
                write_coordinator=coordinator,
            )

            self.assertEqual(state.metadata.final_content, "Done.")
            self.assertEqual((workspace / "generated.txt").read_text(), provider.body)
            self.assertFalse((workspace / "partial.txt").exists())
            self.assertEqual(list(coordinator.staging_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
