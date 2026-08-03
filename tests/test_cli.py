from __future__ import annotations

import json
from pathlib import Path

import pytest

import aeloon_core.__main__ as cli
from aeloon_core.harness import AssistantMessage, ScriptedProvider, TextContent, Usage


def _provider(text: str = "done") -> ScriptedProvider:
    return ScriptedProvider(
        [
            AssistantMessage(
                (TextContent(text),),
                provider="deepseek",
                model="deepseek-v4-flash",
                usage=Usage(input=3, output=1, total_tokens=4),
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["text", "json", "stream-json"])
async def test_run_output_modes_are_stable_and_network_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output: str,
) -> None:
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider())

    code = await cli.async_main(
        [
            "run",
            "hello",
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--no-session",
            "--output",
            output,
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    if output == "text":
        assert captured.out == "done\n"
    elif output == "json":
        payload = json.loads(captured.out)
        assert payload["type"] == "result"
        assert payload["session_id"] is None
        assert payload["final_content"] == "done"
    else:
        lines = [json.loads(line) for line in captured.out.splitlines()]
        assert lines[0]["type"] == "before_agent_start"
        assert lines[1]["type"] == "agent_start"
        assert lines[-1]["type"] == "result"
        assert lines[-1]["status"] == "completed"
    assert not (tmp_path / "data" / "harness-sessions").exists()


@pytest.mark.asyncio
async def test_text_renderer_shows_tool_command_and_result(capsys) -> None:
    renderer = cli.RunRenderer("text")
    await renderer(
        cli.HarnessEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-1",
                "toolName": "bash",
                "args": {"command": "printf 'hello\\n'"},
            },
        )
    )
    await renderer(
        cli.HarnessEvent(
            "tool_execution_end",
            {
                "toolCallId": "call-1",
                "toolName": "bash",
                "result": {
                    "content": [{"type": "text", "text": "hello"}],
                    "details": {"exitCode": 0, "truncated": False},
                    "isError": False,
                },
                "isError": False,
            },
        )
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[tool] bash" in captured.err
    assert "$ printf 'hello\\n'" in captured.err
    assert "[tool result] bash · exit 0" in captured.err
    assert "  hello" in captured.err


@pytest.mark.asyncio
async def test_text_renderer_summarizes_mutation_arguments_and_shows_errors(capsys) -> None:
    renderer = cli.RunRenderer("text")
    await renderer(
        cli.HarnessEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-2",
                "toolName": "write",
                "args": {"path": "result.txt", "content": "do not echo this content"},
            },
        )
    )
    await renderer(
        cli.HarnessEvent(
            "tool_execution_end",
            {
                "toolCallId": "call-2",
                "toolName": "write",
                "result": {
                    "content": [{"type": "text", "text": "permission denied"}],
                    "isError": True,
                },
                "isError": True,
            },
        )
    )

    captured = capsys.readouterr()
    assert "path=result.txt (24 bytes)" in captured.err
    assert "do not echo this content" not in captured.err
    assert "[tool error] write" in captured.err
    assert "permission denied" in captured.err


@pytest.mark.asyncio
async def test_run_session_list_show_and_no_session(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("saved"))
    await cli.async_main(
        [
            "run",
            "persist",
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "--output",
            "json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    session_id = result["session_id"]

    await cli.async_main(["session", "list", "--data-dir", str(data_dir)])
    listed = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in listed] == [session_id]

    await cli.async_main(["session", "show", session_id, "--data-dir", str(data_dir)])
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == session_id
    assert [message["role"] for message in shown["context"]] == ["user", "assistant"]

    before = set((data_dir / "harness-sessions").glob("*.jsonl"))
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("ephemeral"))
    await cli.async_main(
        [
            "run",
            "temporary",
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "--no-session",
            "--output",
            "json",
        ]
    )
    assert set((data_dir / "harness-sessions").glob("*.jsonl")) == before


def test_config_path_init_show_and_set(tmp_path: Path, monkeypatch, capsys) -> None:
    path = tmp_path / "config.json"
    assert cli.main(["config", "path", "--config", str(path)]) == 0
    assert capsys.readouterr().out.strip() == str(path)

    assert (
        cli.main(
            [
                "config",
                "init",
                "--config",
                str(path),
                "--workspace",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    assert cli.main(["config", "set", "max-retries", "5", "--config", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["config", "show", "--config", str(path)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["agent"]["retry"]["max_retries"] == 5
    assert shown["deepseek"]["api_key"] == "***"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["deepseek"]["api_key"] == "no-key"
