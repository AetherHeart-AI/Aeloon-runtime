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
async def test_text_renderer_is_quiet_by_default(capsys) -> None:
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
    assert captured.err == ""


@pytest.mark.asyncio
async def test_text_renderer_summarizes_tool_command_and_result(capsys) -> None:
    renderer = cli.RunRenderer("text", verbose=True)
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
    assert "[run] $ printf 'hello\\n'" in captured.err
    assert "[ok] bash · exit 0" in captured.err
    assert "hello" not in captured.err.splitlines()[-1]


@pytest.mark.asyncio
async def test_text_renderer_reports_read_size_without_echoing_content(capsys) -> None:
    renderer = cli.RunRenderer("text", verbose=True)
    await renderer(
        cli.HarnessEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-read",
                "toolName": "read",
                "args": {"path": "large.md", "offset": 11, "limit": 30},
            },
        )
    )
    await renderer(
        cli.HarnessEvent(
            "tool_execution_end",
            {
                "toolCallId": "call-read",
                "toolName": "read",
                "result": {
                    "content": [{"type": "text", "text": "private file contents"}],
                    "details": {"sizeBytes": 2_048, "selectedLines": 30},
                },
            },
        )
    )

    rendered = capsys.readouterr().err
    assert "[read] large.md · lines 11-40" in rendered
    assert "[ok] read · 2.0 KB · 30 lines" in rendered
    assert "private file contents" not in rendered


@pytest.mark.asyncio
async def test_text_renderer_marks_nonzero_bash_exit_as_failed(capsys) -> None:
    renderer = cli.RunRenderer("text", verbose=True)
    await renderer(
        cli.HarnessEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-failed",
                "toolName": "bash",
                "args": {"command": "false"},
            },
        )
    )
    await renderer(
        cli.HarnessEvent(
            "tool_execution_end",
            {
                "toolCallId": "call-failed",
                "toolName": "bash",
                "result": {
                    "content": [{"type": "text", "text": "Command exited with code 1"}],
                    "details": {"exitCode": 1, "outputBytes": 0},
                    "isError": False,
                },
                "isError": False,
            },
        )
    )

    rendered = capsys.readouterr().err
    assert "[failed] bash · exit 1 · 0 B output" in rendered


@pytest.mark.asyncio
async def test_text_renderer_summarizes_mutation_arguments_and_shows_errors(capsys) -> None:
    renderer = cli.RunRenderer("text", verbose=True)
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
    assert "[write] result.txt · 24 B" in captured.err
    assert "do not echo this content" not in captured.err
    assert "[failed] write" in captured.err
    assert "permission denied" in captured.err


def test_text_renderer_renders_final_markdown_in_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: True)

    cli.RunRenderer("text").finish_text("# Result\n\n- first\n- second")

    rendered = capsys.readouterr().out
    assert "# Result" not in rendered
    assert "Result" in rendered
    assert "• first" in rendered


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


@pytest.mark.asyncio
async def test_cloud_login_reads_hidden_password_and_uses_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    socket_path = tmp_path / "bridge.sock"
    calls: dict[str, object] = {}

    async def fake_ensure_daemon(**kwargs):
        calls["ensure"] = kwargs
        return {"socket_path": str(socket_path), "status": "running"}

    async def fake_bridge_request(path, method, params=None, *, timeout=3.0):
        calls["request"] = {
            "path": path,
            "method": method,
            "params": params,
            "timeout": timeout,
        }
        return {
            "enabled": True,
            "authenticated": True,
            "user": {"id": "alice", "username": "alice", "display_name": "Alice"},
            "base_url": "https://cloud.example",
            "vault_kind": "test",
        }

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "bridge_request", fake_bridge_request)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "terminal-secret")

    code = await cli.async_main(
        [
            "cloud",
            "login",
            "alice",
            "--data-dir",
            str(tmp_path / "data"),
            "--socket",
            str(socket_path),
            "--output",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["user"]["username"] == "alice"
    assert "terminal-secret" not in captured.out
    assert calls["request"] == {
        "path": socket_path,
        "method": "cloud.account.login",
        "params": {"username": "alice", "password": "terminal-secret"},
        "timeout": 60.0,
    }
    assert not hasattr(cli.build_parser().parse_args(["cloud", "login", "alice"]), "password")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "result", "expected"),
    [
        (
            "status",
            {
                "authenticated": True,
                "user": {"username": "alice", "display_name": "Alice"},
            },
            "Signed in to Aeloon Cloud as Alice (@alice).\n",
        ),
        ("status", {"authenticated": False, "user": None}, "Not signed in to Aeloon Cloud.\n"),
        ("logout", {"authenticated": False, "user": None}, "Signed out of Aeloon Cloud.\n"),
    ],
)
async def test_cloud_status_and_logout_use_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    result: dict[str, object],
    expected: str,
) -> None:
    socket_path = tmp_path / "bridge.sock"
    requested: list[str] = []

    async def fake_ensure_daemon(**_kwargs):
        return {"socket_path": str(socket_path), "status": "running"}

    async def fake_bridge_request(_path, method, _params=None, *, timeout=3.0):
        requested.append(method)
        assert timeout == 3.0
        return result

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "bridge_request", fake_bridge_request)

    assert await cli.async_main(["cloud", command, "--socket", str(socket_path)]) == 0
    assert capsys.readouterr().out == expected
    assert requested == [f"cloud.account.{command}"]


@pytest.mark.asyncio
async def test_provider_add_reads_api_key_privately_and_uses_unified_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    socket_path = tmp_path / "bridge.sock"
    request: dict[str, object] = {}

    async def fake_ensure_daemon(**_kwargs):
        return {"socket_path": str(socket_path), "status": "running"}

    async def fake_bridge_request(path, method, params=None, *, timeout=3.0):
        request.update(path=path, method=method, params=params, timeout=timeout)
        return {
            "provider": {
                "id": "ollama",
                "name": "Ollama",
                "kind": "local",
                "base_url": "http://127.0.0.1:11434/v1",
                "model_ids": ["ollama/qwen3-coder"],
            },
            "revision": 2,
        }

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "bridge_request", fake_bridge_request)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "private-local-key")

    assert (
        await cli.async_main(
            [
                "provider",
                "add",
                "ollama",
                "--name",
                "Ollama",
                "--base-url",
                "http://127.0.0.1:11434/v1",
                "--model",
                "qwen3-coder",
                "--socket",
                str(socket_path),
                "--output",
                "json",
            ]
        )
        == 0
    )

    assert request == {
        "path": socket_path,
        "method": "provider.local.add",
        "params": {
            "provider_id": "ollama",
            "name": "Ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "private-local-key",
            "models": ["qwen3-coder"],
        },
        "timeout": 3.0,
    }
    assert "private-local-key" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_provider_login_uses_cloud_rpc_compatible_with_existing_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    socket_path = tmp_path / "bridge.sock"
    calls: dict[str, object] = {}

    async def fake_ensure_daemon(**kwargs):
        calls["ensure"] = kwargs
        return {"socket_path": str(socket_path), "status": "running"}

    async def fake_bridge_request(path, method, params=None, *, timeout=3.0):
        calls["request"] = (path, method, params, timeout)
        return {
            "authenticated": True,
            "user": {"username": "alice", "display_name": "Alice"},
        }

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "bridge_request", fake_bridge_request)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "secret")

    assert (
        await cli.async_main(
            ["provider", "login", "alice", "--socket", str(socket_path), "--output", "json"]
        )
        == 0
    )

    assert calls["request"] == (
        socket_path,
        "cloud.account.login",
        {"username": "alice", "password": "secret"},
        60.0,
    )
    assert calls["ensure"]["required_methods"] == ()  # type: ignore[index]
    assert "secret" not in capsys.readouterr().out
