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
            "--model",
            "deepseek/deepseek-v4-flash",
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
async def test_text_renderer_uses_one_status_line_in_a_terminal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_is_interactive_stderr", lambda: True)
    renderer = cli.RunRenderer("text")

    await renderer(cli.HarnessEvent("agent_start"))
    await renderer(
        cli.HarnessEvent(
            "tool_execution_start",
            {"toolCallId": "call-1", "toolName": "read", "args": {"path": "private.txt"}},
        )
    )
    await renderer(cli.HarnessEvent("agent_end"))

    rendered = capsys.readouterr().err
    assert "Working…" in rendered
    assert "Reading files…" in rendered
    assert "private.txt" not in rendered
    assert "\x1b[2K" in rendered


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
            "--model",
            "deepseek/deepseek-v4-flash",
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
            "--model",
            "deepseek/deepseek-v4-flash",
            "--output",
            "json",
        ]
    )
    assert set((data_dir / "harness-sessions").glob("*.jsonl")) == before


@pytest.mark.asyncio
async def test_session_dir_does_not_change_cloud_account_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config_path = tmp_path / "config.json"
    configured_data_dir = tmp_path / "core-data"
    session_dir = tmp_path / "benchmark-sessions"
    cli.save_config(
        cli.Config(
            workspace=tmp_path,
            data_dir=configured_data_dir,
            deepseek={"api_key": "configured-key"},
        ),
        config_path,
    )
    account_data_dirs: list[Path] = []
    account_type = cli.CloudAccountService

    def cloud_account(config, *, data_dir):
        account_data_dirs.append(data_dir)
        return account_type(config, data_dir=data_dir)

    monkeypatch.setattr(cli, "CloudAccountService", cloud_account)
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("isolated"))

    assert (
        await cli.async_main(
            [
                "run",
                "task",
                "--config",
                str(config_path),
                "--session-dir",
                str(session_dir),
                "--model",
                "deepseek/deepseek-v4-flash",
                "--output",
                "json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert account_data_dirs == [configured_data_dir.resolve()]
    assert len(list((session_dir / "harness-sessions").glob("*.jsonl"))) == 1
    assert not (configured_data_dir / "harness-sessions").exists()


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
    assert shown["agent"]["model"] == ""
    assert shown["deepseek"]["api_key"] == "no-key"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["deepseek"]["api_key"] == "no-key"


def test_fresh_config_ignores_deepseek_environment_and_has_no_default_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-read")

    config = cli.load_config(tmp_path / "missing.json")

    assert config.deepseek.api_key == "no-key"
    assert config.agent.model == ""


def test_run_without_any_connected_model_explains_setup(tmp_path: Path, capsys) -> None:
    assert (
        cli.main(
            [
                "inspect this project",
                "--config",
                str(tmp_path / "missing.json"),
                "--data-dir",
                str(tmp_path / "data"),
                "--ephemeral",
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "No connected model is available" in error
    assert "aeloon local add" in error
    assert "first available model automatically" in error


def test_top_level_help_focuses_on_user_tasks() -> None:
    help_text = cli.build_parser().format_help()

    assert 'aeloon "fix the failing tests"' in help_text
    assert "resume" in help_text
    assert "history" in help_text
    assert "doctor" in help_text
    assert "bridge    " not in help_text
    assert "session   " not in help_text
    assert "==SUPPRESS==" not in help_text


def test_default_task_command_and_explicit_separator_are_normalized() -> None:
    assert cli._normalize_argv(["fix", "the", "tests"]) == ["run", "fix", "the", "tests"]
    assert cli._normalize_argv(["--json", "inspect"]) == ["run", "--json", "inspect"]
    assert cli._normalize_argv(["--", "models"]) == ["run", "models"]
    assert cli._normalize_argv(["resume", "continue"]) == ["resume", "continue"]


def test_new_commands_use_actionable_errors_while_legacy_run_keeps_json(
    tmp_path: Path, capsys
) -> None:
    assert (
        cli.main(
            [
                "resume",
                "continue",
                "-C",
                str(tmp_path),
                "--data-dir",
                str(tmp_path / "data"),
            ]
        )
        == 2
    )
    human = capsys.readouterr().err
    assert human.startswith("Error: No saved task")
    assert "aeloon history" in human

    assert cli.main(["run"]) == 2
    legacy = json.loads(capsys.readouterr().err)
    assert legacy["error"] == "invalid_argument"


@pytest.mark.asyncio
async def test_default_task_runs_without_run_verb(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("direct"))

    code = await cli.async_main(
        [
            "inspect this repository",
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--ephemeral",
            "--model",
            "deepseek/deepseek-v4-flash",
        ]
    )

    assert code == 0
    assert capsys.readouterr().out == "direct\n"


@pytest.mark.asyncio
async def test_effort_flag_overrides_reasoning_effort_for_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    provider = _provider("reasoned")
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: provider)

    code = await cli.async_main(
        [
            "inspect this repository",
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--ephemeral",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--effort",
            "max",
        ]
    )

    assert code == 0
    assert provider.requests[0][2].thinking_level == "max"
    assert capsys.readouterr().out == "reasoned\n"


@pytest.mark.asyncio
async def test_resume_uses_latest_session_in_workspace(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("saved"))
    await cli.async_main(
        [
            "first task",
            "-C",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "--model",
            "deepseek/deepseek-v4-flash",
            "--json",
        ]
    )
    first = json.loads(capsys.readouterr().out)

    await cli.async_main(
        [
            "newer task",
            "-C",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "--model",
            "deepseek/deepseek-v4-flash",
            "--json",
        ]
    )
    second = json.loads(capsys.readouterr().out)
    assert second["session_id"] != first["session_id"]

    repository = cli.JsonlSessionRepository(data_dir)
    first_session = await repository.open(first["session_id"])
    await first_session.append_custom_message(custom_type="note", content="recent activity")

    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("continued"))
    await cli.async_main(
        [
            "resume",
            "continue",
            "-C",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "--json",
        ]
    )
    resumed = json.loads(capsys.readouterr().out)

    assert resumed["session_id"] == first["session_id"]
    assert resumed["final_content"] == "continued"


@pytest.mark.asyncio
async def test_history_is_human_readable_and_supports_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("done"))
    await cli.async_main(
        [
            "remember this task",
            "-C",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
            "--model",
            "deepseek/deepseek-v4-flash",
        ]
    )
    capsys.readouterr()

    assert (
        await cli.async_main(
            ["history", "-C", str(tmp_path), "--data-dir", str(data_dir)]
        )
        == 0
    )
    human = capsys.readouterr().out
    assert "UPDATED\tWORKSPACE\tSUMMARY\tSESSION" in human
    assert "remember this task" in human

    await cli.async_main(
        ["history", "-C", str(tmp_path), "--data-dir", str(data_dir), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["summary"] == "remember this task"

    await cli.async_main(
        ["history", payload[0]["id"][:8], "--data-dir", str(data_dir)]
    )
    detail = capsys.readouterr().out
    assert "remember this task" in detail
    assert f"Session: {payload[0]['id']}" in detail


@pytest.mark.asyncio
async def test_models_and_doctor_offer_human_and_machine_views(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.json"
    cli.save_config(cli.Config(workspace=tmp_path, data_dir=tmp_path / "data"), config_path)

    assert await cli.async_main(["models", "--config", str(config_path)]) == 0
    models = capsys.readouterr().out
    assert "No models are connected" in models
    assert "deepseek/deepseek-v4-flash" not in models

    assert await cli.async_main(["doctor", "--config", str(config_path), "--json"]) == 1
    diagnosis = json.loads(capsys.readouterr().out)
    assert diagnosis["ok"] is False
    model = next(item for item in diagnosis["checks"] if item["name"] == "model")
    assert model["status"] == "error"
    assert "aeloon local add" in model["fix"]


@pytest.mark.asyncio
async def test_first_listed_model_is_automatic_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    cli.save_config(
        cli.Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            local_providers={
                "studio": {
                    "name": "Studio",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "models": [{"id": "first"}, {"id": "second"}],
                }
            },
        ),
        config_path,
    )
    provider = _provider("automatic")
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: provider)

    assert (
        await cli.async_main(
            ["automatic task", "--config", str(config_path), "--ephemeral", "--json"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["final_content"] == "automatic"
    assert provider.requests[0][0].id == "studio/first"

    assert await cli.async_main(["models", "--config", str(config_path), "--json"]) == 0
    models = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in models] == ["studio/first", "studio/second"]
    assert models[0]["selected"] is True
    assert models[0]["automatic"] is True
    assert models[1]["selected"] is False
    assert models[1]["automatic"] is False

    assert await cli.async_main(["doctor", "--config", str(config_path), "--json"]) == 0
    diagnosis = json.loads(capsys.readouterr().out)
    model_check = next(item for item in diagnosis["checks"] if item["name"] == "model")
    assert model_check["status"] == "warning"
    assert "automatically use studio/first" in model_check["message"]


@pytest.mark.asyncio
async def test_explicit_short_model_uses_first_matching_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    cli.save_config(
        cli.Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            local_providers={
                "studio": {
                    "name": "Studio",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "models": [{"id": "deepseek-v4-flash"}],
                },
                "backup": {
                    "name": "Backup",
                    "base_url": "http://127.0.0.1:9000/v1",
                    "models": [{"id": "deepseek-v4-flash"}],
                },
            },
        ),
        config_path,
    )
    provider = _provider("matched")
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: provider)

    assert (
        await cli.async_main(
            [
                "matched task",
                "--config",
                str(config_path),
                "--model",
                "deepseek-v4-flash",
                "--ephemeral",
                "--json",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["final_content"] == "matched"
    assert provider.requests[0][0].id == "studio/deepseek-v4-flash"


@pytest.mark.asyncio
async def test_setup_configures_deepseek_without_exposing_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ignored-environment-secret")
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "setup-secret")

    assert (
        await cli.async_main(
            [
                "setup",
                "--provider",
                "deepseek",
                "--model",
                "deepseek-v4-pro",
                "--config",
                str(config_path),
                "-C",
                str(tmp_path),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "setup-secret" not in output
    assert saved["deepseek"]["api_key"] == "setup-secret"
    assert saved["deepseek"]["api_key"] != "ignored-environment-secret"
    assert saved["agent"]["model"] == "deepseek/deepseek-v4-pro"


@pytest.mark.asyncio
async def test_setup_cloud_selects_an_account_model(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    socket_path = tmp_path / "bridge.sock"
    calls: list[str] = []

    async def fake_cloud_command(args):
        calls.append(args.cloud_command)
        return 0

    async def fake_ensure_daemon(**_kwargs):
        return {"socket_path": str(socket_path), "status": "running"}

    async def fake_bridge_request(_path, method, params=None, **_kwargs):
        if method == "catalog.get":
            return {
                "models": [
                    {
                        "id": "aeloon-cloud/reasoner",
                        "provider_id": "aeloon-cloud",
                    }
                ]
            }
        if method == "settings.get":
            return {"revision": 1}
        assert method == "settings.update"
        assert params == {
            "revision": 1,
            "patch": {"default_model_id": "aeloon-cloud/reasoner"},
        }
        persisted = json.loads(config_path.read_text(encoding="utf-8"))
        persisted["agent"]["model"] = "aeloon-cloud/reasoner"
        config_path.write_text(json.dumps(persisted), encoding="utf-8")
        return {"revision": 2}

    monkeypatch.setattr(cli, "cloud_command", fake_cloud_command)
    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "bridge_request", fake_bridge_request)

    assert (
        await cli.async_main(
            [
                "setup",
                "--provider",
                "cloud",
                "--username",
                "alice",
                "--config",
                str(config_path),
                "-C",
                str(tmp_path),
            ]
        )
        == 0
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert calls == ["login"]
    assert saved["agent"]["model"] == "aeloon-cloud/reasoner"
    assert "Selected aeloon-cloud/reasoner" in capsys.readouterr().out


def test_completion_command_emits_shell_script(capsys) -> None:
    assert cli.main(["completion", "zsh"]) == 0
    rendered = capsys.readouterr().out
    assert rendered.startswith("#compdef aeloon aeloon-core")
    assert "resume history local login" in rendered


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
async def test_local_add_reads_api_key_privately_and_uses_unified_bridge(
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
                "local",
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
    output = capsys.readouterr().out
    assert "private-local-key" not in output
    assert "aeloon models use ollama/qwen3-coder" in output


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
            ["login", "alice", "--socket", str(socket_path), "--output", "json"]
        )
        == 0
    )

    assert calls["request"] == (
        socket_path,
        "cloud.account.login",
        {"username": "alice", "password": "secret"},
        60.0,
    )
    assert "required_methods" not in calls["ensure"]
    assert "secret" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_models_use_sets_default_through_bridge(tmp_path: Path, monkeypatch, capsys) -> None:
    socket_path = tmp_path / "bridge.sock"
    requests: list[tuple[str, object]] = []

    async def fake_ensure_daemon(**_kwargs):
        return {"socket_path": str(socket_path), "status": "running"}

    async def fake_bridge_request(_path, method, params=None, **_kwargs):
        requests.append((method, params))
        if method == "catalog.get":
            return {"models": [{"id": "studio/coder", "provider_id": "studio"}]}
        if method == "settings.get":
            return {"revision": 7}
        assert method == "settings.update"
        return {"revision": 8, "default_model_id": "studio/coder"}

    monkeypatch.setattr(cli, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(cli, "bridge_request", fake_bridge_request)

    assert (
        await cli.async_main(
            ["models", "use", "coder", "--socket", str(socket_path), "--json"]
        )
        == 0
    )
    assert requests[-1] == (
        "settings.update",
        {"revision": 7, "patch": {"default_model_id": "studio/coder"}},
    )
    assert json.loads(capsys.readouterr().out)["default_model_id"] == "studio/coder"
