from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

import aeloon_core.__main__ as cli
from aeloon_core.core import AssistantMessage, Model, TextContent, Usage
from aeloon_core.runtime.providers import ProviderManager as RuntimeProviderManager
from aeloon_core.runtime.providers.testing import ScriptedProvider


def _provider(text: str = "done", *, models=()) -> ScriptedProvider:
    return ScriptedProvider(
        [
            AssistantMessage(
                (TextContent(text),),
                provider="deepseek",
                model="deepseek-v4-flash",
                usage=Usage(input=3, output=1, total_tokens=4),
            )
        ],
        models=models,
    )


def _providers(configured=None, **updates):
    return {**cli.Config().providers, **(configured or {}), **updates}


def _use_scripted_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: ScriptedProvider | dict[str, ScriptedProvider],
) -> None:
    providers = provider if isinstance(provider, dict) else None

    def create_manager(config, *, account=None, driver_factories=None):
        factories = dict(driver_factories or {})
        factories["custom"] = lambda provider_id, *_args: (
            providers[provider_id] if providers is not None else provider
        )
        return RuntimeProviderManager(
            config,
            account=account,
            driver_factories=factories,
        )

    monkeypatch.setattr(cli, "ProviderManager", create_manager)


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
        cli.RunEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-1",
                "toolName": "bash",
                "args": {"command": "printf 'hello\\n'"},
            },
        )
    )
    await renderer(
        cli.RunEvent(
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

    await renderer(cli.RunEvent("agent_start"))
    await renderer(
        cli.RunEvent(
            "tool_execution_start",
            {"toolCallId": "call-1", "toolName": "read", "args": {"path": "private.txt"}},
        )
    )
    await renderer(cli.RunEvent("agent_end"))

    rendered = capsys.readouterr().err
    assert "Working…" in rendered
    assert "Reading files…" in rendered
    assert "private.txt" not in rendered
    assert "\x1b[2K" in rendered


@pytest.mark.asyncio
async def test_text_renderer_summarizes_tool_command_and_result(capsys) -> None:
    renderer = cli.RunRenderer("text", verbose=True)
    await renderer(
        cli.RunEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-1",
                "toolName": "bash",
                "args": {"command": "printf 'hello\\n'"},
            },
        )
    )
    await renderer(
        cli.RunEvent(
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
        cli.RunEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-read",
                "toolName": "read",
                "args": {"path": "large.md", "offset": 11, "limit": 30},
            },
        )
    )
    await renderer(
        cli.RunEvent(
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
        cli.RunEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-failed",
                "toolName": "bash",
                "args": {"command": "false"},
            },
        )
    )
    await renderer(
        cli.RunEvent(
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
        cli.RunEvent(
            "tool_execution_start",
            {
                "toolCallId": "call-2",
                "toolName": "write",
                "args": {"path": "result.txt", "content": "do not echo this content"},
            },
        )
    )
    await renderer(
        cli.RunEvent(
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
async def test_task_history_and_ephemeral_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("saved"))
    await cli.async_main(
        [
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

    await cli.async_main(["history", "--all", "--data-dir", str(data_dir), "--json"])
    listed = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in listed] == [session_id]

    await cli.async_main(["history", session_id, "--data-dir", str(data_dir), "--json"])
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == session_id
    assert [message["role"] for message in shown["messages"]] == ["user", "assistant"]

    before = set((data_dir / "harness-sessions").glob("*.jsonl"))
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda **_kwargs: _provider("ephemeral"))
    await cli.async_main(
        [
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
            providers=_providers(deepseek={"driver": "deepseek", "api_key": "configured-key"}),
        ),
        config_path,
    )
    account_data_dirs: list[Path] = []
    account_type = cli.CloudAccountGateway

    def cloud_account(config, *, data_dir):
        account_data_dirs.append(data_dir)
        return account_type(config, data_dir=data_dir)

    monkeypatch.setattr(cli, "CloudAccountGateway", cloud_account)
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
    assert shown["providers"]["deepseek"]["api_key"] is None
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["providers"]["deepseek"]["api_key"] is None


def test_fresh_config_ignores_deepseek_environment_and_has_no_default_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-read")

    config = cli.load_config(tmp_path / "missing.json")

    assert config.providers["deepseek"].api_key is None
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
    assert "aeloon-core provider add" in error
    assert "first available model automatically" in error


def test_top_level_help_focuses_on_user_tasks() -> None:
    help_text = cli.build_parser().format_help()

    assert 'aeloon-core "fix the failing tests"' in help_text
    assert "resume" in help_text
    assert "history" in help_text
    assert "doctor" in help_text
    assert "rpc    " not in help_text
    assert "session   " not in help_text
    assert "==SUPPRESS==" not in help_text


def test_removed_local_and_provider_login_commands_are_absent() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["local", "add", "studio"])
    with pytest.raises(SystemExit):
        parser.parse_args(["provider", "local", "add", "studio"])
    with pytest.raises(SystemExit):
        parser.parse_args(["provider", "login"])


def test_rpc_serve_has_no_browser_runtime_interface(tmp_path: Path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["rpc", "serve", "--socket", str(tmp_path / "core.sock")])

    assert args.rpc_command == "serve"
    assert not hasattr(args, "browser_runtime_socket")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "rpc",
                "serve",
                "--socket",
                str(tmp_path / "core.sock"),
                "--browser-runtime-socket",
                str(tmp_path / "browser.sock"),
            ]
        )


def test_default_task_command_and_explicit_separator_are_normalized() -> None:
    task = cli._TASK_COMMAND
    assert cli._normalize_argv(["fix", "the", "tests"]) == [task, "fix", "the", "tests"]
    assert cli._normalize_argv(["--json", "inspect"]) == [task, "--json", "inspect"]
    assert cli._normalize_argv(["--", "models"]) == [task, "models"]
    assert cli._normalize_argv(["resume", "continue"]) == ["resume", "continue"]


def test_commands_use_actionable_errors(tmp_path: Path, capsys) -> None:
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
    assert "aeloon-core history" in human

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

    assert await cli.async_main(["history", "-C", str(tmp_path), "--data-dir", str(data_dir)]) == 0
    human = capsys.readouterr().out
    assert "UPDATED\tWORKSPACE\tSUMMARY\tSESSION" in human
    assert "remember this task" in human

    await cli.async_main(["history", "-C", str(tmp_path), "--data-dir", str(data_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["summary"] == "remember this task"

    await cli.async_main(["history", payload[0]["id"][:8], "--data-dir", str(data_dir)])
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
    assert "aeloon-core provider add" in model["fix"]


@pytest.mark.asyncio
async def test_first_listed_model_is_automatic_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    cli.save_config(
        cli.Config(
            workspace=tmp_path,
            data_dir=tmp_path / "data",
            providers=_providers(
                {
                    "studio": {
                        "driver": "custom",
                        "backend": "openai",
                        "name": "Studio",
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "models": [{"id": "first"}, {"id": "second"}],
                    }
                }
            ),
        ),
        config_path,
    )
    provider = _provider(
        "automatic",
        models=(
            Model("studio/first", "first", "studio"),
            Model("studio/second", "second", "studio"),
        ),
    )
    _use_scripted_openai_provider(monkeypatch, provider)

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
            providers=_providers(
                {
                    "studio": {
                        "driver": "custom",
                        "backend": "openai",
                        "name": "Studio",
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "models": [{"id": "shared-model"}],
                    },
                    "backup": {
                        "driver": "custom",
                        "backend": "openai",
                        "name": "Backup",
                        "endpoint": "http://127.0.0.1:9000/v1",
                        "models": [{"id": "shared-model"}],
                    },
                }
            ),
        ),
        config_path,
    )
    provider = _provider(
        "matched",
        models=(Model("studio/shared-model", "shared-model", "studio"),),
    )
    backup = ScriptedProvider(
        (),
        models=(Model("backup/shared-model", "shared-model", "backup"),),
    )
    _use_scripted_openai_provider(
        monkeypatch,
        {"studio": provider, "backup": backup},
    )

    assert (
        await cli.async_main(
            [
                "matched task",
                "--config",
                str(config_path),
                "--model",
                "shared-model",
                "--ephemeral",
                "--json",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["final_content"] == "matched"
    assert provider.requests[0][0].id == "studio/shared-model"


def test_completion_command_emits_shell_script(capsys) -> None:
    assert cli.main(["completion", "zsh"]) == 0
    rendered = capsys.readouterr().out
    assert rendered.startswith("#compdef aeloon-core")
    assert "resume history login logout" in rendered


def test_package_exposes_only_the_aeloon_core_command() -> None:
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["scripts"] == {
        "aeloon-core": "aeloon_core.__main__:main",
    }
