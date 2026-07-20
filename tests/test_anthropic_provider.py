from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import AsyncAnthropic

from aeloon_core.config import AnthropicProviderConfig, load_config
from aeloon_core.providers.anthropic_provider import AnthropicProvider
from aeloon_core.providers.base import GenerationSettings, LLMResponse


def _value(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


class _Stream:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event

    async def aclose(self) -> None:
        self.closed = True


def test_builds_anthropic_messages_request_for_kimi() -> None:
    provider = AnthropicProvider(
        api_key="sk-test",
        base_url="https://api.kimi.com/coding/",
        default_model="k3[1m]",
    )
    kwargs = provider._build_kwargs(
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Read the file."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read",
                        "input": {"path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "contents",
                    }
                ],
            },
        ],
        tools=[
            {
                "name": "read",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )

    assert str(provider._client.base_url) == "https://api.kimi.com/coding/"
    assert kwargs["model"] == "k3"
    assert kwargs["system"] == [{"type": "text", "text": "You are a coding agent."}]
    assert [message["role"] for message in kwargs["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert kwargs["messages"][-1]["content"][0]["type"] == "tool_result"
    assert kwargs["tools"][0]["input_schema"]["required"] == ["path"]
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert "response_format" not in kwargs


def test_prompt_caching_can_be_disabled() -> None:
    assert AnthropicProviderConfig().prompt_caching is True
    assert AnthropicProviderConfig(prompt_caching=False).prompt_caching is False

    provider = AnthropicProvider(api_key="sk-test", prompt_caching=False)
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "Do not cache this request."}]
    )

    assert "cache_control" not in kwargs


def test_prompt_caching_preserves_legacy_positional_constructor_order() -> None:
    generation = GenerationSettings(temperature=0.2, reasoning_effort="high")

    provider = AnthropicProvider(
        "sk-test",
        "https://api.anthropic.com",
        "test-model",
        {},
        None,
        generation,
        17,
    )

    assert provider.generation is generation
    assert provider.chat_timeout == 17
    assert provider.prompt_caching is True


def test_parses_anthropic_tool_use_and_usage() -> None:
    provider = AnthropicProvider(api_key="sk-test")
    response = provider._parse(
        _value(
            content=[
                _value(type="text", text="Checking."),
                _value(
                    type="tool_use",
                    id="toolu_1",
                    name="read",
                    input={"path": "README.md"},
                ),
            ],
            stop_reason="tool_use",
            usage=_value(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=2,
                cache_read_input_tokens=3,
            ),
        )
    )

    assert response.content == "Checking."
    assert response.finish_reason == "tool_use"
    assert response.tool_calls[0].id == "toolu_1"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "prompt_tokens": 15,
        "completion_tokens": 5,
        "total_tokens": 20,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
    }


@pytest.mark.asyncio
async def test_retries_forced_tool_choice_as_auto_when_thinking_rejects_it() -> None:
    provider = AnthropicProvider(api_key="sk-test")
    attempts: list[dict[str, Any]] = []

    async def run(kwargs: dict[str, Any]) -> LLMResponse:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("tool_choice 'specified' is incompatible with thinking enabled")
        return LLMResponse(content="Recovered.")

    response = await provider._create_with_tool_fallback(
        {
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [{"name": "ping", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "ping"},
        },
        run,
    )

    assert response.content == "Recovered."
    assert len(attempts) == 2
    assert "tool_choice" in attempts[0]
    assert "tool_choice" not in attempts[1]


@pytest.mark.parametrize(
    "error_message",
    [
        "Error code: 400 - cache_control: extra inputs are not permitted",
        "Unrecognized request argument supplied: cache_control",
        "cache_control [type=extra_forbidden]",
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_prompt_cache_rejection_falls_back_once_and_is_remembered(
    streaming: bool,
    error_message: str,
) -> None:
    provider = AnthropicProvider(api_key="sk-test")
    attempts: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError(error_message)
        return _value(
            content=[_value(type="text", text="Recovered without caching.")],
            stop_reason="end_turn",
            usage=_value(input_tokens=4, output_tokens=2),
        )

    provider._client = _value(messages=_value(create=create))
    call = provider.chat_stream if streaming else provider.chat

    response = await call(messages=[{"role": "user", "content": "Finish."}])

    assert response.content == "Recovered without caching."
    assert len(attempts) == 2
    assert attempts[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in attempts[1]
    assert attempts[1].get("stream", False) is streaming
    assert "cache_control" not in provider._build_kwargs(
        messages=[{"role": "user", "content": "A later request."}]
    )


@pytest.mark.asyncio
async def test_unrelated_provider_error_does_not_disable_or_retry_prompt_caching() -> None:
    provider = AnthropicProvider(api_key="sk-test")
    attempts: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        attempts.append(kwargs)
        raise RuntimeError("Error code: 400 - max_tokens must be greater than zero")

    provider._client = _value(messages=_value(create=create))

    response = await provider.chat(messages=[{"role": "user", "content": "Finish."}])

    assert response.finish_reason == "error"
    assert "max_tokens must be greater than zero" in str(response.content)
    assert len(attempts) == 1
    assert provider._build_kwargs(
        messages=[{"role": "user", "content": "A later request."}]
    )["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_sdk_posts_anthropic_wire_format_to_kimi_messages_endpoint() -> None:
    captured: dict[str, Any] = {}

    async def handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"request-id": "req_1"},
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "k3[1m]",
                "content": [{"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = AnthropicProvider(
        api_key="sk-kimi",
        base_url="https://api.kimi.com/coding/",
        default_model="k3[1m]",
    )
    provider._client = AsyncAnthropic(
        api_key="sk-kimi",
        base_url="https://api.kimi.com/coding/",
        http_client=http_client,
    )

    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Finish."},
            ],
            tools=[
                {
                    "name": "complete_work",
                    "description": "Finish the task.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
    finally:
        await provider._client.close()

    assert response.content == "Done."
    assert response.finish_reason == "end_turn"
    assert captured["url"] == "https://api.kimi.com/coding/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-kimi"
    assert captured["body"]["system"] == [{"type": "text", "text": "Be concise."}]
    assert captured["body"]["messages"] == [{"role": "user", "content": "Finish."}]
    assert captured["body"]["tools"][0]["input_schema"]["type"] == "object"
    assert captured["body"]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_collects_anthropic_stream_content_thinking_and_tool_input() -> None:
    provider = AnthropicProvider(api_key="sk-test")
    stream = _Stream(
        [
            _value(
                type="message_start",
                message=_value(usage=_value(input_tokens=10, output_tokens=0)),
            ),
            _value(
                type="content_block_start",
                index=0,
                content_block=_value(
                    type="thinking",
                    thinking="",
                    signature="",
                    model_dump=lambda **_: {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                ),
            ),
            _value(
                type="content_block_delta",
                index=0,
                delta=_value(type="thinking_delta", thinking="Plan first."),
            ),
            _value(
                type="content_block_delta",
                index=0,
                delta=_value(type="signature_delta", signature="signed"),
            ),
            _value(type="content_block_start", index=1, content_block=_value(type="text")),
            _value(
                type="content_block_delta",
                index=1,
                delta=_value(type="text_delta", text="Reading."),
            ),
            _value(
                type="content_block_start",
                index=2,
                content_block=_value(type="tool_use", id="toolu_1", name="read", input={}),
            ),
            _value(
                type="content_block_delta",
                index=2,
                delta=_value(type="input_json_delta", partial_json='{"path":'),
            ),
            _value(
                type="content_block_delta",
                index=2,
                delta=_value(type="input_json_delta", partial_json='"README.md"}'),
            ),
            _value(
                type="message_delta",
                delta=_value(stop_reason="tool_use"),
                usage=_value(output_tokens=5),
            ),
            _value(type="message_stop"),
        ]
    )
    text_deltas: list[str] = []
    thinking_deltas: list[str] = []

    async def on_text(delta: str) -> None:
        text_deltas.append(delta)

    async def on_thinking(delta: str) -> None:
        thinking_deltas.append(delta)

    response = await provider._collect_stream(
        stream,
        on_delta=on_text,
        on_reasoning_delta=on_thinking,
    )

    assert stream.closed is True
    assert text_deltas == ["Reading."]
    assert thinking_deltas == ["Plan first."]
    assert response.content == "Reading."
    assert response.reasoning_content == "Plan first."
    assert response.thinking_blocks == [
        {"type": "thinking", "thinking": "Plan first.", "signature": "signed"}
    ]
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.usage["total_tokens"] == 15


def test_loads_claude_and_kimi_environment_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-kimi")
    monkeypatch.setenv("ANTHROPIC_MODEL", "k3[1m]")
    monkeypatch.setenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "1048576")
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "1048576")

    config = load_config(tmp_path / "missing.json")

    assert config.providers.anthropic.base_url == "https://api.kimi.com/coding/"
    assert config.providers.anthropic.api_key == "sk-kimi"
    assert config.agents.defaults.model == "k3[1m]"
    assert config.agents.defaults.context_window_tokens == 1_048_576
    assert config.agents.defaults.context_compaction.trigger_ratio == 1.0


def test_upgrades_persisted_provider_configuration(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "custom": {
                        "api_key": "sk-old",
                        "api_base": "https://example.test/anthropic/",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.providers.anthropic.api_key == "sk-old"
    assert config.providers.anthropic.base_url == "https://example.test/anthropic/"
