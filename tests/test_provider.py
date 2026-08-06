from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from aeloon_core.core import (
    ImageContent,
    InferenceContext,
    InferenceError,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    UserMessage,
)
from aeloon_core.core.inference_runtime import collect_assistant
from aeloon_core.runtime.providers import (
    DEEPSEEK_V4_FLASH,
    DeepSeekProvider,
    OpenAICompatibleProvider,
)


def _sse(*chunks: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


@pytest.mark.asyncio
async def test_deepseek_stream_parses_reasoning_tool_fragments_usage_and_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                {"choices": [{"delta": {"reasoning_content": "think "}}]},
                {"choices": [{"delta": {"content": "hello"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "read", "arguments": '{"pa'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": 'th":"x"}'}}]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "prompt_cache_hit_tokens": 3,
                        "completion_tokens": 7,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 18,
                    },
                },
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DeepSeekProvider(api_key="sk-test", client=client)
    events = [
        event
        async for event in provider.stream(
            DEEPSEEK_V4_FLASH,
            InferenceContext("system", (UserMessage("go"),), (), "session"),
            StreamOptions(thinking_level="high", max_tokens=123),
        )
    ]
    await client.aclose()

    final = events[-1].message
    assert [event.type for event in events] == [
        "start",
        "thinking_delta",
        "text_delta",
        "toolcall_delta",
        "toolcall_delta",
        "done",
    ]
    assert final is not None
    assert isinstance(final.content[0], ThinkingContent)
    assert final.text == "hello"
    assert final.tool_calls == (ToolCall("call_1", "read", {"path": "x"}),)
    assert final.stop_reason == "toolUse"
    assert final.usage.total_tokens == 18
    assert (final.usage.input, final.usage.cache_read, final.usage.reasoning) == (8, 3, 2)
    assert final.usage.cost["total"] > 0
    payload = json.loads(requests[0].content)
    assert payload["max_tokens"] == 123
    assert payload["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_deepseek_retries_retryable_http_and_honors_retry_after() -> None:
    attempts = 0
    retry_events: list[dict[str, object]] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DeepSeekProvider(api_key="sk-test", client=client)
    message = await collect_assistant(
        provider,
        DEEPSEEK_V4_FLASH,
        InferenceContext("", (UserMessage("go"),), (), "session"),
        StreamOptions(
            max_retries=1,
            metadata={"on_retry": lambda event: retry_events.append(event)},
        ),
    )
    await client.aclose()

    assert attempts == 2
    assert message.text == "ok"
    assert [event["stage"] for event in retry_events] == ["start", "end"]
    assert retry_events[0]["delayMs"] == 0


@pytest.mark.asyncio
async def test_deepseek_reports_auth_and_nonretryable_http_errors() -> None:
    provider = DeepSeekProvider(api_key=None)
    with pytest.raises(InferenceError, match="API key is required"):
        await collect_assistant(
            provider,
            DEEPSEEK_V4_FLASH,
            InferenceContext("", (), (), "session"),
            StreamOptions(),
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(400, text="bad input sk-test")
        )
    )
    provider = DeepSeekProvider(api_key="sk-test", client=client)
    with pytest.raises(InferenceError, match="HTTP 400") as error:
        await collect_assistant(
            provider,
            DEEPSEEK_V4_FLASH,
            InferenceContext("", (), (), "session"),
            StreamOptions(),
        )
    assert "sk-test" not in str(error.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_stream_is_cancellable() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, content=_sse())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DeepSeekProvider(api_key="sk-test", client=client)
    task = asyncio.create_task(
        collect_assistant(
            provider,
            DEEPSEEK_V4_FLASH,
            InferenceContext("", (), (), "session"),
            StreamOptions(),
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_serializes_supported_image_input() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse({"choices": [{"delta": {"content": "seen"}, "finish_reason": "stop"}]}),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = Model(
        "studio/vision",
        "Vision",
        "studio",
        input=("text", "image"),
    )
    provider = OpenAICompatibleProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        models=(model,),
        client=client,
    )

    message = await collect_assistant(
        provider,
        model,
        InferenceContext(
            "",
            (UserMessage((TextContent("look"), ImageContent("YWJj", "image/png"))),),
            (),
            "session",
        ),
        StreamOptions(max_retries=0),
    )
    await client.aclose()

    assert message.text == "seen"
    content = json.loads(requests[0].content)["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"data: not-json\n\n", "invalid SSE JSON"),
        (b"data: [DONE]\n\n", "without any chunks"),
    ],
)
async def test_openai_compatible_rejects_abnormal_sse(content: bytes, message: str) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            )
        )
    )
    model = Model("studio/model", "Model", "studio")
    provider = OpenAICompatibleProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        client=client,
    )

    with pytest.raises(InferenceError, match=message):
        await collect_assistant(
            provider,
            model,
            InferenceContext("", (), (), "session"),
            StreamOptions(max_retries=0),
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_applies_timeout_to_request() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"]["read"] == pytest.approx(0.025)
        raise httpx.ReadTimeout("too slow", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    model = Model("studio/model", "Model", "studio")
    provider = OpenAICompatibleProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        client=client,
    )

    with pytest.raises(InferenceError, match="too slow"):
        await collect_assistant(
            provider,
            model,
            InferenceContext("", (), (), "session"),
            StreamOptions(timeout_ms=25, max_retries=0),
        )
    await client.aclose()
