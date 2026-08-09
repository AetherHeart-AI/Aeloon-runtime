from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from aeloon_core.core import InferenceError
from aeloon_core.runtime.providers import CustomProvider


@pytest.mark.asyncio
async def test_custom_discovery_reads_llamacpp_meta_allow_image_without_probe() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "model": "vision.gguf",
                        "capabilities": ["completion", "multimodal"],
                    }
                ],
                "object": "list",
                "data": [
                    {"id": "vision.gguf", "meta": {"allow_image": True}},
                    {"id": "text.gguf", "meta": {"allow_image": False}},
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        client=client,
    )

    models = await provider.discover_models()
    await provider.close()
    await client.aclose()

    by_id = {model.id: model for model in models}
    assert by_id["studio/vision.gguf"].input == ("text", "image")
    assert by_id["studio/text.gguf"].input == ("text",)
    assert [request.method for request in requests] == ["GET"]


@pytest.mark.asyncio
async def test_custom_discovery_reads_common_metadata_and_probes_only_unknown_models() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer local-secret"
        assert request.headers["x-trace"] == "trace-id"
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "models": [
                        "plain-string",
                        {"id": "direct-image", "supportsImage": True},
                        {"id": "direct-text", "supports_image": False},
                        {
                            "model": "architecture-image",
                            "displayName": "Architecture Image",
                            "architecture": {"input_modalities": ["text", "image"]},
                        },
                        {
                            "id": "mapped-modalities",
                            "modalities": {"input": ["text", "image"], "output": ["text"]},
                        },
                        {"model_key": "caps-image", "capabilities": ["tools", "vision"]},
                        {"id": "caps-text", "capabilities": {"tools": True}},
                        {
                            "id": "unknown-image",
                            "contextWindow": 64_000,
                            "maxTokens": 128_000,
                        },
                        {"id": "unknown-text"},
                        {"id": "unknown-image", "supports_image": False},
                        {},
                    ]
                },
            )
        payload = json.loads(request.content)
        assert payload["model"] in {"plain-string", "unknown-image", "unknown-text"}
        assert payload["max_tokens"] == 1
        assert payload["stream"] is False
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        if payload["model"] in {"plain-string", "unknown-image"}:
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return httpx.Response(400, json={"error": {"message": "images are unsupported"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        api_key="local-secret",
        headers={"X-Trace": "trace-id"},
        client=client,
    )

    models = await provider.discover_models()
    await provider.close()
    await client.aclose()

    by_id = {model.id: model for model in models}
    assert list(by_id) == [
        "studio/plain-string",
        "studio/direct-image",
        "studio/direct-text",
        "studio/architecture-image",
        "studio/mapped-modalities",
        "studio/caps-image",
        "studio/caps-text",
        "studio/unknown-image",
        "studio/unknown-text",
    ]
    assert by_id["studio/architecture-image"].name == "Architecture Image"
    assert by_id["studio/unknown-image"].context_window == 64_000
    assert by_id["studio/unknown-image"].max_tokens == 64_000
    assert {
        model.id for model in models if "image" in model.input
    } == {
        "studio/plain-string",
        "studio/direct-image",
        "studio/architecture-image",
        "studio/mapped-modalities",
        "studio/caps-image",
        "studio/unknown-image",
    }
    probe_models = [
        json.loads(request.content)["model"] for request in requests if request.method == "POST"
    ]
    assert probe_models == ["plain-string", "unknown-image", "unknown-text"]


@pytest.mark.asyncio
async def test_custom_discovery_limits_probe_concurrency_to_four() -> None:
    active = 0
    maximum = 0
    probe_counts: dict[str, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [{"id": f"model-{index}"} for index in range(9)]},
            )
        model_id = str(json.loads(request.content)["model"])
        probe_counts[model_id] = probe_counts.get(model_id, 0) + 1
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={"choices": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        client=client,
    )

    models = await provider.discover_models()
    await client.aclose()

    assert len(models) == 9
    assert maximum == 4
    assert probe_counts == {f"model-{index}": 1 for index in range(9)}


@pytest.mark.asyncio
async def test_custom_probe_errors_do_not_fail_model_discovery() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "unknown"}]})
        raise httpx.ReadTimeout("probe timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example/v1",
        client=client,
    )

    models = await provider.discover_models()
    await client.aclose()

    assert models[0].input == ("text",)


@pytest.mark.asyncio
async def test_custom_discovery_failure_sanitizes_secrets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("could not connect with local-secret", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="studio",
        name="Studio",
        endpoint="https://studio.example",
        api_key="local-secret",
        client=client,
    )

    with pytest.raises(InferenceError, match="Could not load models") as captured:
        await provider.discover_models()
    await client.aclose()

    assert "local-secret" not in str(captured.value)
    assert "***" in str(captured.value)
    assert "local-secret" not in str(captured.value.cause)


@pytest.mark.asyncio
async def test_llamacpp_backend_reads_models_and_props_fields() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/models":
            return httpx.Response(200, json={"models": [{"model": "gemma.gguf"}]})
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": 32_768},
                    "modalities": ["text", "image"],
                    "chat_template_caps": {"tools": True},
                },
            )
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="llama",
        name="llama.cpp",
        endpoint="http://127.0.0.1:8080",
        backend="llamacpp",
        client=client,
    )

    models = await provider.discover_models()
    await client.aclose()

    assert requests == ["/models", "/props"]
    assert provider.endpoint == "http://127.0.0.1:8080/v1"
    assert models[0].id == "llama/gemma.gguf"
    assert models[0].context_window == 32_768
    assert models[0].input == ("text", "image")


@pytest.mark.asyncio
async def test_ollama_backend_reads_native_tags_and_show_fields() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma3:latest"}]})
        if request.url.path == "/api/show":
            assert json.loads(request.content)["model"] == "gemma3:latest"
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "vision", "thinking"],
                    "model_info": {"gemma3.context_length": 131_072},
                },
            )
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="desktop",
        name="Ollama",
        endpoint="http://127.0.0.1:11434",
        backend="ollama",
        client=client,
    )

    models = await provider.discover_models()
    await client.aclose()

    assert requests == ["/api/tags", "/api/show"]
    assert provider.endpoint == "http://127.0.0.1:11434/v1"
    assert models[0].id == "desktop/gemma3:latest"
    assert models[0].context_window == 131_072
    assert models[0].reasoning is True
    assert models[0].input == ("text", "image")


@pytest.mark.asyncio
async def test_vllm_backend_reads_model_card_max_model_len() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "served-model",
                            "owned_by": "vllm",
                            "root": "org/model",
                            "parent": None,
                            "max_model_len": 65_536,
                            "supports_image": False,
                        }
                    ]
                },
            )
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CustomProvider(
        provider_id="gpu",
        name="vLLM",
        endpoint="http://127.0.0.1:8000/v1",
        backend="vllm",
        client=client,
    )

    models = await provider.discover_models()
    await client.aclose()

    assert models[0].id == "gpu/served-model"
    assert models[0].context_window == 65_536
    assert models[0].input == ("text",)
