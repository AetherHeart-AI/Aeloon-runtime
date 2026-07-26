from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import SlidingWindow
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

from aeloon_core.config import Config
from aeloon_core.harness_runtime import (
    _workflow_name,
    history_capability,
    master_harness_capabilities,
)
from aeloon_core.model_router import ModelRouter
from aeloon_core.pydantic_runtime import (
    AgentRunSpec,
    CapabilityManifest,
    HarnessAgentRuntime,
)
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.workers import WorkerRegistry


def test_workflow_names_are_valid_python_identifiers() -> None:
    assert _workflow_name("code-reviewer") == "code_reviewer"
    assert _workflow_name("class") == "worker_class"


def test_history_policy_is_owned_by_harness_sliding_window(tmp_path: Path) -> None:
    config = Config(
        workspace=tmp_path,
        agents={
            "defaults": {
                "context_window_tokens": 100_000,
                "context_compaction": {
                    "trigger_ratio": 0.8,
                    "preserve_recent_tokens": 20_000,
                },
            }
        },
    )

    capability = history_capability(config)

    assert isinstance(capability, SlidingWindow)
    assert capability.max_tokens == 80_000
    assert capability.keep_tokens == 20_000


def test_master_always_exposes_ephemeral_worker_catalog(tmp_path: Path) -> None:
    config = Config(
        workspace=tmp_path,
        agents={"harness": {"sub_agent_request_limit": 7}},
    ).normalized()
    registry = WorkerRegistry.discover(tmp_path)
    model = TestModel()

    capabilities = master_harness_capabilities(
        config=config,
        model_router=ModelRouter(config, injected_model=model),
        worker_types=registry,
    )

    assert isinstance(capabilities[0], SlidingWindow)
    workflow = capabilities[1]
    assert isinstance(workflow, DynamicWorkflow)
    assert workflow.max_agent_calls == config.agents.harness.max_agent_calls
    assert workflow.sub_agent_usage_limits.request_limit == 7
    assert workflow.resource_limits == {
        "max_duration_secs": config.agents.harness.workflow_cpu_seconds,
    }
    assert [entry.name for entry in workflow.agents] == [
        snapshot.id.replace("-", "_") for snapshot in registry.list()
    ]
    assert all(entry.agent.model is model for entry in workflow.agents)


@pytest.mark.asyncio
async def test_master_executes_workflow_and_finishes_with_plain_text(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], list[str]]] = []
    lifecycles: list[dict[str, Any]] = []

    class Progress:
        async def on_worker_lifecycle(self, **payload: Any) -> None:
            lifecycles.append(payload)

    async def function(messages: list[ModelMessage], info: Any) -> ModelResponse:
        request = info.model_request_parameters
        function_tools = [tool.name for tool in request.function_tools]
        output_tools = [tool.name for tool in request.output_tools]
        calls.append((function_tools, output_tools))
        if "run_workflow" in function_tools:
            workflow_returned = any(
                isinstance(message, ModelRequest)
                and any(
                    isinstance(part, ToolReturnPart) and part.tool_name == "run_workflow"
                    for part in message.parts
                )
                for message in messages
            )
            if workflow_returned:
                return ModelResponse(parts=[TextPart("workflow complete")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow",
                        {
                            "code": (
                                'result = await builder(task="Inspect the workspace '
                                'and report success")\nresult'
                            )
                        },
                        "workflow",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tools[0],
                    {
                        "summary": "builder finished",
                        "artifacts": [],
                        "evidence": [],
                        "unresolved": [],
                    },
                    "worker-output",
                )
            ]
        )

    config = Config(workspace=tmp_path).normalized()
    model = FunctionModel(function=function)
    router = ModelRouter(config, injected_model=model)
    registry = WorkerRegistry.discover(tmp_path)
    tools = ToolRegistry()
    outcome = await HarnessAgentRuntime().run(
        AgentRunSpec(
            role="master",
            model=model,
            instructions="Use the workflow.",
            prompt="Do it.",
            history=[],
            tools=tools,
            output_type=str,
            terminal_models={},
            capability_manifest=CapabilityManifest.from_registry(
                tools,
                namespace="master",
            ),
            capabilities=master_harness_capabilities(
                config=config,
                model_router=router,
                worker_types=registry,
            ),
            progress=Progress(),
            request_limit=4,
        )
    )

    assert outcome.output == "workflow complete"
    assert outcome.tools_used == ["run_workflow"]
    assert len(calls) == 3
    assert "run_workflow" in calls[0][0]
    assert "read_file" in calls[1][0]
    assert calls[1][1] == ["final_result"]
    assert [event["event"] for event in lifecycles] == [
        "started",
        "completed",
    ]
    assert lifecycles[0]["objective"] == "Inspect the workspace and report success"
    assert lifecycles[1]["summary"] == "builder finished"
