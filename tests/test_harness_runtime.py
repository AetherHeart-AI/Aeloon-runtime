from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import SlidingWindow
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

from aeloon_core.config import Config
from aeloon_core.harness_runtime import (
    history_capability,
    master_harness_capabilities,
    worker_harness_capabilities,
)
from aeloon_core.master_flow_tools import FinishTurnArgs
from aeloon_core.model_router import ModelRouter
from aeloon_core.pydantic_runtime import (
    AgentRunSpec,
    CapabilityManifest,
    HarnessAgentRuntime,
    output_tools,
)
from aeloon_core.tools.registry import ToolRegistry
from aeloon_core.workers import WorkerRegistry


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
    assert worker_harness_capabilities(config) == [capability]


def test_master_harness_exposes_worker_catalog_to_dynamic_workflow(
    tmp_path: Path,
) -> None:
    config = Config(workspace=tmp_path).normalized()
    registry = WorkerRegistry.discover(tmp_path)
    model = TestModel()
    router = ModelRouter(config, injected_model=model)

    capabilities = master_harness_capabilities(
        config=config,
        model_router=router,
        worker_types=registry,
    )

    assert isinstance(capabilities[0], SlidingWindow)
    workflow = capabilities[1]
    assert isinstance(workflow, DynamicWorkflow)
    assert workflow.max_agent_calls == config.agents.harness.max_agent_calls
    assert workflow.resource_limits == {
        "max_duration_secs": config.agents.harness.workflow_cpu_seconds,
        "max_memory": config.agents.harness.workflow_memory_mb * 1024 * 1024,
    }
    assert [entry.name for entry in workflow.agents] == [
        snapshot.id.replace("-", "_") for snapshot in registry.list()
    ]
    assert all(entry.agent.model is model for entry in workflow.agents)


def test_dynamic_workflow_can_be_disabled_without_disabling_compaction(
    tmp_path: Path,
) -> None:
    config = Config(
        workspace=tmp_path,
        agents={"harness": {"dynamic_workflow_enabled": False}},
    ).normalized()

    capabilities = master_harness_capabilities(
        config=config,
        model_router=ModelRouter(config, injected_model=TestModel()),
        worker_types=WorkerRegistry.discover(tmp_path),
    )

    assert len(capabilities) == 1
    assert isinstance(capabilities[0], SlidingWindow)


@pytest.mark.asyncio
async def test_master_executes_dynamic_workflow_and_receives_worker_report(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], list[str]]] = []
    lifecycles: list[dict[str, Any]] = []

    class Progress:
        async def on_worker_lifecycle(self, **payload: Any) -> None:
            lifecycles.append(payload)

    async def function(
        messages: list[ModelMessage],
        info: Any,
    ) -> ModelResponse:
        request = info.model_request_parameters
        function_tools = [tool.name for tool in request.function_tools]
        output_tools = [tool.name for tool in request.output_tools]
        calls.append((function_tools, output_tools))
        if "run_workflow" in function_tools:
            workflow_returned = any(
                isinstance(message, ModelRequest)
                and any(
                    isinstance(part, ToolReturnPart)
                    and part.tool_name == "run_workflow"
                    for part in message.parts
                )
                for message in messages
            )
            if workflow_returned:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            output_tools[0],
                            {"final_content": "workflow complete"},
                            "finish",
                        )
                    ]
                )
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
            output_type=output_tools(
                (FinishTurnArgs, "finish_turn", "Finish the turn.")
            ),
            terminal_models={"finish_turn": FinishTurnArgs},
            capability_manifest=CapabilityManifest.from_registry(
                tools,
                namespace="master",
                terminal_names=("finish_turn",),
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

    assert outcome.output == FinishTurnArgs(final_content="workflow complete")
    assert outcome.tools_used == ["run_workflow"]
    assert len(calls) == 3
    assert "run_workflow" in calls[0][0]
    assert "read_file" in calls[1][0]
    assert calls[1][1] == ["final_result"]
    assert [event["event"] for event in lifecycles] == [
        "created",
        "started",
        "completed",
    ]
    assert lifecycles[1]["objective"] == "Inspect the workspace and report success"
    assert lifecycles[2]["summary"] == "builder finished"
    assert lifecycles[2]["ephemeral"] is True
