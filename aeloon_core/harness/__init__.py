"""Host-owned Harness execution layer.

Project customization belongs in :mod:`aeloon_core.customization`; this package
owns agent construction, budgets, lifecycle integration, and workflow execution.
"""

from aeloon_core.harness.agents import (
    RoleAgentFactory,
    history_capability,
    master_harness_capabilities,
)
from aeloon_core.harness.routing import ModelBinding, ModelRouter
from aeloon_core.harness.runner import WorkflowExecutionResult, WorkflowRunner
from aeloon_core.harness.runtime import (
    AeloonRunDeps,
    AgentRunOutcome,
    AgentRunSpec,
    AgentRunStatus,
    CapabilityManifest,
    HarnessAgentRuntime,
    deserialize_messages,
    serialize_messages,
)
from aeloon_core.harness.workflow_tools import (
    WorkflowDescribeTool,
    WorkflowExecuteTool,
    WorkflowSearchTool,
    workflow_tools,
)

__all__ = [
    "AeloonRunDeps",
    "AgentRunOutcome",
    "AgentRunSpec",
    "AgentRunStatus",
    "CapabilityManifest",
    "HarnessAgentRuntime",
    "ModelBinding",
    "ModelRouter",
    "RoleAgentFactory",
    "WorkflowDescribeTool",
    "WorkflowExecutionResult",
    "WorkflowExecuteTool",
    "WorkflowRunner",
    "WorkflowSearchTool",
    "history_capability",
    "deserialize_messages",
    "master_harness_capabilities",
    "serialize_messages",
    "workflow_tools",
]
