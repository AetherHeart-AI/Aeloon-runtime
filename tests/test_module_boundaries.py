"""Architecture-level compatibility tests for split state and persistence modules."""

from aeloon_core import flow_state, flows, worker_sessions, worker_state


def test_flow_persistence_module_reexports_domain_contracts() -> None:
    assert flows.FlowNodeSpec is flow_state.FlowNodeSpec
    assert flows.MasterFlow is flow_state.MasterFlow
    assert flows.FlowId is flow_state.FlowId
    assert flows.revise_flow_node is flow_state.revise_flow_node


def test_worker_persistence_module_reexports_domain_contracts() -> None:
    assert worker_sessions.ContextEnvelope is worker_state.ContextEnvelope
    assert worker_sessions.ReportText is worker_state.ReportText
    assert worker_sessions.WorkerRunRecord is worker_state.WorkerRunRecord
    assert worker_sessions.WorkerRunStatus is worker_state.WorkerRunStatus
