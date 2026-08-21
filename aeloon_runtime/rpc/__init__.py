"""Aeloon Runtime's private desktop RPC boundary."""

from aeloon_runtime.rpc.protocol import PROTOCOL_NAME, PROTOCOL_VERSION, RpcError
from aeloon_runtime.runtime_adapter import AeloonRpcAdapter

__all__ = ["AeloonRpcAdapter", "PROTOCOL_NAME", "PROTOCOL_VERSION", "RpcError"]
