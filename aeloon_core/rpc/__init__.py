"""Aeloon Core's private desktop RPC boundary."""

from aeloon_core.rpc.adapter import AeloonRpcAdapter
from aeloon_core.rpc.protocol import PROTOCOL_NAME, PROTOCOL_VERSION, RpcError

__all__ = ["AeloonRpcAdapter", "PROTOCOL_NAME", "PROTOCOL_VERSION", "RpcError"]
