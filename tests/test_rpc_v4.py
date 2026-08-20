from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from aeloon_runtime.rpc.protocol import PROTOCOL_NAME, PROTOCOL_VERSION, RPC_CODES


ROOT = Path(__file__).resolve().parents[1]


def test_rpc4_manifest_is_strict_and_current() -> None:
    path = ROOT / "aeloon_runtime" / "rpc" / "aeloon-rpc-v4.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["protocol"] == PROTOCOL_NAME == "aeloon-rpc"
    assert manifest["protocol_version"] == PROTOCOL_VERSION == "4.0.0"
    assert manifest["transport"]["max_frame_bytes"] == 40 * 1024 * 1024
    assert set(RPC_CODES).issubset(manifest["errors"])
    for definition in manifest["$defs"].values():
        Draft202012Validator.check_schema(definition)


def test_rpc4_runtime_manifest_has_workspace_and_claim_methods() -> None:
    path = ROOT / "aeloon_runtime" / "rpc" / "aeloon-rpc-v4.manifest.json"
    methods = json.loads(path.read_text(encoding="utf-8"))["methods"]
    assert {"devices.claim", "workspace.roots", "workspace.list"}.issubset(methods)
