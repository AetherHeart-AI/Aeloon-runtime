from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from aeloon_runtime.rpc.protocol import RPC_CODES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "rpc-v3.json"
MANIFEST = ROOT / "aeloon_runtime" / "rpc" / "aeloon-rpc-v3.manifest.json"


def test_v3_source_and_generated_manifest_are_current() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(source)
    assert list(Draft202012Validator(source).iter_errors(source)) == []
    assert source["additionalProperties"] is False
    methods = [method for group in source["groups"] for method in group["methods"]]
    # 66 first-class methods at 3.0.0, plus devices.list/revoke/enroll at 3.1.0.
    assert len(methods) == 69
    assert manifest["protocol"] == "aeloon-rpc"
    assert manifest["protocol_version"] == source["version"]
    assert manifest["transport"]["max_frame_bytes"] == 40 * 1024 * 1024
    assert manifest["transport"]["file_bytes"] == 25 * 1024 * 1024
    assert set(manifest["methods"]) == {method["name"] for method in methods}
    assert all(
        isinstance(method["params"], dict)
        and isinstance(method["params"].get("$ref"), str)
        and isinstance(method["result"], dict)
        and isinstance(method["result"].get("$ref"), str)
        for method in methods
    )
    assert set(source["event_schemas"]) == set(manifest["events"])
    expected_errors = (set(RPC_CODES) - {"session_not_found"}) | {
        "thread_not_found",
        "unauthorized",
        "forbidden",
        "capability_unavailable",
        "payload_too_large",
    }
    assert set(manifest["errors"]) == expected_errors
    for definition in manifest["$defs"].values():
        Draft202012Validator.check_schema(definition)


def test_v3_method_names_are_two_segment_except_plugin_namespace() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    methods = [method["name"] for group in source["groups"] for method in group["methods"]]
    assert all(name.count(".") == 1 for name in methods)
    plugin_names = [item["name"] for item in source["plugin_contributed"]["methods"]]
    assert all(name.startswith("plugin.") and name.count(".") == 2 for name in plugin_names)


def test_v3_source_metadata_is_strict() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    invalid = {
        **source,
        "removed": [{"name": "old", "why": "merged", "detail": "x", "extra": True}],
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(source).validate(invalid)


def test_v3_generated_shapes_match_runtime_projection_types() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    params = manifest["$defs"]["Params_settings_update_v3"]["properties"]["patch"]
    refresh = manifest["$defs"]["Params_thread_get_v3"]["properties"]["refresh"]
    cancelling = manifest["$defs"]["Result_turn_cancel_v3"]["properties"]["cancelling"]
    assert params["type"] == "object"
    assert refresh["type"] == "boolean"
    assert cancelling["type"] == "boolean"


def test_v3_numeric_boundaries_use_integer_schema_types() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    defs = manifest["$defs"]
    assert defs["Params_fs_read_v3"]["properties"]["max_bytes"]["type"] == "integer"
    assert defs["Params_terminal_resize_v3"]["properties"]["columns"]["type"] == "integer"
    assert defs["Event_operation_completed_v3"]["properties"]["duration_ms"]["type"] == "integer"
    assert defs["Event_terminal_opened_v3"]["properties"]["rows"]["type"] == "integer"
    assert defs["Result_system_health_v3"]["properties"]["uptime_s"]["type"] == "integer"
    assert defs["Result_thread_context_v3"]["properties"]["ratio"]["type"] == ["number", "null"]
