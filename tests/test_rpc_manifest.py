from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from aeloon_core.rpc.manifest import MANIFEST_PATH, build_manifest, render_manifest
from aeloon_core.rpc.protocol import EVENT_REGISTRY, METHOD_REGISTRY


def test_checked_in_protocol_manifest_is_current_and_valid() -> None:
    assert MANIFEST_PATH.read_text(encoding="utf-8") == render_manifest()
    manifest = build_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["protocol"] == "aeloon-rpc-v2"
    assert set(manifest["methods"]) == set(METHOD_REGISTRY)
    assert set(manifest["events"]) == set(EVENT_REGISTRY)
    for definition in manifest["$defs"].values():
        Draft202012Validator.check_schema(definition)


def test_manifest_registry_has_resolvable_exact_shapes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    definitions = manifest["$defs"]
    referenced: set[str] = set()
    for group in (manifest["methods"], manifest["events"]):
        for entry in group.values():
            for reference in entry.values():
                name = reference["$ref"].removeprefix("#/$defs/")
                assert name in definitions
                referenced.add(name)
    assert referenced
    for schema in definitions.values():
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False


def test_production_adapter_does_not_import_manifest_generator() -> None:
    adapter_source = (Path(__file__).parents[1] / "aeloon_core" / "rpc" / "adapter.py").read_text(
        encoding="utf-8"
    )
    assert "rpc.manifest" not in adapter_source
    assert "TypeAdapter" not in adapter_source
    assert "jsonschema" not in adapter_source
