"""Deterministic aeloon-rpc-v2 manifest exporter.

This module is a build/test tool.  The production RPC adapter does not import it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, get_origin

from pydantic import TypeAdapter

from aeloon_core.rpc.protocol import (
    EVENT_SPECS,
    MAX_FRAME_BYTES,
    METHOD_SPECS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RPC_CODES,
)

MANIFEST_PATH = Path(__file__).with_name(f"{PROTOCOL_NAME}.manifest.json")


def _type_name(value: Any) -> str:
    if get_origin(value) is dict or value is dict:
        return "JsonObject"
    name = getattr(value, "__name__", None)
    if not name:
        raise TypeError(f"Protocol type has no stable name: {value!r}")
    return str(name)


def _add_schema(definitions: dict[str, Any], value: Any) -> str:
    name = _type_name(value)
    if name in definitions:
        return name
    schema = TypeAdapter(value).json_schema(ref_template="#/$defs/{model}")
    nested = schema.pop("$defs", {})
    _close_typed_objects(schema)
    for nested_schema in nested.values():
        _close_typed_objects(nested_schema)
    for nested_name, nested_schema in nested.items():
        existing = definitions.get(nested_name)
        if existing is not None and existing != nested_schema:
            raise ValueError(f"Conflicting JSON Schema definition: {nested_name}")
        definitions[nested_name] = nested_schema
    schema.setdefault("title", name)
    definitions[name] = schema
    return name


def _close_typed_objects(value: Any) -> None:
    """Make declared object fields exact while preserving explicit open mappings."""
    if isinstance(value, dict):
        if value.get("type") == "object" and "properties" in value:
            value.setdefault("additionalProperties", False)
        for child in value.values():
            _close_typed_objects(child)
    elif isinstance(value, list):
        for child in value:
            _close_typed_objects(child)


def build_manifest() -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    methods: dict[str, Any] = {}
    events: dict[str, Any] = {}
    for spec in METHOD_SPECS:
        params = _add_schema(definitions, spec.params)
        result = _add_schema(definitions, spec.result)
        methods[spec.name] = {
            "params": {"$ref": f"#/$defs/{params}"},
            "result": {"$ref": f"#/$defs/{result}"},
        }
    for spec in EVENT_SPECS:
        payload = _add_schema(definitions, spec.payload)
        events[spec.name] = {"payload": {"$ref": f"#/$defs/{payload}"}}
    return {
        "schema_version": 1,
        "json_schema_draft": "https://json-schema.org/draft/2020-12/schema",
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "transport": {
            "encoding": "utf-8-json",
            "framing": "uint32-be-length-prefix",
            "max_frame_bytes": MAX_FRAME_BYTES,
        },
        "methods": methods,
        "events": events,
        "errors": dict(sorted(RPC_CODES.items())),
        "$defs": dict(sorted(definitions.items())),
    }


def render_manifest() -> str:
    return json.dumps(build_manifest(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the aeloon RPC protocol manifest")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="write the manifest to this path")
    group.add_argument(
        "--check",
        nargs="?",
        type=Path,
        const=MANIFEST_PATH,
        help="fail if the target differs (defaults to the packaged manifest)",
    )
    args = parser.parse_args(argv)
    rendered = render_manifest()
    if args.check is not None:
        try:
            current = args.check.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Protocol manifest is missing: {args.check}", file=sys.stderr)
            return 1
        if current != rendered:
            print(f"Protocol manifest is stale: {args.check}", file=sys.stderr)
            return 1
        return 0
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
