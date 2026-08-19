#!/usr/bin/env python3
"""Generate the standalone TypeScript protocol package from the v3 manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "aeloon_core" / "rpc" / "aeloon-rpc-v3.manifest.json"
SOURCE = ROOT / "docs" / "rpc-v3.json"
OUTPUT = ROOT / "packages" / "protocol" / "src" / "index.ts"


def ts_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def schema_type(schema: Any, *, required: bool = True) -> str:
    if not isinstance(schema, dict):
        return "unknown"
    if isinstance(schema.get("$ref"), str):
        return ts_name(schema["$ref"])
    if "const" in schema:
        return json.dumps(schema["const"], ensure_ascii=False)
    union = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(union, list):
        return " | ".join(schema_type(item) for item in union) or "unknown"
    if "enum" in schema and isinstance(schema["enum"], list):
        return " | ".join(json.dumps(item, ensure_ascii=False) for item in schema["enum"])
    schema_kind = schema.get("type")
    if isinstance(schema_kind, list):
        return " | ".join(schema_type({"type": item}) for item in schema_kind)
    if schema_kind == "string":
        return "string"
    if schema_kind in {"number", "integer"}:
        return "number"
    if schema_kind == "boolean":
        return "boolean"
    if schema_kind == "null":
        return "null"
    if schema_kind == "array":
        return f"({schema_type(schema.get('items', {}))})[]"
    if schema_kind == "object" or "properties" in schema:
        if not schema.get("properties"):
            return (
                "{ [k: string]: unknown }"
                if schema.get("additionalProperties")
                else "Record<string, never>"
            )
        return "{ [k: string]: unknown }"
    return "unknown"


def interface_body(schema: dict[str, Any]) -> list[str]:
    properties = (
        schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    )
    required = (
        set(schema.get("required", []))
        if isinstance(schema.get("required"), list)
        else set()
    )
    lines: list[str] = []
    if schema.get("additionalProperties") is True:
        lines.append("  [k: string]: unknown;")
    for key in sorted(properties):
        safe = key if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", key) else json.dumps(key)
        optional = "" if key in required else "?"
        lines.append(f"  {safe}{optional}: {schema_type(properties[key])};")
    return lines


def render(manifest: dict[str, Any]) -> str:
    defs = manifest.get("$defs", {})
    lines = [
        "/* eslint-disable */",
        "/** Generated from aeloon_core/rpc/aeloon-rpc-v3.manifest.json. DO NOT EDIT. */",
        "",
    ]
    for name in sorted(defs):
        schema = defs[name]
        if not isinstance(schema, dict):
            continue
        union = schema.get("oneOf") or schema.get("anyOf")
        if isinstance(union, list):
            lines.append(f"export type {name} = {schema_type(schema)};")
        elif schema.get("type") == "object" or "properties" in schema:
            lines.append(f"export interface {name} {{")
            lines.extend(interface_body(schema))
            lines.extend(["}", ""])
        else:
            lines.append(f"export type {name} = {schema_type(schema)};")
            lines.append("")
    lines.extend(["export interface AeloonRuntimeRpcDefinitions {}", ""])
    lines.append("export interface RuntimeRpcMethodMap {")
    for method, spec in sorted(manifest.get("methods", {}).items()):
        params = ts_name(spec["params"]["$ref"])
        result = ts_name(spec["result"]["$ref"])
        lines.append(f"  {json.dumps(method)}: {{ params: {params}; result: {result} }};")
    lines.extend(["}", "export type RuntimeMethod = keyof RuntimeRpcMethodMap;"])
    lines.append(
        'export type RuntimeRpcParams<M extends RuntimeMethod> = RuntimeRpcMethodMap[M]["params"];'
    )
    lines.append(
        'export type RuntimeRpcResult<M extends RuntimeMethod> = RuntimeRpcMethodMap[M]["result"];'
    )
    lines.extend(["", "export interface RuntimePluginMethodMap {"])
    for method, spec in sorted(manifest.get("plugin_methods", {}).items()):
        params = ts_name(spec["params"]["$ref"])
        result = ts_name(spec["result"]["$ref"])
        lines.append(f"  {json.dumps(method)}: {{ params: {params}; result: {result} }};")
    lines.extend(["}", "export type RuntimePluginMethod = keyof RuntimePluginMethodMap;"])
    lines.append(
        'export type RuntimePluginRpcParams<M extends RuntimePluginMethod> = '
        'RuntimePluginMethodMap[M]["params"];'
    )
    lines.append(
        'export type RuntimePluginRpcResult<M extends RuntimePluginMethod> = '
        'RuntimePluginMethodMap[M]["result"];'
    )
    lines.extend(
        [
            "",
            "export interface RuntimeEventBase {",
            "  seq: number;",
            "  time?: string;",
            "  thread_id?: string | null;",
            "  operation_id?: string | null;",
            "  terminal_id?: string | null;",
            "  workspace?: string | null;",
            "}",
        ]
    )
    event_union: list[str] = []
    for event, spec in sorted(manifest.get("events", {}).items()):
        payload = ts_name(spec["payload"]["$ref"])
        event_union.append(
            f'  | (RuntimeEventBase & {{ name: {json.dumps(event)}; payload: {payload} }})'
        )
    lines.extend(
        [
            "export type RuntimeEvent =",
            *event_union,
            ";",
            'export type RuntimeEventName = RuntimeEvent["name"];',
            "",
        ]
    )
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    lines.extend(
        [
            f"export const RUNTIME_RPC_PROTOCOL = {json.dumps(manifest['protocol'])} as const;",
            "export const RUNTIME_RPC_VERSION = "
            f"{json.dumps(manifest['protocol_version'])} as const;",
            f"export const RUNTIME_RPC_MAX_FRAME_BYTES = {source['frame_max_bytes']} as const;",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = render(json.loads(MANIFEST.read_text(encoding="utf-8")))
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if actual != generated:
            raise SystemExit(f"Generated protocol is stale: {OUTPUT}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
