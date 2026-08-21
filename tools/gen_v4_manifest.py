#!/usr/bin/env python3
"""Generate the checked-in aeloon-rpc v4 JSON Schema manifest.

``docs/rpc-v4.json`` is the protocol source. Method entries contain a direct
``$ref`` for both parameters and results; the one-time ``--upgrade-source``
helper converts the original compact signatures into those definitions while
preserving the signatures as documentation fields.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "rpc-v4.json"
DEFAULT_OUTPUT = ROOT / "aeloon_runtime" / "rpc" / "aeloon-rpc-v4.manifest.json"

ERROR_CODES = {
    "protocol_incompatible": -32010,
    "invalid_argument": -32602,
    "thread_not_found": -32020,
    "session_not_found": -32020,
    "operation_not_found": -32021,
    "busy": -32022,
    "invalid_state": -32023,
    "invalid_attachment": -32024,
    "attachment_processing_failed": -32028,
    "revision_conflict": -32025,
    "authentication_failed": -32027,
    "internal_error": -32603,
    "method_not_found": -32601,
    "unauthorized": -32011,
    "forbidden": -32012,
    "capability_unavailable": -32013,
    "payload_too_large": -32014,
}


def _event_object(
    properties: dict[str, dict[str, Any]],
    required: tuple[str, ...] = (),
    *,
    additional_properties: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "object",
        "additionalProperties": additional_properties,
        "properties": properties,
    }
    if required:
        value["required"] = list(required)
    return value


def _event_payload_schema(event: str) -> dict[str, Any] | None:
    """Return the stable v4 payload shape for an event family.

    Event envelopes carry the cursor/thread/operation metadata.  This table
    describes only ``payload`` so the gateway can validate producers without
    making opaque provider/retry diagnostics part of the public contract.
    """

    string = {"type": "string"}
    integer = {"type": "integer"}
    boolean = {"type": "boolean"}
    json_object = {"type": "object", "additionalProperties": True}
    block = {"type": "object", "additionalProperties": True}
    if event.startswith("operation."):
        return _event_object(
            {
                "kind": string,
                "queue_position": integer,
                "skill_id": string,
                "attachment_ids": {"type": "array", "items": string},
                "duration_ms": integer,
                "error": string,
                "code": string,
            },
            ("kind",),
        )
    if event in {"content.started", "tool.started"}:
        return _event_object({"block": block}, ("block",))
    if event == "content.delta":
        return _event_object({"block_id": string, "delta": string}, ("block_id", "delta"))
    if event in {"content.updated", "content.completed", "tool.updated", "tool.completed"}:
        return _event_object({"block_id": string, "patch": json_object}, ("block_id", "patch"))
    if event == "usage.updated":
        return _event_object({"usage": json_object, "stats": json_object})
    if event == "queue.updated":
        return _event_object(
            {
                "queued_operation_ids": {"type": "array", "items": string},
                "active_operation_id": {"type": ["string", "null"]},
            }
        )
    if event == "provider.updated":
        return _event_object({"provider_id": string, "action": string}, ("provider_id", "action"))
    if event == "settings.updated":
        return _event_object({"revision": integer}, ("revision",))
    if event == "system.shutdown":
        return _event_object({"intentional": boolean, "reason": string}, ("intentional", "reason"))
    if event == "log.entry":
        return _event_object(
            {
                "category": string,
                "action": string,
                "runtime_version": string,
                "runtime_commit": string,
                "attachment_id": string,
                "canonical_path": string,
                "metadata": json_object,
                "pages": {"type": "array", "items": {"type": "integer"}},
                "dpi": {"type": "integer"},
                "level": string,
                "message": string,
                "data": json_object,
            }
        )
    if event == "thread.renamed":
        return _event_object(
            {"title": {"type": ["string", "null"]}, "source": string},
            ("title",),
        )
    if event == "terminal.opened":
        return _event_object({"columns": integer, "rows": integer}, ("columns", "rows"))
    if event == "terminal.output":
        return _event_object({"data": string}, ("data",))
    if event == "terminal.exit":
        return _event_object(
            {"status": string, "exit_code": integer, "signal": integer},
            ("status",),
        )
    if event == "capabilities.updated":
        return _event_object(
            {"capabilities": {"type": "array", "items": json_object}},
            ("capabilities",),
        )
    if event == "plugin.cloud.account_updated":
        return _event_object(
            {
                "enabled": boolean,
                "authenticated": boolean,
                "user": {"type": ["object", "null"], "additionalProperties": True},
                "base_url": string,
                "vault_kind": string,
                "ok": boolean,
            },
            ("enabled", "authenticated", "user", "base_url", "vault_kind"),
        )
    # Compaction/navigation, retry diagnostics, turn-created and generic
    # plugin events are deliberately extensible maps by protocol design.
    return None


class ShapeParser:
    def __init__(self, definitions: dict[str, Any]) -> None:
        self.definitions = definitions

    def schema(self, shape: str, name: str) -> dict[str, Any]:
        if shape == "{}":
            value = {"type": "object", "additionalProperties": False, "properties": {}}
        else:
            value = self._value(shape.strip(), name)
        value.setdefault("title", name)
        self.definitions[name] = value
        return {"$ref": f"#/$defs/{name}"}

    def _value(self, value: str, name: str) -> dict[str, Any]:
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            return self._object(value[1:-1], name)
        if value.startswith("[") and value.endswith("]"):
            return {"type": "array", "items": self._value(value[1:-1], f"{name}Item")}
        if value.endswith("[]"):
            return {"type": "array", "items": self._value(value[:-2], f"{name}Item")}
        if "|" in value:
            return {
                "anyOf": [
                    self._value(part, f"{name}Option{index}")
                    for index, part in enumerate(_split(value, "|"))
                ]
            }
        if value.startswith('"') and value.endswith('"'):
            return {"const": value[1:-1]}
        if value in {"true", "false", "boolean"}:
            return {"type": "boolean"}
        if value in {"int", "integer"}:
            return {"type": "integer"}
        if value == "number":
            return {"type": "number"}
        if value in {"object", "map"}:
            return {"type": "object", "additionalProperties": True}
        if value == "string_or_null":
            return {"type": ["string", "null"]}
        if value == "number_or_null":
            return {"type": ["number", "null"]}
        if value == "object_or_null":
            return {"type": ["object", "null"], "additionalProperties": True}
        return {"type": "string"}

    def _object(self, body: str, name: str) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        alternatives: list[list[str]] = []
        for raw in _split(body, ","):
            field = raw.strip()
            if not field:
                continue
            field_name, field_value, optional = _field(field)
            names = [part.strip() for part in field_name.split("|") if part.strip()]
            is_array = False
            normalized_names: list[str] = []
            for item_name in names:
                if item_name.endswith("[]"):
                    is_array = True
                    item_name = item_name[:-2]
                normalized_names.append(item_name)
            names = normalized_names
            field_shape = field_value or _inferred_shape(names[0], is_array=is_array, context=name)
            for item_name in names:
                properties[item_name] = self._value(field_shape, f"{name}_{item_name}")
            if len(names) > 1 and not optional:
                alternatives.append(names)
            elif not optional:
                # Use the normalized property name (without the compact
                # signature's ``[]`` marker) in JSON Schema required lists.
                required.extend(names)
        result: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }
        if required:
            result["required"] = required
        if alternatives:
            # A compact ``a|b`` field means exactly one discriminator is
            # required.  Materialize full object variants so JSON Schema
            # consumers (including the TypeScript generator) retain the
            # mutually-exclusive union instead of collapsing it to an open
            # map.  Shared properties are duplicated intentionally: the
            # generated source remains self-contained and Draft 2020-12
            # validators can enforce the same additionalProperties boundary
            # on either branch.
            variants: list[dict[str, Any]] = []
            for group in alternatives:
                for item in group:
                    variants.append(
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": properties,
                            "required": [*required, item],
                        }
                    )
            result["oneOf"] = variants
        return result


def _field(raw: str) -> tuple[str, str | None, bool]:
    colon = _find_top_level(raw, ":")
    if colon < 0:
        key = raw.rstrip("?").strip()
        return key, None, raw.endswith("?")
    key = raw[:colon].strip()
    optional = key.endswith("?")
    return key.rstrip("?"), raw[colon + 1 :].strip(), optional


def _inferred_shape(name: str, *, is_array: bool, context: str = "") -> str:
    # ``settings.update`` carries a list of structured secret actions even
    # though the compact signature keeps the field name without a trailing
    # ``[]`` marker for readability.
    if name == "secret_actions":
        return "object[]"
    if is_array:
        if name.endswith("_ids") or name in {
            "arguments",
            "branches",
            "data_paths",
            "methods",
            "paths",
            "ports",
            "thread_ids",
            "workspace_roots",
        }:
            return "string[]"
        if name == "events" and "capabilities" in context:
            return "string[]"
        return "object[]"
    if name == "settings_schema":
        return "object_or_null"
    if name == "default_model_id":
        return "string_or_null"
    if name == "branch":
        return "string_or_null"
    if name == "current":
        return "string_or_null"
    if name in {
        "accepted",
        "active",
        "archived",
        "binary",
        "cancelled",
        "cancelling",
        "closed",
        "compacted",
        "deleted",
        "dirty",
        "enabled",
        "exists",
        "replay_complete",
        "authenticated",
        "is_git",
        "opened",
        "prepared",
        "pinned",
        "pushed",
        "read",
        "refresh",
        "removed",
        "truncated",
        "updated",
        "writable",
        "ok",
    }:
        return "boolean"
    if name in {"mtime", "ratio", "maximum"}:
        return "number_or_null" if name in {"maximum", "ratio"} else "number"
    if name in {
        "active_operations",
        "attachments",
        "after_seq",
        "additions",
        "columns",
        "current_seq",
        "estimated_bytes",
        "file_bytes",
        "image_bytes",
        "max_bytes",
        "order",
        "port",
        "prompt_chars",
        "ahead",
        "behind",
        "request_bytes",
        "retained_events",
        "rows",
        "seq",
        "size",
        "size_bytes",
        "deletions",
        "duration_ms",
        "exit_code",
        "queue_position",
        "signal",
        "uptime_s",
        "used",
        "revision",
    }:
        return "integer"
    if name in {
        "capabilities",
        "config",
        "cursor",
        "patch",
        "history",
        "limits",
        "metadata",
        "project",
        "projects",
        "provider",
        "runtime",
        "stats",
        "thread",
        "threads",
        "turn",
        "turns",
        "ui",
        "settings",
    }:
        return "object"
    return "string"


def _find_top_level(value: str, needle: str) -> int:
    depth = 0
    quote = False
    for index, char in enumerate(value):
        if char == '"':
            quote = not quote
        elif not quote and char == "{":
            depth += 1
        elif not quote and char == "}":
            depth -= 1
        elif not quote and depth == 0 and char == needle:
            return index
    return -1


def _split(value: str, separator: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    quote = False
    for index, char in enumerate(value):
        if char == '"':
            quote = not quote
        elif not quote and char == "{":
            depth += 1
        elif not quote and char == "}":
            depth -= 1
        elif not quote and depth == 0 and char == separator:
            pieces.append(value[start:index])
            start = index + 1
    pieces.append(value[start:])
    return pieces


def method_def_name(method: str, side: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", method).strip("_")
    return f"{side}_{safe}_v4"


def _method_schema(
    spec: Any,
    name: str,
    definitions: dict[str, Any],
    parser: ShapeParser,
) -> dict[str, Any]:
    if isinstance(spec, dict) and isinstance(spec.get("$ref"), str):
        return {"$ref": spec["$ref"]}
    return parser.schema(str(spec), name)


def _definition_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            refs.add(reference.removeprefix("#/$defs/"))
        for child in value.values():
            refs.update(_definition_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_definition_refs(child))
    return refs


def upgrade_source(source: dict[str, Any]) -> dict[str, Any]:
    """Materialize compact method signatures as source-level ``$ref`` defs."""

    definitions = dict(source.get("$defs") or {})
    parser = ShapeParser(definitions)
    for group in source["groups"]:
        for method in group["methods"]:
            params_signature = method.get("params_signature", method["params"])
            result_signature = method.get("result_signature", method["result"])
            params_name = method_def_name(method["name"], "Params")
            result_name = method_def_name(method["name"], "Result")
            method["params_signature"] = params_signature
            method["result_signature"] = result_signature
            method["params"] = (
                params_signature
                if isinstance(params_signature, dict) and "$ref" in params_signature
                else parser.schema(params_signature, params_name)
            )
            method["result"] = (
                result_signature
                if isinstance(result_signature, dict) and "$ref" in result_signature
                else parser.schema(result_signature, result_name)
            )
            if isinstance(method["params_signature"], dict):
                method["params_signature"] = "See the referenced parameter schema."
            if isinstance(method["result_signature"], dict):
                method["result_signature"] = "See the referenced result schema."
    event_schemas: dict[str, Any] = {}
    event_catalog = source.get("events", {})
    event_names = [
        *event_catalog.get("kept_from_core", []),
        *event_catalog.get("kept_from_workbench", []),
        *(item["to"] for item in event_catalog.get("renamed_from_core", [])),
        *(item["name"] for item in event_catalog.get("new", [])),
    ]
    for event in dict.fromkeys(event_names):
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", event).strip("_")
        name = f"Event_{safe}_v4"
        # Once the source has an explicit event definition, it is the single
        # protocol authority. The family table remains a fallback for draft
        # inputs that did not carry event `$defs`.
        shape = definitions.get(name) or _event_payload_schema(event)
        if shape is not None:
            shape["title"] = name
            definitions[name] = shape
        else:
            definitions.setdefault(
                name,
                {"title": name, "type": "object", "additionalProperties": True},
            )
        event_schemas[event] = {"$ref": f"#/$defs/{name}"}
    source["event_schemas"] = event_schemas
    source["$defs"] = definitions
    return source


def build(source: dict[str, Any]) -> dict[str, Any]:
    source_definitions = source.get("$defs") or {}
    definitions: dict[str, Any] = {
        key: value
        for key, value in source_definitions.items()
        if key.startswith(("Params_", "Result_"))
    }

    # Keep every custom definition reachable from a generated request/result
    # shape.  The source intentionally contains transport/catalog helper
    # definitions that do not belong in the runtime manifest, but dropping a
    # nested `$ref` would leave the generated boundary validator unusable.
    pending = list(definitions)
    while pending:
        name = pending.pop()
        for reference in _definition_refs(definitions[name]):
            if reference in definitions or reference not in source_definitions:
                continue
            definitions[reference] = source_definitions[reference]
            pending.append(reference)
    parser = ShapeParser(definitions)
    methods: dict[str, Any] = {}
    for group in source["groups"]:
        for method in group["methods"]:
            params = method_def_name(method["name"], "Params")
            result = method_def_name(method["name"], "Result")
            methods[method["name"]] = {
                "params": _method_schema(method["params"], params, definitions, parser),
                "result": _method_schema(method["result"], result, definitions, parser),
            }
    plugin_methods: dict[str, Any] = {}
    for method in source.get("plugin_contributed", {}).get("methods", []):
        if not isinstance(method, dict) or not isinstance(method.get("name"), str):
            continue
        params_spec = method.get("params")
        result_spec = method.get("result")
        if not isinstance(params_spec, dict) or not isinstance(result_spec, dict):
            continue
        params_ref = params_spec.get("$ref")
        result_ref = result_spec.get("$ref")
        if not isinstance(params_ref, str) or not isinstance(result_ref, str):
            continue
        plugin_methods[method["name"]] = {
            "params": {"$ref": params_ref},
            "result": {"$ref": result_ref},
        }
    events: dict[str, Any] = {}
    declared_events = source.get("event_schemas") or {}
    if declared_events:
        for event, spec in declared_events.items():
            events[event] = {"payload": spec}
            if isinstance(spec, dict) and isinstance(spec.get("$ref"), str):
                ref_name = spec["$ref"].removeprefix("#/$defs/")
                if ref_name in source_definitions:
                    shape = source_definitions[ref_name] or _event_payload_schema(event)
                    if shape is not None:
                        shape["title"] = ref_name
                    definitions[ref_name] = shape or source_definitions[ref_name]
        return {
            "schema_version": 1,
            "json_schema_draft": source["json_schema_draft"],
            "protocol": source["protocol"],
            "protocol_version": source["version"],
            "transport": source["transport"],
            "methods": methods,
            "plugin_methods": plugin_methods,
            "events": events,
            "errors": ERROR_CODES,
            "$defs": dict(sorted(definitions.items())),
        }
    for event in source.get("events", {}).get("kept_from_core", []):
        name = f"Event_{re.sub(r'[^a-zA-Z0-9]+', '_', event)}_v4"
        definitions.setdefault(name, {"type": "object", "additionalProperties": True})
        events[event] = {"payload": {"$ref": f"#/$defs/{name}"}}
    for event in source.get("events", {}).get("kept_from_workbench", []):
        name = f"Event_{re.sub(r'[^a-zA-Z0-9]+', '_', event)}_v4"
        definitions.setdefault(name, {"type": "object", "additionalProperties": True})
        events[event] = {"payload": {"$ref": f"#/$defs/{name}"}}
    for item in source.get("events", {}).get("renamed_from_core", []):
        event = item["to"]
        name = f"Event_{re.sub(r'[^a-zA-Z0-9]+', '_', event)}_v4"
        definitions.setdefault(name, {"type": "object", "additionalProperties": True})
        events[event] = {"payload": {"$ref": f"#/$defs/{name}"}}
    for item in source.get("events", {}).get("new", []):
        event = item["name"]
        name = f"Event_{re.sub(r'[^a-zA-Z0-9]+', '_', event)}_v4"
        definitions.setdefault(name, {"type": "object", "additionalProperties": True})
        events[event] = {"payload": {"$ref": f"#/$defs/{name}"}}
    return {
        "schema_version": 1,
        "json_schema_draft": source["json_schema_draft"],
        "protocol": source["protocol"],
        "protocol_version": source["version"],
        "transport": source["transport"],
        "methods": methods,
        "plugin_methods": plugin_methods,
        "events": events,
        "errors": ERROR_CODES,
        "$defs": dict(sorted(definitions.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--upgrade-source",
        action="store_true",
        help="Materialize compact signatures as source-level $defs/$ref entries.",
    )
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    # The checked-in source is itself a Draft 2020-12 document containing the
    # protocol metadata. Validate both the schema vocabulary and the document
    # instance before any generated registry can be written.
    Draft202012Validator.check_schema(source)
    errors = sorted(
        Draft202012Validator(source).iter_errors(source),
        key=lambda error: tuple(str(item) for item in error.path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "$"
        raise SystemExit(f"RPC v4 source is invalid at {location}: {first.message}")
    if args.upgrade_source:
        upgraded = upgrade_source(source)
        SOURCE.write_text(
            json.dumps(upgraded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    value = build(source)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Generated v4 manifest is stale: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
