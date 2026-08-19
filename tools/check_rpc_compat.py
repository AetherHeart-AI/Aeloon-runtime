#!/usr/bin/env python3
"""Check additive/removal compatibility between two generated RPC manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    breaking: list[str] = []
    additions: list[str] = []
    for key in ("methods", "plugin_methods", "events", "errors"):
        old_values = old.get(key, {})
        new_values = new.get(key, {})
        for name in sorted(set(old_values) - set(new_values)):
            breaking.append(f"removed {key[:-1]} {name}")
        for name in sorted(set(new_values) - set(old_values)):
            additions.append(f"added {key[:-1]} {name}")
        for name in sorted(set(old_values) & set(new_values)):
            old_shape = _expand_refs(old_values[name], old.get("$defs", {}))
            new_shape = _expand_refs(new_values[name], new.get("$defs", {}))
            if not _compatible_shape(old_shape, new_shape):
                breaking.append(f"changed {key[:-1]} {name}")
    return {"compatible": not breaking, "breaking": breaking, "additions": additions}


def _expand_refs(value: Any, definitions: Any, seen: set[str] | None = None) -> Any:
    """Resolve local manifest references before comparing method/event shapes."""

    if not isinstance(value, (dict, list)):
        return value
    if seen is None:
        seen = set()
    if isinstance(value, list):
        return [_expand_refs(item, definitions, seen) for item in value]
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        name = reference.removeprefix("#/$defs/")
        target = definitions.get(name) if isinstance(definitions, dict) else None
        if isinstance(target, (dict, list)) and name not in seen:
            return _expand_refs(target, definitions, {*seen, name})
    return {
        key: _expand_refs(item, definitions, seen)
        for key, item in value.items()
    }


def _compatible_shape(old: Any, new: Any) -> bool:
    """Return whether ``new`` preserves every value accepted by ``old``.

    This deliberately models the additive minor-version rule: new optional
    object properties are compatible, while removing properties, adding a
    required property, narrowing an enum/type, or changing a scalar is major.
    """

    if isinstance(old, dict) and isinstance(new, dict):
        old_required = set(old.get("required", []))
        new_required = set(new.get("required", []))
        if not old_required.issubset(new_required) or new_required - old_required:
            return False
        old_type = old.get("type")
        new_type = new.get("type")
        if old_type is not None:
            old_types = set(old_type) if isinstance(old_type, list) else {old_type}
            new_types = set(new_type) if isinstance(new_type, list) else {new_type}
            if not old_types.issubset(new_types):
                return False
        if "enum" in old:
            if not set(old["enum"]).issubset(set(new.get("enum", []))):
                return False
        old_properties = old.get("properties", {})
        new_properties = new.get("properties", {})
        if set(old_properties) - set(new_properties):
            return False
        for key in ("const", "enum", "items", "anyOf", "oneOf", "allOf", "$ref"):
            if key in old and key not in new:
                return False
        if "additionalProperties" in old and "additionalProperties" in new:
            old_extra = old["additionalProperties"]
            new_extra = new["additionalProperties"]
            if old_extra is True and new_extra is False:
                return False
            if isinstance(old_extra, dict) and not _compatible_shape(old_extra, new_extra):
                return False
        return all(
            _compatible_shape(old_properties[name], new_properties[name])
            for name in old_properties
        ) and all(
            _compatible_shape(old_value, new.get(key, old_value))
            for key, old_value in old.items()
            if key not in {
                "required",
                "properties",
                "type",
                "enum",
                "additionalProperties",
            }
            and key in new
        )
    if isinstance(old, list) and isinstance(new, list):
        return all(any(_compatible_shape(item, candidate) for candidate in new) for item in old)
    return old == new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.old.read_text()), json.loads(args.new.read_text()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
