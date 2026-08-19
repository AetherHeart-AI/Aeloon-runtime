from __future__ import annotations

from tools.check_rpc_compat import compare


def test_rpc_compatibility_allows_additive_methods() -> None:
    old = {"methods": {"system.health": {"params": 1}}, "events": {}, "errors": {}}
    new = {
        "methods": {"system.health": {"params": 1}, "system.snapshot": {}},
        "events": {},
        "errors": {},
    }
    result = compare(old, new)
    assert result["compatible"] is True
    assert result["additions"] == ["added method system.snapshot"]


def test_rpc_compatibility_rejects_removed_methods() -> None:
    result = compare(
        {"methods": {"system.health": {}}, "events": {}, "errors": {}},
        {"methods": {}, "events": {}, "errors": {}},
    )
    assert result["compatible"] is False


def test_rpc_compatibility_covers_plugin_contributed_methods() -> None:
    result = compare(
        {
            "methods": {},
            "plugin_methods": {"plugin.cloud.account_status": {}},
            "events": {},
            "errors": {},
        },
        {"methods": {}, "plugin_methods": {}, "events": {}, "errors": {}},
    )
    assert result["compatible"] is False


def test_rpc_compatibility_allows_optional_field_addition_but_rejects_required() -> None:
    old = {
        "methods": {
            "system.health": {
                "result": {"type": "object", "properties": {"ok": {"type": "boolean"}}}
            }
        },
        "events": {},
        "errors": {},
    }
    optional = {
        "methods": {
            "system.health": {
                "result": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}, "uptime": {"type": "number"}},
                }
            }
        },
        "events": {},
        "errors": {},
    }
    assert compare(old, optional)["compatible"] is True
    required = optional.copy()
    required["methods"] = {
        "system.health": {
            "result": {
                **optional["methods"]["system.health"]["result"],
                "required": ["uptime"],
            }
        }
    }
    assert compare(old, required)["compatible"] is False


def test_rpc_compatibility_treats_enum_addition_as_minor_and_removal_as_breaking() -> None:
    old = {
        "methods": {
            "thread.list": {
                "params": {"type": "string", "enum": ["active", "archived"]}
            }
        },
        "events": {},
        "errors": {},
    }
    added = {
        "methods": {
            "thread.list": {
                "params": {
                    "type": "string",
                    "enum": ["active", "archived", "all"],
                }
            }
        },
        "events": {},
        "errors": {},
    }
    removed = {
        "methods": {
            "thread.list": {
                "params": {"type": "string", "enum": ["active"]}
            }
        },
        "events": {},
        "errors": {},
    }
    assert compare(old, added)["compatible"] is True
    assert compare(old, removed)["compatible"] is False


def test_rpc_compatibility_resolves_manifest_refs_before_comparing() -> None:
    old = {
        "$defs": {
            "Result_health": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            }
        },
        "methods": {
            "system.health": {"result": {"$ref": "#/$defs/Result_health"}}
        },
        "events": {},
        "errors": {},
    }
    new = {
        "$defs": {
            "Result_health": {
                "type": "object",
                "properties": {"ok": {"type": "string"}},
            }
        },
        "methods": {
            "system.health": {"result": {"$ref": "#/$defs/Result_health"}}
        },
        "events": {},
        "errors": {},
    }
    assert compare(old, new)["compatible"] is False
