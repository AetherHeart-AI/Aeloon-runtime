#!/usr/bin/env python3
"""Generate a compact human-readable index from the protocol source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "docs" / "rpc-v4.json").read_text(encoding="utf-8"))
lines = [
    f"# Aeloon RPC {source['version']}",
    "",
    "This file is generated from `docs/rpc-v4.json`; do not edit by hand.",
    "",
    f"- Protocol: `{source['protocol']}`",
    f"- Frame: `{source['frame_max_bytes'] // (1024 * 1024)} MiB`",
    "- File/image: "
    f"`{source['file_max_bytes'] // (1024 * 1024)} MiB` / "
    f"`{source['image_max_bytes'] // (1024 * 1024)} MiB`",
    "",
]
for group in source["groups"]:
    lines.extend([f"## {group['id']}", ""])
    for method in group["methods"]:
        params = method.get("params_signature", method["params"])
        result = method.get("result_signature", method["result"])
        lines.append(f"- `{method['name']}` — `{params}` → `{result}`")
    lines.append("")

plugin_methods = source.get("plugin_contributed", {}).get("methods", [])
if plugin_methods:
    lines.extend(["## Plugin-contributed methods", ""])
    for method in plugin_methods:
        lines.append(f"- `{method['name']}` — contributed namespace ({method.get('from', '')})")
    lines.append("")

events = source.get("events", {})
event_names = [
    *events.get("kept_from_core", []),
    *events.get("kept_from_workbench", []),
    *(item["to"] for item in events.get("renamed_from_core", [])),
    *(item["name"] for item in events.get("new", [])),
]
lines.extend(["## Events", ""])
for name in sorted(dict.fromkeys(event_names)):
    lines.append(f"- `{name}`")
lines.extend(["", "## Errors", ""])
error_catalog = source.get("errors", {})
for name in error_catalog.get("kept", []):
    lines.append(f"- `{name}` — retained")
for item in error_catalog.get("added", []):
    lines.append(f"- `{item['code']}` ({item['num']}) — added")
for item in error_catalog.get("renamed", []):
    lines.append(f"- `{item['from']}` → `{item['to']}` — renamed")

outputs = {
    ROOT / "docs" / "rpc-v4.md": "\n".join(lines) + "\n",
}
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
for output, rendered in outputs.items():
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Generated RPC documentation is stale: {output}")
    else:
        output.write_text(rendered, encoding="utf-8")
        print(output)
