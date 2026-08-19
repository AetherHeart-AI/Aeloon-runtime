#!/usr/bin/env python3
"""Generate a compact human-readable index from the protocol source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "docs" / "rpc-v3.json").read_text(encoding="utf-8"))
lines = [
    f"# Aeloon RPC {source['version']}",
    "",
    "This file is generated from `docs/rpc-v3.json`; do not edit by hand.",
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

report_lines = [
    f"# Aeloon RPC {source['version']} compatibility report",
    "",
    "This file is generated from `docs/rpc-v3.json`; do not edit by hand.",
    "",
    "Minor-compatible changes are additive optional methods, events, fields, and error codes.",
    (
        "Removing an existing item, narrowing a shape, or changing existing semantics "
        "requires a major version."
    ),
    "",
    "## Method differences",
    "",
    "| Method | Change | Source | Note |",
    "| --- | --- | --- | --- |",
]
for group in source["groups"]:
    for method in group["methods"]:
        note = str(method.get("note", "")).replace("|", "\\|").replace("\n", " ")
        report_lines.append(
            f"| `{method['name']}` | `{method.get('change', 'same')}` | "
            f"`{method.get('src', '')}` | {note} |"
        )
if plugin_methods:
    report_lines.extend(["", "## Plugin-contributed methods", ""])
    for method in plugin_methods:
        availability = (
            "available as the hard-wired Cloud capability"
            if method["name"].startswith("plugin.cloud.")
            else "returns capability_unavailable in the base release"
        )
        report_lines.append(
            f"- `{method['name']}` — contributed namespace; {availability}"
        )
removed = source.get("removed", [])
if removed:
    report_lines.extend(["", "## Removed or collapsed legacy surface", ""])
    for item in removed:
        report_lines.append(f"- `{item['name']}` — {item.get('why', '')}: {item.get('detail', '')}")
report_lines.extend(["", "## Event differences", ""])
for item in events.get("renamed_from_core", []):
    report_lines.append(f"- `{item['from']}` → `{item['to']}`")
for item in events.get("new", []):
    report_lines.append(f"- `{item['name']}` — new")
for item in events.get("removed", []):
    report_lines.append(f"- `{item['name']}` — removed: {item.get('why', '')}")
if events.get("envelope_change"):
    report_lines.extend(["", f"Envelope: {events['envelope_change']}"])
report_lines.extend(["", "## Error differences", ""])
for item in error_catalog.get("added", []):
    report_lines.append(f"- `{item['code']}` ({item['num']}) — {item.get('when', '')}")
for item in error_catalog.get("renamed", []):
    report_lines.append(f"- `{item['from']}` → `{item['to']}` — {item.get('note', '')}")

outputs = {
    ROOT / "docs" / "rpc-v3.md": "\n".join(lines) + "\n",
    ROOT / "docs" / "rpc-v3-compatibility.md": "\n".join(report_lines) + "\n",
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
