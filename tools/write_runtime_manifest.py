#!/usr/bin/env python3
"""Write the manifest consumed by the desktop Runtime launcher.

The archive builder deliberately treats this file as input data. Keeping the
manifest generation here makes the release workflow unable to silently publish
an archive whose executable hashes or Core site tree do not describe its bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--runtime-version", default="0.1.0")
    parser.add_argument("--app-version", default="0.0.17")
    parser.add_argument("--core-version", default="0.1.0")
    parser.add_argument("--core-commit", default=os.environ.get("GITHUB_SHA", "unknown"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    required = {
        "python": ("components/python", "bin/python3", "executable"),
        "uv": ("components/uv", "uv", "executable"),
        "ripgrep": ("components/ripgrep", "rg", "executable"),
    }
    components: dict[str, object] = {}
    for name, (relative_root, executable, field) in required.items():
        component_root = root / relative_root
        executable_path = component_root / executable
        if not executable_path.is_file():
            raise SystemExit(f"missing Runtime component executable: {executable_path}")
        components[name] = {
            "version": os.environ.get(f"{name.upper()}_VERSION", "bundled"),
            "root": relative_root,
            field: executable,
            "executableSha256": _sha256(executable_path),
        }
    core_site = root / "core-site"
    tree_hash = _tree_sha256(core_site)
    manifest = {
        "schemaVersion": 5,
        "runtimeVersion": args.runtime_version,
        "appVersion": args.app_version,
        "rpcProtocol": "aeloon-rpc",
        "core": {
            "version": args.core_version,
            "repository": "AetherHeart-AI/Aeloon-core",
            "commit": args.core_commit,
            "sitePackages": "core-site",
            "treeSha256": tree_hash,
        },
        "toolVersions": {
            "python": components["python"]["version"],
            "uv": components["uv"]["version"],
            "ripgrep": components["ripgrep"]["version"],
            "core": args.core_version,
        },
        "defaultSources": {
            "python": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            "npm": "https://registry.npmmirror.com/",
        },
        "platform": args.platform,
        "components": components,
    }
    (root / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
