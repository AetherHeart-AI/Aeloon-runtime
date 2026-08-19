#!/usr/bin/env python3
"""Update one UI Runtime bundle pin from the exact archive to publish."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--platform", choices=("darwin-aarch64", "linux-aarch64"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    lock_path = args.lock.expanduser().resolve(strict=True)
    archive = args.archive.expanduser().resolve(strict=True)
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != 1 or value.get("runtimeVersion") != "0.1.0":
        raise SystemExit("Unsupported Runtime bundle lock")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    value.setdefault("platforms", {})[args.platform] = {
        "url": args.url,
        "sha256": digest,
        "size_bytes": archive.stat().st_size,
    }
    lock_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"platform": args.platform, "url": args.url, "sha256": digest}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
