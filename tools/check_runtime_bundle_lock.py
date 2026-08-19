#!/usr/bin/env python3
"""Validate the UI Runtime bundle lock before a release is attempted."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

PLATFORMS = {"darwin-aarch64", "linux-aarch64"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate(value: object, *, release: bool = False) -> None:
    if not isinstance(value, dict):
        raise ValueError("runtime bundle lock must be an object")
    if value.get("schemaVersion") != 1 or value.get("runtimeVersion") != "0.1.0":
        raise ValueError("unsupported runtime bundle lock schema/version")
    if value.get("protocol") != "aeloon-rpc":
        raise ValueError("runtime bundle lock has an invalid protocol")
    platforms = value.get("platforms")
    if not isinstance(platforms, dict) or set(platforms) != PLATFORMS:
        raise ValueError("runtime bundle lock must pin both ARM64 platforms")
    for platform, artifact in platforms.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"{platform} artifact is not an object")
        url = artifact.get("url")
        digest = artifact.get("sha256")
        if not isinstance(url, str) or urlparse(url).scheme not in {"https", "file"}:
            raise ValueError(f"{platform} artifact URL is invalid")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"{platform} artifact SHA-256 is invalid")
        if release and digest == "0" * 64:
            raise ValueError(f"{platform} artifact has no published SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    validate(json.loads(args.lock.read_text(encoding="utf-8")), release=args.release)
    print(f"validated {args.lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
