#!/usr/bin/env python3
"""Build a reproducible Runtime bundle archive and print its lock entry.

The platform CPython/uv/ripgrep files are supplied by release CI.  This tool
only packages a prepared tree, so a local developer cannot accidentally claim
an x86 tree is a macOS/Linux ARM release.  The resulting digest is always
computed from the bytes that are actually published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--platform", choices=("darwin-aarch64", "linux-aarch64"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-version", default="0.1.0")
    parser.add_argument("--protocol-range", default=">=3.0.0-draft.3 <4.0.0")
    args = parser.parse_args()
    root = args.runtime_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SystemExit("--runtime-root must be a directory")
    _assert_runtime_tree(root)
    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="aeloon-runtime-bundle-") as temporary:
        staging = Path(temporary) / "aeloon-runtime"
        shutil.copytree(root, staging, copy_function=shutil.copy2)
        metadata = {
            "runtime_version": args.runtime_version,
            "protocol_range": args.protocol_range,
            "platform": args.platform,
        }
        (staging / "runtime-release.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _normalize_tree(staging)
        _tar_zstd(staging, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "url": "https://github.com/AetherHeart-AI/Aeloon-core/releases/download/"
                f"runtime-v{args.runtime_version}/aeloon-runtime-{args.platform}.tar.zst",
                "sha256": digest,
                "size_bytes": output.stat().st_size,
            },
            sort_keys=True,
        )
    )
    return 0


def _assert_runtime_tree(root: Path) -> None:
    candidates = (
        root / "aeloon-runtime",
        root / "bin" / "aeloon-runtime",
        root / "runtime" / "aeloon-runtime",
    )
    if not any(item.is_file() for item in candidates):
        raise SystemExit("prepared tree does not contain an aeloon-runtime executable")
    forbidden = {"bun", "bun-pty", "sidecar", "workbench"}
    for path in root.rglob("*"):
        if path.name.lower() in forbidden:
            raise SystemExit(f"prepared tree contains removed component: {path}")


def _normalize_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        os.utime(path, (0, 0), follow_symlinks=False)
        if path.is_file():
            path.chmod(0o755 if os.access(path, os.X_OK) else 0o600)
        elif path.is_dir():
            path.chmod(0o700)


def _tar_zstd(root: Path, output: Path) -> None:
    tar_path = output.with_suffix(output.suffix + ".tar")
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root.parent)
            info = archive.gettarinfo(str(path), arcname=str(relative))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    try:
        subprocess.run(["zstd", "-q", "-19", "-f", str(tar_path), "-o", str(output)], check=True)
    finally:
        tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
