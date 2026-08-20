#!/usr/bin/env python3
"""Build the release-only wheel with the single ``aeloon-runtime`` command.

The source checkout and the published wheel deliberately expose exactly one
executable entry point: ``aeloon-runtime``.  This helper keeps the release
assertion close to the archive-building path so an old alias cannot return.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="aeloon-runtime-wheel-") as temporary:
        staging = Path(temporary) / "source"
        shutil.copytree(
            source,
            staging,
            ignore=shutil.ignore_patterns(".git", ".venv", "dist", "build", "*.egg-info"),
        )
        metadata = (staging / "pyproject.toml").read_text(encoding="utf-8")
        metadata = metadata.replace('name = "aeloon-runtime"', 'name = "aeloon-runtime"', 1)
        metadata = metadata.replace('version = "0.1.0"', 'version = "0.1.0"', 1)
        if '[project.scripts]\naeloon-runtime = "aeloon_runtime.__main__:main"' not in metadata:
            raise SystemExit("source metadata must expose only the aeloon-runtime script")
        (staging / "pyproject.toml").write_text(metadata, encoding="utf-8")
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(output)], cwd=staging, check=True
        )
    wheels = sorted(output.glob("aeloon_runtime-0.1.0-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Runtime wheel build did not produce exactly one wheel")
    print(wheels[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
