#!/usr/bin/env python3
"""Deterministically sanitize a Runtime JSONL trace for human review.

Raw traces stay local and are already secret-redacted by ``TraceRecorder``.
This second pass replaces identifiers, timestamps, process endpoints and
absolute paths with stable symbols so a reviewed fixture does not contain
machine-specific values.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
from pathlib import Path
from typing import Any

_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SHA = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_ISO_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_ID_KEY = re.compile(
    r"(?:^|_)(?:id|ids|operation|thread|project|attachment|terminal)(?:$|_)",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


class Sanitizer:
    def __init__(self) -> None:
        self.ids: dict[str, str] = {}
        self.paths: dict[str, str] = {}
        self.times: dict[str, str] = {}
        self.shas: dict[str, str] = {}
        self.next_id = 1
        self.next_path = 1
        self.next_time = 1
        self.next_sha = 1

    def value(self, value: Any, key: str = "") -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if re.fullmatch(r"(?:pid|port)", key, re.IGNORECASE):
                return self._mapped(
                    self.ids,
                    f"{key}:{value}",
                    lambda: self._next_endpoint(key),
                )
            return value
        if isinstance(value, str):
            return self.string(value, key)
        if isinstance(value, list):
            return [self.value(item, key) for item in value]
        if isinstance(value, dict):
            return {
                child_key: self.value(child_value, child_key)
                for child_key, child_value in value.items()
            }
        return value

    def string(self, value: str, key: str = "") -> str:
        if _ISO_TIME.fullmatch(value):
            return self._mapped(self.times, value, self._next_time)
        if (
            _UUID.fullmatch(value)
            and (
                _ID_KEY.search(key)
                or re.search(r"(?:id|ids)$", key, re.IGNORECASE)
                or not key
            )
        ):
            return self._mapped(self.ids, value, self._next_id)
        if _ID_KEY.search(key) and _IDENTIFIER.fullmatch(value):
            return self._mapped(self.ids, value, self._next_id)
        if _SHA.fullmatch(value) and re.search(r"(?:commit|sha|hash)", key, re.IGNORECASE):
            return self._mapped(self.shas, value, self._next_sha)
        if self.looks_like_path(value):
            normalized = posixpath.normpath(value.replace("\\", "/"))
            basename = posixpath.basename(normalized) or "."
            return self._mapped(
                self.paths,
                normalized,
                lambda: self._next_path_symbol(basename),
            )
        return value

    @staticmethod
    def looks_like_path(value: str) -> bool:
        return value.startswith(("/", "./", "../")) or "/." in value or "\\." in value

    def _mapped(self, mapping: dict[str, str], value: str, create: Any) -> str:
        if value not in mapping:
            mapping[value] = create()
        return mapping[value]

    def _next_id(self) -> str:
        value = f"<id:{self.next_id}>"
        self.next_id += 1
        return value

    def _next_endpoint(self, key: str) -> str:
        value = f"<{key.lower()}:{self.next_id}>"
        self.next_id += 1
        return value

    def _next_time(self) -> str:
        value = f"<time:{self.next_time}>"
        self.next_time += 1
        return value

    def _next_sha(self) -> str:
        value = f"<sha:{self.next_sha}>"
        self.next_sha += 1
        return value

    def _next_path_symbol(self, basename: str) -> str:
        value = f"<path:{self.next_path}/{basename}"
        self.next_path += 1
        return f"{value}>"


def sanitize_lines(lines: list[str]) -> list[str]:
    sanitizer = Sanitizer()
    output: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        output.append(
            json.dumps(
                sanitizer.value(json.loads(line)),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw JSONL trace")
    parser.add_argument("output", type=Path, help="sanitized JSONL trace")
    args = parser.parse_args()

    output = args.output.expanduser().resolve(strict=False)
    parent_existed = output.parent.exists()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        output.parent.chmod(0o700)
    records = sanitize_lines(args.input.read_text(encoding="utf-8").splitlines())
    payload = "\n".join(records) + ("\n" if records else "")
    output.write_text(payload, encoding="utf-8")
    os.chmod(output, 0o600)
    print(f"Wrote {len(records)} sanitized records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
