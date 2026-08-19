from __future__ import annotations

import base64
import json
from pathlib import Path

from aeloon_core.trace import TraceRecorder
from tools.sanitize_trace import Sanitizer, sanitize_lines


def test_trace_recorder_is_opt_in_redacted_and_content_addressed(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    recorder = TraceRecorder(directory)
    payload = base64.b64encode(b"attachment bytes").decode()
    recorder.request(
        "request-1",
        "attachment.upload",
        {
            "api_key": "never-write",
            "authorization": "Bearer never-write",
            "detail": "auth_header=never-write",
            "headers": [{"name": "X-Api-Key", "value": "never-write"}],
            "data_base64": payload,
            "path": "/Users/private/workspace/note.txt",
        },
    )
    recorder.response("request-1", "attachment.upload", {"attachment_id": "attachment-1"})
    recorder.error("request-2", "provider.refresh", "authentication_failed", "token=never-write")
    recorder.event("operation.completed", {"thread_id": "thread-1"})
    recorder.close()

    trace_files = sorted(directory.glob("*.jsonl"))
    assert len(trace_files) == 1
    records = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    assert [record["sequence"] for record in records] == [1, 2, 3, 4, 5]
    serialized = json.dumps(records, ensure_ascii=False)
    assert "never-write" not in serialized
    assert records[1]["params"]["api_key"] == "[REDACTED]"
    assert records[1]["params"]["data_base64"]["sha256"]
    assert trace_files[0].stat().st_mode & 0o777 == 0o600
    blob = directory / records[1]["params"]["data_base64"]["path"]
    assert blob.read_bytes() == b"attachment bytes"
    assert blob.stat().st_mode & 0o777 == 0o600


def test_trace_sanitizer_symbolizes_dynamic_values_deterministically() -> None:
    lines = [
        json.dumps(
            {
                "thread_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2026-08-19T00:00:00.000Z",
                "workspace": "/private/tmp/first/project",
                "pid": 1234,
                "port": 4317,
                "commit": "0123456789012345678901234567890123456789",
            }
        ),
        json.dumps(
            {
                "thread_id": "123e4567-e89b-12d3-a456-426614174000",
                "workspace": "/private/tmp/first/project",
            }
        ),
    ]
    records = [json.loads(line) for line in sanitize_lines(lines)]
    assert records[0] == {
        "thread_id": "<id:1>",
        "created_at": "<time:1>",
        "workspace": "<path:1/project>",
        "pid": "<pid:2>",
        "port": "<port:3>",
        "commit": "<sha:1>",
    }
    assert records[1] == {
        "thread_id": "<id:1>",
        "workspace": "<path:1/project>",
    }


def test_trace_sanitizer_state_can_be_reused_for_mapping_consistency() -> None:
    sanitizer = Sanitizer()
    first = sanitizer.value({"operation_id": "operation-9"})
    second = sanitizer.value({"operation_id": "operation-9"})
    assert first == second == {"operation_id": "<id:1>"}
