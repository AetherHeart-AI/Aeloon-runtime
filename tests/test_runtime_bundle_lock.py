from __future__ import annotations

from tools.check_runtime_bundle_lock import validate


def test_runtime_bundle_lock_rejects_unpublished_digest() -> None:
    value = {
        "schemaVersion": 1,
        "runtimeVersion": "0.1.0",
        "protocol": "aeloon-rpc",
        "platforms": {
            "darwin-aarch64": {"url": "https://example.test/a", "sha256": "0" * 64},
            "linux-aarch64": {"url": "https://example.test/b", "sha256": "1" * 64},
        },
        "core": {
            "schemaVersion": 1,
            "repository": "org/core",
            "commit": "a" * 40,
            "version": "0.1.0",
            "rpcProtocol": "aeloon-rpc",
        },
    }
    validate(value)
    try:
        validate(value, release=True)
    except ValueError as exc:
        assert "published SHA" in str(exc)
    else:
        raise AssertionError("release validation accepted a zero digest")
