from __future__ import annotations

import json
from pathlib import Path

from aeloon_runtime.pairing import (
    AuthLimiter,
    DeviceStore,
    EnrollmentVault,
    certificate_fingerprint,
    generate_self_signed_cert,
    hash_token,
    is_loopback_host,
    issue_token,
    normalize_enrollment_code,
    pairing_url,
    parse_pairing_url,
)


def test_device_store_persists_hashes_not_tokens(tmp_path: Path) -> None:
    store = DeviceStore(tmp_path)
    record, token = store.issue(name="MacBook", platform="darwin")
    payload = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert token not in (tmp_path / "devices.json").read_text(encoding="utf-8")
    assert payload["devices"][0]["token_sha256"] == hash_token(token)
    assert store.verify(token)["id"] == record["id"]
    assert store.verify("not-the-token") is None
    assert (tmp_path / "devices.json").stat().st_mode & 0o777 == 0o600


def test_enrollment_code_is_single_use_and_in_memory() -> None:
    vault = EnrollmentVault(ttl_s=60)
    code, expires_at = vault.issue()
    assert len(code) == 10
    assert expires_at.endswith("Z")
    assert vault.consume(code.lower()) is True
    assert vault.consume(code) is False


def test_enrollment_normalizes_crockford_lookalikes() -> None:
    assert normalize_enrollment_code("ilo-") == "110"
    vault = EnrollmentVault(ttl_s=60)
    vault._code = "1100000000"
    vault._expires_at = 10**18
    assert vault.consume("ILOOOOOOOO") is True


def test_auth_limiter_grows_to_cap() -> None:
    limiter = AuthLimiter(initial_s=1, maximum_s=30)
    assert limiter.delay_for("10.0.0.1") == 0
    assert limiter.record_failure("10.0.0.1") == 1
    assert limiter.record_failure("10.0.0.1") == 2
    assert limiter.record_failure("10.0.0.1") == 4
    while limiter.record_failure("10.0.0.1") < 30:
        pass
    assert limiter.record_failure("10.0.0.1") == 30
    limiter.record_success("10.0.0.1")
    assert limiter.delay_for("10.0.0.1") == 0


def test_self_signed_certificate_fingerprint_and_pairing_url(tmp_path: Path) -> None:
    cert, key = generate_self_signed_cert(tmp_path / "tls", "127.0.0.1")
    assert cert.stat().st_mode & 0o777 == 0o600
    assert key.stat().st_mode & 0o777 == 0o600
    fingerprint = certificate_fingerprint(cert)
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == len("sha256:") + 64
    assert fingerprint == fingerprint.lower()
    url = pairing_url(host="127.0.0.1", port=7420, fingerprint=fingerprint, code="0123456789")
    parsed = parse_pairing_url(url)
    assert parsed["host"] == "127.0.0.1"
    assert parsed["port"] == "7420"
    assert parsed["fp"] == fingerprint
    assert parsed["code"] == "0123456789"


def test_loopback_detection() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.5")


def test_issued_token_is_unpadded_base64url() -> None:
    token = issue_token()
    assert "=" not in token
    assert len(token) >= 43
