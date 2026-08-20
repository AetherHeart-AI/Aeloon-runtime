"""Device pairing, token hashing, enrollment codes, and listen TLS.

The pairing store never keeps a token, only its SHA-256. An enrollment code is
a ten-character Crockford value that lives in memory for ten minutes and is
consumed once — restarting the process is supposed to forget it. Listen TLS is
generated here so the fingerprint in the pairing URL is the certificate the
client will pin.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import secrets
import socket
import ssl
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

SCHEMA_VERSION = 4
ENROLLMENT_TTL_S = 10 * 60
TOKEN_BYTES = 32
CODE_LENGTH = 10
CERT_DAYS = 825
AUTH_BACKOFF_INITIAL_S = 1.0
AUTH_BACKOFF_MAX_S = 30.0
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CROCKFORD_NORMALIZE = str.maketrans(
    {
        "I": "1",
        "L": "1",
        "O": "0",
        "i": "1",
        "l": "1",
        "o": "0",
    }
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_stamp(moment: datetime | None = None) -> str:
    value = moment or utc_now()
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def issue_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(TOKEN_BYTES)).decode("ascii").rstrip("=")


def generate_enrollment_code() -> str:
    # Ten Crockford characters are 50 bits. A LAN attacker still needs the
    # backoff on failed attempts; the code itself is only a short-lived secret.
    n = secrets.randbits(5 * CODE_LENGTH)
    chars = []
    for _ in range(CODE_LENGTH):
        chars.append(CROCKFORD_ALPHABET[n & 31])
        n >>= 5
    return "".join(reversed(chars))


def normalize_enrollment_code(value: str) -> str:
    return value.replace("-", "").replace(" ", "").translate(_CROCKFORD_NORMALIZE).upper()


def advertised_host(bind_host: str) -> str:
    if bind_host in {"", "0.0.0.0", "::", "[::]"}:
        return socket.gethostname()
    return bind_host


def is_loopback_host(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def pairing_url(*, host: str, port: int, fingerprint: str, code: str) -> str:
    endpoint_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    endpoint = f"wss://{endpoint_host}:{port}"
    query = urlencode(
        {
            "v": "2",
            "endpoint": endpoint,
            "fingerprint": fingerprint.lower(),
            "code": code,
        },
        safe=":",
    )
    return f"aeloon://pair?{query}"


def parse_pairing_url(value: str) -> dict[str, str]:
    text = value.strip()
    if not text.startswith("aeloon://pair?"):
        raise ValueError("Pairing string must start with aeloon://pair?")
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(text)
    if (
        parsed.scheme != "aeloon"
        or parsed.netloc != "pair"
        or parsed.path not in ("", "/")
        or parsed.fragment
    ):
        raise ValueError("Pairing string must start with aeloon://pair?")
    fields = {
        key: values[-1]
        for key, values in parse_qs(parsed.query, keep_blank_values=False).items()
    }
    if fields.get("v") != "2":
        raise ValueError("Pairing string version must be v=2")
    for key in ("endpoint", "fingerprint", "code"):
        if not fields.get(key):
            raise ValueError(f"Pairing string is missing {key}")
    try:
        endpoint = urlparse(fields["endpoint"])
    except ValueError:
        raise ValueError("Pairing string endpoint is invalid") from None
    if (
        endpoint.scheme != "wss"
        or endpoint.username
        or endpoint.password
        or endpoint.path not in ("", "/")
        or endpoint.params
        or endpoint.query
        or endpoint.fragment
        or not endpoint.hostname
        or endpoint.port is None
        or not 1 <= endpoint.port <= 65535
        or not _valid_pairing_host(endpoint.hostname)
    ):
        raise ValueError("Pairing string endpoint must be a bare wss:// URL")
    host = endpoint.hostname
    endpoint_host = f"[{host}]" if ":" in host else host
    fields["endpoint"] = f"wss://{endpoint_host}:{endpoint.port}"
    fingerprint = fields["fingerprint"].lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint):
        raise ValueError("Pairing string fingerprint must be sha256:<hex>")
    fields["fingerprint"] = fingerprint
    code = normalize_enrollment_code(fields["code"])
    if len(code) != CODE_LENGTH or any(char not in CROCKFORD_ALPHABET for char in code):
        raise ValueError("Pairing string code is invalid")
    fields["code"] = code
    return fields


def _valid_pairing_host(host: str) -> bool:
    if not host or len(host) > 253 or any(ord(char) < 0x21 for char in host):
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if "/" in host or "\\" in host or ":" in host:
        return False
    labels = host.rstrip(".").split(".")
    return bool(labels) and all(
        label
        and len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(char.isalnum() or char == "-" for char in label)
        for label in labels
    )


class DeviceStore:
    """JSON device list under the incompatible v4 store (mode 0600).

    The old ``devices.json`` file is intentionally never read. A destructive
    v4 upgrade requires every client to pair again.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve(strict=False)
        self.path = self.data_dir / "devices-v4.json"
        self._devices: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._devices = []
            return
        except (OSError, json.JSONDecodeError, TypeError):
            self._devices = []
            return
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            self._devices = []
            return
        records = payload.get("devices")
        self._devices = (
            [item for item in records if self._valid_record(item)]
            if isinstance(records, list)
            else []
        )

    @staticmethod
    def _valid_record(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and bool(item["id"].strip())
            and isinstance(item.get("token_sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", item["token_sha256"]))
            and isinstance(item.get("name"), str)
            and isinstance(item.get("platform"), str)
        )

    def has_devices(self) -> bool:
        return bool(self._devices)

    def list_devices(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._devices]

    def get(self, device_id: str) -> dict[str, Any] | None:
        for item in self._devices:
            if item.get("id") == device_id:
                return dict(item)
        return None

    def issue(self, *, name: str, platform: str) -> tuple[dict[str, Any], str]:
        token = issue_token()
        now = utc_stamp()
        record = {
            "id": str(uuid.uuid4()),
            "name": name.strip() or "device",
            "platform": platform.strip() or "unknown",
            "token_sha256": hash_token(token),
            "paired_at": now,
            "last_seen_at": now,
        }
        self._devices.append(record)
        try:
            self.save()
        except Exception:
            self._devices.pop()
            raise
        return dict(record), token

    def verify(self, token: str) -> dict[str, Any] | None:
        digest = hash_token(token)
        for item in self._devices:
            if secrets.compare_digest(str(item.get("token_sha256") or ""), digest):
                return dict(item)
        return None

    def touch(self, device_id: str) -> None:
        now = utc_stamp()
        updated = [dict(item) for item in self._devices]
        for item in updated:
            if item.get("id") == device_id:
                item["last_seen_at"] = now
                self._save(updated)
                self._devices = updated
                return

    def revoke(self, device_id: str) -> bool:
        updated = [item for item in self._devices if item.get("id") != device_id]
        if len(updated) == len(self._devices):
            return False
        self._save(updated)
        self._devices = updated
        return True

    def save(self) -> None:
        self._save(self._devices)

    def _save(self, devices: list[dict[str, Any]]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.data_dir.chmod(0o700)
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "devices": devices},
            ensure_ascii=False,
            indent=2,
        )
        fd, temporary = tempfile.mkstemp(prefix=".devices.", dir=self.data_dir)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                fd = -1
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            directory_fd = os.open(self.data_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise


class EnrollmentVault:
    """One in-memory enrollment code. Restarting the process forgets it."""

    def __init__(self, *, ttl_s: float = ENROLLMENT_TTL_S) -> None:
        self.ttl_s = ttl_s
        self._code: str | None = None
        self._expires_at: float = 0.0

    def issue(self) -> tuple[str, str]:
        self._code = generate_enrollment_code()
        expires = utc_now() + timedelta(seconds=self.ttl_s)
        self._expires_at = expires.timestamp()
        return self._code, utc_stamp(expires)

    def consume(self, code: str) -> bool:
        offered = normalize_enrollment_code(code)
        current = self._code
        if current is None or time.time() >= self._expires_at:
            self._code = None
            return False
        if not secrets.compare_digest(normalize_enrollment_code(current), offered):
            return False
        self._code = None
        self._expires_at = 0.0
        return True

    def peek(self) -> str | None:
        if self._code is None or time.time() >= self._expires_at:
            return None
        return self._code


class AuthLimiter:
    """Exponential backoff for consecutive authentication failures from one source."""

    def __init__(
        self,
        *,
        initial_s: float = AUTH_BACKOFF_INITIAL_S,
        maximum_s: float = AUTH_BACKOFF_MAX_S,
    ) -> None:
        self.initial_s = initial_s
        self.maximum_s = maximum_s
        self._failures: dict[str, int] = {}
        self._blocked_until: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def lock_for(self, source: str) -> asyncio.Lock:
        return self._locks.setdefault(source, asyncio.Lock())

    def delay_for(self, source: str) -> float:
        remaining = self._blocked_until.get(source, 0.0) - time.monotonic()
        return remaining if remaining > 0 else 0.0

    def record_failure(self, source: str) -> float:
        count = self._failures.get(source, 0) + 1
        self._failures[source] = count
        delay = min(self.maximum_s, self.initial_s * (2 ** (count - 1)))
        self._blocked_until[source] = time.monotonic() + delay
        return delay

    def record_success(self, source: str) -> None:
        self._failures.pop(source, None)
        self._blocked_until.pop(source, None)


def certificate_fingerprint(cert_path: Path) -> str:
    data = cert_path.read_bytes()
    try:
        cert = x509.load_pem_x509_certificate(data)
    except ValueError:
        cert = x509.load_der_x509_certificate(data)
    digest = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return f"sha256:{digest}"


def fingerprint_der(der: bytes) -> str:
    return f"sha256:{hashlib.sha256(der).hexdigest()}"


def _san_entries(bind_host: str) -> x509.SubjectAlternativeName:
    names: list[x509.GeneralName] = []
    seen_dns: set[str] = set()
    seen_ip: set[str] = set()

    def add_dns(value: str) -> None:
        if value and value not in seen_dns:
            seen_dns.add(value)
            names.append(x509.DNSName(value))

    def add_ip(value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
        text = str(value)
        if text not in seen_ip:
            seen_ip.add(text)
            names.append(x509.IPAddress(value))

    candidates = [bind_host, advertised_host(bind_host), "localhost"]
    for candidate in candidates:
        if not candidate or candidate in {"0.0.0.0", "::", "[::]"}:
            continue
        try:
            address = ipaddress.ip_address(candidate.strip("[]"))
        except ValueError:
            add_dns(candidate)
        else:
            if not address.is_unspecified:
                add_ip(address)
    add_ip(ipaddress.ip_address("127.0.0.1"))
    add_ip(ipaddress.ip_address("::1"))
    return x509.SubjectAlternativeName(names)


def generate_self_signed_cert(tls_dir: Path, bind_host: str) -> tuple[Path, Path]:
    tls_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    tls_dir.chmod(0o700)
    cert_path = tls_dir / "cert.pem"
    key_path = tls_dir / "key.pem"
    key = ec.generate_private_key(ec.SECP256R1())
    host = advertised_host(bind_host)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
    now = utc_now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=CERT_DAYS))
        .add_extension(_san_entries(bind_host), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_bytes = cert.public_bytes(serialization.Encoding.PEM)
    _write_secret(key_path, key_bytes)
    _write_secret(cert_path, cert_bytes)
    return cert_path, key_path


def ensure_tls_files(
    data_dir: Path,
    bind_host: str,
    *,
    certificate: Path | None = None,
    key: Path | None = None,
) -> tuple[Path, Path]:
    if certificate is not None or key is not None:
        if certificate is None or key is None:
            raise ValueError("--tls-cert and --tls-key must be provided together")
        return certificate, key
    tls_dir = data_dir.expanduser().resolve(strict=False) / "tls"
    cert_path = tls_dir / "cert.pem"
    key_path = tls_dir / "key.pem"
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path
    return generate_self_signed_cert(tls_dir, bind_host)


def build_server_tls_context(certificate: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(certificate), keyfile=str(key))
    return context


def _write_secret(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


class PairingState:
    """Per-Runtime pairing store, enrollment vault, limiter, and listen TLS."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve(strict=False)
        self.store = DeviceStore(self.data_dir)
        self.enrollment = EnrollmentVault()
        self.limiter = AuthLimiter()
        self.certificate: Path | None = None
        self.key: Path | None = None
        self.fingerprint: str | None = None
        self.listen_host: str | None = None
        self.listen_port: int | None = None
        self.advertise_url: str | None = None

    def prepare_listen(
        self,
        host: str,
        port: int,
        *,
        certificate: Path | None = None,
        key: Path | None = None,
        advertise_url: str | None = None,
    ) -> ssl.SSLContext:
        cert_path, key_path = ensure_tls_files(
            self.data_dir, host, certificate=certificate, key=key
        )
        self.certificate = cert_path
        self.key = key_path
        self.fingerprint = certificate_fingerprint(cert_path)
        self.advertise_url = _normalize_advertise_url(advertise_url) if advertise_url else None
        if self.advertise_url:
            advertised = urlparse(self.advertise_url)
            self.listen_host = advertised.hostname
            self.listen_port = advertised.port
        else:
            self.listen_host = advertised_host(host)
            self.listen_port = port
        return build_server_tls_context(cert_path, key_path)

    def issue_enrollment(self) -> tuple[str, str, str]:
        if self.listen_host is None or self.listen_port is None or self.fingerprint is None:
            raise RuntimeError("Runtime is not listening; start with --listen to enroll a device")
        code, expires_at = self.enrollment.issue()
        url = pairing_url(
            host=self.listen_host,
            port=self.listen_port,
            fingerprint=self.fingerprint,
            code=code,
        )
        return code, expires_at, url


def _normalize_advertise_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme != "wss"
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not parsed.hostname
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
        or not _valid_pairing_host(parsed.hostname)
    ):
        raise ValueError("--advertise-url must be a bare wss://host:port URL")
    host = parsed.hostname
    endpoint_host = f"[{host}]" if ":" in host else host
    return f"wss://{endpoint_host}:{parsed.port}"


__all__ = [
    "AUTH_BACKOFF_INITIAL_S",
    "AUTH_BACKOFF_MAX_S",
    "AuthLimiter",
    "DeviceStore",
    "EnrollmentVault",
    "PairingState",
    "advertised_host",
    "build_server_tls_context",
    "certificate_fingerprint",
    "ensure_tls_files",
    "fingerprint_der",
    "generate_enrollment_code",
    "generate_self_signed_cert",
    "hash_token",
    "is_loopback_host",
    "issue_token",
    "normalize_enrollment_code",
    "pairing_url",
    "parse_pairing_url",
    "utc_stamp",
]
