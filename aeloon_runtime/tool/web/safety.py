from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlsplit


async def validate_url_target(url: str) -> str:
    # This preflight blocks ordinary SSRF targets. httpx resolves the host again when
    # connecting, so a hostile authoritative DNS server can still attempt rebinding;
    # fully closing that TOCTOU gap requires a transport pinned to the validated IP.
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Only public HTTP(S) URLs are allowed")
    try:
        addresses = [ipaddress.ip_address(parsed.hostname)]
    except ValueError:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, parsed.port or 443)
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    if any(not address.is_global for address in addresses):
        raise ValueError("Private, local, and reserved network targets are not allowed")
    return url
