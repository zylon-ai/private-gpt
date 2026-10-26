import io
import ipaddress
import socket
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlparse, urlsplit

import requests
import urllib3

from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import settings

DEFAULT_URL_TIMEOUT_SECONDS: tuple[float, float] = (10.0, 60.0)
_DOWNLOAD_CHUNK_SIZE = 64 * 1024
_MAX_REDIRECTS = 5


def _public_url_target(url: str) -> tuple[str, str, int, str, str]:
    """Resolve once and reject every nonpublic address before connecting."""
    if (
        url != url.strip()
        or "\\" in url
        or any(ord(c) < 32 or ord(c) == 127 for c in url)
    ):
        raise ValueError("Invalid remote file URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "%" in parsed.hostname
    ):
        raise ValueError("Remote files require an HTTP(S) URL without credentials")

    # Use the same URL normalization as Requests (IDNA, quoting, IPv6).
    prepared = requests.Request("GET", url).prepare()
    parsed = urlsplit(prepared.url)
    host = parsed.hostname
    if not host:
        raise ValueError("Remote file URL requires a hostname")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("Remote file hostname did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            raise ValueError("Remote file URLs must resolve only to public addresses")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return parsed.scheme, host, port, addresses[0][4][0], target


def _load_file_from_url(
    url: str,
    *,
    timeout: float | tuple[float, float] | None = None,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> BinaryIO:
    """Download a public HTTP(S) URL with bounded redirects, time and size.

    Each hop connects to its validated IP, preventing DNS rebinding. Direct
    pools deliberately ignore environment proxies and netrc credentials; HTTPS
    still verifies the original hostname and sends it as SNI and Host.
    """
    del kwargs
    if timeout is None:
        timeout = DEFAULT_URL_TIMEOUT_SECONDS
    if max_bytes is None:
        max_bytes = settings().chat.maximum_blob_size

    request_timeout = (
        urllib3.Timeout(connect=timeout[0], read=timeout[1])
        if isinstance(timeout, tuple)
        else urllib3.Timeout(connect=timeout, read=timeout)
    )
    for hop in range(_MAX_REDIRECTS + 1):
        scheme, host, port, address, target = _public_url_target(url)
        pool = (
            urllib3.HTTPSConnectionPool(
                address,
                port,
                server_hostname=host,
                assert_hostname=host,
                cert_reqs="CERT_REQUIRED",
                ca_certs=requests.certs.where(),
            )
            if scheme == "https"
            else urllib3.HTTPConnectionPool(address, port)
        )
        authority = f"[{host}]" if ":" in host else host
        if port != (443 if scheme == "https" else 80):
            authority += f":{port}"
        with (
            pool,
            pool.urlopen(
                "GET",
                target,
                headers={"Host": authority},
                redirect=False,
                retries=False,
                preload_content=False,
                timeout=request_timeout,
            ) as response,
        ):
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location or hop == _MAX_REDIRECTS:
                    raise ValueError("Invalid or excessive remote file redirects")
                url = urljoin(url, location)
                continue
            if response.status >= 400:
                raise requests.HTTPError(f"Remote file returned HTTP {response.status}")
            if 300 <= response.status < 400:
                raise ValueError("Unexpected remote file redirect")
            buffer = io.BytesIO()
            total = 0
            for chunk in response.stream(_DOWNLOAD_CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"Remote file exceeds the maximum allowed size of {max_bytes} bytes."
                    )
                buffer.write(chunk)
            buffer.seek(0)
            return buffer
    raise ValueError("Excessive remote file redirects")


def _load_file_from_base64(base64_str: str, **kwargs: Any) -> BinaryIO:
    import base64

    # Handle data URI format: data:image/png;base64,iVBORw0KG...
    if base64_str.startswith("data:"):
        _, encoded = base64_str.split(",", 1)
        content = base64.b64decode(encoded)
    else:
        content = base64.b64decode(base64_str.strip())

    return io.BytesIO(content)


def _load_file_from_disk(url: str, **kwargs: Any) -> BinaryIO:
    with open(url, "rb") as source:
        content = source.read()
    return io.BytesIO(content)


def _is_url(uri: str) -> bool:
    try:
        result = urlparse(uri)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def _is_base64(uri: str) -> bool:
    import base64

    if uri.startswith("data:"):
        return True

    try:
        base64.b64decode(uri.strip(), validate=True)
        return True
    except Exception:
        return False


def load_file_from_uri(
    uri: str, *, allow_local: bool = False, **kwargs: Any
) -> BinaryIO:
    """Load remote or encoded content; disk reads require a trusted caller."""
    if uri.startswith("s3://"):
        # S3 URI
        s3_helper = get_global_injector().get(S3Helper)
        return s3_helper.load_file_from_s3(uri, **kwargs)
    elif _is_url(uri):
        # Public URL
        return _load_file_from_url(uri, **kwargs)
    elif _is_base64(uri):
        # Base64 encoded data
        return _load_file_from_base64(uri, **kwargs)
    elif allow_local:
        # Only server-owned development mount sources may read local files.
        return _load_file_from_disk(uri, **kwargs)
    raise ValueError("Local file paths and unsupported URI schemes are not allowed")
