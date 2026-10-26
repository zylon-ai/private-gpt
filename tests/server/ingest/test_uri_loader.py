"""URI loading rejects SSRF before connecting and preserves bounded downloads."""

import io
import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import urllib3

from private_gpt.server.ingest.uri_loader import load_file_from_uri

PUBLIC_IP = "93.184.216.34"


def _addresses(*ips: str) -> list[tuple]:
    return [
        (
            socket.AF_INET6 if ":" in ip else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (ip, 443),
        )
        for ip in ips
    ]


def _response(
    status: int = 200,
    chunks: tuple[bytes, ...] = (b"body",),
    location: str | None = None,
) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.headers = {"Location": location} if location else {}
    response.stream.return_value = iter(chunks)
    response.__enter__.return_value = response
    return response


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    monkeypatch.setattr(
        socket, "getaddrinfo", MagicMock(return_value=_addresses(PUBLIC_IP))
    )
    pool = MagicMock()
    pool.__enter__.return_value = pool
    factory = MagicMock(return_value=pool)
    monkeypatch.setattr(urllib3, "HTTPConnectionPool", factory)
    monkeypatch.setattr(urllib3, "HTTPSConnectionPool", factory)
    # No test may accidentally contact an external or internal HTTP service.
    monkeypatch.setattr(
        requests, "get", MagicMock(side_effect=AssertionError("unvalidated transport"))
    )
    return factory, pool


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "255.255.255.255",
        "224.0.0.1",
        "192.0.2.1",
        "::1",
        "::",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
def test_private_destination_rejected_before_connect(
    address: str,
    transport: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = transport
    monkeypatch.setattr(
        socket, "getaddrinfo", MagicMock(return_value=_addresses(address))
    )
    host = f"[{address}]" if ":" in address else address
    with pytest.raises(ValueError, match="public addresses"):
        load_file_from_uri(f"http://{host}/internal")
    factory.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://files.example.com/test",
        "file:///etc/passwd",
        "/etc/passwd",
        "../../etc/passwd",
        "http://user:secret@files.example.com/test",
        "http://files.example.com\\@localhost/test",
        "http://[fe80::1%25en0]/",
        "https://files.example.com/\r\nInjected: true",
    ],
)
def test_invalid_uri_rejected_before_connect(
    url: str, transport: tuple[MagicMock, MagicMock]
) -> None:
    factory, _ = transport
    with pytest.raises(ValueError):
        load_file_from_uri(url)
    factory.assert_not_called()


@pytest.mark.parametrize("answers", [(PUBLIC_IP, "10.0.0.1"), ("::1", PUBLIC_IP), ()])
def test_all_dns_answers_must_be_public(
    answers: tuple[str, ...],
    transport: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = transport
    monkeypatch.setattr(
        socket, "getaddrinfo", MagicMock(return_value=_addresses(*answers))
    )
    with pytest.raises(ValueError):
        load_file_from_uri("https://files.example.com/test")
    factory.assert_not_called()


def test_dns_failure_does_not_connect(
    transport: tuple[MagicMock, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, _ = transport
    monkeypatch.setattr(
        socket, "getaddrinfo", MagicMock(side_effect=socket.gaierror("no answer"))
    )
    with pytest.raises(socket.gaierror):
        load_file_from_uri("https://files.example.com/test")
    factory.assert_not_called()


def test_connection_pins_dns_and_preserves_https_identity(
    transport: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, pool = transport
    resolver = MagicMock(side_effect=[_addresses(PUBLIC_IP), _addresses("127.0.0.1")])
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    pool.urlopen.return_value = _response(chunks=(b"ok",))
    assert (
        load_file_from_uri("https://files.example.com:8443/a%20b?q=1#fragment").read()
        == b"ok"
    )
    assert resolver.call_count == 1
    assert factory.call_args.args == (PUBLIC_IP, 8443)
    assert factory.call_args.kwargs == {
        "server_hostname": "files.example.com",
        "assert_hostname": "files.example.com",
        "cert_reqs": "CERT_REQUIRED",
        "ca_certs": requests.certs.where(),
    }
    args, kwargs = pool.urlopen.call_args
    assert args == ("GET", "/a%20b?q=1")
    assert kwargs["headers"] == {"Host": "files.example.com:8443"}
    assert kwargs["redirect"] is False
    assert kwargs["retries"] is False
    assert kwargs["preload_content"] is False
    assert kwargs["timeout"].connect_timeout == 10
    assert kwargs["timeout"].read_timeout == 60


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/secret",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ],
)
def test_redirect_cannot_reach_private_or_local_target(
    location: str,
    transport: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, pool = transport
    resolver = MagicMock(side_effect=[_addresses(PUBLIC_IP), _addresses("127.0.0.1")])
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    response = _response(302, location=location)
    pool.urlopen.return_value = response
    with pytest.raises(ValueError):
        load_file_from_uri("https://files.example.com/start")
    assert factory.call_count == 1
    assert pool.urlopen.call_count == 1
    response.__exit__.assert_called_once()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_public_relative_redirect_is_revalidated(
    status: int, transport: tuple[MagicMock, MagicMock]
) -> None:
    _, pool = transport
    pool.urlopen.side_effect = [
        _response(status, location="../final?token=ok"),
        _response(chunks=(b"download",)),
    ]
    assert load_file_from_uri("https://files.example.com/a/start").read() == b"download"
    assert socket.getaddrinfo.call_count == 2
    assert pool.urlopen.call_args.args == ("GET", "/final?token=ok")


def test_redirect_loop_is_bounded(transport: tuple[MagicMock, MagicMock]) -> None:
    _, pool = transport
    pool.urlopen.side_effect = [_response(302, location="/again") for _ in range(6)]
    with pytest.raises(ValueError, match="redirects"):
        load_file_from_uri("https://files.example.com/start")
    assert pool.urlopen.call_count == 6


def test_url_loader_rejects_http_error_pages(
    transport: tuple[MagicMock, MagicMock],
) -> None:
    _, pool = transport
    response = _response(404, (b"not found",))
    pool.urlopen.return_value = response
    with pytest.raises(requests.HTTPError):
        load_file_from_uri("https://files.example.com/test")
    response.stream.assert_not_called()
    response.__exit__.assert_called_once()


def test_url_loader_enforces_size_limit_while_streaming(
    transport: tuple[MagicMock, MagicMock],
) -> None:
    _, pool = transport
    response = _response(chunks=(b"x" * 1024,) * 8)
    pool.urlopen.return_value = response
    with pytest.raises(ValueError, match="exceeds"):
        load_file_from_uri("https://files.example.com/big.bin", max_bytes=2048)
    assert next(response.stream.return_value) == b"x" * 1024
    response.__exit__.assert_called_once()


def test_url_loader_returns_body_within_limit(
    transport: tuple[MagicMock, MagicMock],
) -> None:
    _, pool = transport
    pool.urlopen.return_value = _response(chunks=(b"abc", b"", b"def"))
    data = load_file_from_uri(
        "https://files.example.com/ok.txt", max_bytes=6, timeout=(1.5, 2.5)
    )
    assert isinstance(data, io.BytesIO)
    assert data.read() == b"abcdef"
    assert pool.urlopen.call_args.kwargs["timeout"].connect_timeout == 1.5
    assert pool.urlopen.call_args.kwargs["timeout"].read_timeout == 2.5


@pytest.mark.parametrize("uri", ["aGVsbG8=", "data:text/plain;base64,aGVsbG8="])
def test_base64_compatibility(uri: str, transport: tuple[MagicMock, MagicMock]) -> None:
    factory, _ = transport
    assert load_file_from_uri(uri).read() == b"hello"
    factory.assert_not_called()


def test_s3_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    injector = MagicMock()
    injector.get.return_value.load_file_from_s3.return_value = io.BytesIO(b"s3 data")
    monkeypatch.setattr(
        "private_gpt.server.ingest.uri_loader.get_global_injector", lambda: injector
    )
    assert load_file_from_uri("s3://bucket/key").read() == b"s3 data"
    injector.get.return_value.load_file_from_s3.assert_called_once_with(
        "s3://bucket/key"
    )


def test_local_disk_requires_explicit_trusted_caller(tmp_path: Path) -> None:
    source = tmp_path / "local.txt"
    source.write_bytes(b"local data")
    with pytest.raises(ValueError, match="Local file paths"):
        load_file_from_uri(str(source))
    assert load_file_from_uri(str(source), allow_local=True).read() == b"local data"


@pytest.mark.asyncio
async def test_trusted_development_mount_still_loads_local_file(tmp_path: Path) -> None:
    from private_gpt.components.sandbox.mount import UriSource

    source = tmp_path / "local.txt"
    source.write_bytes(b"local mount")
    files = await UriSource.from_uri(str(source), allow_local=True).fetch()
    assert len(files) == 1
    assert files[0].content == b"local mount"


@pytest.mark.parametrize("unreachable_first", [False, True])
def test_real_http_transport_connects_only_to_validated_ip(
    monkeypatch: pytest.MonkeyPatch,
    unreachable_first: bool,
) -> None:
    """Exercise actual urllib3, redirect handling, streaming and Host headers.

    A socket shim maps the synthetic public IP to a local test server. Production
    destination validation stays enabled; every attempted connection is recorded.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from urllib3.util import connection

    requests_seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests_seen.append((self.path, self.headers.get("Host")))
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/final?q=1")
                self.end_headers()
            elif self.path == "/private":
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1/secret")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"downloaded bytes")

        def log_message(self, *_: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_resolver = socket.getaddrinfo
    original_connect = connection.create_connection
    attempted: list[tuple[str, int]] = []

    def resolve(host: str, port: int, *args: object, **kwargs: object) -> list[tuple]:
        if host == "files.example.com":
            return (
                _addresses(PUBLIC_IP, "1.1.1.1")
                if unreachable_first
                else _addresses(PUBLIC_IP)
            )
        return original_resolver(host, port, *args, **kwargs)

    def connect(
        address: tuple[str, int], *args: object, **kwargs: object
    ) -> socket.socket:
        attempted.append(address)
        assert address[0] in {PUBLIC_IP, "1.1.1.1"}
        assert address[1] == server.server_port
        if unreachable_first and address[0] == PUBLIC_IP:
            raise ConnectionRefusedError("synthetic unreachable first public address")
        return original_connect(("127.0.0.1", server.server_port), *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(connection, "create_connection", connect)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    try:
        base = f"http://files.example.com:{server.server_port}"
        assert load_file_from_uri(base + "/start").read() == b"downloaded bytes"
        with pytest.raises(ValueError, match="public addresses"):
            load_file_from_uri(base + "/private")
        assert len(attempted) == (6 if unreachable_first else 3)
        assert requests_seen == [
            ("/start", f"files.example.com:{server.server_port}"),
            ("/final?q=1", f"files.example.com:{server.server_port}"),
            ("/private", f"files.example.com:{server.server_port}"),
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "uri", ["/etc/passwd", "file:///etc/passwd", "http://127.0.0.1/internal"]
)
def test_uri_artifact_entry_point_rejects_unsafe_sources(
    uri: str,
    transport: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from private_gpt.server.utils.artifact_input import UriArtifact

    factory, _ = transport
    monkeypatch.setattr(
        socket, "getaddrinfo", MagicMock(return_value=_addresses("127.0.0.1"))
    )
    with pytest.raises(ValueError):
        UriArtifact(value=uri).to_binary_content()
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "file",
        "file_uri",
        "directory",
        "directory_uri",
        "file_symlink",
        "directory_symlink",
    ],
)
async def test_api_mount_cannot_hydrate_from_host_files(
    tmp_path: Path, kind: str
) -> None:
    """The actual API MountEntry path never inherits local source privileges."""
    from types import SimpleNamespace

    from private_gpt.components.environment.hydration import HydratingEnvironmentManager
    from private_gpt.components.filesystems.mount_entry import MountEntry
    from private_gpt.components.filesystems.mount_resolver import MountResolver
    from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
    from private_gpt.components.filesystems.path_resolver import PathResolver
    from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "synthetic.txt"
    secret.write_bytes(b"SYNTHETIC_ONLY")
    root = tmp_path / "namespace"
    root.mkdir()
    config = NamespaceConfig(root=str(root), hydration=True)
    registry = NamespaceRegistry(
        settings=SimpleNamespace(
            filesystems=FilesystemsSettings(namespaces={"artifacts": config})
        )
    )
    resolver = MountResolver(
        registry=registry, resolver=PathResolver(registry=registry)
    )
    source = outside if kind.startswith("directory") else secret
    if kind.endswith("_symlink"):
        alias = root / "source-alias"
        alias.symlink_to(source)
        source = alias
    uri = source.as_uri() if kind.endswith("_uri") else str(source)
    entry = MountEntry(
        namespace="artifacts",
        scope="thread",
        path="content",
        uri=uri,
        target="/mnt/content/" if kind.startswith("directory") else "/mnt/content.txt",
    )
    mounts = resolver.resolve([entry])
    assert len(mounts) == 1
    delegate = MagicMock()
    manager = HydratingEnvironmentManager(
        manager=delegate, namespaces={"artifacts": config}
    )
    with pytest.raises(ValueError, match="Local file paths"):
        await manager.acquire("session", mounts)
    delegate.acquire.assert_not_called()
    assert not (root / "thread" / "content").exists()
    assert secret.read_bytes() == b"SYNTHETIC_ONLY"


@pytest.mark.asyncio
@pytest.mark.parametrize("as_uri", [False, True])
async def test_server_owned_local_directory_requires_explicit_opt_in(
    tmp_path: Path, as_uri: bool
) -> None:
    from private_gpt.components.sandbox.mount import UriSource

    directory = tmp_path / "server-owned"
    directory.mkdir()
    (directory / "a.txt").write_bytes(b"trusted directory")
    source = directory.as_uri() if as_uri else str(directory)
    with pytest.raises(ValueError, match="Local file paths"):
        await UriSource.from_uri(source).fetch()
    files = await UriSource.from_uri(source, allow_local=True).fetch()
    assert [(f.path, f.content) for f in files] == [("a.txt", b"trusted directory")]


@pytest.mark.asyncio
async def test_serialized_uri_source_cannot_enable_local_reads(tmp_path: Path) -> None:
    from private_gpt.components.sandbox.mount import UriSource

    source = tmp_path / "local.txt"
    source.write_bytes(b"server owned")
    trusted = UriSource.from_uri(str(source), allow_local=True)
    data = trusted.model_dump()
    assert "allow_local" not in data
    data["allow_local"] = True
    restored = UriSource.model_validate(data)
    with pytest.raises(ValueError, match="Local file paths"):
        await restored.fetch()


def test_https_fqdn_retains_host_but_normalizes_tls_identity(
    transport: tuple[MagicMock, MagicMock],
) -> None:
    factory, pool = transport
    pool.urlopen.return_value = _response()
    assert load_file_from_uri("https://files.example.com./test").read() == b"body"
    assert factory.call_args.kwargs["server_hostname"] == "files.example.com"
    assert factory.call_args.kwargs["assert_hostname"] == "files.example.com"
    assert factory.call_args.kwargs["cert_reqs"] == "CERT_REQUIRED"
    assert pool.urlopen.call_args.kwargs["headers"] == {"Host": "files.example.com."}


def test_all_validated_connection_addresses_exhausted_without_dns_retry(
    transport: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, pool = transport
    resolver = MagicMock(return_value=_addresses(PUBLIC_IP, "1.1.1.1"))
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    pool.urlopen.side_effect = urllib3.exceptions.NewConnectionError(
        None, "synthetic connection failure"
    )
    with pytest.raises(urllib3.exceptions.NewConnectionError):
        load_file_from_uri("https://files.example.com/test")
    assert resolver.call_count == 1
    assert [call.args[0] for call in factory.call_args_list] == [PUBLIC_IP, "1.1.1.1"]


def test_body_read_failure_does_not_retry_another_address(
    transport: tuple[MagicMock, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, pool = transport
    monkeypatch.setattr(
        socket, "getaddrinfo", MagicMock(return_value=_addresses(PUBLIC_IP, "1.1.1.1"))
    )
    response = _response()
    response.stream.side_effect = urllib3.exceptions.ReadTimeoutError(
        None, None, "synthetic read timeout"
    )
    pool.urlopen.return_value = response
    with pytest.raises(urllib3.exceptions.ReadTimeoutError):
        load_file_from_uri("https://files.example.com/test")
    assert factory.call_count == 1
    response.__exit__.assert_called_once()
