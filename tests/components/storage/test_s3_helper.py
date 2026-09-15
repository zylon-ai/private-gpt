from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import unquote

from private_gpt.components.storage.s3_helper import S3Helper


def test_upload_file_encodes_unicode_filename_in_metadata() -> None:
    s3_client = MagicMock()
    helper = S3Helper.__new__(S3Helper)
    helper._s3_client = s3_client

    result = helper.upload_file_to_s3(
        filename="sámple_¡™£¢∞§.txt",
        bytes_data=b"content",
        bucket_name="test-bucket",
        object_name="object-id",
    )

    put_args = s3_client.put_object.call_args.kwargs
    encoded_filename = put_args["Metadata"]["file_name"]
    assert encoded_filename.isascii()
    assert unquote(encoded_filename) == "sámple_¡™£¢∞§.txt"
    assert result == "s3://test-bucket/object-id"


async def test_async_upload_file_encodes_unicode_filename_in_metadata() -> None:
    s3_client = MagicMock()
    s3_client.put_object = AsyncMock()

    @asynccontextmanager
    async def get_async_s3_client() -> Any:
        yield s3_client

    helper = S3Helper.__new__(S3Helper)
    helper._get_async_s3_client = get_async_s3_client  # type: ignore[method-assign]

    result = await helper.async_upload_file_to_s3(
        filename="sámple_¡™£¢∞§.txt",
        bytes_data=b"content",
        bucket_name="test-bucket",
        object_name="object-id",
    )

    put_args = s3_client.put_object.await_args.kwargs
    encoded_filename = put_args["Metadata"]["file_name"]
    assert encoded_filename.isascii()
    assert unquote(encoded_filename) == "sámple_¡™£¢∞§.txt"
    assert result == "s3://test-bucket/object-id"


def test_sync_s3_client_is_built_with_timeouts(monkeypatch: Any) -> None:
    import boto3

    from private_gpt.settings.settings import S3Settings

    captured: dict[str, Any] = {}

    def fake_client(*args: Any, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(boto3, "client", fake_client)
    monkeypatch.setattr(boto3, "set_stream_logger", lambda *a, **k: None)
    s3_settings = S3Settings(
        endpoint_url="http://s3.local",
        public_endpoint_url="http://s3.local",
        access_key_id="key",
        secret_access_key="secret",
        durable_bucket_name="durable",
        temporary_bucket_name="temporary",
        connect_timeout_seconds=7,
        read_timeout_seconds=42,
    )

    S3Helper._get_s3_client(s3_settings)

    config = captured.get("config")
    assert config is not None, "boto3 client built without a botocore Config"
    assert config.connect_timeout == 7
    assert config.read_timeout == 42


def test_async_s3_client_is_built_with_timeouts(monkeypatch: Any) -> None:
    import aiobotocore.session

    from private_gpt.settings.settings import S3Settings

    captured: dict[str, Any] = {}
    session = MagicMock()

    def create_client(*args: Any, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    session.create_client.side_effect = create_client
    monkeypatch.setattr(aiobotocore.session, "get_session", lambda: session)

    helper = S3Helper.__new__(S3Helper)
    helper._s3_settings = S3Settings(
        endpoint_url="http://s3.local",
        public_endpoint_url="http://s3.local",
        access_key_id="key",
        secret_access_key="secret",
        durable_bucket_name="durable",
        temporary_bucket_name="temporary",
        connect_timeout_seconds=7,
        read_timeout_seconds=42,
    )

    helper._get_async_s3_client()

    config = captured.get("config")
    assert config is not None, "aiobotocore client built without an AioConfig"
    assert config.connect_timeout == 7
    assert config.read_timeout == 42
