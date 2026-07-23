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
