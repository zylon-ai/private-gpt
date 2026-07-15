from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any, BinaryIO

from injector import inject, singleton

from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

    from private_gpt.settings.settings import S3Settings


DEBUG_MODE = settings().server.debug_mode


@singleton
class S3Helper:
    _s3_client: S3Client | None
    _s3_settings: S3Settings

    @inject
    def __init__(self) -> None:
        self._s3_settings = settings().s3
        self._s3_client = self._get_s3_client(self._s3_settings)

    def is_available(self) -> bool:
        return self._s3_client is not None

    @staticmethod
    def _get_s3_client(s3_settings: S3Settings) -> S3Client | None:
        if (
            not s3_settings
            or not s3_settings.access_key_id
            or not s3_settings.secret_access_key
            or not s3_settings.endpoint_url
        ):
            return None

        import boto3  # ty:ignore[unresolved-import]

        boto3.set_stream_logger(
            "botocore", logging.INFO if DEBUG_MODE else logging.ERROR
        )
        client: S3Client = boto3.client(
            "s3",
            verify=False,
            endpoint_url=s3_settings.endpoint_url,
            aws_access_key_id=s3_settings.access_key_id,
            aws_secret_access_key=s3_settings.secret_access_key,
        )
        return client

    def upload_file_to_s3(
        self,
        filename: str,
        bytes_data: bytes,
        bucket_name: str,
        object_name: str | None = None,
        mime_type: str | None = None,
    ) -> str:
        """Uploads a file to an S3 bucket.

        :param filename: Name to use for the uploaded file
        :param bytes_data: File content as bytes
        :param bucket_name: Name of the S3 bucket
        :param object_name: S3 object name. If not specified then filename is used
        :param content_type: MIME type of the file.
        """
        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")
        if object_name is None:
            object_name = filename

        put_args: dict[str, Any] = {
            "Bucket": bucket_name,
            "Key": object_name,
            "Body": bytes_data,
            "ContentType": mime_type,
            "Metadata": {
                "file_name": filename,
                "content_type": mime_type,
            },
        }

        s3_client.put_object(**put_args)
        return f"s3://{bucket_name}/{object_name}"

    def load_file_from_s3(self, s3_url: str, **kwargs: Any) -> BinaryIO:
        s3_path = s3_url[5:]
        s3_components = s3_path.split("/")
        s3_bucket = s3_components[0]
        s3_key = ""
        if len(s3_components) > 1:
            s3_key = "/".join(s3_components[1:])

        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        content = response["Body"].read()
        return io.BytesIO(content)

    def remove_file_from_s3(self, s3_url: str) -> None:
        """Removes a file from an S3 bucket.

        :param s3_url: S3 URL of the file to remove
        """
        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")
        s3_path = s3_url[5:]
        s3_components = s3_path.split("/", 1)
        s3_bucket = s3_components[0]
        s3_key = s3_components[1] if len(s3_components) > 1 else ""

        s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)

    def head_object(self, bucket_name: str, key: str) -> dict[str, Any] | None:
        """Return metadata for a single S3 object, or None if it doesn't exist."""
        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=key)
            return {
                "content_length": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified"),
                "content_type": response.get(
                    "ContentType",
                    response.get("Metadata", {}).get(
                        "content_type", "application/octet-stream"
                    ),
                ),
            }
        except Exception as exc:
            # botocore ClientError with 404 → not found
            error_response = getattr(exc, "response", None)
            if isinstance(error_response, dict) and error_response.get("Error", {}).get(
                "Code"
            ) in ("404", "NoSuchKey"):
                return None
            raise

    def delete_key(self, bucket_name: str, key: str) -> bool:
        """Delete a single S3 object. Returns True if it existed, False if not."""
        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")
        existed = self.head_object(bucket_name, key) is not None
        if existed:
            s3_client.delete_object(Bucket=bucket_name, Key=key)
        return existed

    def list_objects_by_prefix(self, bucket_name: str, prefix: str) -> list[str]:
        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")

        paginator = s3_client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(str(key))
        return keys

    def generate_presigned_download_url(self, uri: str, expiration: int = 3600) -> str:
        """Generate a presigned download URL for an S3 object.

        :param uri: S3 URI (s3://bucket/path)
        :param expiration: URL validity in seconds (default: 1 hour)
        :return: Presigned URL with temporary access
        """
        s3_client = self._s3_client
        if not s3_client:
            raise RuntimeError("Failed to create S3 client")

        # Parse s3://bucket/key
        s3_path = uri[5:]  # Remove 's3://'
        bucket, key = s3_path.split("/", 1)

        # Generate presigned URL with internal endpoint
        url: str = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiration
        )

        # Replace internal endpoint with public endpoint and add path prefix
        prefix = (
            f"/{self._s3_settings.path_prefix}" if self._s3_settings.path_prefix else ""
        )
        public_endpoint = self._s3_settings.public_endpoint_url + prefix
        return url.replace(self._s3_settings.endpoint_url, public_endpoint)
