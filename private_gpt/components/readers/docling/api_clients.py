import asyncio
import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Self

import aiohttp
from pydantic import BaseModel, Field

from private_gpt.components.readers.docling.common import (
    EMBEDDED_IMAGES,
    calculate_file_priority,
    get_ocr_langs,
)
from private_gpt.settings.settings import DoclingSettings, settings
from private_gpt.utils.retry import retry

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

_MAX_RETRIES = 5
_JITTER = (5.0, 15.0)
RETRYABLE_EXCEPTIONS = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
    asyncio.TimeoutError,
)


class DoclingConfig(DoclingSettings):
    has_image_multimodal_model: bool = Field(
        description="Whether the Docling server has a multimodal model", default=False
    )


class ResourceNotFoundError(Exception):
    """Exception raised when a requested resource is not found (HTTP 404)."""

    def __init__(self, resource: str):
        self.resource = resource
        self.status_code = 404
        super().__init__(f"Resource not found: {resource}")


class _TaskStatusResponse(BaseModel):
    task_id: str
    task_status: str
    task_position: int | None = None


class _DoclingFileSource(BaseModel):
    base64_string: str = Field(...)
    filename: str = Field(...)


class _DoclingSource(_DoclingFileSource):
    kind: Literal["file"] = Field(default="file")


class _DoclingApiTarget(BaseModel):
    kind: Literal["zip"] = Field(default="zip")


class _DoclingApiDocumentOptionsInput(BaseModel):
    from_formats: list[str] | None = Field(None)
    to_formats: list[str] | None = Field(None)
    pdf_backend: str = Field("dlparse_v2")
    do_ocr: bool | None = Field(None)
    force_ocr: bool = Field(False)
    ocr_engine: str | None = Field(None)
    ocr_lang: list[str] | None = Field(None)
    do_table_structure: bool = Field(True)
    table_mode: str | None = Field(None)
    table_cell_matching: bool | None = Field(None)
    page_range: tuple[int, int] | None = Field(None)
    image_export_mode: str | None = Field(None)
    include_images: bool | None = Field(None)
    images_scale: float = Field(2.0)
    do_code_enrichment: bool = Field(False)
    do_formula_enrichment: bool = Field(False)
    do_picture_classification: bool = Field(False)
    do_picture_description: bool = Field(False)
    abort_on_error: bool = Field(True)
    return_as_file: bool | None = Field(None)

    def enable_for_image_multimodality(self) -> Self:
        self.do_ocr = True
        self.include_images = True
        self.image_export_mode = "embedded"
        self.do_picture_classification = False
        self.do_picture_description = False
        return self


class _DoclingApiDocumentInput(BaseModel):
    file_sources: list[_DoclingFileSource] = Field(default_factory=list)
    options: _DoclingApiDocumentOptionsInput = Field(
        description="Options for the document conversion"
    )
    priority: int | None = Field(
        0, description="Priority of the task, lower means more priority"
    )


class _DoclingApiDocumentOutput(BaseModel):
    md_content: str | None = Field("")
    json_content: dict[str, Any] | None = Field({})
    html_content: str | None = Field("")
    text_content: str | None = Field("")
    doctags_content: str | None = Field("")


class DoclingApiOutputModel(BaseModel):
    document: _DoclingApiDocumentOutput = Field(default=...)
    status: str = Field(...)
    processing_time: float = Field(0.0)
    timings: dict[str, Any] = Field({})
    errors: list[str] = Field([])


def get_value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _prepare_conversion_options(
    docling_settings: DoclingConfig,
    api_version: Literal["v1alpha", "v1"],
    from_format: list[str] | None = None,
    to_formats: list[str] | None = None,
    do_ocr: bool | None = None,
    force_ocr: bool | None = None,
    ocr_engine: str | None = None,
    ocr_lang: list[str] | None = None,
    pdf_backend: str | None = None,
    table_mode: str | None = None,
    do_table_structure: bool = True,
    page_range: tuple[int, int] | None = None,
    pages: int | None = None,
    include_images: bool | None = None,
    images_scale: float | None = None,
    do_code_enrichment: bool = False,
    do_formula_enrichment: bool = False,
    do_picture_classification: bool = False,
    do_picture_description: bool = False,
    abort_on_error: bool = True,
    return_as_file: bool = False,
    **kwargs: Any,
) -> _DoclingApiDocumentOptionsInput:
    config = _DoclingApiDocumentOptionsInput(
        from_formats=from_format,
        to_formats=to_formats,
        # Backend
        pdf_backend=pdf_backend or "dlparse_v2",
        # OCR config
        do_ocr=get_value_or_default(do_ocr, docling_settings.use_ocr),
        force_ocr=get_value_or_default(force_ocr, docling_settings.force_full_page_ocr),
        ocr_engine=get_value_or_default(ocr_engine, docling_settings.ocr_model),
        ocr_lang=ocr_lang or get_ocr_langs(),
        # Tables
        table_mode=get_value_or_default(table_mode, docling_settings.table_mode),
        do_table_structure=get_value_or_default(
            do_table_structure, docling_settings.table_mode != "none"
        ),
        table_cell_matching=(
            docling_settings.do_cell_matching if api_version == "v1" else None
        ),
        page_range=(
            page_range
            if page_range is not None
            else (1, pages)
            if api_version == "v1" and pages
            else None
        ),
        # Images configuration
        image_export_mode=docling_settings.image_mode,
        include_images=get_value_or_default(include_images, EMBEDDED_IMAGES),
        images_scale=get_value_or_default(images_scale, 2.0),
        # Code enrichment
        do_code_enrichment=get_value_or_default(
            do_code_enrichment, docling_settings.code_mode != "none"
        ),
        # Formula enrichment
        do_formula_enrichment=get_value_or_default(
            do_formula_enrichment, docling_settings.math_mode != "none"
        ),
        # Picture classification
        do_picture_classification=get_value_or_default(
            do_picture_classification, docling_settings.image_classifier == "docling"
        ),
        # Picture description
        do_picture_description=get_value_or_default(
            do_picture_description, docling_settings.image_descriptor == "docling"
        ),
        # Other config
        abort_on_error=abort_on_error,
        return_as_file=return_as_file if api_version == "v1alpha" else None,
    )

    # If the server has a multimodal model, adjust settings accordingly
    if docling_settings.has_image_multimodal_model:
        config = config.enable_for_image_multimodality()

    return config


class BaseDoclingClient(ABC):
    @abstractmethod
    async def convert_from_bytes(
        self, file_name: str, file_bytes: bytes, **kwargs: Any
    ) -> DoclingApiOutputModel:
        pass


def _build_api_base_url(api_base: str, api_version: Literal["v1alpha", "v1"]) -> str:
    return f"{api_base.rstrip('/')}/{api_version}"


def _build_request_headers(docling_settings: DoclingConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    if docling_settings.api_key:
        headers["X-Api-Key"] = docling_settings.api_key
    if docling_settings.tenant_id:
        headers["X-Tenant-Id"] = docling_settings.tenant_id
    return headers


def _build_source_request_payload(
    *,
    api_version: Literal["v1alpha", "v1"],
    options: _DoclingApiDocumentOptionsInput,
    file_name: str,
    file_base64: str,
    priority: int,
    return_as_file: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "options": options.model_dump(exclude_none=True),
        "priority": priority,
    }

    if api_version == "v1":
        payload["sources"] = [
            _DoclingSource(base64_string=file_base64, filename=file_name).model_dump(
                exclude_none=True
            )
        ]
        if return_as_file:
            payload["target"] = _DoclingApiTarget().model_dump()
        return payload

    payload["file_sources"] = [
        _DoclingFileSource(base64_string=file_base64, filename=file_name).model_dump(
            exclude_none=True
        )
    ]
    return payload


class DoclingClient(BaseModel, BaseDoclingClient):
    docling_settings: DoclingConfig = Field(description="Docling settings")
    base_url: str = Field(description="Base URL for the Docling API")

    def __init__(
        self,
        settings: DoclingConfig,
        api_base: str | None = None,
    ):
        super().__init__(
            docling_settings=settings,
            base_url=_build_api_base_url(
                api_base or settings.api_base, settings.api_version
            ),
        )

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def _convert_file_with_multipart(
        self,
        file_name: str,
        file_bytes: bytes,
        **kwargs: Any,
    ) -> DoclingApiOutputModel:
        file_name = file_name
        headers = _build_request_headers(self.docling_settings)
        form_data = _prepare_conversion_options(
            self.docling_settings,
            api_version=self.docling_settings.api_version,
            **kwargs,
        ).model_dump(exclude_none=True)

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "files",
                file_bytes,
                filename=file_name,
                content_type="application/octet-stream",
            )

            for key, value in form_data.items():
                if isinstance(value, list):
                    for item in value:
                        data.add_field(key, str(item))
                elif isinstance(value, dict):
                    data.add_field(key, json.dumps(value))
                else:
                    data.add_field(key, str(value))

            if self.docling_settings.api_version == "v1" and kwargs.get(
                "return_as_file", False
            ):
                data.add_field("target_type", "zip")

            async with session.post(
                f"{self.base_url}/convert/file", data=data, headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return DoclingApiOutputModel(**result)

    @retry(
        is_async=True,
        tries=_MAX_RETRIES,
        jitter=_JITTER,
        logger=logger,
        exceptions=RETRYABLE_EXCEPTIONS,
    )
    async def _convert_file_using_base64(
        self,
        file_name: str,
        file_bytes: bytes,
        **kwargs: Any,
    ) -> DoclingApiOutputModel:
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        headers = _build_request_headers(self.docling_settings)
        priority = int(
            kwargs.get("priority", await calculate_file_priority(file_bytes, **kwargs))
        )
        return_as_file = bool(kwargs.get("return_as_file", False))

        payload = _build_source_request_payload(
            api_version=self.docling_settings.api_version,
            options=_prepare_conversion_options(
                self.docling_settings,
                api_version=self.docling_settings.api_version,
                **kwargs,
            ),
            file_name=file_name,
            file_base64=file_base64,
            priority=priority,
            return_as_file=return_as_file,
        )

        async with aiohttp.ClientSession() as session, session.post(
            f"{self.base_url}/convert/source", json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return DoclingApiOutputModel(**result)

    async def convert_from_bytes(
        self, file_name: str, file_bytes: bytes, **kwargs: Any
    ) -> DoclingApiOutputModel:
        result: DoclingApiOutputModel = await self._convert_file_using_base64(
            file_name, file_bytes, **kwargs
        )
        return result


class AsyncDoclingClient(BaseModel, BaseDoclingClient):
    docling_settings: DoclingConfig = Field(description="Docling settings")
    base_url: str = Field(description="Base URL for the Docling API")
    poll_interval: float = Field(description="Polling interval in seconds", default=5.0)
    poll_timeout: float | None = Field(
        description="Polling timeout in seconds", default=None
    )

    def __init__(
        self,
        settings: DoclingConfig,
        api_base: str | None = None,
        poll_interval: float | None = None,
        poll_timeout: float | None = None,
    ):
        api_base = api_base or settings.api_base
        poll_interval = poll_interval or settings.pool_interval
        poll_timeout = poll_timeout or settings.pool_timeout

        if not api_base or not poll_interval:
            raise ValueError(
                "API base URL and poll interval must be provided in async mode"
            )

        super().__init__(
            docling_settings=settings,
            base_url=_build_api_base_url(
                api_base or settings.api_base, settings.api_version
            ),
            poll_interval=poll_interval or settings.pool_interval,
            poll_timeout=poll_timeout or settings.pool_timeout,
        )

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def _submit_task(
        self, file_name: str, file_bytes: bytes, **kwargs: Any
    ) -> str:
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        headers = _build_request_headers(self.docling_settings)
        priority = int(
            kwargs.get("priority", await calculate_file_priority(file_bytes, **kwargs))
        )
        return_as_file = bool(kwargs.get("return_as_file", False))
        payload = _build_source_request_payload(
            api_version=self.docling_settings.api_version,
            options=_prepare_conversion_options(
                self.docling_settings,
                api_version=self.docling_settings.api_version,
                **kwargs,
            ),
            file_name=file_name,
            file_base64=file_base64,
            priority=priority,
            return_as_file=return_as_file,
        )

        async with aiohttp.ClientSession() as session, session.post(
            f"{self.base_url}/convert/source/async", json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            result = await response.json()
            task_status = _TaskStatusResponse(**result)
            return task_status.task_id

    @retry(
        is_async=True,
        tries=_MAX_RETRIES,
        jitter=_JITTER,
        logger=logger,
        exceptions=RETRYABLE_EXCEPTIONS,
    )
    async def _poll_task_status(
        self, task_id: str, wait: float = 0.0
    ) -> _TaskStatusResponse:
        headers = _build_request_headers(self.docling_settings)
        async with aiohttp.ClientSession() as session:
            params = {"wait": wait} if wait > 0 else {}
            async with session.get(
                f"{self.base_url}/status/poll/{task_id}", params=params, headers=headers
            ) as response:
                if response.status == 404:
                    # If the task is not found, it means that the server
                    # was reset or the task was deleted.
                    raise ResourceNotFoundError(task_id)

                response.raise_for_status()
                result = await response.json()
                return _TaskStatusResponse(**result)

    @retry(
        is_async=True,
        tries=_MAX_RETRIES,
        jitter=_JITTER,
        logger=logger,
        exceptions=RETRYABLE_EXCEPTIONS,
    )
    async def _get_task_result(self, task_id: str) -> DoclingApiOutputModel:
        headers = _build_request_headers(self.docling_settings)
        async with aiohttp.ClientSession() as session, session.get(
            f"{self.base_url}/result/{task_id}", headers=headers
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return DoclingApiOutputModel(**result)

    async def _wait_for_completion(self, task_id: str) -> DoclingApiOutputModel:
        start_time = time.time()
        while not self.poll_timeout or time.time() - start_time < self.poll_timeout:
            status = await self._poll_task_status(task_id)
            if status.task_status == "success":
                task_result: DoclingApiOutputModel = await self._get_task_result(
                    task_id
                )
                return task_result
            if status.task_status in ["failure", "skipped"]:
                raise ValueError(f"Task failed with status: {status.task_status}")

            await asyncio.sleep(self.poll_interval)

        raise TimeoutError(f"Task did not complete within {self.poll_timeout} seconds")

    @retry(
        is_async=True,
        tries=_MAX_RETRIES,
        jitter=_JITTER,
        logger=logger,
        exceptions=ResourceNotFoundError,
    )
    async def convert_from_bytes(
        self, file_name: str, file_bytes: bytes, **kwargs: Any
    ) -> DoclingApiOutputModel:
        task_id = await self._submit_task(file_name, file_bytes, **kwargs)
        return await self._wait_for_completion(task_id)


class DoclingClientFactory:
    @staticmethod
    def create(config: DoclingConfig, async_client: bool = False) -> BaseDoclingClient:
        return AsyncDoclingClient(config) if async_client else DoclingClient(config)
