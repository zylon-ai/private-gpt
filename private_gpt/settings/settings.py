import inspect
import json
from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator

from private_gpt.settings.settings_loader import load_active_settings


class CorsSettings(BaseModel):
    """CORS configuration.

    For more details on the CORS configuration, see:
    # * https://fastapi.tiangolo.com/tutorial/cors/
    # * https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    """

    enabled: bool = Field(
        description="Flag indicating if CORS headers are set or not."
        "If set to True, the CORS headers will be set to allow all origins, methods and headers.",
        default=False,
    )
    allow_credentials: bool = Field(
        description="Indicate that cookies should be supported for cross-origin requests",
        default=False,
    )
    allow_origins: list[str] = Field(
        description="A list of origins that should be permitted to make cross-origin requests.",
        default=[],
    )
    allow_origin_regex: list[str] | None = Field(
        description="A regex string to match against origins that should be permitted to make cross-origin requests.",
        default=None,
    )
    allow_methods: list[str] = Field(
        description="A list of HTTP methods that should be allowed for cross-origin requests.",
        default=[
            "GET",
        ],
    )
    allow_headers: list[str] = Field(
        description="A list of HTTP request headers that should be supported for cross-origin requests.",
        default=[],
    )


class AuthSettings(BaseModel):
    """Authentication configuration.

    The implementation of the authentication strategy must
    """

    enabled: bool = Field(
        description="Flag indicating if authentication is enabled or not.",
        default=False,
    )
    secret: str = Field(
        description="The secret to be used for authentication. "
        "It can be any non-blank string. For HTTP basic authentication, "
        "this value should be the whole 'Authorization' header that is expected",
        repr=False,
    )


class ApiDocSettings(BaseModel):
    """Swagger configuration.

    For more details on the Swagger configuration, see:
    # * https://fastapi.tiangolo.com
    """

    enabled: bool = Field(
        description="Flag indicating if Swagger UI is enabled or not.",
        default=False,
    )
    swagger_url: str = Field(
        description="The URL for the Swagger UI.",
        default="/docs",
    )
    redoc_url: str = Field(
        description="The URL for the ReDoc UI.",
        default="/redoc",
    )
    openapi_url: str = Field(
        description="The URL for the OpenAPI schema.",
        default="/openapi.json",
    )


class UiSettings(BaseModel):
    """Static UI hosting configuration."""

    enabled: bool = Field(
        description="Flag indicating if the bundled static UI is enabled or not.",
        default=False,
    )
    path: str = Field(
        description="The URL path where the bundled static UI is mounted.",
        default="/ui",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value:
            raise ValueError("UI path cannot be empty")
        if not value.startswith("/"):
            raise ValueError("UI path must start with '/'")
        return value.rstrip("/") or "/"


class ProxySettings(BaseModel):
    """Proxy configuration for HTTP/HTTPS requests.

    Supports HTTP, HTTPS, SOCKS4, and SOCKS5 proxies with optional authentication.
    Credentials can be embedded in the URL or provided separately.
    """

    enabled: bool = Field(
        default=False,
        description="Flag indicating if proxy is enabled or not.",
    )
    http_proxy: str | None = Field(
        default=None,
        description=(
            "HTTP proxy server URL. Can include credentials in URL format. "
            "Examples: 'http://proxy.example.com:8080', 'http://user:pass@proxy.example.com:8080'"
        ),
    )
    https_proxy: str | None = Field(
        default=None,
        description=(
            "HTTPS proxy server URL. Can include credentials in URL format. "
            "Examples: 'http://proxy.example.com:8080', 'socks5://user:pass@proxy.example.com:1080'"
        ),
    )
    username: str | None = Field(
        default=None,
        description="Username for proxy authentication (overrides URL credentials)",
        repr=False,
    )
    password: str | None = Field(
        default=None,
        description="Password for proxy authentication (overrides URL credentials)",
        repr=False,
    )
    bypass: str | None = Field(
        default=None,
        description=(
            "Comma-separated list of domains/IPs to bypass proxy. "
            "Supports wildcards like '*.example.com'. "
            "Example: 'localhost,127.0.0.1,*.internal.com'"
        ),
    )

    @property
    def http_server(self) -> AnyUrl | None:
        """Computed HTTP proxy server URL with credentials."""
        if not self.http_proxy:
            return None
        return self._build_proxy_url(self.http_proxy)

    @property
    def https_server(self) -> AnyUrl | None:
        """Computed HTTPS proxy server URL with credentials."""
        if not self.https_proxy:
            return None
        return self._build_proxy_url(self.https_proxy)

    def _build_proxy_url(self, proxy_url: str) -> AnyUrl:
        """Build proxy URL with credentials if provided separately.

        Args:
            proxy_url: The base proxy URL

        Returns:
            URL with credentials injected if username/password are set
        """
        if not self.username or not self.password:
            return AnyUrl(proxy_url)

        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(proxy_url)
        username = parsed.username or self.username
        password = parsed.password or self.password

        netloc = f"{username}:{password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"

        new_url = urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

        return AnyUrl(new_url)


class SSLSettings(BaseModel):
    """SSL/TLS certificate configuration for secure connections.

    Supports custom CA certificates and certificate directories.
    """

    cert_file: str | None = Field(
        default=None,
        description=(
            "Path to custom CA certificate bundle (PEM format). "
            "Example: '/etc/ssl/certs/custom-ca-bundle.crt'"
        ),
    )
    cert_dir: str | None = Field(
        default=None,
        description=("Directory containing CA certificates. Example: '/etc/ssl/certs'"),
    )
    verify_ssl: bool = Field(
        default=True,
        description=(
            "Flag indicating if SSL certificates should be verified. "
            "Set to False to ignore certificate errors (not recommended for production)."
        ),
    )

    @field_validator("cert_file")
    @classmethod
    def validate_cert_file(cls, v: str | None) -> str | None:
        """Validate that certificate file exists and is readable."""
        if not v:
            return v

        from pathlib import Path

        cert_path = Path(v)

        if not cert_path.exists():
            raise ValueError(f"Certificate file does not exist: {v}")

        if not cert_path.is_file():
            raise ValueError(f"Certificate path is not a file: {v}")

        return v

    @field_validator("cert_dir")
    @classmethod
    def validate_cert_dir(cls, v: str | None) -> str | None:
        """Validate that certificate directory exists."""
        if not v:
            return v

        from pathlib import Path

        cert_dir_path = Path(v)

        if not cert_dir_path.exists():
            raise ValueError(f"Certificate directory does not exist: {v}")

        if not cert_dir_path.is_dir():
            raise ValueError(f"Certificate path is not a directory: {v}")

        return v


class NetworkSettings(BaseModel):
    """Network configuration including proxy and SSL settings."""

    offline_mode: bool = Field(
        default=False,
        description="Flag indicating if the application should operate in offline mode.",
    )
    proxy: ProxySettings = Field(
        default_factory=lambda: ProxySettings(),
        description="Proxy configuration for HTTP/HTTPS requests",
    )
    ssl: SSLSettings = Field(
        default_factory=lambda: SSLSettings(),
        description="SSL/TLS certificate configuration",
    )


class IngestionSettings(BaseModel):
    """Ingestion configuration.

    This configuration is used to control the ingestion of data into the system
    using non-server methods. This is useful for local development and testing;
    or to ingest in bulk from a folder.

    Please note that this configuration is not secure and should be used in
    a controlled environment only (setting right permissions, etc.).
    """

    enabled: bool = Field(
        description="Flag indicating if local ingestion is enabled or not.",
        default=False,
    )
    allow_ingest_from: list[str] = Field(
        description="A list of folders that should be permitted to make ingest requests.",
        default=[],
    )


class ServerSettings(BaseModel):
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    root_path: str = Field(
        default="",
        description="Root path for the FastAPI server",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host of PrivateGPT FastAPI server, defaults to 0.0.0.0",
    )
    port: int = Field(description="Port of PrivateGPT FastAPI server, defaults to 8080")
    cors: CorsSettings = Field(description="CORS configuration", default=CorsSettings())
    auth: AuthSettings = Field(
        description="Authentication configuration",
        default_factory=lambda: AuthSettings(enabled=False, secret="secret-key"),
    )
    network: NetworkSettings = Field(
        description="Network configuration",
        default_factory=lambda: NetworkSettings(),
    )
    api_doc: ApiDocSettings = Field(
        description="Swagger configuration",
        default_factory=lambda: ApiDocSettings(enabled=False),
    )
    ui: UiSettings = Field(
        description="Static UI hosting configuration",
        default_factory=lambda: UiSettings(),
    )
    debug_mode: bool = Field(
        description="Flag indicating if debug mode is enabled or not.",
        default=False,
    )
    max_workers: int | None = Field(
        description="The maximum number of workers to use for the server.",
        default=None,
    )

    def model_post_init(self, __context: dict[str, Any]) -> None:
        # Cast max_workers to int if it is not None
        if self.max_workers is not None:
            self.max_workers = int(self.max_workers)
            # Set max_workers to None if it is less than or equal to 0
            if self.max_workers <= 0:
                self.max_workers = None
        super().model_post_init(__context)


class FileLimitSettings(BaseModel):
    max_file_size: int = Field(
        default=100 * 1024 * 1024,
        description="The maximum file size in bytes that can be ingested.",
    )
    max_file_pages: int = Field(
        default=100,
        description="The maximum number of pages that can be ingested from a file.",
    )


class DataSettings(BaseModel):
    local_ingestion: IngestionSettings = Field(
        description="Ingestion configuration",
        default_factory=lambda: IngestionSettings(allow_ingest_from=["*"]),
    )
    local_data_folder: str = Field(
        description="Path to local storage."
        "It will be treated as an absolute path if it starts with /"
    )
    limits: FileLimitSettings = Field(
        description="File limit settings", default_factory=lambda: FileLimitSettings()
    )
    reader: str = Field(
        description="The reader selection mode to use for ingestion. "
        "Set to 'auto' to try the registered readers for the file extension in order.",
        default="auto",
    )
    enable_fake_progress: bool = Field(
        description="Flag indicating if fake progress should be enabled or not.",
        default=False,
    )
    enable_reuse_generated_nodes_before: bool = Field(
        description="Flag indicating if generated nodes should be reused when same file was ingested before.",
        default=False,
    )
    enable_vision_fallback: bool = Field(
        default=False,
        description=(
            "Retry PDF extraction with the vision reader when the primary reader "
            "raises ExtractionUnsuccessfulError. Requires a configured VLM."
        ),
    )
    enable_term_extractor: bool = Field(
        description="Flag indicating if term extraction should be enabled or not.",
        default=False,
    )
    max_num_nodes: int | None = Field(
        description="The maximum number of nodes to ingest.",
        default=None,
    )
    use_async: bool = Field(
        description="Flag indicating if async mode should be used for ingestion.",
        default=True,
    )


class RetrievalSettings(BaseModel):
    top_k: int = Field(
        default=32,
        description="The number of top results to return from the vector store.",
    )
    maximize_top_k: bool = Field(
        default=True,
        description="Flag indicating if the top k results should be maximized or not.",
    )
    max_merging_recalculations: int = Field(
        default=3,
        description="The maximum number of margin recalculations to perform.",
    )


class PreprocessTypeSettings(BaseModel):
    """Per-type preprocessing settings (extensible for future options)."""

    max_concurrency: int | None = Field(
        description="The maximum number of concurrent workers to use.",
        default=None,
    )
    return_type: Literal["user_message", "tool_result"] = Field(
        default="user_message",
        description=(
            "Where to store the preprocessed content. "
            "'user_message' appends it directly to the user message; "
            "'tool_result' carries it as a tool-use/result pair in the history."
        ),
    )


class PreprocessSettings(BaseModel):
    documents: PreprocessTypeSettings = Field(
        default_factory=lambda: PreprocessTypeSettings(),
        description="Settings for document block preprocessing.",
    )
    multimodal: PreprocessTypeSettings = Field(
        default_factory=lambda: PreprocessTypeSettings(),
        description="Settings for image/audio block preprocessing.",
    )


class ChatSettings(BaseModel):
    allow_use_default_prompt: bool = Field(
        True,
        description="Flag indicating if the chat engine should use default prompts or not.",
    )
    allow_generate_citations: bool = Field(
        True,
        description="Flag indicating if the chat engine should generate citations or not.",
    )
    allow_reasoning: bool = Field(
        True,
        description="Flag indicating if the chat engine should use reasoning or not.",
    )
    return_missing_citations: bool = Field(
        False,
        description="Flag indicating if the chat engine should return missing citations or not. Only used if `allow_generate_citations` is set to True.",
    )
    add_context_to_system_prompt: bool = Field(
        False,
        description="Flag indicating if the chat engine should add context to the system prompt or not.",
    )
    deduplicate_context_in_history: bool = Field(
        False,
        description="Flag indicating if the chat engine should deduplicate context in the chat history or not.",
    )
    force_to_return_citations: bool = Field(
        False,
        description="Flag indicating if the chat engine should force to return citations or not. Only used if `allow_generate_citations` is set to True.",
    )
    numerical_shorter_citations: bool = Field(
        False,
        description="Flag indicating if the chat engine should use numerical shorter citations or not. Only used if `allow_generate_citations` is set to True.",
    )
    maximum_context_length: int | None = Field(
        None,
        description="The maximum context length tokens that it can be used for the chat engine in context mode.",
    )
    assistant_name: str = Field("Zylon", description="The assistant name")
    assistant_description: str = Field("Zylon", description="The assistant description")
    condense_strategy: Literal["none", "condenser",] = Field(
        "none",
        description=(
            "The strategy to use for condensing the chat history.\n"
            "If `none` - do not condense the chat history.\n"
            "If `condenser` - use the last user message as the context.\n"
        ),
    )
    format_context_strategy: Literal["list", "xml", "json"] = Field(
        "list",
        description=(
            "The strategy to use for formatting the context.\n"
            "If `list` - format the context as a list of messages.\n"
            "If `xml` - format the context as XML.\n"
            "If `json` - format the context as JSON."
        ),
    )
    tldr_timeout: int | None = Field(
        None,
        description=(
            "The timeout in seconds for the TDLR processor to condense the chat history.\n"
            "If the condense strategy is set to `condenser`, this value will be used to limit the time spent on condensing."
        ),
    )
    tldr_minimum_threshold_seconds: float | None = Field(
        default=None,
        description=(
            "If the start and stop happens in less than this number, it will be ignored.\n"
            "By default, it will emit all the TLDR"
        ),
    )
    multiplexing_threshold: int | None = Field(
        None,
        description="The threshold for the number of context items to switch to multiplexing mode.",
    )
    maximum_concurrent_requests: int | None = Field(
        None,
        description="The maximum number of concurrent requests that can be handled by the chat engine.",
    )
    maximum_blob_size: int = Field(
        25 * 1024 * 1024,
        description="The maximum size in bytes of a blob that can be processed by the chat engine.",
    )
    preprocess: PreprocessSettings = Field(
        default_factory=PreprocessSettings,
        description="Concurrency settings for in-message preprocessing (documents, multimodal).",
    )

    def model_post_init(self, __context: Any) -> None:
        # If maximum_context_length is set to 0, set it to None
        if self.maximum_context_length == 0:
            self.maximum_context_length = None
        # If tldr_timeout is set to 0, set it to None
        if self.tldr_timeout == 0:
            self.tldr_timeout = None
        super().model_post_init(__context)


class StreamSettings(BaseModel):
    broker: Literal["redis", "memory"] = Field(
        default="redis",
        description="The broker to use for streaming events. If set to `redis`, the events will be streamed using Redis Streams. If set to `memory`, the events will be streamed using a memory queue.",
    )
    stream_expiration: int = Field(
        default=3600,
        description="The expiration time in seconds for the stream. "
        "If set to 0, the stream will not expire. ",
    )
    maximum_stream_length: int = Field(
        default=10000, description="The maximum length of the stream. "
    )
    stream_prefix: str = Field(
        default="stream", description="The prefix to use for the stream keys in Redis. "
    )
    status_prefix: str = Field(
        default="status", description="The prefix to use for the status keys in Redis. "
    )
    minimum_connections: int | None = Field(
        default=1,
        description="The minimum number of connections to keep in the Redis connection pool. ",
    )


class SamplingParams(BaseModel):
    seed: int = Field(
        default=0,
        description="Seed for the random number generator. If not set, a random seed will be used.",
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature is used to control the randomness of the model's output. "
        "A higher value (e.g., 1.0) will lead to more diverse text, "
        "while a lower value (e.g., 0.1) will generate more focused and conservative text.",
    )
    max_new_tokens: int = Field(
        default=1024,
        description="Maximum number of new tokens to generate. "
        "If not set, the model will generate tokens until it reaches the end of the input or the maximum context length.",
    )
    min_p: float = Field(
        default=0.1,
        description="Minimum probability for the model to generate a token. "
        "A value of 0.1 means that the model will only generate tokens with a probability of at least 0.1.",
    )
    top_p: float = Field(
        default=0.7,
        description="Top P sampling is used to reduce the impact of less probable tokens from the output. "
        "A higher value (e.g., 0.95) will lead to more diverse text, "
        "while a lower value (e.g., 0.5) will generate more focused and conservative text.",
    )
    top_k: int = Field(
        default=50,
        description="Top K sampling is used to increase of tokens to be considered for generation. "
        "A higher value (e.g., 100) will give more diverse answers, "
        "while a lower value (e.g., 10) will be more conservative.",
    )
    repetition_penalty: float = Field(
        default=1.2,
        description="Repetition penalty is used to penalize the model for generating the same token multiple times. "
        "A higher value (e.g., 1.5) will penalize repetitions more strongly, "
        "while a lower value (e.g., 0.9) will be more lenient.",
    )
    presence_penalty: float = Field(
        default=0.0,
        description="Presence penalty is used to penalize the model for generating tokens "
        "that are already present in the context. A higher value (e.g., 1.5) will penalize "
        "repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        description="Frequency penalty is used to penalize the model for generating tokens that "
        "are already present in the context. A higher value (e.g., 1.5) "
        "will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
    )


class LLMModelConfig(BaseModel):
    type: Literal["llm"] = Field(
        default="llm",
        description="The type of model. This config is specifically for 'llm' models.",
    )
    name: str = Field(
        description="Internal name identifier for the model",
    )
    mode: str = Field(
        description="The LLM provider mode to use for this model",
    )
    provider: str | None = Field(
        default=None,
        description="Provider detected during model auto-discovery.",
    )
    enabled: bool = Field(
        True,
        description="Flag indicating if the model is enabled or not.",
    )
    alias: str | None = Field(
        default=None,
        description="The alias to use for the model. This is the actual model identifier used by the provider.",
    )
    context_window: int = Field(
        8096,
        description="The maximum number of context tokens for the model.",
    )
    tokenizer: str | None = Field(
        default=None,
        description="The model id of a predefined tokenizer hosted inside a model repo on huggingface.co.",
    )
    prompt_style: str | None = Field(
        default=None,
        description="The prompt style to use for the chat engine.",
    )
    tokenizer_mode: str | None = Field(
        default=None,
        description="The tokenizer mode to use for the chat engine.",
    )
    support_image: int | None = Field(
        default=None,
        description="The number of image tokens that the model can process. If None, the model does not support image inputs.",
    )
    support_audio: int | None = Field(
        default=None,
        description="The number of audio tokens that the model can process. If None, the model does not support audio inputs.",
    )
    api_type: Literal["chat_completions", "responses"] = Field(
        default="chat_completions",
        description="The OpenAI API type to use: 'chat_completions' (default) or 'responses' (Responses API).",
    )
    support_tools: bool | None = Field(
        default=None,
        description="Flag indicating if the model supports tools. If None, the model does not support tools.",
    )
    support_reasoning: bool | None = Field(
        default=None,
        description="Flag indicating if the model supports reasoning or not.",
    )
    sampling_params: SamplingParams = Field(
        default_factory=lambda: SamplingParams(),
        description="Default sampling parameters for this model",
    )
    reasoning_sampling_params: SamplingParams = Field(
        default_factory=lambda: SamplingParams(),
        description="Sampling parameters to use when reasoning is enabled",
    )
    tags: set[str] = Field(
        default_factory=set,
        description="Tags for model categorization and selection",
    )

    class Config:
        # don't validate
        validate_assignment = False

    def __init__(self, **data: Any) -> None:
        if "tags" in data:
            data["tags"] = (
                json.loads(data["tags"])
                if isinstance(data["tags"], str)
                else data["tags"]
            )
        super().__init__(**data)


class EmbeddingModelConfig(BaseModel):
    type: Literal["embedding"] = Field(
        default="embedding",
        description="The type of model. This embedding model config always uses 'embedding'.",
    )
    name: str = Field(
        description="Internal name identifier for the model",
    )
    mode: str = Field(
        description="The embedding provider mode to use for this model",
    )
    provider: str | None = Field(
        default=None,
        description="Provider detected during model auto-discovery.",
    )
    enabled: bool = Field(
        True,
        description="Flag indicating if the model is enabled or not.",
    )
    alias: str | None = Field(
        default=None,
        description="The alias to use for the model. This is the actual model identifier used by the provider.",
    )
    context_window: int = Field(
        description="The maximum number of context tokens for the model.",
    )
    embed_dim: int | None = Field(
        default=None,
        description="Embedding vector dimension produced by this model.",
    )
    embedding_batch_size: int | None = Field(
        default=None,
        description="The batch size to use for the embedding model. "
        "If not set, the default batch size will be used.",
    )
    prefix_text: str | None = Field(
        default=None,
        description="The prefix text to use for embeddings.",
    )
    prefix_query: str | None = Field(
        default=None,
        description="The prefix query to use for embeddings.",
    )
    tags: set[str] = Field(
        default_factory=set,
        description="Tags for model categorization and selection",
    )

    def __init__(self, **data: Any) -> None:
        if "tags" in data:
            data["tags"] = (
                json.loads(data["tags"])
                if isinstance(data["tags"], str)
                else data["tags"]
            )
        super().__init__(**data)


ModelConfigType = Annotated[
    LLMModelConfig | EmbeddingModelConfig,
    Field(discriminator="type"),
]


class LLMSettings(BaseModel):
    default_model: str = Field(
        description="Default model identifier to use from the models dictionary",
    )
    auto_discover_models: bool = Field(
        True,
        description="Flag indicating if the system should automatically discover models or not",
    )


class VectorstoreSettings(BaseModel):
    database: str
    embed_dim: int = Field(
        default=1536,  # OpenAI embeddings dimension
        description="Embedding dimension for the configured vector store.",
    )
    multitenancy: Literal["collection", "logical"]
    default_collection: str | None = Field(
        None,
        description="The default collection to use for the vector store.",
    )


class NodeStoreSettings(BaseModel):
    index_store: str
    doc_store: str
    node_store: str | None = Field(
        None,
        description=(
            "The node store to use. If not set, it will use the same store as the doc store."
        ),
    )


class HuggingFaceSettings(BaseModel):
    username: str | None = Field(
        None,
        description="Huggingface username, required to create/download some models",
        repr=False,
    )
    access_token: str | None = Field(
        None,
        description="Huggingface access token, required to download some models",
        repr=False,
    )
    trust_remote_code: bool = Field(
        False,
        description="If set to True, the code from the remote model will be trusted and executed.",
    )


class EmbeddingSettings(BaseModel):
    default_model: str = Field(
        description="Default model identifier to use from the models dictionary",
    )
    auto_discover_models: bool = Field(
        True,
        description="Flag indicating if the system should automatically discover models or not",
    )
    ingest_mode: Literal["simple", "batch", "parallel"] = Field(
        "simple",
        description=(
            "The ingest mode to use for the embedding engine:\n"
            "If `simple` - ingest files sequentially and one by one. It is the historic behaviour.\n"
            "If `batch` - if multiple files, parse all the files in parallel, "
            "and send them in batch to the embedding model.\n"
            "In `pipeline` - The Embedding engine is kept as busy as possible\n"
            "If `parallel` - parse the files in parallel using multiple cores, and embedd them in parallel.\n"
            "`parallel` is the fastest mode for local setup, as it parallelize IO RW in the index.\n"
            "For modes that leverage parallelization, you can specify the number of "
            "workers to use with `count_workers`.\n"
        ),
    )
    count_workers: int | None = Field(
        2,
        description=(
            "The number of workers to use for file ingestion.\n"
            "In `batch` mode, this is the number of workers used to parse the files.\n"
            "In `parallel` mode, this is the number of workers used to parse the files and embed them.\n"
            "In `pipeline` mode, this is the number of workers that can perform embeddings.\n"
            "This is only used if `ingest_mode` is not `simple`.\n"
            "Do not go too high with this number, as it might cause memory issues. (especially in `parallel` mode)\n"
            "Do not set it higher than your number of threads of your CPU."
        ),
    )


class OpenAISettings(BaseModel):
    api_base: str = Field(
        "https://api.openai.com/v1",
        description="Base URL of OpenAI API. Example: 'https://api.openai.com/v1'.",
        repr=False,
    )
    api_key: str = Field(
        description="API key for OpenAI API.",
        repr=False,
    )
    request_timeout: float = Field(
        7200.0,
        description=(
            "Time elapsed until the OpenAI-compatible server times out the request. "
            "Default is 120s. Format is float. "
        ),
    )
    embedding_api_base: str | None = Field(
        None,
        description="Base URL of OpenAI API. Example: 'https://api.openai.com/v1'.",
    )
    embedding_api_key: str | None = Field(
        None,
        description="API key for OpenAI API. Required if `embedding_api_base` is set.",
        repr=False,
    )


class ObservabilitySettings(BaseModel):
    mode: Literal["simple", "arize_phoenix", "opik", "none"]


class QdrantSettings(BaseModel):
    location: str | None = Field(
        None,
        description=(
            "If `:memory:` - use in-memory Qdrant instance.\n"
            "If `str` - use it as a `url` parameter.\n"
        ),
    )
    url: str | None = Field(
        None,
        description=(
            "Either host or str of 'Optional[scheme], host, Optional[port], Optional[prefix]'."
        ),
    )
    port: int | None = Field(6333, description="Port of the REST API interface.")
    grpc_port: int | None = Field(6334, description="Port of the gRPC interface.")
    prefer_grpc: bool | None = Field(
        False,
        description="If `true` - use gRPC interface whenever possible in custom methods.",
    )
    https: bool | None = Field(
        None,
        description="If `true` - use HTTPS(SSL) protocol.",
    )
    api_key: str | None = Field(
        None,
        description="API key for authentication in Qdrant Cloud.",
        repr=False,
    )
    prefix: str | None = Field(
        None,
        description=(
            "Prefix to add to the REST URL path."
            "Example: `service/v1` will result in "
            "'http://localhost:6333/service/v1/{qdrant-endpoint}' for REST API."
        ),
    )
    timeout: float | None = Field(
        None,
        description="Timeout for REST and gRPC API requests.",
    )
    host: str | None = Field(
        None,
        description="Host name of Qdrant service. If url and host are None, set to 'localhost'.",
    )
    path: str | None = Field(None, description="Persistence path for QdrantLocal.")
    force_disable_check_same_thread: bool | None = Field(
        True,
        description=(
            "For QdrantLocal, force disable check_same_thread. Default: `True`"
            "Only use this if you can guarantee that you can resolve the thread safety outside QdrantClient."
        ),
    )
    hybrid_search: bool = Field(
        False, description="Flag indicating if hybrid search is enabled or not. "
    )
    distance_metric: str = Field(
        "cosine",
        description="Distance metric to use. Default: `cosine`",
    )
    hnsw_m: int = Field(
        0,
        description="The number of neighbors to search. "
        "It applies when `logical_multitenancy` is enabled. Default: `0`",
    )
    hnsw_payload_m: int = Field(
        16,
        description="The number of neighbors to search for payload."
        "It applies when `logical_multitenancy` is enabled. Default: `16`",
    )
    check_compatibility: bool = Field(
        False,
        description="Flag indicating if the system "
        "should check the compatibility of the Qdrant version or not. ",
    )

    def get_parameters(
        self, class_type: type[Any], exclude_none: bool = True
    ) -> dict[str, Any]:
        """Get the parameters that are valid for the given class type."""
        valid_keys = set(inspect.signature(class_type.__init__).parameters.keys())
        settings_dict = self.model_dump(exclude_none=exclude_none)
        return {key: value for key, value in settings_dict.items() if key in valid_keys}

    @field_validator("location", "url", "host", "path")
    @classmethod
    def empty_str_to_none(cls, v: str | None) -> str | None:
        if v == "":
            return None
        return v


class RabbitMQSettings(BaseModel):
    host: str = Field(description="RabbitMQ host")
    username: str = Field(description="RabbitMQ username")
    password: str = Field(description="RabbitMQ password", repr=False)
    ssl: bool = Field(description="RabbitMQ SSL flag (amqps or amqp)")

    @property
    def url(self) -> str:
        return f"amqp{'s' if self.ssl else ''}://{self.username}:{self.password}@{self.host}"


class DatabaseSettings(BaseModel):
    host: str = Field(description="Database host")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password", repr=False)
    provider: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Provider used to persist schema migration state.",
    )
    schema_name: str = Field(
        default="zgpt",
        alias="schema",
        description="Database schema used by application tables on postgres.",
    )
    local_path: str = Field(
        default="local_data/private_gpt/skills",
        description="Local folder path used by components configured to use sqlite.",
    )

    @property
    def url_without_protocol(self) -> str:
        """Database URL without the protocol."""
        return f"{self.username}:{self.password}@{self.host}/{self.database}"


class CelerySettings(BaseModel):
    use_workers: bool = Field(
        description="Flag indicating if workers are used or tasks are executed in the calling process",
        default=True,
    )
    broker_mode: Literal["local", "rabbitmq", "redis"] = Field(
        description="The broker to use for Celery",
        default="redis",
    )
    backend_mode: Literal["local", "rabbitmq", "redis"] = Field(
        description="The backend to use for Celery",
        default="redis",
    )
    acks_late: bool = Field(
        description="Flag indicating if tasks should be acknowledged after they are executed, rather than before",
        default=False,
    )
    soft_time_limit: int | None = Field(
        description="The soft time limit for tasks in seconds",
        default=None,
    )
    hard_time_limit: int | None = Field(
        description="The hard time limit for tasks in seconds",
        default=None,
    )
    visibility_timeout: int | None = Field(
        description="The visibility timeout for tasks in seconds",
        default=None,
    )

    def __init__(self, **data: Any) -> None:
        if "soft_time_limit" in data:
            data["soft_time_limit"] = (
                int(data["soft_time_limit"]) if data["soft_time_limit"] else None
            )
        if "hard_time_limit" in data:
            data["hard_time_limit"] = (
                int(data["hard_time_limit"]) if data["hard_time_limit"] else None
            )
        if "visibility_timeout" in data:
            data["visibility_timeout"] = (
                int(data["visibility_timeout"]) if data["visibility_timeout"] else None
            )
        super().__init__(**data)

    def validate_config(self) -> bool:
        # Check if visibility timeout is set when broker or backend is redis
        if self.broker_mode == "redis" or self.backend_mode == "redis":
            if not self.visibility_timeout:
                raise ValueError(
                    "Visibility timeout should be set when broker or backend is Redis"
                )

        if self.soft_time_limit:
            if self.hard_time_limit and self.soft_time_limit > self.hard_time_limit:
                raise ValueError(
                    "Soft time limit should be less than or equal to hard time limit"
                )

            if (
                self.visibility_timeout
                and self.visibility_timeout < self.soft_time_limit
            ):
                raise ValueError(
                    "Visibility timeout should be greater than soft time limit"
                )

        if self.hard_time_limit:
            if (
                self.visibility_timeout
                and self.visibility_timeout < self.hard_time_limit
            ):
                raise ValueError(
                    "Visibility timeout should be greater than hard time limit"
                )

        return True


class RedisSettings(BaseModel):
    host: str = Field(description="Redis host")
    username: str | None = Field(default=None, description="Redis username")
    password: str | None = Field(default=None, description="Redis password")
    database: str | None = Field(default=None, description="Redis name")

    @property
    def url(self) -> str:
        """Database URL without the protocol."""
        username_path = f"{self.username}" if self.username else ""
        password_path = f"{self.password}" if self.password else ""
        credentials = (
            f"{username_path}:{password_path}"
            if username_path and password_path
            else ""
        )
        return f"redis://{credentials}@{self.host}"

    @property
    def url_with_default_database(self) -> str:
        database_path = f"/{self.database}" if self.database else ""
        return f"{self.url}{database_path}"


class DoclingSettings(BaseModel):
    mode: Literal["api"] = "api"
    api_base: str = Field(
        "http://localhost:5001",
        description="Base URL of Docling API. Example: 'http://localhost:5001'.",
    )
    api_version: Literal["v1alpha", "v1"] = Field(
        "v1alpha",
        description=(
            "Docling API version to target. Use 'v1alpha' for older 0.x servers "
            "and 'v1' for the stable API in newer releases."
        ),
    )
    api_key: str | None = Field(
        None,
        description="Optional API key sent as the X-Api-Key header to Docling.",
        repr=False,
    )
    tenant_id: str | None = Field(
        None,
        description="Optional tenant id sent as the X-Tenant-Id header to Docling.",
    )
    num_threads: int = Field(
        4,
        description="Number of threads to use for the PDF pipeline.",
    )
    use_ocr: bool = Field(
        True,
        description="Flag indicating if OCR should be used for PDFs.",
    )
    use_gpu: bool = Field(
        True,
        description="Flag indicating if GPU should be used for OCR.",
    )
    ocr_model: Literal["easyocr", "tesseract", "rapidocr", "ocrmac"] = Field(
        "easyocr",
        description="The OCR model to use for PDFs.",
    )
    force_full_page_ocr: bool = Field(
        False,
        description="Flag indicating if full page OCR should be used.",
    )
    do_cell_matching: bool = Field(
        True,
        description="Flag indicating if cell matching should be used.",
    )
    bitmap_area_threshold: float = Field(
        0.2,
        description="Percentage of the area for a bitmap to processed with OCR.",
    )
    image_mode: Literal["embedded", "placeholder"] = Field(
        "placeholder",
        description="If we want to extract images, use ImageRefMode.EMBEDDED",
    )
    table_mode: Literal["none", "fast", "accurate"] = Field(
        "accurate",
        description="The mode to use for table extraction.",
    )
    code_mode: Literal["none", "code"] = Field(
        "none",
        description="The mode to use for code extraction.",
    )
    math_mode: Literal["none", "formula"] = Field(
        "none",
        description="The mode to use for math extraction.",
    )
    image_classifier: Literal["none", "docling"] = Field(
        "none",
        description="The image classifier to use for images.",
    )
    image_descriptor: Literal["none", "docling", "zylon"] = Field(
        "none",
        description="The image descriptor to use for images.",
    )
    langs: list[str] = Field(
        ["en-US", "fr-FR", "de-DE", "es-ES"],
        description="List of languages to use for OCR. Always pass as ISO 639-1 codes.",
    )
    use_async: bool = Field(
        False,
        description="Flag indicating if async should be used for Docling API.",
    )
    pool_interval: int | None = Field(
        10,
        description="Interval in seconds to wait before checking for new tasks.",
    )
    pool_timeout: int | None = Field(
        None,
        description="Timeout in seconds for the Docling API requests.",
    )

    failure_threshold: float = Field(
        0.3,
        description=(
            "Ratio of unmapped-glyph characters over total characters above which a "
            "document extraction is considered unsuccessful."
        ),
    )

    def __init__(self, **data: Any) -> None:
        # Convert a string in langs to a list (consider as a json)
        if "langs" in data:
            data["langs"] = (
                json.loads(data["langs"])
                if isinstance(data["langs"], str)
                else data["langs"]
            )
        super().__init__(**data)


class S3Settings(BaseModel):
    endpoint_url: str = Field(description="S3 endpoint override")
    public_endpoint_url: str = Field(description="Public S3 endpoint override")
    path_prefix: str = Field(default="", description="Prefix of the S3 path.")
    access_key_id: str = Field(description="S3 access key", repr=False)
    secret_access_key: str = Field(description="S3 secret key", repr=False)
    durable_bucket_name: str = Field(
        description="Default durable S3 bucket name for persisted application data."
    )
    temporary_bucket_name: str = Field(description="S3 temporary bucket name")


class ArizePhoenixSettings(BaseModel):
    url: str = Field(description="Arize Phoenix host")


class OpikSettings(BaseModel):
    workspace: str = Field(description="Opik workspace")
    project_name: str = Field(description="Opik project name")
    host: str | None = Field(description="Opik host", default="http://localhost:5173/")
    api_key: str | None = Field(description="Opik API key", repr=False)
    task_threads: int = Field(
        default=4, description="Number of threads to use for Opik tasks."
    )


class ZGPTSettings(BaseModel):
    api_url: str = Field(
        default="http://localhost:8001",
        description="Base URL for the ZGPT API. Example: 'http://localhost:8001'.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for ZGPT API (optional, can be obtained via login)",
        repr=False,
    )


class BackendSettings(BaseModel):
    api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the Backend API. Example: 'http://localhost:8000'.",
    )
    project_id: str | None = Field(
        default=None,
        description="Project ID for Backend API (optional, can be created if project_name is provided)",
    )
    cookie: str | None = Field(
        default=None,
        description="Session cookie for Backend API (optional, can be obtained via login)",
    )
    username: str | None = Field(
        default=None,
        description="Username for Backend API authentication (used if cookie not provided)",
    )
    password: str | None = Field(
        default=None,
        description="Password for Backend API authentication (used if cookie not provided)",
    )
    project_name: str | None = Field(
        default=None,
        description="Project name for Backend API (used to create/get project if project_id not provided)",
    )


class WebFetchSettings(BaseModel):
    enabled: bool = Field(
        default=False, description="Flag indicating if web page fetching is enabled."
    )
    timeout_seconds: int = Field(
        default=15, description="Timeout in seconds for web page fetching."
    )
    pool_size: int = Field(
        default=5, description="Number of browser instances to keep in pool."
    )
    pool_idle_timeout_seconds: int = Field(
        default=300, description="Seconds before idle browser is terminated."
    )
    pool_max_pages_per_browser: int = Field(
        default=5, description="Max concurrent pages per browser instance."
    )


class BraveSearchSettings(BaseModel):
    api_key: str = Field(description="Brave API key")
    rate_limit: float = Field(
        default=1.0, description="Minimum seconds between API requests"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


class WebSearchParams(BaseModel):
    max_timeout_seconds: int | None = Field(
        default=70, description="Maximum timeout to spend in webseach."
    )
    max_summary_timeout_seconds: int | None = Field(
        default=20,
        description="Maximum timeout to spend in webseach summary generation.",
    )
    num_references: int = Field(
        default=3,
        description="Number of web search results to use as references.",
    )
    num_concurrent_consumers: int = Field(
        default=6,
        description="Number of concurrent consumers to use for web search processing.",
    )
    max_parallel_summary: int = Field(
        default=2,
        description="Maximum number of parallel summaries to generate.",
    )
    index_weight: float | None = Field(
        default=1,
        description="Weight to give to the index score when combining with the web search score.",
    )
    token_weight: float | None = Field(
        default=1,
        description="Weight to give to the web search score when combining with the index score.",
    )
    token_exponent_penalty: float | None = Field(
        default=2,
        description="Exponent penalty to apply to the number of tokens in the web search result.",
    )


class DatabaseQuerySettings(BaseModel):
    timeout_seconds: int | None = Field(
        default=1000, description="Timeout in seconds for database querying."
    )
    batch_size: int = Field(
        default=1000,
        description="Number batch for each query.",
    )
    max_mb_result: int | None = Field(
        default=50,
        description="Maximum number of results in MB.",
    )


class WebSearchSettings(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Flag indicating if the web search is enabled.",
    )
    provider: Literal["brave", "mock"] = Field(
        default="brave",
        description="The web search provider to use.",
    )

    processor: Literal[
        "simple_text", "scraped_content", "clean_content", "best_links"
    ] = Field(
        default="simple_text",
        description="The web search proccesor to use.",
    )

    cached: bool = Field(
        default=False,
        description="Flag indicating if the web search results should be cached.",
    )

    num_links: int = Field(
        default=20,
        description="Number of link search using the provider.",
    )

    mode_quality: Literal["fast", "accurate"] = Field(
        default="fast",
        description="The web search quality mode to use when processing the links.",
    )

    fast_params: WebSearchParams = Field(
        default_factory=lambda: WebSearchParams(),
        description="Parameters to use when quality_mode is 'fast'.",
    )
    accurate_params: WebSearchParams = Field(
        default_factory=lambda: WebSearchParams(),
        description="Parameters to use when quality_mode is 'accurate'.",
    )
    context_token: int | None = Field(
        default=None,
        description="Number of tokens to reserve for tool usage in the LLM prompt.",
    )


class TasksResultsBroker(BaseModel):
    mode: Literal["none", "rabbitmq"]


class SkillSettings(BaseModel):
    """Skill management configuration."""

    database: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Database backend for skills metadata.",
    )
    storage_provider: Literal["local", "s3"] = Field(
        default="local",
        description="File storage backend for skill bundles. Uses global storage config and feature-based prefixes.",
    )
    skill_injection_mode: Literal["system_prompt", "tool_result"] = Field(
        default="system_prompt",
        description=(
            "How lazy-loaded skill instructions are injected into the model context."
        ),
    )
    maximum_loaded_skills: int = Field(
        default=1,
        ge=1,
        description="Default max number of concurrently loaded skills per chat.",
    )
    max_bundle_size_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum total size in bytes allowed for a skill bundle upload (all files combined). "
            "If None (default), no size limit is enforced."
        ),
    )
    volume_root: str | None = Field(
        default=None,
        description=(
            "Host filesystem root for skill bundle volumes. "
            "When set, VolumeContentMounter bind-mounts skill files from "
            "{volume_root}/{storage_prefix}/ into the sandbox at /mnt/skills/{name}/. "
            "In production this path should be backed by a FUSE/S3FS DaemonSet mount. "
            "When absent or empty the mounter fetches files from the storage backend "
            "and caches them locally before creating the bind-mount."
        ),
    )

    @field_validator("max_bundle_size_bytes", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class SandboxSettings(BaseModel):
    provider: str | None = Field(
        default=None,
        description="Sandbox provider registered by the application layer. "
        "Defaults to null (disabled); set explicitly to enable sandbox usage.",
    )
    timeout: int = Field(
        default=60,
        description="Default sandbox operation timeout in seconds.",
    )

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_empty_provider(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PrincipalSettings(BaseModel):
    forwarded_headers: list[str] = Field(
        default_factory=lambda: ["authorization"],
        description="HTTP request headers to capture in the Principal. "
        "When set via env var, use a comma-separated string: "
        "'authorization, x-custom-header'.",
    )
    forwarded_cookies: list[str] = Field(
        default_factory=list,
        description="HTTP request cookies to capture in the Principal. "
        "When set via env var, use a comma-separated string: "
        "'session, csrf-token'.",
    )

    @field_validator("forwarded_headers", "forwarded_cookies", mode="before")
    @classmethod
    def _parse_list(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [h.strip().lower() for h in value.split(",") if h.strip()]
        if not isinstance(value, list):
            raise ValueError("must be a list or comma-separated string")
        return [str(h).strip().lower() for h in value if h]


class BashSettings(BaseModel):
    cpu_limit_seconds: int = Field(
        default=30,
        description="RLIMIT_CPU applied to each isolated bash subprocess.",
    )
    memory_limit_mb: int = Field(
        default=512,
        description="RLIMIT_AS in MB applied to each isolated bash subprocess.",
    )
    fsize_limit_mb: int = Field(
        default=50,
        description="RLIMIT_FSIZE in MB applied to each isolated bash subprocess.",
    )
    nproc_limit: int = Field(
        default=50,
        description="RLIMIT_NPROC applied to each isolated bash subprocess.",
    )
    output_cap_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Hard cap on raw subprocess output bytes before LLM truncation.",
    )


class CodeExecutionSettings(BaseModel):
    provider: str | None = Field(
        default="local",
        description="Code execution provider registered by the application layer. "
        "Defaults to local.",
    )
    workspace_path: str | None = Field(
        default=None,
        description="Optional filesystem path used for persistent code execution workspaces. "
        "Defaults to the local data folder when unset.",
    )
    timeout: int = Field(
        default=60,
        description="Default code execution timeout in seconds.",
    )
    max_output_bytes: int = Field(
        default=1_048_576,
        description="Maximum output size to return from code execution tools.",
    )
    session_ttl_seconds: int = Field(
        default=1800,
        description="Idle TTL in seconds before a local session kernel is destroyed. "
        "Workspace files are preserved for restart.",
    )
    vfs_sessions_prefix: str = Field(
        default="sessions",
        description="Path prefix inside the storage bucket for session workspace data.",
    )
    volume_root: str | None = Field(
        default=None,
        description="Host filesystem root for session volumes. "
        "When set, session upload/output directories are created under "
        "{volume_root}/{vfs_sessions_prefix}/{session_id}/. "
        "Required by the Files API when storage_provider is 'local'.",
    )
    storage_provider: Literal["local", "s3"] = Field(
        default="local",
        description="Storage backend for session files (Files API). "
        "Use 'local' with volume_root set, or 's3' with s3.durable_bucket_name set.",
    )

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_empty_provider(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ReaderSettings(BaseModel):
    retry_number: int = Field(
        default=3,
        description="Number of retries in case of failure during reader processing.",
    )
    max_concurrent: int = Field(  # Preset
        default=4,
        description="Maximum number of concurrent workers for reader processing.",
    )
    batch_size: int = Field(
        default=8,
        description="Batch size for image processing during reader pipeline.",
    )
    max_iterations: int = Field(  # Preset
        default=2,
        description="Maximum number of iterations for processing each element.",
    )
    enable_semaphore: bool = Field(  # Preset
        default=False,
        description="Flag indicating if semaphore should be used to limit concurrent processing.",
    )


class VisionSettings(ReaderSettings):
    enable_evaluation: bool = Field(  # Preset
        default=True,
        description="Flag indicating if validation of extracted information is enabled.",
    )

    mode: Literal["none", "lite", "deep"] = Field(  # Preset
        default="none",
        description="The vision processing mode to use.",
    )

    @property
    def is_enabled(self) -> bool:
        return self.mode != "none"

    def get_vision_mode(self, vllm_enabled: bool = False) -> str:
        return self.mode if vllm_enabled else "none"

    @property
    def active_modes(self) -> str:
        return self.mode if self.is_enabled else "none"


class TransformationReadersSettings(BaseModel):
    vision: VisionSettings = Field(default_factory=VisionSettings)

    @property
    def is_enabled(self) -> bool:
        # Return true if any of the readers is enabled (for the moment
        # we only have vision, but in the future we can add more readers here)
        return self.vision.is_enabled


class TransformationSettings(BaseModel):
    pptx: TransformationReadersSettings = Field(
        default_factory=lambda: TransformationReadersSettings(),
        description="Settings for PPTX file processing during ingestion.",
    )

    docling: TransformationReadersSettings = Field(
        default_factory=lambda: TransformationReadersSettings(),
        description="Settings for Docling file processing during ingestion.",
    )

    vision_documents: TransformationReadersSettings = Field(
        default_factory=lambda: TransformationReadersSettings(),
        description="Settings for vision processing of documents during ingestion.",
    )


class SemaphoreSettings(BaseModel):
    mode: Literal["memory", "redis"] = Field(
        default="memory",
        description="The backend to use for the semaphore.",
    )


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    server: ServerSettings
    data: DataSettings
    chat: ChatSettings
    stream: StreamSettings
    retrieval: RetrievalSettings
    observability: ObservabilitySettings
    models: list[ModelConfigType]
    llm: LLMSettings
    embedding: EmbeddingSettings
    huggingface: HuggingFaceSettings
    openai: OpenAISettings
    docling: DoclingSettings
    vectorstore: VectorstoreSettings
    node_store: NodeStoreSettings
    qdrant: QdrantSettings
    rabbitmq: RabbitMQSettings
    database: DatabaseSettings
    celery: CelerySettings
    redis: RedisSettings
    tasks_results_broker: TasksResultsBroker
    s3: S3Settings
    phoenix: ArizePhoenixSettings
    opik: OpikSettings
    principal: PrincipalSettings
    sandbox: SandboxSettings
    bash: BashSettings
    code_execution: CodeExecutionSettings
    web_fetch: WebFetchSettings
    web_search: WebSearchSettings
    database_query: DatabaseQuerySettings
    brave: BraveSearchSettings
    skills: SkillSettings
    transformation: TransformationSettings
    semaphore: SemaphoreSettings


"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_settings = load_active_settings()

"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_typed_settings = Settings(**unsafe_settings)


def settings() -> Settings:
    """Get the current loaded settings from the DI container.

    This method exists to keep compatibility with the existing code,
    that require global access to the settings.

    For regular components use dependency injection instead.
    """
    from private_gpt.di import get_global_injector

    return get_global_injector().get(Settings)
