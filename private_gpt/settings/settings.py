from typing import Literal

from pydantic import BaseModel, Field

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
    allow_origin_regex: list[str] = Field(
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
        "this value should be the whole 'Authorization' header that is expected"
    )


class ServerSettings(BaseModel):
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    port: int = Field(description="Port of PrivateGPT FastAPI server, defaults to 8001")
    cors: CorsSettings = Field(
        description="CORS configuration", default=CorsSettings(enabled=False)
    )
    auth: AuthSettings = Field(
        description="Authentication configuration",
        default_factory=lambda: AuthSettings(enabled=False, secret="secret-key"),
    )


class DataSettings(BaseModel):
    local_data_folder: str = Field(
        description="Path to local storage."
        "It will be treated as an absolute path if it starts with /"
    )


class LLMSettings(BaseModel):
    mode: Literal[
        "llamacpp", "openai", "openailike", "azopenai", "sagemaker", "mock", "ollama"
    ]
    max_new_tokens: int = Field(
        256,
        description="The maximum number of token that the LLM is authorized to generate in one completion.",
    )
    context_window: int = Field(
        3900,
        description="The maximum number of context tokens for the model.",
    )
    tokenizer: str = Field(
        None,
        description="The model id of a predefined tokenizer hosted inside a model repo on "
        "huggingface.co. Valid model ids can be located at the root-level, like "
        "`bert-base-uncased`, or namespaced under a user or organization name, "
        "like `HuggingFaceH4/zephyr-7b-beta`. If not set, will load a tokenizer matching "
        "gpt-3.5-turbo LLM.",
    )
    temperature: float = Field(
        0.1,
        description="The temperature of the model. Increasing the temperature will make the model answer more creatively. A value of 0.1 would be more factual.",
    )


class VectorstoreSettings(BaseModel):
    database: Literal["chroma", "qdrant", "postgres"]


class NodeStoreSettings(BaseModel):
    database: Literal["simple", "postgres"]


class LlamaCPPSettings(BaseModel):
    llm_hf_repo_id: str
    llm_hf_model_file: str
    prompt_style: Literal["default", "llama2", "tag", "mistral", "chatml"] = Field(
        "llama2",
        description=(
            "The prompt style to use for the chat engine. "
            "If `default` - use the default prompt style from the llama_index. It should look like `role: message`.\n"
            "If `llama2` - use the llama2 prompt style from the llama_index. Based on `<s>`, `[INST]` and `<<SYS>>`.\n"
            "If `tag` - use the `tag` prompt style. It should look like `<|role|>: message`. \n"
            "If `mistral` - use the `mistral prompt style. It shoudl look like <s>[INST] {System Prompt} [/INST]</s>[INST] { UserInstructions } [/INST]"
            "`llama2` is the historic behaviour. `default` might work better with your custom models."
        ),
    )

    tfs_z: float = Field(
        1.0,
        description="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting.",
    )
    top_k: int = Field(
        40,
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
    )
    top_p: float = Field(
        0.9,
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
    )
    repeat_penalty: float = Field(
        1.1,
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
    )


class HuggingFaceSettings(BaseModel):
    embedding_hf_model_name: str = Field(
        description="Name of the HuggingFace model to use for embeddings"
    )
    access_token: str = Field(
        None,
        description="Huggingface access token, required to download some models",
    )


class EmbeddingSettings(BaseModel):
    mode: Literal["huggingface", "openai", "azopenai", "sagemaker", "ollama", "mock"]
    ingest_mode: Literal["simple", "batch", "parallel", "pipeline"] = Field(
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
    count_workers: int = Field(
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
    embed_dim: int = Field(
        384,
        description="The dimension of the embeddings stored in the Postgres database",
    )


class SagemakerSettings(BaseModel):
    llm_endpoint_name: str
    embedding_endpoint_name: str


class OpenAISettings(BaseModel):
    api_base: str = Field(
        None,
        description="Base URL of OpenAI API. Example: 'https://api.openai.com/v1'.",
    )
    api_key: str
    model: str = Field(
        "gpt-3.5-turbo",
        description="OpenAI Model to use. Example: 'gpt-4'.",
    )


class OllamaSettings(BaseModel):
    api_base: str = Field(
        "http://localhost:11434",
        description="Base URL of Ollama API. Example: 'https://localhost:11434'.",
    )
    embedding_api_base: str = Field(
        "http://localhost:11434",
        description="Base URL of Ollama embedding API. Example: 'https://localhost:11434'.",
    )
    llm_model: str = Field(
        None,
        description="Model to use. Example: 'llama2-uncensored'.",
    )
    embedding_model: str = Field(
        None,
        description="Model to use. Example: 'nomic-embed-text'.",
    )
    keep_alive: str = Field(
        "5m",
        description="Time the model will stay loaded in memory after a request. examples: 5m, 5h, '-1' ",
    )
    tfs_z: float = Field(
        1.0,
        description="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting.",
    )
    num_predict: int = Field(
        None,
        description="Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)",
    )
    top_k: int = Field(
        40,
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
    )
    top_p: float = Field(
        0.9,
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
    )
    repeat_last_n: int = Field(
        64,
        description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
    )
    repeat_penalty: float = Field(
        1.1,
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
    )
    request_timeout: float = Field(
        120.0,
        description="Time elapsed until ollama times out the request. Default is 120s. Format is float. ",
    )


class AzureOpenAISettings(BaseModel):
    api_key: str
    azure_endpoint: str
    api_version: str = Field(
        "2023_05_15",
        description="The API version to use for this operation. This follows the YYYY-MM-DD format.",
    )
    embedding_deployment_name: str
    embedding_model: str = Field(
        "text-embedding-ada-002",
        description="OpenAI Model to use. Example: 'text-embedding-ada-002'.",
    )
    llm_deployment_name: str
    llm_model: str = Field(
        "gpt-35-turbo",
        description="OpenAI Model to use. Example: 'gpt-4'.",
    )


class UISettings(BaseModel):
    enabled: bool
    path: str
    default_chat_system_prompt: str = Field(
        None,
        description="The default system prompt to use for the chat mode.",
    )
    default_query_system_prompt: str = Field(
        None, description="The default system prompt to use for the query mode."
    )
    delete_file_button_enabled: bool = Field(
        True, description="If the button to delete a file is enabled or not."
    )
    delete_all_files_button_enabled: bool = Field(
        False, description="If the button to delete all files is enabled or not."
    )


class RerankSettings(BaseModel):
    enabled: bool = Field(
        False,
        description="This value controls whether a reranker should be included in the RAG pipeline.",
    )
    model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
        description="Rerank model to use. Limited to SentenceTransformer cross-encoder models.",
    )
    top_n: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline.",
    )


class RagSettings(BaseModel):
    similarity_top_k: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline or considered for reranking if enabled.",
    )
    similarity_value: float = Field(
        None,
        description="If set, any documents retrieved from the RAG must meet a certain match score. Acceptable values are between 0 and 1.",
    )
    rerank: RerankSettings


class PostgresSettings(BaseModel):
    host: str = Field(
        "localhost",
        description="The server hosting the Postgres database",
    )
    port: int = Field(
        5432,
        description="The port on which the Postgres database is accessible",
    )
    user: str = Field(
        "postgres",
        description="The user to use to connect to the Postgres database",
    )
    password: str = Field(
        "postgres",
        description="The password to use to connect to the Postgres database",
    )
    database: str = Field(
        "postgres",
        description="The database to use to connect to the Postgres database",
    )
    schema_name: str = Field(
        "public",
        description="The name of the schema in the Postgres database to use",
    )


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


class Settings(BaseModel):
    server: ServerSettings
    data: DataSettings
    ui: UISettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    llamacpp: LlamaCPPSettings
    huggingface: HuggingFaceSettings
    sagemaker: SagemakerSettings
    openai: OpenAISettings
    ollama: OllamaSettings
    azopenai: AzureOpenAISettings
    vectorstore: VectorstoreSettings
    nodestore: NodeStoreSettings
    rag: RagSettings
    qdrant: QdrantSettings | None = None
    postgres: PostgresSettings | None = None


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
    from private_gpt.di import global_injector

    return global_injector.get(Settings)
