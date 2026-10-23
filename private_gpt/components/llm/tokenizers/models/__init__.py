"""Model Discovery, Cache, and Download System.

Clear separation of responsibilities:
- model_cache.py: All cache utilities (finding, checking, configuration)
- model_downloader.py: Only downloading functions (network I/O)
- model_discovery.py: Entry point, orchestration, and CLI
- auto_discovery.py: Decorator for automatic model resolution

Public API:
- discover_model: Main async entry point for model resolution
- configure_model_path: Async path resolver for offline backends
- auto_discover_model: Decorator for automatic model resolution
"""

# Decorator
from private_gpt.components.llm.tokenizers.models.auto_discovery import (
    auto_discover_model,
)

# Cache utilities
from private_gpt.components.llm.tokenizers.models.model_cache import (
    configure_model_path,
    find_local_cache_model,
    find_local_model,
    find_repo_candidates,
    has_all_safetensors,
    has_tokenizer_files,
    validate_model_path,
)
from private_gpt.components.llm.tokenizers.models.model_discovery import (
    discover_model,
)

# Main entry points
from private_gpt.components.llm.tokenizers.models.model_discovery import (
    main as cli_main,
)

# Download functions
from private_gpt.components.llm.tokenizers.models.model_downloader import (
    download_from_hf,
    download_model,
)

__all__ = [
    "auto_discover_model",
    "cli_main",
    "configure_model_path",
    "discover_model",
    "download_from_hf",
    "download_model",
    "find_local_cache_model",
    "find_local_model",
    "find_repo_candidates",
    "has_all_safetensors",
    "has_tokenizer_files",
    "validate_model_path",
]
